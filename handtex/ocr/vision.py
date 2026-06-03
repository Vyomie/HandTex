"""Vision-model OCR backend (the real recognizer).

Sends the whiteboard image to a Claude vision model and asks it to return the
same JSON schema the :class:`~handtex.ocr.mock.MockBackend` consumes — a list
of glyphs, each with a canonical ``name``, literal ``text`` and a pixel
``bbox``. Keeping the contract identical means layout/LaTeX/render never know
which backend produced the glyphs.

Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` in the
environment. Both are imported/checked lazily so offline runs never touch
this module.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import List

from ..types import RecognizedGlyph
from .base import OCRBackend
from .mock import glyphs_from_json
from ..font import symbols

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = """You are a precise handwriting OCR engine for math and text on a \
whiteboard. Identify every individual glyph (letter, digit, math symbol, \
punctuation mark) and report its tight pixel bounding box.

Return ONLY a JSON object of the form:
{"glyphs": [{"name": <symbol-name>, "text": <character>, \
"bbox": [x, y, w, h], "confidence": <0..1>}, ...]}

Rules:
- bbox is [x, y, width, height] in PIXELS, origin at the TOP-LEFT of the image.
- Report each glyph SEPARATELY, even small marks (commas, apostrophes, \
primes, exponents). Do NOT pre-interpret superscripts/subscripts — just give \
accurate boxes; a later stage infers position from geometry.
- Prefer canonical names from the provided vocabulary when one applies; \
otherwise use the literal character as both name and text.
- Order glyphs left-to-right, top-to-bottom."""


def _vocab_hint() -> str:
    pairs = [f"{s.name}={s.char}" for s in symbols.all_symbols()]
    return "Known symbol names: " + ", ".join(pairs)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    # Strip ```json fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model response")
    return json.loads(text[start : end + 1])


class VisionBackend(OCRBackend):
    name = "vision"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def recognize(self, image_path: str) -> List[RecognizedGlyph]:
        if not self._api_key:
            raise RuntimeError(
                "VisionBackend needs ANTHROPIC_API_KEY (or pass api_key=...). "
                "Use the 'mock' backend for offline runs."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "VisionBackend needs the 'anthropic' package: pip install anthropic"
            ) from exc

        with open(image_path, "rb") as fh:
            img_b64 = base64.standard_b64encode(fh.read()).decode("ascii")
        media_type = mimetypes.guess_type(image_path)[0] or "image/png"

        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=_SYSTEM + "\n\n" + _vocab_hint(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": "Recognize all glyphs."},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return glyphs_from_json(_extract_json(text))
