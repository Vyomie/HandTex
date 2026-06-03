"""The OCR backend contract."""

from __future__ import annotations

import abc
from typing import List

from ..types import RecognizedGlyph


class OCRBackend(abc.ABC):
    """Recognize handwritten glyphs in a whiteboard image.

    Implementations must return glyphs with **pixel bounding boxes** in the
    coordinate space of the input image (origin top-left, y grows down). The
    layout engine relies on those boxes for baseline / super-subscript logic,
    so position matters as much as identity.
    """

    name: str = "base"

    @abc.abstractmethod
    def recognize(self, image_path: str) -> List[RecognizedGlyph]:
        """Return recognized glyphs for the image at ``image_path``."""
        raise NotImplementedError
