"""Core data types shared across the pipeline.

The whole pipeline is a series of transforms over these structures::

    image --ocr--> list[RecognizedGlyph]
          --layout--> Document
          --latex/render--> .tex / .png
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in image pixel coordinates (y grows down)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def vertical_overlap(self, other: "BBox") -> float:
        """Height of the vertical overlap between two boxes (>=0)."""
        return max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))


@dataclass
class RecognizedGlyph:
    """One symbol returned by an OCR backend.

    ``name`` is the canonical symbol name from :mod:`handtex.font.symbols`
    (e.g. ``"alpha"``, ``"x"``, ``"plus"``). ``text`` is the literal character.
    """

    name: str
    text: str
    bbox: BBox
    confidence: float = 1.0


class Role(str, Enum):
    """Vertical role of a token relative to its line / base glyph."""

    BASE = "base"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"


@dataclass
class Token:
    """A laid-out glyph: the recognized glyph plus its resolved role."""

    glyph: RecognizedGlyph
    role: Role = Role.BASE
    # Super/subscripts that attach to this base, in reading order.
    superscript: List["Token"] = field(default_factory=list)
    subscript: List["Token"] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.glyph.name

    @property
    def text(self) -> str:
        return self.glyph.text


@dataclass
class Line:
    """A horizontal line of base tokens (super/subscripts nested inside)."""

    tokens: List[Token] = field(default_factory=list)
    baseline: Optional[float] = None
    x_height: Optional[float] = None


@dataclass
class Document:
    """The structured result of reading a whiteboard."""

    lines: List[Line] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(line.tokens for line in self.lines)
