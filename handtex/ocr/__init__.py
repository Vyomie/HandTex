"""Pluggable OCR backends.

A backend turns a whiteboard image into ``list[RecognizedGlyph]`` (each with
a bounding box). The rest of the pipeline never cares which backend ran, so
you can swap the offline :class:`MockBackend` for a real vision model without
touching layout, LaTeX or rendering.
"""

from .base import OCRBackend
from .mock import MockBackend

__all__ = ["OCRBackend", "MockBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> OCRBackend:
    """Factory: resolve a backend by name (``"mock"``, ``"local"`` or ``"vision"``)."""
    name = (name or "mock").lower()
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "local":
        from .local import LocalBackend  # imported lazily; needs torch + weights
        return LocalBackend(**kwargs)
    if name == "vision":
        from .vision import VisionBackend  # imported lazily; needs a network/SDK
        return VisionBackend(**kwargs)
    raise ValueError(f"unknown OCR backend: {name!r} (try 'mock', 'local' or 'vision')")
