"""Small local OCR model: connected-component segmentation + a tiny CNN.

Imports are kept lazy (the heavy ``torch`` dependency only loads when you
actually use this backend), so the rest of HandTex runs without it.
"""

from .backend import LocalBackend

__all__ = ["LocalBackend"]
