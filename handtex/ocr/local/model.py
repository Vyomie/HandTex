"""The small glyph-classifier CNN.

Tiny by design (~3 conv blocks): it classifies one already-segmented,
normalized glyph, because the layout engine handles all spatial reasoning.
Weights live next to this module so the backend works out of the box.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .normalize import OUT_SIZE

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "glyph_cnn.pt")
LABELS_PATH = os.path.join(WEIGHTS_DIR, "labels.json")


class GlyphCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def save_model(model: GlyphCNN, labels: List[str], weights_path: str = WEIGHTS_PATH,
               labels_path: str = LABELS_PATH) -> None:
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(labels, fh)


def load_model(weights_path: str = WEIGHTS_PATH, labels_path: str = LABELS_PATH,
               device: str = "cpu") -> Tuple[GlyphCNN, List[str]]:
    with open(labels_path, "r", encoding="utf-8") as fh:
        labels = json.load(fh)
    model = GlyphCNN(num_classes=len(labels))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, labels


def has_weights(weights_path: str = WEIGHTS_PATH, labels_path: str = LABELS_PATH) -> bool:
    return os.path.exists(weights_path) and os.path.exists(labels_path)


@torch.no_grad()
def predict(model: GlyphCNN, glyphs: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Classify a batch of normalized glyphs ``(N, OUT_SIZE, OUT_SIZE)``.

    Returns ``(indices, confidences)``.
    """
    x = torch.from_numpy(glyphs.astype(np.float32)).view(-1, 1, OUT_SIZE, OUT_SIZE).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    return idx.cpu().numpy(), conf.cpu().numpy()


@torch.no_grad()
def predict_topk(model: GlyphCNN, glyphs: np.ndarray, k: int = 3,
                 device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-``k`` ``(indices, probabilities)`` per glyph, ``(N, k)``.

    Used by the refinement step to demote a duplicate to its next-best guess.
    """
    x = torch.from_numpy(glyphs.astype(np.float32)).view(-1, 1, OUT_SIZE, OUT_SIZE).to(device)
    probs = torch.softmax(model(x), dim=1)
    k = min(k, probs.shape[1])
    conf, idx = probs.topk(k, dim=1)
    return idx.cpu().numpy(), conf.cpu().numpy()
