"""Train the glyph classifier on synthetic font data.

    python -m handtex.ocr.local.train --epochs 12

Saves ``weights/glyph_cnn.pt`` + ``weights/labels.json`` next to the model.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import build_dataset
from .model import GlyphCNN, save_model
from .normalize import OUT_SIZE


def train(
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    per_class_per_font: int = 4,
    max_fonts: int = 40,
    val_frac: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> float:
    """Train and persist the model. Returns final validation accuracy."""
    torch.manual_seed(seed)
    t0 = time.time()
    if verbose:
        print("synthesizing dataset...")
    X, y, labels = build_dataset(per_class_per_font=per_class_per_font, max_fonts=max_fonts, seed=seed)
    if verbose:
        print(f"  {X.shape[0]} samples, {len(labels)} classes, {time.time()-t0:.1f}s")

    # shuffle + split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_val = int(len(X) * val_frac)
    Xtr, ytr = X[n_val:], y[n_val:]
    Xva, yva = X[:n_val], y[:n_val]

    def loader(Xs, ys, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xs).view(-1, 1, OUT_SIZE, OUT_SIZE),
            torch.from_numpy(ys),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_dl = loader(Xtr, ytr, True)
    val_dl = loader(Xva, yva, False)

    model = GlyphCNN(num_classes=len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs}  val_acc={val_acc:.3f}")

    save_model(model, labels)
    if verbose:
        print(f"saved model ({len(labels)} classes), val_acc={val_acc:.3f}, "
              f"{time.time()-t0:.1f}s total")
    return val_acc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="train the HandTex glyph classifier")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-class-per-font", type=int, default=4)
    p.add_argument("--max-fonts", type=int, default=40)
    args = p.parse_args(argv)
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          per_class_per_font=args.per_class_per_font, max_fonts=args.max_fonts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
