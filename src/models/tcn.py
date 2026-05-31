"""Temporal Convolutional Network model and training utilities.

Extracted from t3a4_tcn_baseline.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

# Default training hyper-parameters
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
N_FILTERS = 64
N_LAYERS = 4
KERNEL_SIZE = 3


class TCNBlock(nn.Module):
    """Single dilated convolutional block with batch norm + residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class TCNClassifier(nn.Module):
    """TCN with dilated conv1d layers + global average pooling + dense head."""

    def __init__(
        self,
        n_features: int,
        n_filters: int = N_FILTERS,
        kernel_size: int = KERNEL_SIZE,
        n_layers: int = N_LAYERS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Conv1d(n_features, n_filters, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(n_filters, n_filters, 2**i, kernel_size, dropout)
                for i in range(n_layers)
            ]
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(n_filters, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class WindowDataset(Dataset):
    """In-memory window dataset for TCN training."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    size_label: str = "",
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    n_filters: int = N_FILTERS,
    n_layers: int = N_LAYERS,
    kernel_size: int = KERNEL_SIZE,
    verbose: bool = True,
) -> tuple[TCNClassifier, list[dict]]:
    """Train TCN with early stopping. Returns (model, history)."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val]).float()

    if verbose:
        print(
            f"    pos_weight={pos_weight.item():.2f} "
            f"(n_pos={int(n_pos):,}, n_neg={int(n_neg):,})"
        )

    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val) if len(X_val) > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_ds
        else None
    )

    model = TCNClassifier(
        n_features=n_features,
        n_filters=n_filters,
        kernel_size=kernel_size,
        n_layers=n_layers,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"    Model params: {n_params:,}")
        print(f"    Training (n={len(X_train):,}, val={len(X_val):,})")

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_auc = 0.0
        if val_loader and len(X_val) > 0:
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    logits = model(X_b)
                    val_loss += criterion(logits, y_b).item()
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(y_b.cpu().numpy())
            val_loss /= max(len(val_loader), 1)
            y_prob = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            val_auc = roc_auc_score(y_true, y_prob)
            scheduler.step(val_auc)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_auc": round(val_auc, 4),
            }
        )

        if verbose:
            print(
                f"      Epoch {epoch + 1:2d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_tcn(
    model: TCNClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Evaluate on test set, return metrics dict."""
    if len(X_test) == 0:
        return {"n_test": 0, "error": "No test data"}

    model.eval()
    test_ds = WindowDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            all_preds.append(torch.sigmoid(model(X_b)).cpu().numpy())
            all_labels.append(y_b.cpu().numpy())

    y_prob = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
        "test_high_risk_ratio": float(y_true.mean()),
        "test_pred_high_risk_ratio": float(y_prob.mean()),
    }
