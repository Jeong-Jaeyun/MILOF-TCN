from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.fog_tcn_ae import TCNAutoEncoder


class NumpySeqDataset(Dataset):
    def __init__(self, X: np.ndarray):
        # X: [N, L, C]
        self.X = np.asarray(X, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx])


@dataclass
class TrainResult:
    model: TCNAutoEncoder
    best_val_loss: float


def train_tcn_ae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n_features: int,
    channels: list[int],
    kernel_size: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    save_path: str | None = None,
) -> TrainResult:
    device = device if torch.cuda.is_available() and "cuda" in device else "cpu"

    model = TCNAutoEncoder(
        n_features=n_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss(reduction="mean")

    train_loader = DataLoader(NumpySeqDataset(X_train), batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(NumpySeqDataset(X_val), batch_size=int(batch_size), shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []
        for xb in train_loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                loss = loss_fn(recon, xb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[TCN-AE] epoch={ep:03d} train_mse={tr:.6f} val_mse={va:.6f} best_val={best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, save_path)

    return TrainResult(model=model, best_val_loss=best_val)


@torch.no_grad()
def score_tcn_ae(model: TCNAutoEncoder, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    model = model.to(device)
    model.eval()

    loader = DataLoader(NumpySeqDataset(X), batch_size=int(batch_size), shuffle=False, drop_last=False)
    scores = []
    for xb in loader:
        xb = xb.to(device)
        recon = model(xb)
        s = TCNAutoEncoder.recon_error(xb, recon, reduce="mean")
        scores.append(s.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)
