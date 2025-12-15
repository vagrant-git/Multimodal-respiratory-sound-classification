import argparse
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# Label normalization mapping
DROP_TOKEN = "drop"
LABEL_MAP: Dict[str, str] = {
    "no secretion": "no_secretion",
    "no secretion sound": "no_secretion",
    "no secretion sound (with hemf)": "no_secretion",
    "3ml secretion": "secretion",
    "3ml secretion m4": "secretion",
    "5ml secretion m4": "secretion",
    "5ml secretion": "secretion",
    "3ml secretion (with hemf)": DROP_TOKEN,
}


class DropLabel(Exception):
    """Raised when a sample is configured to be dropped."""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enforce deterministic CuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Make DataLoader workers deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_label(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in LABEL_MAP:
        norm = LABEL_MAP[key]
        if norm == DROP_TOKEN:
            raise DropLabel(key)
        return norm
    if key in LABEL_MAP.values():
        if key == DROP_TOKEN:
            raise DropLabel(key)
        return key
    raise ValueError(f"Unknown label: {raw}")


def pick_column(cols: List[str], hint: str) -> str:
    for c in cols:
        if c.startswith(hint):
            return c
    for c in cols:
        if hint.lower() in c.lower():
            return c
    raise ValueError(f"Column with hint '{hint}' not found in {cols}")


class NpzSegmentDataset(Dataset):
    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        self.classes = sorted(set(v for v in LABEL_MAP.values() if v != DROP_TOKEN))
        self.label2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        data = np.load(path, allow_pickle=True)
        raw_label = data["label"].item() if np.ndim(data["label"]) == 0 else data["label"]
        norm_label = normalize_label(raw_label)
        label_id = self.label2id[norm_label]

        sensor_cols = [str(c) for c in data["sensor_cols"]]
        flow_col = pick_column(sensor_cols, "F_")
        press_col = pick_column(sensor_cols, "P_")

        values = np.array(data["sensor_values"], dtype=np.float32)
        flow_idx = sensor_cols.index(flow_col)
        press_idx = sensor_cols.index(press_col)
        seq = values[:, [flow_idx, press_idx]]  # (T, 2)

        # Per-sample standardization
        seq = seq - seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True) + 1e-6
        seq = seq / std

        # Shape to (C, T) for Conv1d
        tensor = torch.from_numpy(seq.T)  # (2, T)
        return tensor, label_id


def collate_batch(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors, labels = zip(*batch)
    lengths = [t.shape[1] for t in tensors]  # original time dimension

    # pad_sequence pads on the sequence dimension, so temporarily move time to dim 0
    seq_first = [t.permute(1, 0) for t in tensors]  # (T, C)
    padded = pad_sequence(seq_first, batch_first=True, padding_value=0.0)  # (B, T_max, C)
    padded = padded.permute(0, 2, 1)  # (B, C, T_max) for Conv1d

    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


class Simple1DCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        feats = self.net(x).squeeze(-1)
        return self.head(feats)


def find_npz_files(root: Path) -> List[Path]:
    pattern = str(root / "**" / "*.npz")
    files = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    return files


def filter_drop_labels(paths: List[Path]) -> Tuple[List[Path], int]:
    """
    Remove any .npz whose normalized label maps to DROP_TOKEN.
    Returns (kept_paths, dropped_count).
    """
    kept: List[Path] = []
    dropped = 0
    for p in paths:
        data = np.load(p, allow_pickle=True)
        raw_label: Optional[str] = data.get("label")
        if raw_label is None:
            dropped += 1
            continue
        if isinstance(raw_label, np.ndarray) and raw_label.shape == ():
            raw_label = raw_label.item()
        try:
            _ = normalize_label(str(raw_label))
        except DropLabel:
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped


def split_dataset(paths: List[Path], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary 1D-CNN training on NPZ segments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/MMDataset_segments"), help="Root directory containing .npz files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    dl_generator = torch.Generator().manual_seed(args.seed)

    all_paths = find_npz_files(args.data_dir)
    if not all_paths:
        raise RuntimeError(f"No .npz files found under {args.data_dir}")

    filtered_paths, dropped = filter_drop_labels(all_paths)
    if dropped:
        print(f"Dropped {dropped} segments due to label rules (e.g., '{DROP_TOKEN}').")
    if not filtered_paths:
        raise RuntimeError("All .npz samples were dropped by label filtering.")

    train_paths, val_paths = split_dataset(filtered_paths, train_ratio=args.train_ratio, seed=args.seed)
    print(f"Found {len(all_paths)} segments -> train {len(train_paths)} | val {len(val_paths)}")

    train_ds = NpzSegmentDataset(train_paths)
    val_ds = NpzSegmentDataset(val_paths)
    if train_ds.label2id != val_ds.label2id:
        val_ds.label2id = train_ds.label2id  # align mapping

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    device = torch.device(args.device)
    model = Simple1DCNN(num_classes=len(train_ds.label2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | best={best_acc:.4f}")

    save_path = Path("checkpoints") / "binary_1dcnn_npz.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(), "label2id": train_ds.label2id}, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
