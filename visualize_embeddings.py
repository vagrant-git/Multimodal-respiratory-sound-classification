import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.ReadSegments import ReadSegments, find_segment_paths
from train_compare import MultiModalLateFusionNet
from train_supcon_stack import (
    AudioResNetEncoder as SupConAudioResNetEncoder,
    InceptionTimeEncoder as SupConSensorEncoder,
)
from util.label_processor import LabelProcessor, DropLabel
from util.seed import set_seed

try:
    import umap  # type: ignore
except ImportError:
    umap = None


def build_label_processor() -> LabelProcessor:
    """
    Use the same label mapping as train_compare.py to stay consistent with the checkpoint.
    """
    negative_token = "no_secretion"
    return LabelProcessor(
        raw_to_norm={
            "no secretion": negative_token,
            "no secretion sound": negative_token,
            "no secretion sound (with hemf)": negative_token,
            "3ml secretion": "secretion",
            "3ml secretion m4": "secretion",
            "5ml secretion m4": "secretion",
            "5ml secretion": "secretion",
            "3ml secretion (with hemf)": "secretion",
        },
        fail_on_unknown=True,
    )


def collate_with_labels(batch):
    """
    Pad audio and sensor sequences (same logic as train_compare.collate_fn) and keep labels/paths.
    """
    audio_list = [b["audio"] for b in batch]
    audio_lengths = [a.shape[0] for a in audio_list]
    max_T_audio = max(audio_lengths)
    B = len(batch)

    audio = torch.zeros(B, max_T_audio, dtype=torch.float32)
    for i, (a, L) in enumerate(zip(audio_list, audio_lengths)):
        audio[i, :L] = a

    sensor_seqs = []
    lengths = []
    paths = []
    raw_labels = []
    norm_labels = []

    for b in batch:
        P = b["P"]
        Q = b["Q"]

        if P is None or Q is None:
            T = b["audio"].shape[0]
            sensor = torch.zeros(T, 2, dtype=torch.float32)
        else:
            P_2d = P.unsqueeze(-1) if P.dim() == 1 else P
            Q_2d = Q.unsqueeze(-1) if Q.dim() == 1 else Q
            P_main = P_2d[:, 0:1]
            Q_main = Q_2d[:, 0:1]
            sensor = torch.cat([P_main, Q_main], dim=-1)

        channel_mean = sensor.mean(dim=0, keepdim=True)
        channel_std = sensor.std(dim=0, keepdim=True)
        channel_std = torch.where(channel_std < 1e-6, torch.ones_like(channel_std), channel_std)
        sensor = (sensor - channel_mean) / channel_std

        sensor_seqs.append(sensor)
        lengths.append(sensor.shape[0])
        paths.append(b["path"])
        raw_labels.append(b["raw_label"])
        norm_labels.append(b["norm_label"])

    max_len = max(lengths)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    label_ids = torch.stack([b["label_id"] for b in batch], dim=0)

    return {
        "audio": audio,
        "sensor": sensor_padded,
        "sensor_lengths": lengths_tensor,
        "label": label_ids,
        "paths": paths,
        "raw_label": raw_labels,
        "norm_label": norm_labels,
    }


def collate_with_labels_supcon(batch):
    """
    Match train_supcon_stack collate behavior (no extra normalization here).
    """
    audio_list = [b["audio"] for b in batch]
    audio_lengths = [a.shape[0] for a in audio_list]
    max_T_audio = max(audio_lengths)
    B = len(batch)

    audio = torch.zeros(B, max_T_audio, dtype=torch.float32)
    for i, (a, L) in enumerate(zip(audio_list, audio_lengths)):
        audio[i, :L] = a

    sensor_seqs = []
    lengths = []
    paths = []
    raw_labels = []
    norm_labels = []
    label_ids = []

    for b in batch:
        P = b["P"]
        Q = b["Q"]

        if P is None or Q is None:
            T = b["audio"].shape[0]
            sensor = torch.zeros(T, 2, dtype=torch.float32)
        else:
            P_2d = P.unsqueeze(-1) if P.dim() == 1 else P
            Q_2d = Q.unsqueeze(-1) if Q.dim() == 1 else Q
            P_main = P_2d[:, 0:1]
            Q_main = Q_2d[:, 0:1]
            sensor = torch.cat([P_main, Q_main], dim=-1)

        sensor_seqs.append(sensor)
        lengths.append(sensor.shape[0])
        paths.append(b["path"])
        raw_labels.append(b["raw_label"])
        norm_labels.append(b["norm_label"])
        label_ids.append(b["label_id"])

    max_len = max(lengths)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    label_tensor = torch.stack(label_ids, dim=0)

    return {
        "audio": audio,
        "sensor": sensor_padded,
        "sensor_lengths": lengths_tensor,
        "label": label_tensor,
        "paths": paths,
        "raw_label": raw_labels,
        "norm_label": norm_labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings and plot UMAP/t-SNE/PCA.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_multimodal.pt", help="Checkpoint path")
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory or glob for .npz segments")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--output-dir", type=str, default="figures/embeddings", help="Where to save plots/npz")
    parser.add_argument("--target-sr", type=int, default=22050, help="Resample audio to this rate")
    parser.add_argument("--fmax", type=float, default=4000.0, help="Max frequency for MelSpectrogram")
    parser.add_argument(
        "--methods",
        type=str,
        default="umap,tsne,pca",
        help="Comma-separated list among: umap, tsne, pca",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="norm,raw,date",
        help="Color by: norm, raw, date (comma-separated)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def date_from_path(path_str: str) -> str:
    """
    Extract the parent-of-parent directory name as a date token.
    Example: data/segments/231107/foo_win00001.npz -> '231107'.
    """
    p = Path(path_str)
    if p.parent.parent.name:
        return p.parent.parent.name
    return p.parent.name or "unknown"


class EncoderBundle(nn.Module):
    def __init__(self, audio_encoder: nn.Module, sensor_encoder: nn.Module) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.sensor_encoder = sensor_encoder


class SensorEncoderIgnoreLengths(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, sensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _ = lengths
        return self.encoder(sensor)


def _merge_audio_cfg(cfg: Dict[str, float], default_sr: int, default_fmax: float) -> Dict[str, float]:
    merged = {
        "sample_rate": default_sr,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 512,
        "out_dim": 128,
        "f_min": 0.0,
        "f_max": default_fmax,
    }
    merged.update(cfg)
    return merged


def _merge_sensor_cfg(cfg: Dict[str, float]) -> Dict[str, float]:
    merged = {
        "input_dim": 2,
        "out_dim": 128,
        "n_filters": 32,
        "kernel_sizes": (9, 19, 39),
        "bottleneck_channels": 32,
        "n_blocks": 6,
        "use_residual": True,
    }
    merged.update(cfg)
    return merged


def load_model_from_state(
    state: Dict[str, object],
    num_classes: int,
    target_sr: int,
    fmax: float,
    device: torch.device,
) -> Tuple[nn.Module, str, bool, int, float]:
    if "model_state" in state:
        model = MultiModalLateFusionNet(
            audio_feat_dim=128,
            sensor_feat_dim=128,
            num_classes=num_classes,
            sample_rate=target_sr,
            f_max=fmax,
        )
        model.load_state_dict(state["model_state"])
        ckpt_type = "multimodal"
        group_normalize = bool(state.get("group_normalize", False))
        resolved_sr = target_sr
        resolved_fmax = fmax
    elif "audio_encoder_state" in state and "sensor_encoder_state" in state:
        audio_cfg = _merge_audio_cfg(state.get("audio_encoder_cfg", {}), target_sr, fmax)
        sensor_cfg = _merge_sensor_cfg(state.get("sensor_encoder_cfg", {}))
        resolved_sr = int(audio_cfg["sample_rate"])
        resolved_fmax = float(audio_cfg["f_max"])
        audio_encoder = SupConAudioResNetEncoder(**audio_cfg)
        sensor_encoder = SupConSensorEncoder(**sensor_cfg)
        audio_encoder.load_state_dict(state["audio_encoder_state"])
        sensor_encoder.load_state_dict(state["sensor_encoder_state"])
        model = EncoderBundle(audio_encoder, SensorEncoderIgnoreLengths(sensor_encoder))
        ckpt_type = "supcon_stack"
        group_normalize = bool(state.get("group_normalize", True))
    else:
        raise ValueError("Checkpoint format not recognized.")

    model.to(device)
    model.eval()
    return model, ckpt_type, group_normalize, resolved_sr, resolved_fmax


def extract_embeddings(
    model: MultiModalLateFusionNet,
    loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    audio_feats: List[np.ndarray] = []
    sensor_feats: List[np.ndarray] = []
    fused_feats: List[np.ndarray] = []
    label_ids: List[int] = []
    raw_labels: List[str] = []
    norm_labels: List[str] = []
    paths: List[str] = []
    date_labels: List[str] = []

    processed = 0
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            sensor = batch["sensor"].to(device)
            lengths = batch["sensor_lengths"].to(device)
            labels = batch["label"]

            fa = model.audio_encoder(audio)
            fs = model.sensor_encoder(sensor, lengths)
            fused = torch.cat([fa, fs], dim=-1)

            audio_feats.append(fa.cpu().numpy())
            sensor_feats.append(fs.cpu().numpy())
            fused_feats.append(fused.cpu().numpy())
            label_ids.extend(labels.tolist())
            raw_labels.extend(batch["raw_label"])
            norm_labels.extend(batch["norm_label"])
            paths.extend(batch["paths"])
            date_labels.extend(date_from_path(p) for p in batch["paths"])

            processed += labels.size(0)
            if max_samples is not None and processed >= max_samples:
                break

    def _stack(xs: Sequence[np.ndarray]) -> np.ndarray:
        if not xs:
            return np.empty((0, 1))
        cat = np.concatenate(xs, axis=0)
        if max_samples is not None:
            cat = cat[:max_samples]
        return cat

    size = min(processed, max_samples) if max_samples is not None else processed
    return {
        "audio": _stack(audio_feats),
        "sensor": _stack(sensor_feats),
        "fused": _stack(fused_feats),
        "label_ids": np.array(label_ids[:size]),
        "raw_labels": np.array(raw_labels[:size]),
        "norm_labels": np.array(norm_labels[:size]),
        "paths": np.array(paths[:size]),
        "date_labels": np.array(date_labels[:size]),
    }


def reduce_2d(name: str, data: np.ndarray, seed: int = 42) -> np.ndarray:
    if data.shape[0] == 0:
        return np.empty((0, 2))
    name = name.lower()
    if name == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif name == "tsne":
        reducer = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30)
    elif name == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown reducer: {name}")
    return reducer.fit_transform(data)


def plot_embedding(coords: np.ndarray, labels: Sequence, label_to_name: Dict, title: str, out_file: Path) -> None:
    if coords.shape[0] == 0:
        print(f"Skip plotting {out_file} (no samples).")
        return
    unique_vals = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(unique_vals))]

    plt.figure(figsize=(8, 6))
    for idx, val in enumerate(unique_vals):
        mask = [l == val for l in labels]
        pts = coords[mask]
        if pts.size == 0:
            continue
        name = label_to_name[val] if val in label_to_name else str(val)
        plt.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.8, color=colors[idx], label=name)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")


def main():
    args = parse_args()
    set_seed(42)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    color_modes = [c.strip().lower() for c in args.color_by.split(",") if c.strip()]
    for required in ["raw", "date"]:
        if required not in color_modes:
            color_modes.append(required)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    if "label2id" not in state:
        raise KeyError(f"Checkpoint {ckpt_path} missing label2id.")
    label2id = state["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    label_processor = build_label_processor()
    model, ckpt_type, group_normalize, resolved_sr, resolved_fmax = load_model_from_state(
        state,
        num_classes=len(label2id),
        target_sr=args.target_sr,
        fmax=args.fmax,
        device=device,
    )

    # Load dataset
    all_paths = find_segment_paths(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .npz files found under {args.data_root}")

    # Filter out samples whose label is configured to drop
    filtered_paths = []
    dropped = 0
    for p in all_paths:
        try:
            d = np.load(p, allow_pickle=True)
            lab = d["label"]
            if isinstance(lab, np.ndarray) and lab.shape == ():
                lab = lab.item()
            label_processor(str(lab))
            filtered_paths.append(p)
        except DropLabel:
            dropped += 1
            continue
    if dropped:
        print(f"Filtered out {dropped} samples due to drop_label rules.")

    dataset = ReadSegments(
        filtered_paths,
        target_sample_rate=resolved_sr,
        label_normalizer=label_processor,
        label2id=label2id,
        label_map_paths=filtered_paths,
        group_normalize=group_normalize,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_labels_supcon if ckpt_type == "supcon_stack" else collate_with_labels,
    )

    feats = extract_embeddings(
        model,
        loader,
        device=device,
        max_samples=args.max_samples,
    )

    np.savez_compressed(
        output_dir / "embeddings.npz",
        audio=feats["audio"],
        sensor=feats["sensor"],
        fused=feats["fused"],
        label_ids=feats["label_ids"],
        raw_labels=feats["raw_labels"],
        norm_labels=feats["norm_labels"],
        paths=feats["paths"],
        date_labels=feats["date_labels"],
    )
    print(f"Saved embeddings to {output_dir / 'embeddings.npz'}")

    def prettify_raw_label(label: str) -> str:
        s = " ".join(str(label).split())
        s = re.sub(r"(\d+)\\s*[mM][lL]", lambda m: f"{m.group(1)}ML", s)
        return s

    label_arrays = {
        "norm": feats["norm_labels"],
        "raw": feats["raw_labels"],
        "date": feats["date_labels"],
    }
    label_name_maps = {
        "norm": {lab: lab for lab in np.unique(feats["norm_labels"])},
        "raw": {lab: prettify_raw_label(lab) for lab in np.unique(feats["raw_labels"])},
        "date": {lab: lab for lab in np.unique(feats["date_labels"])},
        "id": {i: id2label[i] for i in id2label},
    }

    for feat_name, data in feats.items():
        if feat_name not in {"audio", "sensor", "fused"}:
            continue
        for reducer in methods:
            try:
                coords = reduce_2d(reducer, data)
            except ImportError as e:
                print(f"Skip {reducer}: {e}")
                continue
            except ValueError as e:
                print(f"Skip {reducer} on {feat_name}: {e}")
                continue

            for mode in color_modes:
                if mode not in label_arrays:
                    continue
                labels = label_arrays[mode]
                title = f"{feat_name} | {reducer.upper()} | color={mode}"
                out_file = output_dir / f"{feat_name}_{reducer}_{mode}.png"
                plot_embedding(coords, labels.tolist(), label_name_maps[mode], title, out_file)

            # Always save a label_id-colored plot for debugging
            title = f"{feat_name} | {reducer.upper()} | color=id"
            out_file = output_dir / f"{feat_name}_{reducer}_id.png"
            plot_embedding(coords, feats["label_ids"].tolist(), label_name_maps["id"], title, out_file)


if __name__ == "__main__":
    main()
