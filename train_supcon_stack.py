import atexit
import copy
import os
import random
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
import torchaudio
import torchvision.models as models

from src.ReadSegments import ReadSegments, find_segment_paths
from util.label_processor import LabelProcessor, DropLabel
from util.seed import set_seed, seed_worker


class GroupBalancedBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        groups,
        batch_size: int = 64,
        groups_per_class: int = 2,
        seed: int = 42,
    ):
        self.labels = np.asarray(labels)
        self.groups = np.asarray(groups)
        if self.labels.shape[0] != self.groups.shape[0]:
            raise ValueError("labels and groups must have the same length.")

        self.class_labels = sorted(set(self.labels.tolist()))
        if len(self.class_labels) != 2:
            raise ValueError("GroupBalancedBatchSampler expects exactly 2 classes.")

        self.num_classes = len(self.class_labels)
        if batch_size % (self.num_classes * groups_per_class) != 0:
            raise ValueError(
                "batch_size must be divisible by (num_classes * groups_per_class)."
            )
        self.batch_size = batch_size
        self.groups_per_class = groups_per_class
        self.samples_per_group = batch_size // (self.num_classes * groups_per_class)
        self.seed = seed
        self.epoch = 0

        self.class_group_indices: Dict[int, Dict[str, List[int]]] = {}
        for lab in self.class_labels:
            group_map: Dict[str, List[int]] = {}
            for idx in np.where(self.labels == lab)[0].tolist():
                g = self.groups[idx]
                group_map.setdefault(g, []).append(idx)
            self.class_group_indices[int(lab)] = group_map

        self._group_chunk_counts: Dict[int, Dict[str, int]] = {}
        for lab, group_map in self.class_group_indices.items():
            counts: Dict[str, int] = {
                g: len(idxs) // self.samples_per_group for g, idxs in group_map.items()
            }
            valid_groups = [g for g, c in counts.items() if c > 0]
            if len(valid_groups) < groups_per_class:
                raise ValueError(
                    f"Label {lab} only has {len(valid_groups)} groups with at least "
                    f"{self.samples_per_group} samples; need {groups_per_class}."
                )
            self._group_chunk_counts[lab] = counts

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        class_chunks: Dict[int, Dict[str, List[List[int]]]] = {}
        for lab in self.class_labels:
            group_chunks: Dict[str, List[List[int]]] = {}
            for g, idxs in self.class_group_indices[int(lab)].items():
                idxs = list(idxs)
                rng.shuffle(idxs)
                chunks = [
                    idxs[i : i + self.samples_per_group]
                    for i in range(0, len(idxs) - self.samples_per_group + 1, self.samples_per_group)
                ]
                if chunks:
                    group_chunks[g] = chunks
            class_chunks[int(lab)] = group_chunks

        available = {
            lab: [g for g, chunks in group_chunks.items() if chunks]
            for lab, group_chunks in class_chunks.items()
        }

        while all(
            len(available[int(lab)]) >= self.groups_per_class for lab in self.class_labels
        ):
            batch: List[int] = []
            for lab in self.class_labels:
                groups = available[int(lab)]
                groups_sorted = sorted(
                    groups,
                    key=lambda g: (-len(class_chunks[int(lab)][g]), rng.random()),
                )
                selected = groups_sorted[: self.groups_per_class]
                for g in selected:
                    batch.extend(class_chunks[int(lab)][g].pop())
                    if not class_chunks[int(lab)][g]:
                        available[int(lab)].remove(g)
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        counts = {
            lab: dict(group_counts) for lab, group_counts in self._group_chunk_counts.items()
        }
        batches = 0
        while all(
            sum(1 for c in counts[int(lab)].values() if c > 0) >= self.groups_per_class
            for lab in self.class_labels
        ):
            for lab in self.class_labels:
                groups_sorted = sorted(
                    counts[int(lab)].items(), key=lambda kv: kv[1], reverse=True
                )
                selected = [g for g, c in groups_sorted if c > 0][
                    : self.groups_per_class
                ]
                for g in selected:
                    counts[int(lab)][g] -= 1
            batches += 1
        return batches


def _group_key_from_path(path: str) -> str:
    base = os.path.basename(path)
    if "_win" in base:
        return base.split("_win")[0]
    return os.path.splitext(base)[0]


def _infer_group_label(group_paths: List[str], label_normalizer) -> str:
    norm_label = None
    for p in group_paths:
        d = np.load(p, allow_pickle=True)
        if "label" not in d:
            raise KeyError(f"Missing 'label' field in {p}")
        lab = d["label"]
        if isinstance(lab, np.ndarray) and lab.shape == ():
            lab = lab.item()
        current = label_normalizer(str(lab))
        if norm_label is None:
            norm_label = current
        elif norm_label != current:
            raise ValueError(
                f"Mixed labels inside group {p}: '{norm_label}' vs '{current}'"
            )
    if norm_label is None:
        raise ValueError("Group contained no labels; cannot infer class")
    return norm_label


def split_paths_by_group(
    paths: List[str],
    train_ratio: float = 0.6,
    seed: int = 42,
    label_normalizer=None,
) -> Tuple[List[str], List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in paths:
        key = _group_key_from_path(p)
        groups.setdefault(key, []).append(p)

    rng = random.Random(seed)

    if label_normalizer is not None:
        label_to_keys: Dict[str, List[str]] = {}
        for k, ps in groups.items():
            try:
                norm_label = _infer_group_label(ps, label_normalizer)
            except DropLabel:
                continue
            label_to_keys.setdefault(norm_label, []).append(k)

        train_keys: List[str] = []
        val_keys: List[str] = []
        for _, keys in sorted(label_to_keys.items()):
            keys_sorted = sorted(keys)
            rng.shuffle(keys_sorted)
            n_train = int(train_ratio * len(keys_sorted))
            if len(keys_sorted) > 1:
                n_train = min(max(n_train, 1), len(keys_sorted) - 1)
            train_keys.extend(keys_sorted[:n_train])
            val_keys.extend(keys_sorted[n_train:])
        rng.shuffle(train_keys)
        rng.shuffle(val_keys)
    else:
        keys = sorted(groups.keys())
        rng.shuffle(keys)
        n_train = int(train_ratio * len(keys))
        if len(keys) > 1:
            n_train = min(max(n_train, 1), len(keys) - 1)
        train_keys = keys[:n_train]
        val_keys = keys[n_train:]

    train_paths = [p for k in train_keys for p in sorted(groups[k])]
    val_paths = [p for k in val_keys for p in sorted(groups[k])]
    return train_paths, val_paths


def collate_fn(
    batch: List[Dict[str, Any]],
    max_length: Optional[int] = None,
    random_crop: bool = False,
) -> Dict[str, torch.Tensor]:
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
    labels = []

    for b in batch:
        P = b["P"]
        Q = b["Q"]

        if P is None or Q is None:
            T = b["audio"].shape[0]
            sensor = torch.zeros(T, 2, dtype=torch.float32)
        else:
            if P.dim() == 1:
                P_2d = P.unsqueeze(-1)
            else:
                P_2d = P
            if Q.dim() == 1:
                Q_2d = Q.unsqueeze(-1)
            else:
                Q_2d = Q
            P_main = P_2d[:, 0:1]
            Q_main = Q_2d[:, 0:1]
            sensor = torch.cat([P_main, Q_main], dim=-1)

        sensor_seqs.append(sensor)
        lengths.append(sensor.shape[0])
        labels.append(b["label_id"])
        paths.append(b["path"])

    max_len = max(lengths)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    labels_tensor = torch.stack(labels, dim=0)
    groups = [_group_key_from_path(p) for p in paths]

    return {
        "audio": audio,
        "sensor": sensor_padded,
        "label": labels_tensor,
        "group": groups,
        "paths": paths,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def collate_sensor(
    batch: List[Dict[str, Any]],
    max_length: Optional[int] = None,
    random_crop: bool = False,
) -> Dict[str, torch.Tensor]:
    sensors = []
    lengths = []
    paths = []
    labels = []

    for b in batch:
        P = b["P"]
        Q = b["Q"]
        if P is not None and P.dim() == 2:
            P = P[:, 0]
        if Q is not None and Q.dim() == 2:
            Q = Q[:, 0]

        p_len = P.shape[0] if P is not None else 0
        q_len = Q.shape[0] if Q is not None else 0
        T = max(p_len, q_len)
        if T == 0:
            T = 1

        if max_length is not None and T > max_length:
            if random_crop:
                start = random.randint(0, T - max_length)
            else:
                start = (T - max_length) // 2
            end = start + max_length
            if P is not None:
                P = P[start:end]
            if Q is not None:
                Q = Q[start:end]
            T = max_length

        dtype = (
            P.dtype
            if P is not None
            else (Q.dtype if Q is not None else torch.float32)
        )
        if P is None:
            P = torch.zeros(T, dtype=dtype)
        if Q is None:
            Q = torch.zeros(T, dtype=dtype)
        if P.shape[0] < T:
            P = nn.functional.pad(P, (0, T - P.shape[0]))
        if Q.shape[0] < T:
            Q = nn.functional.pad(Q, (0, T - Q.shape[0]))

        sensor = torch.stack([P, Q], dim=-1)
        sensors.append(sensor)
        lengths.append(T)
        labels.append(b["label_id"])
        paths.append(b["path"])

    max_T = max(lengths)
    sensors_padded = torch.zeros(len(batch), max_T, 2, dtype=torch.float32)
    for i, (sensor, T) in enumerate(zip(sensors, lengths)):
        sensors_padded[i, :T, :] = sensor

    labels_tensor = torch.stack(labels, dim=0)
    groups = [_group_key_from_path(p) for p in paths]

    return {
        "sensor": sensors_padded,
        "label": labels_tensor,
        "group": groups,
        "paths": paths,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> None:
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
                affine=True,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups=num_groups)


class AudioResNetEncoder(nn.Module):
    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        out_dim: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        max_freq = sample_rate / 2 if f_max is None else f_max
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=max_freq,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _replace_bn_with_gn(self.backbone, num_groups=8)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        x = x.squeeze(1)
        mel = self.melspec(x)
        mel_db = self.db(mel)
        mel_db = mel_db.unsqueeze(1)
        # per-sample normalize
        mean = mel_db.mean(dim=(2, 3), keepdim=True)
        std = mel_db.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        mel_db = (mel_db - mean) / std

        # light SpecAugment
        if self.training:
            # time mask
            T = mel_db.size(-1)
            w = max(1, int(0.05 * T))
            t0 = torch.randint(0, max(1, T - w), (1,), device=mel_db.device).item()
            mel_db[:, :, :, t0 : t0 + w] = 0

            # freq mask
            F = mel_db.size(-2)
            fw = max(1, int(0.1 * F))
            f0 = torch.randint(0, max(1, F - fw), (1,), device=mel_db.device).item()
            mel_db[:, :, f0 : f0 + fw, :] = 0
        feat = self.backbone(mel_db)
        return feat


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: Tuple[int, int, int],
        bottleneck_channels: int,
    ):
        super().__init__()
        use_bottleneck = bottleneck_channels > 0 and in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if use_bottleneck
            else None
        )
        conv_in = bottleneck_channels if use_bottleneck else in_channels
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    n_filters,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        outs = [conv(x) for conv in self.convs]
        outs.append(self.pool_conv(self.maxpool(x_in)))
        x = torch.cat(outs, dim=1)
        x = self.bn(x)
        return self.relu(x)


class InceptionTimeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        out_dim: int = 128,
        n_filters: int = 32,
        kernel_sizes: Tuple[int, int, int] = (9, 19, 39),
        bottleneck_channels: int = 32,
        n_blocks: int = 6,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleDict()

        in_channels = input_dim
        res_in_channels = input_dim
        for i in range(n_blocks):
            block = InceptionBlock1D(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            self.blocks.append(block)
            out_channels = n_filters * (len(kernel_sizes) + 1)
            if self.use_residual and (i + 1) % 3 == 0:
                self.residuals[str(i)] = nn.Sequential(
                    nn.Conv1d(res_in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
                )
                res_in_channels = out_channels
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_channels, out_dim)
        self.res_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for i, block in enumerate(self.blocks):
            if self.use_residual and i % 3 == 0:
                res_input = x
            x = block(x)
            if self.use_residual and (i + 1) % 3 == 0:
                res = self.residuals[str(i)](res_input)
                x = self.res_relu(x + res)
        feats = self.global_pool(x).squeeze(-1)
        return self.proj(feats)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    groups: List[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    z = nn.functional.normalize(z, dim=1)
    labels = labels.view(-1, 1)
    B = z.size(0)
    sim = torch.mm(z, z.t()) / temperature
    sim = sim - torch.eye(B, device=sim.device) * 1e9

    same_label = labels.eq(labels.t())
    uniq = {g: i for i, g in enumerate(sorted(set(groups)))}
    group_ids = torch.tensor([uniq[g] for g in groups], device=labels.device).view(-1, 1)
    diff_group = ~group_ids.eq(group_ids.t())
    pos_mask = same_label & diff_group

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_per_anchor = pos_mask.sum(dim=1)
    valid = pos_per_anchor > 0
    valid_ratio = valid.float().mean().item()
    avg_pos = pos_per_anchor.float().mean().item()
    print(f"[SupCon stats] valid_ratio={valid_ratio:.3f} avg_pos={avg_pos:.2f}")
    pos_counts = pos_per_anchor.clamp(min=1)
    loss = -(pos_mask.float() * log_prob).sum(dim=1) / pos_counts
    if valid.any():
        return loss[valid].mean()
    return loss.mean()


@torch.no_grad()
def encode_batch(
    encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    modality: str,
) -> torch.Tensor:
    if modality == "audio":
        return encoder(batch["audio"])
    if modality == "sensor":
        return encoder(batch["sensor"])
    raise ValueError(f"Unknown modality: {modality}")


def pretrain_supcon(
    encoder: nn.Module,
    projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    epochs: int = 20,
    lr: float = 3e-4,
    log_fn=None,
) -> None:
    encoder.train()
    projector.train()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=1e-4,
        foreach=False,
    )
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        for batch in loader:
            labels = batch["label"].to(device)
            groups = batch["group"]
            if modality == "audio":
                x = batch["audio"].to(device, non_blocking=True)
            else:
                x = batch["sensor"].to(device, non_blocking=True)
            feats = encoder(x)
            proj = projector(feats)
            loss = supervised_contrastive_loss(proj, labels, groups)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs
        msg = f"[SupCon:{modality}] epoch={epoch:02d} loss={total_loss / max(1, total):.4f}"
        if log_fn is None:
            print(msg)
        else:
            log_fn(msg)


def train_classifier(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    out_dim: int,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    log_fn=None,
) -> nn.Module:
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    head = nn.Linear(out_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4, foreach=False)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        for batch in loader:
            labels = batch["label"].to(device)
            if modality == "audio":
                x = batch["audio"].to(device, non_blocking=True)
            else:
                x = batch["sensor"].to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoder(x)
            logits = head(feats)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs
        msg = f"[Cls:{modality}] epoch={epoch:02d} loss={total_loss / max(1, total):.4f}"
        if log_fn is None:
            print(msg)
        else:
            log_fn(msg)
    return head


@torch.no_grad()
def eval_classifier(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    num_classes: int,
) -> Tuple[float, float, torch.Tensor]:
    encoder.eval()
    head.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for batch in loader:
        labels = batch["label"].to(device)
        if modality == "audio":
            x = batch["audio"].to(device, non_blocking=True)
        else:
            x = batch["sensor"].to(device, non_blocking=True)
        feats = encoder(x)
        logits = head(feats)
        loss = ce(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.item(), p.item()] += 1
    return total_loss / max(1, total), correct / max(1, total), confusion


def finetune_classifier(
    encoder: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    modality: str,
    num_classes: int,
    epochs: int = 8,
    encoder_lr: float = 1e-4,
    head_lr: float = 1e-3,
    patience: int = 3,
    min_epochs: int = 3,
    log_fn=None,
) -> nn.Module:
    for p in encoder.parameters():
        p.requires_grad = True
    for p in head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=1e-4,
        foreach=False,
    )
    ce = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            labels = batch["label"].to(device)
            if modality == "audio":
                x = batch["audio"].to(device, non_blocking=True)
            else:
                x = batch["sensor"].to(device, non_blocking=True)
            logits = head(encoder(x))
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

        train_loss = total_loss / max(1, total)
        val_loss, val_acc, confusion = eval_classifier(
            encoder, head, val_loader, device, modality, num_classes
        )
        msg = (
            f"[Finetune:{modality}] epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if log_fn is None:
            print(msg)
            print(f"[Finetune:{modality}] confusion:\n{confusion}")
        else:
            log_fn(msg)
            log_fn(f"[Finetune:{modality}] confusion:\n{confusion}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = (
                copy.deepcopy(encoder.state_dict()),
                copy.deepcopy(head.state_dict()),
            )
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch >= min_epochs and epochs_since_best >= patience:
            break

    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    return head


@torch.no_grad()
def collect_logits(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    head.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        labels = batch["label"].to(device)
        if modality == "audio":
            x = batch["audio"].to(device, non_blocking=True)
        else:
            x = batch["sensor"].to(device, non_blocking=True)
        feats = encoder(x)
        logits = head(feats)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def eval_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_logit_stacking(
    logit_audio: torch.Tensor,
    logit_sensor: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    la = logit_audio[:, 1] - logit_audio[:, 0]
    ls = logit_sensor[:, 1] - logit_sensor[:, 0]
    x = torch.stack([la, ls], dim=1)
    y = labels.float()

    w = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        logits = w[0] + w[1] * x[:, 0] + w[2] * x[:, 1]
        loss = bce(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return w.detach(), x


def eval_stacking(
    weights: torch.Tensor,
    logit_audio: torch.Tensor,
    logit_sensor: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    la = logit_audio[:, 1] - logit_audio[:, 0]
    ls = logit_sensor[:, 1] - logit_sensor[:, 0]
    logits = weights[0] + weights[1] * la + weights[2] * ls
    preds = (logits > 0).long()
    acc = (preds == labels).float().mean().item()
    return acc


def main():
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{Path(__file__).stem}.log"
    log_file = log_path.open("w", encoding="utf-8")
    atexit.register(log_file.close)

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 80)
    log(f"Run started at {datetime.now().isoformat(timespec='seconds')}")

    all_paths = sorted(find_segment_paths("./data"))
    if not all_paths:
        raise RuntimeError("No .npz found! Please run preprocess_segments.py first.")

    seed = 42
    set_seed(seed)
    dl_generator = torch.Generator().manual_seed(seed)

    target_sr = 22050
    negative_token = "no_secretion"
    label_processor = LabelProcessor(
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
    log(label_processor.describe())

    train_paths, val_paths = split_paths_by_group(
        all_paths, train_ratio=0.6, seed=seed, label_normalizer=label_processor
    )
    n = len(all_paths)
    log(
        f"Total segments: {n}, train={len(train_paths)}, val={len(val_paths)}, "
        f"groups={len(set(_group_key_from_path(p) for p in all_paths))}"
    )

    train_ds = ReadSegments(
        train_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
        group_normalize=True,
    )
    val_ds = ReadSegments(
        val_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
        label2id=train_ds.label2id,
        label_map_paths=train_paths,
        group_normalize=True,
    )

    num_classes = len(train_ds.label2id)
    log("num_classes = {} label2id = {}".format(num_classes, train_ds.label2id))

    train_label_ids = torch.tensor(train_ds.get_label_id_list(), dtype=torch.long)
    class_counts = torch.bincount(train_label_ids, minlength=num_classes).float()
    class_weights = 1.0 / class_counts.clamp(min=1.0)
    sample_weights = class_weights[train_label_ids]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    log(
        "Class counts: {} | sampler weights: {}".format(
            class_counts.tolist(), class_weights.tolist()
        )
    )

    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    max_seconds = float(os.environ.get("MAX_SECONDS", "0"))
    max_length = int(max_seconds * target_sr) if max_seconds > 0 else None
    collate_train = partial(collate_fn, max_length=max_length, random_crop=True)
    collate_eval = partial(collate_fn, max_length=max_length, random_crop=False)
    collate_sensor_train = partial(
        collate_sensor, max_length=max_length, random_crop=True
    )
    if max_length is not None:
        log(f"Max crop length: {max_length} samples ({max_seconds:.2f}s)")

    train_groups = [_group_key_from_path(p) for p in train_ds.paths]
    supcon_sampler = GroupBalancedBatchSampler(
        train_label_ids.tolist(),
        train_groups,
        batch_size=batch_size,
        groups_per_class=2,
        seed=seed,
    )
    train_loader_supcon = DataLoader(
        train_ds,
        batch_sampler=supcon_sampler,
        num_workers=0,
        collate_fn=collate_train,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    train_loader_supcon_sensor = DataLoader(
        train_ds,
        batch_sampler=supcon_sampler,
        num_workers=0,
        collate_fn=collate_sensor_train,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    train_loader_cls = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_train,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    train_loader_plain = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    if force_cpu:
        torch.backends.cudnn.enabled = False

    def _move_to_cpu(*modules: nn.Module) -> None:
        for module in modules:
            module.to("cpu")

    def _move_to_device(*modules: nn.Module) -> None:
        for module in modules:
            module.to(device)

    def _maybe_empty_cache() -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    audio_encoder = AudioResNetEncoder(
        sample_rate=target_sr,
        out_dim=128,
        f_max=4000,
    )
    sensor_encoder = InceptionTimeEncoder(
        input_dim=2,
        out_dim=128,
    )

    audio_projector = ProjectionHead(128, proj_dim=128)
    sensor_projector = ProjectionHead(128, proj_dim=128)

    log("Stage 1: supervised contrastive pretrain (audio)")
    _move_to_cpu(sensor_encoder, sensor_projector)
    _move_to_device(audio_encoder, audio_projector)
    _maybe_empty_cache()
    pretrain_supcon(
        audio_encoder,
        audio_projector,
        train_loader_supcon,
        device,
        modality="audio",
        epochs=20,
        log_fn=log,
    )
    log("Stage 2: supervised contrastive pretrain (sensor)")
    _move_to_cpu(audio_encoder, audio_projector)
    _maybe_empty_cache()
    _move_to_device(sensor_encoder, sensor_projector)
    pretrain_supcon(
        sensor_encoder,
        sensor_projector,
        train_loader_supcon_sensor,
        device,
        modality="sensor",
        epochs=20,
        log_fn=log,
    )

    log("Stage 3: linear classifier (audio)")
    _move_to_cpu(sensor_encoder, sensor_projector)
    _maybe_empty_cache()
    _move_to_device(audio_encoder)
    audio_head = train_classifier(
        audio_encoder,
        train_loader_cls,
        device,
        "audio",
        out_dim=128,
        num_classes=num_classes,
        log_fn=log,
    )
    log("Stage 3b: finetune (audio)")
    audio_head = finetune_classifier(
        audio_encoder,
        audio_head,
        train_loader_cls,
        val_loader,
        device,
        "audio",
        num_classes=num_classes,
        log_fn=log,
    )
    log("Stage 4: linear classifier (sensor)")
    _move_to_cpu(audio_encoder, audio_head)
    _maybe_empty_cache()
    _move_to_device(sensor_encoder)
    sensor_head = train_classifier(
        sensor_encoder,
        train_loader_cls,
        device,
        "sensor",
        out_dim=128,
        num_classes=num_classes,
        log_fn=log,
    )
    log("Stage 4b: finetune (sensor)")
    sensor_head = finetune_classifier(
        sensor_encoder,
        sensor_head,
        train_loader_cls,
        val_loader,
        device,
        "sensor",
        num_classes=num_classes,
        log_fn=log,
    )

    log("Stage 5: logit stacking")
    _move_to_device(audio_encoder, audio_head)
    _move_to_cpu(sensor_encoder, sensor_head)
    _maybe_empty_cache()
    train_audio_logits, train_labels = collect_logits(
        audio_encoder, audio_head, train_loader_plain, device, "audio"
    )
    _move_to_cpu(audio_encoder, audio_head)
    _maybe_empty_cache()
    _move_to_device(sensor_encoder, sensor_head)
    train_sensor_logits, _ = collect_logits(
        sensor_encoder, sensor_head, train_loader_plain, device, "sensor"
    )
    log(f"Audio train acc: {eval_accuracy(train_audio_logits, train_labels):.4f}")
    log(f"Sensor train acc: {eval_accuracy(train_sensor_logits, train_labels):.4f}")
    weights, _ = train_logit_stacking(
        train_audio_logits, train_sensor_logits, train_labels
    )
    log(f"Stacking weights: w0={weights[0]:.4f} w1={weights[1]:.4f} w2={weights[2]:.4f}")

    _move_to_device(audio_encoder, audio_head)
    _move_to_cpu(sensor_encoder, sensor_head)
    _maybe_empty_cache()
    val_audio_logits, val_labels = collect_logits(
        audio_encoder, audio_head, val_loader, device, "audio"
    )
    _move_to_cpu(audio_encoder, audio_head)
    _maybe_empty_cache()
    _move_to_device(sensor_encoder, sensor_head)
    val_sensor_logits, _ = collect_logits(
        sensor_encoder, sensor_head, val_loader, device, "sensor"
    )
    log(f"Audio val acc: {eval_accuracy(val_audio_logits, val_labels):.4f}")
    log(f"Sensor val acc: {eval_accuracy(val_sensor_logits, val_labels):.4f}")
    val_acc = eval_stacking(weights, val_audio_logits, val_sensor_logits, val_labels)
    log(f"Stacked val acc: {val_acc:.4f}")

    log_file.close()


if __name__ == "__main__":
    main()
