import atexit
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchaudio
import torchvision.models as models

from src.ReadSegments import ReadSegments, find_segment_paths
from util.label_processor import LabelProcessor, DropLabel
from util.seed import set_seed, seed_worker


# ===========================
# 1. collate_fn：打 batch + domain
# ===========================

def _group_key_from_path(path: str) -> str:
    base = os.path.basename(path)
    if "_win" in base:
        return base.split("_win")[0]
    return os.path.splitext(base)[0]


def build_domain_map(paths: List[str]) -> Dict[str, int]:
    keys = sorted({_group_key_from_path(p) for p in paths})
    return {k: i for i, k in enumerate(keys)}


def collate_fn(
    batch: List[Dict[str, Any]],
    domain_map: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    batch 里的每个元素是 ReadSegments.__getitem__ 返回的 dict：
      - "audio": [T_audio]
      - "P": [T] 或 [T, n_P] 或 None
      - "Q": [T] 或 [T, n_Q] 或 None
      - "label_id": scalar tensor
      - "path": 样本路径（字符串）
    这里我们只用 P/Q + 路径推断 domain。
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
    domain_ids: List[int] = []

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
        path = b["path"]
        paths.append(path)
        if domain_map is not None:
            key = _group_key_from_path(path)
            domain_ids.append(domain_map.get(key, -1))

    max_len = max(lengths)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack([b["label_id"] for b in batch], dim=0)

    if domain_map is None:
        domain_tensor = torch.full((B,), -1, dtype=torch.long)
    else:
        domain_tensor = torch.tensor(domain_ids, dtype=torch.long)

    return {
        "audio": audio,
        "sensor": sensor_padded,
        "sensor_lengths": lengths_tensor,
        "label": labels,
        "domain": domain_tensor,
        "paths": paths,
    }


# ===========================
# 2. split：按录制单位分组
# ===========================

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
            raise ValueError(f"Mixed labels inside group {p}: '{norm_label}' vs '{current}'")
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


# ===========================
# 3. 模型定义（DANN）
# ===========================

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
        use_pretrained: bool = True,
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

        if use_pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)
        _replace_bn_with_gn(self.backbone, num_groups=8)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1).squeeze(1)
        mel = self.melspec(x)
        mel_db = self.db(mel).unsqueeze(1)
        return self.backbone(mel_db)


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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _ = lengths
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


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: torch.Tensor, coeff: float) -> torch.Tensor:
        return GradientReversalFn.apply(x, float(coeff))


class DANNMultiModalNet(nn.Module):
    def __init__(
        self,
        audio_feat_dim: int,
        sensor_feat_dim: int,
        num_classes: int,
        num_domains: int,
        sample_rate: int = 48000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.audio_encoder = AudioResNetEncoder(
            sample_rate=sample_rate,
            out_dim=audio_feat_dim,
            f_min=f_min,
            f_max=f_max,
            use_pretrained=use_pretrained,
        )
        self.sensor_encoder = InceptionTimeEncoder(
            input_dim=2,
            out_dim=sensor_feat_dim,
        )
        fused_dim = audio_feat_dim + sensor_feat_dim
        self.class_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.use_domain = num_domains > 1
        self.domain_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, max(1, num_domains)),
        )
        self.grl = GradientReversal()

    def forward(
        self,
        audio: torch.Tensor,
        sensor: torch.Tensor,
        sensor_lengths: torch.Tensor,
        grl_coeff: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fa = self.audio_encoder(audio)
        fs = self.sensor_encoder(sensor, sensor_lengths)
        feats = torch.cat([fa, fs], dim=-1)
        class_logits = self.class_head(feats)
        if self.use_domain:
            domain_logits = self.domain_head(self.grl(feats, grl_coeff))
        else:
            domain_logits = None
        return class_logits, domain_logits


# ===========================
# 4. 训练 & 验证
# ===========================

def dann_coeff(progress: float, max_coeff: float = 1.0) -> float:
    return max_coeff * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def train_one_epoch(
    model: DANNMultiModalNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    max_grl_coeff: float,
    domain_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls = 0.0
    total_dom = 0.0
    total_samples = 0
    correct = 0
    dom_correct = 0
    dom_total = 0
    steps = max(1, len(loader))

    for step, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        sensor = batch["sensor"].to(device)
        lengths = batch["sensor_lengths"].to(device)
        labels = batch["label"].to(device)
        domain_ids = batch["domain"].to(device)

        progress = (epoch - 1 + step / steps) / max(1, num_epochs)
        grl_coeff = dann_coeff(progress, max_coeff=max_grl_coeff)

        optimizer.zero_grad()
        class_logits, domain_logits = model(
            audio, sensor, lengths, grl_coeff=grl_coeff
        )
        cls_loss = ce(class_logits, labels)
        loss = cls_loss

        dom_loss = torch.tensor(0.0, device=device)
        if domain_logits is not None:
            dom_mask = domain_ids >= 0
            if dom_mask.any():
                dom_loss = ce(domain_logits[dom_mask], domain_ids[dom_mask])
                loss = loss + domain_loss_weight * dom_loss
                dom_preds = domain_logits.argmax(dim=-1)
                dom_correct += (dom_preds[dom_mask] == domain_ids[dom_mask]).sum().item()
                dom_total += int(dom_mask.sum().item())

        loss.backward()
        optimizer.step()

        preds = class_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        bs = labels.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_dom += dom_loss.item() * bs

    metrics = {
        "loss": total_loss / max(1, total_samples),
        "cls_loss": total_cls / max(1, total_samples),
        "dom_loss": total_dom / max(1, total_samples),
        "acc": correct / max(1, total_samples),
        "dom_acc": dom_correct / max(1, dom_total) if dom_total > 0 else 0.0,
    }
    return metrics


@torch.no_grad()
def eval_one_epoch(
    model: DANNMultiModalNet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in loader:
        audio = batch["audio"].to(device)
        sensor = batch["sensor"].to(device)
        lengths = batch["sensor_lengths"].to(device)
        labels = batch["label"].to(device)

        logits, _ = model(audio, sensor, lengths, grl_coeff=0.0)
        loss = ce(logits, labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        bs = labels.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        for t, p in zip(labels, preds):
            confusion[t.item(), p.item()] += 1

    precision = confusion.diag().float() / confusion.sum(0).clamp(min=1).float()
    recall = confusion.diag().float() / confusion.sum(1).clamp(min=1).float()
    acc = correct / max(1, total_samples)
    return total_loss / max(1, total_samples), acc, confusion, precision, recall


# ===========================
# 5. main：串起来
# ===========================

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

    domain_map = build_domain_map(train_paths)
    log(f"num_domains = {len(domain_map)}")

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

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, domain_map=domain_map),
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, domain_map=domain_map),
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    if force_cpu:
        torch.backends.cudnn.enabled = False

    use_pretrained = os.environ.get("USE_IMAGENET", "1") == "1"
    model = DANNMultiModalNet(
        audio_feat_dim=128,
        sensor_feat_dim=128,
        num_classes=num_classes,
        num_domains=len(domain_map),
        sample_rate=target_sr,
        f_max=4000,
        use_pretrained=use_pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=1e-4, foreach=False
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 40
    min_epochs = 30
    early_stop_patience = 5
    best_val_acc = 0.0
    epochs_since_best = 0
    max_grl_coeff = 1.0
    domain_loss_weight = 1.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            num_epochs,
            max_grl_coeff,
            domain_loss_weight,
        )
        val_loss, val_acc, confusion, precision, recall = eval_one_epoch(
            model, val_loader, device, num_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:02d}: lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['cls_loss']:.4f}, dom={train_metrics['dom_loss']:.4f}) | "
            f"train_acc={train_metrics['acc']:.4f} dom_acc={train_metrics['dom_acc']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        log("  Confusion matrix (rows=true, cols=pred):")
        log(str(confusion))
        log("  Per-class precision/recall:")
        for i, name in enumerate([train_ds.id2label[i] for i in range(num_classes)]):
            log(
                f"    {name}: precision={precision[i].item():.3f}, "
                f"recall={recall[i].item():.3f}"
            )

        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join("checkpoints", "best_multimodal_dann.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2id": train_ds.label2id,
                    "domain_map": domain_map,
                },
                ckpt_path,
            )
            log(f"  -> New best val_acc={val_acc:.4f}! Saved to {ckpt_path}")
            epochs_since_best = 0
            improved = True
        else:
            epochs_since_best += 1

        scheduler.step(val_acc)

        if (not improved) and epoch >= min_epochs and epochs_since_best >= early_stop_patience:
            log(
                f"Early stopping triggered at epoch {epoch}: "
                f"no val-accuracy improvement for {early_stop_patience} epochs."
            )
            break

    log_file.close()


if __name__ == "__main__":
    main()
