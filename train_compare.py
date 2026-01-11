import atexit
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
# 1. collate_fn：打 batch
# ===========================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    batch 里的每个元素是 ReadSegments.__getitem__ 返回的 dict：
      - "audio": [T_audio]
      - "P": [T] 或 [T, n_P] 或 None
      - "Q": [T] 或 [T, n_Q] 或 None
      - "label_id": scalar tensor
      - "path": 样本路径（字符串）
    这里我们只用 P 和 Q（温度先不管）。
    """

    # ========= audio: 变长，padding =========
    audio_list = [b["audio"] for b in batch]     # list of [T_audio]
    audio_lengths = [a.shape[0] for a in audio_list]
    max_T_audio = max(audio_lengths)
    B = len(batch)

    # [B, max_T_audio]
    audio = torch.zeros(B, max_T_audio, dtype=torch.float32)
    for i, (a, L) in enumerate(zip(audio_list, audio_lengths)):
        audio[i, :L] = a

    # ========= sensor: P/Q 做 padding =========
    sensor_seqs = []
    lengths = []
    paths = []

    for b in batch:
        P = b["P"]  # [T] or [T, n] or None
        Q = b["Q"]  # [T] or [T, n] or None

        if P is None or Q is None:
            # 极端情况：如果某个 segment 没有 P 或 Q，就用全 0 占位
            T = b["audio"].shape[0]
            sensor = torch.zeros(T, 2, dtype=torch.float32)
        else:
            # 把 1D 变成 [T,1]
            if P.dim() == 1:
                P_2d = P.unsqueeze(-1)
            else:
                P_2d = P  # [T, n_P]
            if Q.dim() == 1:
                Q_2d = Q.unsqueeze(-1)
            else:
                Q_2d = Q  # [T, n_Q]

            # 这里只取第一列作为主 P/Q 通道；如果你有多列可以再拼
            P_main = P_2d[:, 0:1]  # [T,1]
            Q_main = Q_2d[:, 0:1]  # [T,1]
            sensor = torch.cat([P_main, Q_main], dim=-1)  # [T,2]

        sensor_seqs.append(sensor)
        lengths.append(sensor.shape[0])
        paths.append(b["path"])

    max_len = max(lengths)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)  # [B, T_max, 2]
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)  # [B]

    # labels
    labels = torch.stack([b["label_id"] for b in batch], dim=0)  # [B]

    return {
        "audio": audio,               # [B, T_audio_max]
        "sensor": sensor_padded,      # [B, T_sensor_max, 2]
        "sensor_lengths": lengths_tensor,
        "label": labels,
        "paths": paths,
    }

# ===========================
# 0. 数据集分组工具（按录制源拆分，避免同一录音片段分到不同划分）
# ===========================

def _group_key_from_path(path: str) -> str:
    """
    默认用文件名去掉 _winxxxxx 的前缀作为录制单位，
    例如 xxx__sensor_win00001.npz -> xxx__sensor。
    如果没有 _win，退化为去掉扩展名。
    """
    base = os.path.basename(path)
    if "_win" in base:
        return base.split("_win")[0]
    return os.path.splitext(base)[0]

def _infer_group_label(group_paths: List[str], label_normalizer) -> str:
    """
    Infer and validate that a group of segment paths all share the same label.
    Raises if mixed labels are found.
    """
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
    """
    Group segments by recording unit (filename without _win suffix) then split into
    train/val groups. If label_normalizer is provided, perform a stratified split
    at the group level to keep class balance while still preventing leakage.
    """
    groups: Dict[str, List[str]] = {}
    for p in paths:
        key = _group_key_from_path(p)
        groups.setdefault(key, []).append(p)

    rng = random.Random(seed)

    if label_normalizer is not None:
        # Stratified split by group label to avoid dumping most negatives into val
        label_to_keys: Dict[str, List[str]] = {}
        for k, ps in groups.items():
            try:
                norm_label = _infer_group_label(ps, label_normalizer)
            except DropLabel:
                # Skip groups whose labels are configured to be dropped
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
        # 1. 用稳定顺序（比如按 key 排序），再洗牌
        keys = sorted(groups.keys())
        rng.shuffle(keys)

        n_train = int(train_ratio * len(keys))
        if len(keys) > 1:
            n_train = min(max(n_train, 1), len(keys) - 1)

        # 2. 不要用 set，当 list 用，保证顺序可控
        train_keys = keys[:n_train]
        val_keys   = keys[n_train:]

    # 3. 也可以顺带把组内的 path 排序一下，进一步稳定
    train_paths = [p for k in train_keys for p in sorted(groups[k])]
    val_paths   = [p for k in val_keys   for p in sorted(groups[k])]

    return train_paths, val_paths


# ===========================
# 2. 模型定义
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
    """
    waveform -> MelSpectrogram -> ResNet18 -> audio feature
    """
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

        # resnet18 backbone（加载 ImageNet 预训练以提升收敛速度）
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _replace_bn_with_gn(self.backbone, num_groups=8)
        # 把输入改成单通道
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 替换最后一层为 out_dim
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [B, T] waveform
        返回: [B, out_dim]
        """
        # [B, T] -> [B, 1, T]
        x = audio.unsqueeze(1)
        x = x.squeeze(1)  # [B, T]
        mel = self.melspec(x)   # [B, n_mels, T_frames]
        mel_db = self.db(mel)   # [B, n_mels, T_frames]
        mel_db = mel_db.unsqueeze(1)  # [B, 1, n_mels, T_frames]

        feat = self.backbone(mel_db)  # [B, out_dim]
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
    """
    P/Q 序列 -> InceptionTime -> sensor feature
    """

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
        """
        x: [B, T, 2]
        lengths: [B] 真实长度（Inception 使用零填充，lengths 仅保留接口兼容）
        返回: [B, out_dim]
        """
        _ = lengths
        x = x.transpose(1, 2)  # [B, 2, T]
        for i, block in enumerate(self.blocks):
            if self.use_residual and i % 3 == 0:
                res_input = x
            x = block(x)
            if self.use_residual and (i + 1) % 3 == 0:
                res = self.residuals[str(i)](res_input)
                x = self.res_relu(x + res)
        feats = self.global_pool(x).squeeze(-1)
        return self.proj(feats)


class MultiModalLateFusionNet(nn.Module):
    """
    - audio_encoder: ResNet18 over Mel
    - sensor_encoder: GRU over P/Q
    - late fusion: concat -> MLP -> logits
    """
    def __init__(
        self,
        audio_feat_dim: int = 128,
        sensor_feat_dim: int = 128,
        num_classes: int = 2,
        sample_rate: int = 48000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.audio_encoder = AudioResNetEncoder(
            sample_rate=sample_rate,
            out_dim=audio_feat_dim,
            f_min=f_min,
            f_max=f_max,
        )
        self.sensor_encoder = InceptionTimeEncoder(
            input_dim=2,
            out_dim=sensor_feat_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(audio_feat_dim + sensor_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        audio: torch.Tensor,
        sensor: torch.Tensor,
        sensor_lengths: torch.Tensor,
        drop_audio: bool = False,
        drop_sensor: bool = False,
    ) -> torch.Tensor:
        """
        audio: [B, T_audio]
        sensor: [B, T_max, 2]
        sensor_lengths: [B]

        drop_audio=True  -> 把 audio 分支置零，看只有 sensor 的表现
        drop_sensor=True -> 把 sensor 分支置零，看只有 audio 的表现
        """
        fa = self.audio_encoder(audio)                    # [B, A]
        fs = self.sensor_encoder(sensor, sensor_lengths)  # [B, S]

        if drop_audio:
            fa = torch.zeros_like(fa)
        if drop_sensor:
            fs = torch.zeros_like(fs)

        fused = torch.cat([fa, fs], dim=-1)               # [B, A+S]
        logits = self.fusion(fused)                       # [B, num_classes]
        return logits


# ===========================
# 3. 训练 & 验证
# ===========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        audio = batch["audio"].to(device)               # [B, T_audio]
        sensor = batch["sensor"].to(device)             # [B, T_max, 2]
        lengths = batch["sensor_lengths"].to(device)    # [B]
        labels = batch["label"].to(device)              # [B]

        optimizer.zero_grad()
        logits = model(audio, sensor, lengths, drop_audio=False, drop_sensor=False)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def eval_modal_contribution(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    id2label: Dict[int, str],
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Dict[str, str]],
]:
    """
    返回：
      - (full_loss, full_acc): audio+sensor
      - (audio_loss, audio_acc): 只用 audio（sensor 置零）
      - (sensor_loss, sensor_acc): 只用 sensor（audio 置零）
      - confusion: [C, C] full 模式混淆矩阵（行真值，列预测）
      - precision: [C] full 模式逐类 precision
      - recall: [C] full 模式逐类 recall
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    stats = {
        "full": {"loss": 0.0, "correct": 0, "total": 0},
        "audio_only": {"loss": 0.0, "correct": 0, "total": 0},
        "sensor_only": {"loss": 0.0, "correct": 0, "total": 0},
    }

    full_preds_all = []
    full_labels_all = []
    misclassified_samples: List[Dict[str, str]] = []

    for batch in loader:
        audio = batch["audio"].to(device)
        sensor = batch["sensor"].to(device)
        lengths = batch["sensor_lengths"].to(device)
        labels = batch["label"].to(device)
        paths = batch["paths"]
        bs = labels.size(0)

        # 1) full
        logits_full = model(audio, sensor, lengths,
                            drop_audio=False, drop_sensor=False)
        loss_full = ce(logits_full, labels)
        preds_full = logits_full.argmax(dim=-1)
        stats["full"]["loss"] += loss_full.item() * bs
        correct_mask = (preds_full == labels)
        stats["full"]["correct"] += correct_mask.sum().item()
        stats["full"]["total"] += bs
        full_preds_all.append(preds_full.detach().cpu())
        full_labels_all.append(labels.detach().cpu())
        for idx_in_batch in range(bs):
            if not correct_mask[idx_in_batch]:
                true_id = labels[idx_in_batch].item()
                pred_id = preds_full[idx_in_batch].item()
                misclassified_samples.append(
                    {
                        "path": paths[idx_in_batch],
                        "true_label": id2label.get(true_id, str(true_id)),
                        "pred_label": id2label.get(pred_id, str(pred_id)),
                    }
                )

        # 2) audio only
        logits_a = model(audio, sensor, lengths,
                         drop_audio=False, drop_sensor=True)
        loss_a = ce(logits_a, labels)
        preds_a = logits_a.argmax(dim=-1)
        stats["audio_only"]["loss"] += loss_a.item() * bs
        stats["audio_only"]["correct"] += (preds_a == labels).sum().item()
        stats["audio_only"]["total"] += bs

        # 3) sensor only
        logits_s = model(audio, sensor, lengths,
                         drop_audio=True, drop_sensor=False)
        loss_s = ce(logits_s, labels)
        preds_s = logits_s.argmax(dim=-1)
        stats["sensor_only"]["loss"] += loss_s.item() * bs
        stats["sensor_only"]["correct"] += (preds_s == labels).sum().item()
        stats["sensor_only"]["total"] += bs

    def _summ(name):
        loss = stats[name]["loss"] / max(1, stats[name]["total"])
        acc = stats[name]["correct"] / max(1, stats[name]["total"])
        return loss, acc

    full_loss, full_acc = _summ("full")
    a_loss, a_acc = _summ("audio_only")
    s_loss, s_acc = _summ("sensor_only")

    if full_preds_all:
        preds_cat = torch.cat(full_preds_all)
        labels_cat = torch.cat(full_labels_all)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(labels_cat, preds_cat):
            confusion[t.item(), p.item()] += 1
        precision = confusion.diag().float() / confusion.sum(0).clamp(min=1).float()
        recall = confusion.diag().float() / confusion.sum(1).clamp(min=1).float()
    else:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        precision = torch.zeros(num_classes, dtype=torch.float32)
        recall = torch.zeros(num_classes, dtype=torch.float32)

    return (
        (full_loss, full_acc),
        (a_loss, a_acc),
        (s_loss, s_acc),
        confusion,
        precision,
        recall,
        misclassified_samples,
    )


# ===========================
# 4. main：串起来
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
    # 1) 找所有 segment .npz
    # 你可以改成更具体的目录，比如 "data/processed/segments_10s"
    all_paths = sorted(find_segment_paths("./data"))

    if not all_paths:
        raise RuntimeError("No .npz found! Please run preprocess_segments.py first.")

    # 固定随机种子，保证可复现
    seed = 42
    set_seed(seed)
    # Deterministic DataLoader shuffling
    dl_generator = torch.Generator().manual_seed(seed)

    target_sr = 22050  # 统一重采样到 22.05kHz（要和 ReadSegments 里的一致）

    # Label 配置：raw 文本 -> 类别 token（字符串）。未匹配将直接报错（fail fast）。
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

    # 按“录制单位”分组切分，避免同一录音/传感器的多个 10s 片段落入不同划分造成泄漏。
    # 同时按 label 分层，防止几乎所有负样本被分到验证集。
    train_paths, val_paths = split_paths_by_group(
        all_paths, train_ratio=0.6, seed=seed, label_normalizer=label_processor
    )
    n = len(all_paths)
    log(
        f"Total segments: {n}, train={len(train_paths)}, val={len(val_paths)}, "
        f"groups={len(set(_group_key_from_path(p) for p in all_paths))}"
    )

    # 2) 构建 Dataset
    train_ds = ReadSegments(
        train_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
        group_normalize=True,
    )
    # 验证集复用训练集的 label2id，防止类别排序/缺失造成指标错位
    val_ds   = ReadSegments(
        val_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
        label2id=train_ds.label2id,
        label_map_paths=train_paths,
        group_normalize=True,
    )

    num_classes = len(train_ds.label2id)
    log("num_classes = {} label2id = {}".format(num_classes, train_ds.label2id))

    # Compute class-balanced sampling weights
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

    # 3) DataLoader
    # Use single-worker DataLoaders for strict determinism; multi-worker + shared RNG
    # can introduce subtle nondeterminism even with seeding.
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    # 4) 模型 & 优化器
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    if force_cpu:
        # Avoid cuDNN/CUDA to get a fully deterministic CPU path when needed
        torch.backends.cudnn.enabled = False
    model = MultiModalLateFusionNet(
        audio_feat_dim=128,
        sensor_feat_dim=128,
        num_classes=num_classes,
        sample_rate=target_sr,
        f_max=4000
    ).to(device)

    # Disable foreach/fused variants to avoid GPU atomic nondeterminism on some PyTorch builds
    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=1e-4, foreach=False
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    os.makedirs("checkpoints", exist_ok=True)

    # 5) 训练若干 epoch + 模态贡献对比
    num_epochs = 40
    min_epochs = 30
    early_stop_patience = 5
    best_val_full_acc = 0.0
    epochs_since_best = 0
    full_acc_history: List[float] = []
    audio_acc_history: List[float] = []
    sensor_acc_history: List[float] = []
    best_misclassified_samples: List[Dict[str, str]] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        (
            (full_loss, full_acc),
            (a_loss, a_acc),
            (s_loss, s_acc),
            confusion,
            precision,
            recall,
            misclassified_samples,
        ) = eval_modal_contribution(
            model, val_loader, device, num_classes, train_ds.id2label
        )
        full_acc_history.append(full_acc)
        audio_acc_history.append(a_acc)
        sensor_acc_history.append(s_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:02d}: lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"full: loss={full_loss:.4f}, acc={full_acc:.4f} | "
            f"audio_only: loss={a_loss:.4f}, acc={a_acc:.4f} | "
            f"sensor_only: loss={s_loss:.4f}, acc={s_acc:.4f}"
        )
        label_names = [train_ds.id2label[i] for i in range(num_classes)]
        log("  Confusion matrix (rows=true, cols=pred):")
        log(str(confusion))
        log("  Per-class precision/recall:")
        for i, name in enumerate(label_names):
            log(
                f"    {name}: precision={precision[i].item():.3f}, "
                f"recall={recall[i].item():.3f}"
            )

        improved = False
        # 用 full 模式的 acc 作为 best 标准
        if full_acc > best_val_full_acc:
            best_val_full_acc = full_acc
            ckpt_path = os.path.join("checkpoints", "best_multimodal.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2id": train_ds.label2id,
                },
                ckpt_path,
            )
            log(f"  -> New best full_acc={full_acc:.4f}! Saved to {ckpt_path}")
            epochs_since_best = 0
            improved = True
            best_misclassified_samples = list(misclassified_samples)
        else:
            epochs_since_best += 1

        scheduler.step(full_acc)

        if (not improved) and epoch >= min_epochs and epochs_since_best >= early_stop_patience:
            log(
                f"Early stopping triggered at epoch {epoch}: "
                f"no full-accuracy improvement for {early_stop_patience} epochs."
            )
            break

    if full_acc_history:
        log("Run summary:")
        log(
            "  Full acc mean={:.4f}, best={:.4f}".format(
                float(np.mean(full_acc_history)), float(np.max(full_acc_history))
            )
        )
        log(
            "  Audio-only acc mean={:.4f}, best={:.4f}".format(
                float(np.mean(audio_acc_history)), float(np.max(audio_acc_history))
            )
        )
        log(
            "  Sensor-only acc mean={:.4f}, best={:.4f}".format(
                float(np.mean(sensor_acc_history)), float(np.max(sensor_acc_history))
            )
        )
    if best_misclassified_samples:
        log("Misclassified samples from best epoch:")
        for item in best_misclassified_samples:
            log(
                f"  path={item['path']} | true={item['true_label']} | pred={item['pred_label']}"
            )
    else:
        log("Misclassified samples from best epoch: none (perfect accuracy).")

    log_file.close()


if __name__ == "__main__":
    main()
