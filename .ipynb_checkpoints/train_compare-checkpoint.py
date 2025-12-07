import os
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import torchaudio
import torchvision.models as models

from src.ReadSegments import ReadSegments, find_segment_paths
from util.label_processor import LabelProcessor
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


def split_paths_by_group(paths: List[str], train_ratio: float = 0.6, seed: int = 42) -> Tuple[List[str], List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in paths:
        key = _group_key_from_path(p)
        groups.setdefault(key, []).append(p)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_train = int(train_ratio * len(keys))
    if len(keys) > 1:
        n_train = min(max(n_train, 1), len(keys) - 1)  # 保证两边都有数据
    train_keys = set(keys[:n_train])

    train_paths = [p for k in train_keys for p in groups[k]]
    val_paths = [p for k in keys[n_train:] for p in groups[k]]
    return train_paths, val_paths


# ===========================
# 2. 模型定义
# ===========================

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

        # resnet18 backbone（这里用了 ImageNet 预训练，如果你不想下权重可以改成 weights=None）
        # self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = models.resnet18(weights=None)
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


class SensorGRUEncoder(nn.Module):
    """
    P/Q 序列 -> GRU -> sensor feature
    """
    def __init__(
        self,
        input_dim: int = 2,   # P 和 Q 两个通道
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        out_dim: int = 128,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2]
        lengths: [B] 真实长度
        返回: [B, out_dim]
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: [num_layers * num_directions, B, H]

        if self.bidirectional:
            h_forward = h_n[-2, :, :]  # [B, H]
            h_backward = h_n[-1, :, :] # [B, H]
            h = torch.cat([h_forward, h_backward], dim=-1)  # [B, 2H]
        else:
            h = h_n[-1, :, :]  # [B, H]

        out = self.proj(h)  # [B, out_dim]
        return out


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
        self.sensor_encoder = SensorGRUEncoder(
            input_dim=2,
            hidden_dim=128,
            bidirectional=True,
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
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    返回：
      - (full_loss, full_acc): audio+sensor
      - (audio_loss, audio_acc): 只用 audio（sensor 置零）
      - (sensor_loss, sensor_acc): 只用 sensor（audio 置零）
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    stats = {
        "full": {"loss": 0.0, "correct": 0, "total": 0},
        "audio_only": {"loss": 0.0, "correct": 0, "total": 0},
        "sensor_only": {"loss": 0.0, "correct": 0, "total": 0},
    }

    for batch in loader:
        audio = batch["audio"].to(device)
        sensor = batch["sensor"].to(device)
        lengths = batch["sensor_lengths"].to(device)
        labels = batch["label"].to(device)
        bs = labels.size(0)

        # 1) full
        logits_full = model(audio, sensor, lengths,
                            drop_audio=False, drop_sensor=False)
        loss_full = ce(logits_full, labels)
        preds_full = logits_full.argmax(dim=-1)
        stats["full"]["loss"] += loss_full.item() * bs
        stats["full"]["correct"] += (preds_full == labels).sum().item()
        stats["full"]["total"] += bs

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

    return (full_loss, full_acc), (a_loss, a_acc), (s_loss, s_acc)


# ===========================
# 4. main：串起来
# ===========================

def main():
    # 1) 找所有 segment .npz
    # 你可以改成更具体的目录，比如 "data/processed/segments_10s"
    all_paths = find_segment_paths("./data")
    if not all_paths:
        raise RuntimeError("No .npz found! Please run preprocess_segments.py first.")

    # 固定随机种子，保证可复现
    seed = 42
    set_seed(seed)
    # Deterministic DataLoader shuffling
    dl_generator = torch.Generator().manual_seed(seed)

    # 按“录制单位”分组切分，避免同一录音/传感器的多个 10s 片段落入不同划分造成泄漏
    train_paths, val_paths = split_paths_by_group(all_paths, train_ratio=0.6, seed=seed)
    n = len(all_paths)
    print(
        f"Total segments: {n}, train={len(train_paths)}, val={len(val_paths)}, "
        f"groups={len(set(_group_key_from_path(p) for p in all_paths))}"
    )

    target_sr = 16000  # 统一重采样到 16kHz（要和 ReadSegments 里的一致）

    # Label 配置：raw 文本 -> 类别 token（字符串）。未匹配将直接报错（fail fast）。
    label_processor = LabelProcessor(
        raw_to_norm={
            "no secretion": "no_secretion",
            "no secretion sound": "no_secretion",
            "3ml secretion": "secretion",
            "3ml secretion m4": "secretion",
            "5ml secretion m4": "secretion",
            "3ml secretion (no hemf)": "drop",  # 示例：直接丢弃 no HEMF 样本
        },
        fail_on_unknown=True,
    )
    print(label_processor.describe())

    # 2) 构建 Dataset
    train_ds = ReadSegments(
        train_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
    )
    # 验证集复用训练集的 label2id，防止类别排序/缺失造成指标错位
    val_ds   = ReadSegments(
        val_paths,
        target_sample_rate=target_sr,
        label_normalizer=label_processor,
        label2id=train_ds.label2id,
        label_map_paths=train_paths,
    )

    num_classes = len(train_ds.label2id)
    print("num_classes =", num_classes, "label2id =", train_ds.label2id)

    # 3) DataLoader
    # Use single-worker DataLoaders for strict determinism; multi-worker + shared RNG
    # can introduce subtle nondeterminism even with seeding.
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
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
        model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False
    )

    os.makedirs("checkpoints", exist_ok=True)

    # 5) 训练若干 epoch + 模态贡献对比
    num_epochs = 10
    best_val_full_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        (full_loss, full_acc), (a_loss, a_acc), (s_loss, s_acc) = eval_modal_contribution(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} | "
            f"full: loss={full_loss:.4f}, acc={full_acc:.4f} | "
            f"audio_only: loss={a_loss:.4f}, acc={a_acc:.4f} | "
            f"sensor_only: loss={s_loss:.4f}, acc={s_acc:.4f}"
        )

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
            print(f"  -> New best full_acc={full_acc:.4f}! Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
