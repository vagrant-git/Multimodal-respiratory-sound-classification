import os
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchaudio
import torchvision.models as models

# 这里按你前面约定的路径来，注意路径
from src.ReadSegments import ReadSegments, find_segment_paths
from util.label_processor import LabelProcessor
from util.seed import set_seed


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
    # 音频：10s segment，长度应该都一样，直接 stack
    audio_list = [b["audio"] for b in batch]  # list of [T_audio]
    audio = torch.stack(audio_list, dim=0)    # [B, T_audio]

    # 传感器：P 和 Q 可能是 [T] 或 [T,1]，我们统一成 [T, 1] 再拼成 [T, 2]
    sensor_seqs = []
    lengths = []

    for b in batch:
        P = b["P"]  # [T] or [T, n] or None
        Q = b["Q"]  # [T] or [T, n] or None

        if P is None or Q is None:
            # 极端情况：如果某个 segment 没有 P 或 Q，就用全 0 占位
            # 这里假设至少有 audio，长度作为时间长度基准
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

    # 按当前 batch 的最大长度做 padding
    max_len = max(lengths)
    B = len(batch)
    sensor_padded = torch.zeros(B, max_len, 2, dtype=torch.float32)  # [B, T_max, 2]
    for i, (seq, L) in enumerate(zip(sensor_seqs, lengths)):
        sensor_padded[i, :L, :] = seq

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)  # [B]

    # labels
    labels = torch.stack([b["label_id"] for b in batch], dim=0)  # [B]

    return {
        "audio": audio,               # [B, T_audio]
        "sensor": sensor_padded,      # [B, T_max, 2]
        "sensor_lengths": lengths_tensor,
        "label": labels,
    }


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

        # resnet18 backbone
        # self.backbone = models.resnet18(weights=None)  # 不加载预训练，避免联网
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
        # 逐 batch 计算 Mel
        # torchaudio 的 MelSpectrogram 要求 [B, T]，所以先 squeeze 掉 channel
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
        # pack
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: [num_layers * num_directions, B, H]

        if self.bidirectional:
            # 取最后一层的正向 + 反向 hidden 拼接
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
    ) -> torch.Tensor:
        """
        audio: [B, T_audio]
        sensor: [B, T_max, 2]
        sensor_lengths: [B]
        """
        fa = self.audio_encoder(audio)                    # [B, A]
        fs = self.sensor_encoder(sensor, sensor_lengths)  # [B, S]
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
        logits = model(audio, sensor, lengths)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        sensor = batch["sensor"].to(device)
        lengths = batch["sensor_lengths"].to(device)
        labels = batch["label"].to(device)

        logits = model(audio, sensor, lengths)
        loss = ce(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc


# ===========================
# 4. main：串起来
# ===========================

def main():
    # 1) 找所有 segment .npz
    all_paths = find_segment_paths("./data")
    if not all_paths:
        raise RuntimeError("No .npz found! Please run preprocess_segments.py first.")

    seed = 42
    set_seed(seed)
    random.shuffle(all_paths)
    n = len(all_paths)
    n_train = int(0.8 * n)
    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:]

    print(f"Total segments: {n}, train={len(train_paths)}, val={len(val_paths)}")

    target_sr = 22050

    # Label 配置：raw 文本 -> 类别 token（字符串）。未匹配将直接报错（fail fast）。
    label_processor = LabelProcessor(
        raw_to_norm={
            "no secretion": "no_secretion",
            "no secretion sound": "no_secretion",
            "3ml secretion": "secretion_3ml",
            "3ml secretion m4": "secretion_3ml_m4",
            "5ml secretion m4": "secretion_5ml_m4",
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
    )   # 使用自定义 label_processor
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
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # 4) 模型 & 优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalLateFusionNet(
        audio_feat_dim=128,
        sensor_feat_dim=128,
        num_classes=num_classes,
        sample_rate=target_sr,   # 如果你在预处理里降采样了，这里要改
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("checkpoints", exist_ok=True)

    # 5) 训练若干 epoch
    num_epochs = 20
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        # 简单保存 best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join("checkpoints", "best_multimodal.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label2id": train_ds.label2id,
                },
                ckpt_path,
            )
            print(f"  -> New best! Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
