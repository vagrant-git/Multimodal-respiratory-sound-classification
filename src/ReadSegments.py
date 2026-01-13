# src/ReadSegments.py
import glob
from collections import Counter
from typing import List, Sequence, Dict, Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.functional as AF

from util.label_processor import LabelProcessor, DropLabel


DEFAULT_LABEL_PROCESSOR = LabelProcessor()


def find_segment_paths(root_or_pattern: str) -> List[str]:
    """
    给一个目录或 glob pattern，返回所有 .npz segment 的路径列表。

    例子：
        paths = find_segment_paths("data/processed/segments_10s/**/*.npz")
        paths = find_segment_paths("data/processed/segments_10s")
    """
    # 如果传的是目录，就自动加上递归通配
    if not any(ch in root_or_pattern for ch in ["*", "?", "["]):
        pattern = root_or_pattern.rstrip("/\\") + "/**/*.npz"
    else:
        pattern = root_or_pattern

    paths = sorted(glob.glob(pattern, recursive=True))
    return paths


def read_labels_from_segments(paths: Sequence[str]) -> List[str]:
    """
    读取一批 .npz segment 里的原始 label（字符串），返回一个列表。
    只是“如实读取”，不做去重、不做映射。
    """
    labels = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if "label" not in d:
            continue
        lab = d["label"]
        # 可能是 0-dim numpy array
        if isinstance(lab, np.ndarray) and lab.shape == ():
            lab = lab.item()
        if lab is None:
            continue
        labels.append(str(lab))
    return labels

class ReadSegments(Dataset):
    """
    读取 .npz segment 的 Dataset：

      - audio: 波形 [T_audio_resampled]
      - sensor_all: 所有传感器通道 [T_sensor, C]
      - P: 从 sensor 里拆出来的压力通道 (根据列名里是否包含 "P_")
      - Q: 从 sensor 里拆出来的流量通道 (根据列名里是否包含 "F_")
      - T: 温度通道 (列名包含 "T_")，如果有的话
      - label_id: 分类用的整数标签
      - label_ml: 以 mL 数值表示的标签 (float)，比如 0.0 / 3.0
      - raw_label: 原始字符串
      - norm_label: 归一化后的字符串（这里就是提取出来的数字字符串）
      - sample_rate: 当前 audio 的采样率（重采样后）

    内置 label 规则（默认）：
      - "no secretion" / "no" / "nosecretion" 等 -> "0"
      - 含有 "3mL", "3 ml" 等 -> "3"
      - 其他没提取到数字的情况 -> "0"
    """

    def __init__(
        self,
        paths: Sequence[str],
        label_normalizer: Optional[Callable[[str], str]] = None,
        target_sample_rate: Optional[int] = None,
        label2id: Optional[Dict[str, int]] = None,
        label_map_paths: Optional[Sequence[str]] = None,
        group_normalize: bool = False,
    ):
        """
        paths: 一批 .npz 文件的路径
        label_normalizer: 可选，自定义的 label 归一化函数(raw_label -> norm_label字符串)。
                          如果为 None，默认使用 util.label_processor.LabelProcessor()，基于显式字典表映射，
                          并对未知标签直接报错（fail fast）。
        target_sample_rate: 如果不为 None，会把每个 audio 从它自己的 sr 重采样到这个 sr。
        label2id: 可选，直接重用已经构建的 label->id 映射，例如使用训练集的映射，避免验证集重建造成缺失
        label_map_paths: 可选，指定用来构建 label 映射的 paths，默认使用自身 self.paths
        """
        raw_paths = list(paths)
        if len(raw_paths) == 0:
            raise ValueError("ReadSegments: no .npz files found!")

        self.target_sample_rate = target_sample_rate
        self.group_normalize = group_normalize

        if label_normalizer is None:
            self.label_processor = DEFAULT_LABEL_PROCESSOR
            label_normalizer = self.label_processor
        else:
            self.label_processor = label_normalizer

        if not callable(label_normalizer):
            raise TypeError("label_normalizer must be callable: raw_label -> normalized str")
        self.label_normalizer = label_normalizer

        # 预过滤：根据 label_normalizer 丢弃配置为 drop 的样本
        self.paths, dropped_count = self._filter_paths(raw_paths)
        if dropped_count > 0:
            print(f"Filtered out {dropped_count} samples due to label drop rules.")
        if len(self.paths) == 0:
            raise ValueError("ReadSegments: all samples were dropped by label rules.")

        # 扫一遍所有文件，建立或重用 label -> id 映射
        if label2id is not None:
            self.label2id = dict(label2id)
            self.id2label = {i: lab for lab, i in self.label2id.items()}
        else:
            paths_for_map = list(label_map_paths) if label_map_paths is not None else self.paths
            # 对映射构建也做同样的 drop 过滤
            paths_for_map, dropped_map = self._filter_paths(paths_for_map)
            if dropped_map > 0:
                print(f"(label_map_paths) Filtered out {dropped_map} samples due to label drop rules.")
            self.label2id, self.id2label = self._build_label_map(paths_for_map)
            print("Label map (normalized_label -> id):")
            for lab, idx in self.label2id.items():
                print(f"  id={idx}: {lab}")
            if isinstance(self.label_normalizer, LabelProcessor):
                print(self.label_normalizer.describe())

        self.group_stats = None
        if self.group_normalize:
            self.group_stats = self._compute_group_stats(self.paths)

    # ---------- 内置：从所有 segment 里读原始 label ----------

    @staticmethod
    def _read_labels_from_segments(paths: Sequence[str]) -> List[str]:
        labels: List[str] = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if "label" not in d:
                continue
            lab = d["label"]
            # 可能是 0-dim numpy array
            if isinstance(lab, np.ndarray) and lab.shape == ():
                lab = lab.item()
            if lab is None:
                continue
            labels.append(str(lab))
        return labels

    def _filter_paths(self, paths: Sequence[str]) -> Tuple[List[str], int]:
        """
        根据 label_normalizer 过滤掉标记为 drop 的样本。返回 (保留的路径, 丢弃数量)。
        """
        kept: List[str] = []
        dropped = 0
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if "label" not in d:
                dropped += 1
                continue
            lab = d["label"]
            if isinstance(lab, np.ndarray) and lab.shape == ():
                lab = lab.item()
            if lab is None:
                dropped += 1
                continue
            try:
                _ = self.label_normalizer(str(lab))
            except DropLabel:
                dropped += 1
                continue
            kept.append(p)
        return kept, dropped

    # ---------- 内置 label 归一化规则 ----------

    @staticmethod
    def _default_label_normalizer(raw: str) -> str:
        """
        默认行为：委托给 util.label_processor.LabelProcessor() 的默认映射（fail fast，可包含 drop）。
        """
        return DEFAULT_LABEL_PROCESSOR(raw)

    @staticmethod
    def _group_key_from_path(path: str) -> str:
        base = path.split("/")[-1]
        if "_win" in base:
            return base.split("_win")[0]
        return base.rsplit(".", 1)[0]

    @staticmethod
    def _extract_pq(sensor_all: np.ndarray, col_names: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        p_idx = [i for i, name in enumerate(col_names) if "P_" in name]
        q_idx = [i for i, name in enumerate(col_names) if "F_" in name]

        def _take_or_none(idxs):
            if len(idxs) == 0:
                return None
            vals = sensor_all[:, idxs]
            if vals.shape[1] == 1:
                vals = vals[:, 0]
            return vals

        return _take_or_none(p_idx), _take_or_none(q_idx)

    def _compute_group_stats(self, paths: Sequence[str]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if "sensor_values" not in d or "sensor_cols" not in d:
                continue
            sensor_all = d["sensor_values"].astype(np.float32)
            sensor_all = np.nan_to_num(sensor_all, nan=0.0, posinf=0.0, neginf=0.0)
            col_names = [str(c) for c in d["sensor_cols"]]
            P, Q = self._extract_pq(sensor_all, col_names)
            if P is None and Q is None:
                continue
            key = self._group_key_from_path(p)
            bucket = stats.setdefault(
                key,
                {"p_sum": 0.0, "p_sumsq": 0.0, "p_count": 0,
                 "q_sum": 0.0, "q_sumsq": 0.0, "q_count": 0},
            )
            if P is not None:
                p_vals = P[:, 0] if P.ndim == 2 else P
                bucket["p_sum"] += float(p_vals.sum())
                bucket["p_sumsq"] += float((p_vals ** 2).sum())
                bucket["p_count"] += int(p_vals.shape[0])
            if Q is not None:
                q_vals = Q[:, 0] if Q.ndim == 2 else Q
                bucket["q_sum"] += float(q_vals.sum())
                bucket["q_sumsq"] += float((q_vals ** 2).sum())
                bucket["q_count"] += int(q_vals.shape[0])

        out: Dict[str, Dict[str, float]] = {}
        for key, bucket in stats.items():
            p_count = max(1, bucket["p_count"])
            q_count = max(1, bucket["q_count"])
            p_mean = bucket["p_sum"] / p_count
            q_mean = bucket["q_sum"] / q_count
            p_var = max(0.0, bucket["p_sumsq"] / p_count - p_mean ** 2)
            q_var = max(0.0, bucket["q_sumsq"] / q_count - q_mean ** 2)
            out[key] = {
                "p_mean": p_mean,
                "p_std": max(p_var ** 0.5, 1e-6),
                "q_mean": q_mean,
                "q_std": max(q_var ** 0.5, 1e-6),
            }
        return out


        


    # ---------- label 映射 ----------

    def _build_label_map(self, paths_for_map: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        raw_labels = self._read_labels_from_segments(paths_for_map)
        if not raw_labels:
            raise ValueError("No labels found in .npz files (field 'label' missing or empty).")

        norm_labels = []
        dropped = 0
        for l in raw_labels:
            try:
                norm_labels.append(self.label_normalizer(str(l)))
            except DropLabel:
                dropped += 1
                continue
        if not norm_labels:
            raise ValueError("All labels were dropped by label rules when building label map.")

        counts = Counter(norm_labels)
        unique = sorted(set(norm_labels))
        label2id = {lab: i for i, lab in enumerate(unique)}
        id2label = {i: lab for lab, i in label2id.items()}
        print("Label counts (normalized):")
        for lab, cnt in counts.most_common():
            print(f"  {lab}: {cnt}")
        if dropped > 0:
            print(f"  dropped (not mapped): {dropped}")
        return label2id, id2label

    # ---------- Dataset 接口 ----------

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)

        # ===== 1) audio + 重采样 =====
        audio = d["audio"].astype(np.float32)  # [T_audio]
        # 从 npz 里拿原始采样率；如果没有这个字段就默认 48000
        sr = int(d.get("audio_rate_hz", 48000))

        # 如果指定了 target_sample_rate 且不相等，就重采样
        if self.target_sample_rate is not None and sr != self.target_sample_rate:
            wave = torch.from_numpy(audio)  # [T]
            wave = AF.resample(wave, sr, self.target_sample_rate)  # [T_resampled]
            audio = wave.numpy().astype(np.float32)
            sr = self.target_sample_rate

        audio_t = torch.from_numpy(audio)  # [T_audio_resampled]

        # ===== 2) sensor 全部数据 + 列名 =====
        sensor_all = d["sensor_values"].astype(np.float32)  # [T_sensor, C]
        sensor_all = np.nan_to_num(sensor_all, nan=0.0, posinf=0.0, neginf=0.0)
        sensor_cols = d["sensor_cols"]                      # array of str
        col_names = [str(c) for c in sensor_cols]

        # 根据列名拆出 P / Q / T
        P, Q = self._extract_pq(sensor_all, col_names)
        t_idx = [i for i, name in enumerate(col_names) if "T_" in name]
        T = sensor_all[:, t_idx] if len(t_idx) > 0 else None
        if T is not None and T.shape[1] == 1:
            T = T[:, 0]

        if self.group_normalize and self.group_stats is not None:
            key = self._group_key_from_path(path)
            stats = self.group_stats.get(key)
            if stats is not None:
                if P is not None:
                    p_vals = P[:, 0] if P.ndim == 2 else P
                    p_vals = (p_vals - stats["p_mean"]) / stats["p_std"]
                    P = p_vals if P.ndim == 1 else p_vals[:, None]
                if Q is not None:
                    q_vals = Q[:, 0] if Q.ndim == 2 else Q
                    q_vals = (q_vals - stats["q_mean"]) / stats["q_std"]
                    Q = q_vals if Q.ndim == 1 else q_vals[:, None]
        else:
            if P is not None:
                p_vals = P[:, 0] if P.ndim == 2 else P
                p_mean = float(p_vals.mean())
                p_std = float(p_vals.std()) or 1e-6
                p_vals = (p_vals - p_mean) / p_std
                P = p_vals if P.ndim == 1 else p_vals[:, None]
            if Q is not None:
                q_vals = Q[:, 0] if Q.ndim == 2 else Q
                q_mean = float(q_vals.mean())
                q_std = float(q_vals.std()) or 1e-6
                q_vals = (q_vals - q_mean) / q_std
                Q = q_vals if Q.ndim == 1 else q_vals[:, None]

        sensor_all_t = torch.from_numpy(sensor_all)
        P_t = torch.from_numpy(P) if P is not None else None
        Q_t = torch.from_numpy(Q) if Q is not None else None
        T_t = torch.from_numpy(T) if T is not None else None

        # ===== 3) label 处理：raw -> norm -> id & ml =====
        raw_label = d["label"]
        if isinstance(raw_label, np.ndarray) and raw_label.shape == ():
            raw_label = raw_label.item()
        raw_label = str(raw_label)

        try:
            norm_label = self.label_normalizer(raw_label)     # 比如 "no_secretion" / "secretion_3ml"
        except DropLabel:
            raise KeyError(f"Label '{raw_label}' is configured to be dropped but appeared in dataset; path={path}")
        if norm_label not in self.label2id:
            raise KeyError(
                f"Label '{norm_label}' (raw='{raw_label}') 不在传入的 label2id 映射中，"
                f"请确认验证/测试集的标签来源于训练集；path={path}"
            )
        label_id = self.label2id[norm_label]
        try:
            label_ml_value = float(norm_label)
        except ValueError:
            label_ml_value = 0.0

        sample = {
            "audio": audio_t,              # [T_audio_resampled]
            "sensor_all": sensor_all_t,    # [T_sensor, C]
            "P": P_t,                      # [T] or [T, n_P] or None
            "Q": Q_t,                      # [T] or [T, n_Q] or None
            "T": T_t,                      # [T] or None
            "label_id": torch.tensor(label_id, dtype=torch.long),       # 分类标签
            "label_ml": torch.tensor(label_ml_value, dtype=torch.float32),  # mL 数值
            "raw_label": raw_label,
            "norm_label": norm_label,
            "sample_rate": sr,
            "path": path,
        }
        return sample

    def get_label_id_list(self) -> List[int]:
        """
        Return a list of label ids aligned with self.paths.
        """
        label_ids: List[int] = []
        for path in self.paths:
            d = np.load(path, allow_pickle=True)
            raw_label = d["label"]
            if isinstance(raw_label, np.ndarray) and raw_label.shape == ():
                raw_label = raw_label.item()
            raw_label = str(raw_label)
            norm_label = self.label_normalizer(raw_label)
            label_ids.append(self.label2id[norm_label])
        return label_ids
