
"""
multimodal_align.py  (many-to-many capable)

Adds support for:
- Multiple sensor EDF-like TSV files (each may overlap multiple WAVs)
- Multiple WAVs (each may overlap multiple EDFs)
- Per-row alignment to the best candidate audio among many, with tolerance
- Pairing report of interval overlaps (which EDF overlaps which WAV, and for how long)

Public API (in addition to previous functions):
- read_many_sensor_tsv_edf(filepaths) -> list[SensorMeta], dict[file->DataFrame]
- read_many_audio_with_json(wav_paths) -> list[AudioMeta]
- compute_pairings(sensor_metas, audio_metas) -> pd.DataFrame
- align_sensor_to_many_audios(sensor_df, audio_metas, tolerance_ms=20.0, sensor_time_col='Epoch_UTC')
- align_many_to_many(sensor_dict, audio_metas, tolerance_ms=20.0, sensor_time_col='Epoch_UTC')

The single-file readers remain:
- read_sensor_tsv_edf(filepath)
- read_audio_with_json(wav_path, json_path=None)
- align_sensor_to_audio(sensor_df, audio_meta, tolerance_ms=20.0, sensor_time_col='Epoch_UTC')

Assumptions follow the user's original description.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import wave
import contextlib
import os

# -------------------------
# Existing helpers/classes
# -------------------------

def _parse_iso8601_utc(s: str) -> float:
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def _safe_round(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))

@dataclass
class AudioMeta:
    wav_path: str
    json_path: str
    rate_hz: int
    channels: int
    dtype: str
    num_frames: int
    start_utc_s: float
    end_utc_s: Optional[float]
    wav_header_frames: int
    wav_duration_s: float

    @property
    def first_sample_utc(self) -> float:
        return self.start_utc_s

    @property
    def last_sample_utc_by_frames(self) -> float:
        return self.start_utc_s + (self.num_frames - 1) / float(self.rate_hz)

    @property
    def time_range(self) -> Tuple[float, float]:
        # Prefer frames-derived end for alignment
        return (self.first_sample_utc, self.last_sample_utc_by_frames)

    @property
    def audio_id(self) -> str:
        return os.path.basename(self.wav_path)

@dataclass
class SensorMeta:
    filepath: str
    start_utc_s: float
    end_utc_s: float
    n_rows: int

    @property
    def sensor_id(self) -> str:
        return os.path.basename(self.filepath)

    @property
    def time_range(self) -> Tuple[float, float]:
        return (self.start_utc_s, self.end_utc_s)

# -------------------------
# Single-file readers (as before)
# -------------------------

def read_sensor_tsv_edf(filepath: str) -> pd.DataFrame:
    header_line_index = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.startswith('#'):
                header_line_index = i
                break
    if header_line_index is None:
        raise ValueError("No header line found (first non-# line).")

    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=header_line_index
    )
    if 'Epoch_UTC' not in df.columns:
        candidates = [c for c in df.columns if c.lower().strip() in ('epoch_utc', 'epoch', 'utc_epoch')]
        if not candidates:
            raise ValueError("Column 'Epoch_UTC' not found.")
        df = df.rename(columns={candidates[0]: 'Epoch_UTC'})
    df['Epoch_UTC'] = df['Epoch_UTC'].astype(float)
    return df

def read_audio_with_json(wav_path: str, json_path: Optional[str] = None) -> AudioMeta:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        wav_nchannels = wf.getnchannels()
        wav_rate = wf.getframerate()
        wav_nframes = wf.getnframes()
        wav_duration = wav_nframes / float(wav_rate) if wav_rate > 0 else 0.0

    rate_hz = wav_rate
    channels = wav_nchannels
    dtype = 'int16'
    num_frames = wav_nframes
    start_utc_s = None
    end_utc_s = None
    jp = ""

    if json_path is None:
        base, _ = os.path.splitext(wav_path)
        candidate = base + '.json'
        if os.path.exists(candidate):
            json_path = candidate

    if json_path is not None and os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            j = json.load(f)
        jp = json_path
        rate_hz = int(j.get('rate_hz', rate_hz))
        channels = int(j.get('channels', channels))
        dtype = str(j.get('dtype', dtype))
        num_frames = int(j.get('num_frames', num_frames))
        start_utc_s = _parse_iso8601_utc(j['start_utc']) if 'start_utc' in j else None
        end_utc_s = _parse_iso8601_utc(j['end_utc']) if 'end_utc' in j else None

    if start_utc_s is None:
        base = os.path.basename(wav_path)
        stem, _ = os.path.splitext(base)
        token = stem.split('_')[-1]
        if token.endswith('Z') and 'T' in token:
            try:
                start_utc_s = _parse_iso8601_utc(
                    f"{token[0:4]}-{token[4:6]}-{token[6:8]}T{token[9:11]}:{token[11:13]}:{token[13:15]}Z"
                )
            except Exception:
                start_utc_s = None
    if start_utc_s is None:
        raise ValueError("Audio start UTC is required but not found in JSON or filename.")

    return AudioMeta(
        wav_path=wav_path,
        json_path=jp,
        rate_hz=rate_hz,
        channels=channels,
        dtype=dtype,
        num_frames=num_frames,
        start_utc_s=start_utc_s,
        end_utc_s=end_utc_s,
        wav_header_frames=wav_nframes,
        wav_duration_s=wav_duration
    )

# -------------------------
# Many-file readers & indexers
# -------------------------

def read_many_sensor_tsv_edf(filepaths: List[str]) -> Tuple[List[SensorMeta], Dict[str, pd.DataFrame]]:
    """
    Read multiple sensor EDF-like files.
    Returns (list of SensorMeta, dict from filepath -> DataFrame).
    """
    metas: List[SensorMeta] = []
    dfs: Dict[str, pd.DataFrame] = {}
    for fp in filepaths:
        df = read_sensor_tsv_edf(fp)
        dfs[fp] = df
        t0 = float(df['Epoch_UTC'].min())
        t1 = float(df['Epoch_UTC'].max())
        metas.append(SensorMeta(filepath=fp, start_utc_s=t0, end_utc_s=t1, n_rows=len(df)))
    return metas, dfs

def read_many_audio_with_json(wav_paths: List[str]) -> List[AudioMeta]:
    return [read_audio_with_json(p) for p in wav_paths]

# -------------------------
# Pairing: which EDF overlaps which WAV
# -------------------------

def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def compute_pairings(sensor_metas: List[SensorMeta], audio_metas: List[AudioMeta]) -> pd.DataFrame:
    """
    Return a table with all EDF–WAV pairs that have time overlap.
    Columns: sensor_id, audio_id, sensor_file, wav_path, sensor_start, sensor_end, audio_start, audio_end, overlap_sec
    """
    rows = []
    for sm in sensor_metas:
        s_rng = sm.time_range
        for am in audio_metas:
            a_rng = am.time_range
            ov = _interval_overlap(s_rng, a_rng)
            if ov > 0.0:
                rows.append(dict(
                    sensor_id=sm.sensor_id,
                    audio_id=am.audio_id,
                    sensor_file=sm.filepath,
                    wav_path=am.wav_path,
                    sensor_start=sm.start_utc_s,
                    sensor_end=sm.end_utc_s,
                    audio_start=am.first_sample_utc,
                    audio_end=am.last_sample_utc_by_frames,
                    overlap_sec=ov
                ))
    return pd.DataFrame(rows, columns=[
        'sensor_id','audio_id','sensor_file','wav_path',
        'sensor_start','sensor_end','audio_start','audio_end','overlap_sec'
    ])

# -------------------------
# Alignment: sensor -> best audio among many
# -------------------------

def align_sensor_to_many_audios(
    sensor_df: pd.DataFrame,
    audio_metas: List[AudioMeta],
    tolerance_ms: float = 20.0,
    sensor_time_col: str = 'Epoch_UTC'
) -> pd.DataFrame:
    """
    For each sensor timestamp, choose the best matching audio among many:
    - Compute nearest sample index and error for each audio
    - Only consider indices within audio [0, num_frames-1]
    - Pick the candidate with smallest |error|
    - Mark in_window if |error| <= tolerance_ms

    Returns sensor_df copy with added columns:
      audio_id, audio_sample_idx, mapped_audio_time_utc, align_error_ms, in_window
    If no audio matches (all out of range), audio_id='', audio_sample_idx=-1, in_window=False.
    """
    if sensor_time_col not in sensor_df.columns:
        raise ValueError(f"Column '{sensor_time_col}' not found in sensor_df.")

    ts = sensor_df[sensor_time_col].to_numpy(dtype=float)  # shape (N,)
    N = ts.shape[0]

    if len(audio_metas) == 0:
        out = sensor_df.copy()
        out['audio_id'] = ''
        out['audio_sample_idx'] = -1
        out['mapped_audio_time_utc'] = np.nan
        out['align_error_ms'] = np.nan
        out['in_window'] = False
        return out

    # Build arrays for vectorized computation over audios
    rates = np.array([float(a.rate_hz) for a in audio_metas])        # (A,)
    t0s   = np.array([float(a.first_sample_utc) for a in audio_metas])# (A,)
    lens  = np.array([int(a.num_frames) for a in audio_metas])        # (A,)
    ids   = np.array([a.audio_id for a in audio_metas], dtype=object) # (A,)

    # For each audio, compute indices and errors: we'll do broadcasting
    # idx_float: shape (A, N) = (ts - t0) * rate
    idx_float = (ts[None, :] - t0s[:, None]) * rates[:, None]
    idx = np.rint(idx_float).astype(np.int64)

    # Valid mask per audio: 0 <= idx < len
    valid = (idx >= 0) & (idx < lens[:, None])

    # Mapped audio time for those indices
    mapped_t = t0s[:, None] + (idx / rates[:, None])  # (A, N)
    err_ms = (ts[None, :] - mapped_t) * 1000.0        # (A, N)

    # For invalid positions, set error to large value so they won't be chosen
    LARGE = 1e18
    abs_err = np.where(valid, np.abs(err_ms), LARGE)

    # Choose best audio by minimal abs error across audios
    best_audio_ix = np.argmin(abs_err, axis=0)        # (N,)
    best_abs_err = abs_err[best_audio_ix, np.arange(N)]
    best_err_ms = err_ms[best_audio_ix, np.arange(N)]
    best_idx = idx[best_audio_ix, np.arange(N)]
    best_mapped_t = mapped_t[best_audio_ix, np.arange(N)]
    best_valid = valid[best_audio_ix, np.arange(N)]

    # Tolerance check
    in_window = best_valid & (best_abs_err <= tolerance_ms)

    # Fill outputs
    out = sensor_df.copy()
    out['audio_id'] = ids[best_audio_ix]
    out['audio_sample_idx'] = np.where(in_window, best_idx, -1)
    out['mapped_audio_time_utc'] = np.where(in_window, best_mapped_t, np.nan)
    out['align_error_ms'] = np.where(best_valid, best_err_ms, np.nan)
    out['in_window'] = in_window
    return out

def align_many_to_many(
    sensor_dict: Dict[str, pd.DataFrame],
    audio_metas: List[AudioMeta],
    tolerance_ms: float = 20.0,
    sensor_time_col: str = 'Epoch_UTC'
) -> pd.DataFrame:
    """
    Align multiple EDF DataFrames to multiple audios. Returns one concatenated tidy DataFrame with
    columns: sensor_id, audio_id, audio_sample_idx, mapped_audio_time_utc, align_error_ms, in_window, and original sensor columns.
    """
    rows = []
    for fp, df in sensor_dict.items():
        aligned = align_sensor_to_many_audios(df, audio_metas, tolerance_ms=tolerance_ms, sensor_time_col=sensor_time_col)
        aligned.insert(0, 'sensor_id', os.path.basename(fp))
        rows.append(aligned)
    if rows:
        return pd.concat(rows, axis=0, ignore_index=True)
    else:
        return pd.DataFrame()

# -------------------------
# (Optional) WAV segment reader reused from previous version
# -------------------------

def read_audio_segment_samples(wav_path: str, start_idx: int, num_samples: int) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.int16)
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        if nch != 1:
            raise ValueError(f"Currently only mono WAV is supported (found channels={nch}).")
        if sampwidth not in (2,):
            raise ValueError(f"Currently only 16-bit PCM is supported (sampwidth={sampwidth}).")

        s = max(0, min(start_idx, nframes))
        e = max(0, min(start_idx + num_samples, nframes))
        num = max(0, e - s)
        wf.setpos(s)
        raw = wf.readframes(num)
        seg = np.frombuffer(raw, dtype='<i2')
        return seg

# -------------------------
# Directory scanning utilities
# -------------------------

def scan_modal_files(
    root_dir: str,
    recursive: bool = True,
    sensor_suffixes: Tuple[str, ...] = ('.edf',),
    wav_suffixes: Tuple[str, ...] = ('.wav',),
) -> Dict[str, object]:
    """
    Recursively scan a directory for sensor (.edf) and audio (.wav) files.
    For each WAV, check for a same-basename JSON sidecar.

    Returns a dict with:
      - 'sensor_files': List[str] of sensor file paths
      - 'wav_files':    List[str] of wav file paths
      - 'wav_json_map': Dict[str,str|None] mapping wav_path -> json_path (or None if absent)
    """
    sensor_files: List[str] = []
    wav_files: List[str] = []
    wav_json_map: Dict[str, Optional[str]] = {}

    def _iterdir():
        if recursive:
            for dirpath, _, filenames in os.walk(root_dir):
                for fn in filenames:
                    yield os.path.join(dirpath, fn)
        else:
            for fn in os.listdir(root_dir):
                p = os.path.join(root_dir, fn)
                if os.path.isfile(p):
                    yield p

    for path in _iterdir():
        lower = path.lower()
        if any(lower.endswith(suf) for suf in sensor_suffixes):
            sensor_files.append(path)
        elif any(lower.endswith(suf) for suf in wav_suffixes):
            wav_files.append(path)

    # For every WAV, attempt to locate sidecar JSON with the same stem
    for w in wav_files:
        base, _ = os.path.splitext(w)
        candidate = base + '.json'
        wav_json_map[w] = candidate if os.path.exists(candidate) else None

    sensor_files.sort()
    wav_files.sort()
    return {
        'sensor_files': sensor_files,
        'wav_files': wav_files,
        'wav_json_map': wav_json_map,
    }

import os
from typing import Tuple, List, Dict

def load_from_dir(
    root_dir: str,
    recursive: bool = True,
) -> Tuple[List[SensorMeta], Dict[str, pd.DataFrame], List[AudioMeta]]:
    """
    Convenience wrapper:
      1) scan directory to find all .edf and .wav (+ optional same-name .json)
      2) read all sensors and audios
      3) return (sensor_metas, sensor_dict, audio_metas)

    Example:
        sensor_metas, sensors, audios = load_from_dir("/data/session01")
    """
    scan = scan_modal_files(root_dir, recursive=recursive)
    sensor_files: List[str] = scan['sensor_files']
    wav_files: List[str] = scan['wav_files']
    wav_json_map: Dict[str, Optional[str]] = scan['wav_json_map']

    sensor_metas, sensor_dict = read_many_sensor_tsv_edf(sensor_files)
    audio_metas: List[AudioMeta] = []
    for w in wav_files:
        # ⚠️ 这里不要再动 meta.audio_id 之类的东西
        audio_metas.append(read_audio_with_json(w, wav_json_map.get(w)))

    return sensor_metas, sensor_dict, audio_metas






# -------------------------
# Segmentation: slice WAV + EDF into aligned 10s segments and save .npz
# -------------------------

def slice_wav_and_sensor_to_segments(
    audio_meta: AudioMeta,
    sensor_df: pd.DataFrame,
    out_dir: str,
    sensor_time_col: str = "Epoch_UTC",
    sensor_value_cols: Optional[List[str]] = None,
    window_sec: float = 10.0,
    hop_sec: Optional[float] = None,
    min_sensor_rows: int = 1,
    sensor_id: Optional[str] = None,
    sensor_source_path: Optional[str] = None,
    base_prefix: Optional[str] = None,
    save_dtype_audio: str = "int16",
    save_dtype_sensor: str = "float32",
    include_win_keyword: bool = True,
) -> List[str]:
    """
    Slice a single WAV (audio_meta) and a single sensor EDF/TSV (sensor_df)
    into fixed-length time windows, and save each window as one .npz file.

    Each .npz will contain (by default):
      - audio: int16 [T_audio]
      - audio_rate_hz: int
      - audio_start_utc: float (epoch seconds, window start)
      - audio_end_utc: float (epoch seconds, window end)
      - audio_id: str (basename of wav)
      - sensor_time_epoch: float [T_sensor]
      - sensor_values: float32 [T_sensor, num_sensor_cols]
      - sensor_cols: str [num_sensor_cols]
      - sensor_id: str
      - sensor_source_path: str (if provided)
      - label: str (if WAV json has "label")

    Parameters
    ----------
    audio_meta : AudioMeta
        Metadata for the WAV + JSON (from read_audio_with_json).
    sensor_df : pd.DataFrame
        EDF-like DataFrame with Epoch_UTC + sensor channels.
    out_dir : str
        Directory to write .npz files.
    sensor_time_col : str, default "Epoch_UTC"
        Name of the column with UTC epoch seconds in sensor_df.
    sensor_value_cols : list[str] or None
        Which columns to treat as sensor channels. If None, use all except sensor_time_col.
    window_sec : float, default 10.0
        Window length in seconds.
    hop_sec : float or None
        Hop between windows in seconds. If None, defaults to 50% overlap (window_sec * 0.5).
    min_sensor_rows : int, default 1
        Skip a window if the sensor segment inside [start,end) has fewer than this many rows.
    sensor_id : str or None
        Optional sensor identifier to store in the npz.
    sensor_source_path : str or None
        Optional original EDF/TSV path to store for traceability.
    base_prefix : str or None
        Prefix for output file names. If None, uses "<audio_id>__<sensor_id>".
    save_dtype_audio : {"int16","float32"}, default "int16"
        Dtype used when storing audio segment.
    save_dtype_sensor : {"float32","float64"}, default "float32"
        Dtype used when storing sensor_values.
    include_win_keyword : bool, default True
        If True, output files are named "<prefix>_win00000.npz". If False,
        the "win" tag is omitted and files are "<prefix>_00000.npz".

    Returns
    -------
    List[str]
        List of paths to the .npz files created.
    """
    if sensor_time_col not in sensor_df.columns:
        raise ValueError(f"sensor_time_col '{sensor_time_col}' not in sensor_df.columns")

    os.makedirs(out_dir, exist_ok=True)

    # Decide hop (default 50% overlap)
    if hop_sec is None:
        hop_sec = window_sec * 0.5

    # Sort sensor by time just in case
    sensor_df = sensor_df.sort_values(sensor_time_col).reset_index(drop=True)

    # Sensor channels to keep
    if sensor_value_cols is None:
        # 只选数值类型的列（自动排除 Local_Date_Time 这种字符串）
        numeric_cols = sensor_df.select_dtypes(include=["number"]).columns.tolist()
        # 再把时间列去掉
        sensor_value_cols = [c for c in numeric_cols if c != sensor_time_col]

        if not sensor_value_cols:
            raise ValueError(
                f"[slice_wav_and_sensor_to_segments] No numeric sensor columns found "
                f"after excluding time column '{sensor_time_col}'."
            )
    else:
        # 如果显式传了 sensor_value_cols，就检查哪些存在
        existing = [c for c in sensor_value_cols if c in sensor_df.columns]
        missing = [c for c in sensor_value_cols if c not in sensor_df.columns]
        if missing:
            print(
                f"[slice_wav_and_sensor_to_segments] WARNING: "
                f"these sensor_value_cols are missing in {sensor_id or sensor_source_path}: {missing}"
            )
        if not existing:
            print(
                f"[slice_wav_and_sensor_to_segments] No valid sensor columns for this EDF. Skipping."
            )
            return []
        sensor_value_cols = existing

    # Figure out label from audio JSON (if exists)
    label = None
    if audio_meta.json_path and os.path.exists(audio_meta.json_path):
        try:
            with open(audio_meta.json_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            label = j.get("label", None)  # e.g. "3mL secretion(no HEMF)"
        except Exception as e:
            # Silently ignore label if JSON cannot be read
            print(f"[slice_wav_and_sensor_to_segments] Failed to read label from {audio_meta.json_path}: {e}")

    # Time range of this audio (in UTC epoch seconds)
    audio_start_utc = audio_meta.first_sample_utc
    audio_end_utc = audio_meta.last_sample_utc_by_frames  # derived from num_frames

    if audio_end_utc <= audio_start_utc:
        return []

    fs = float(audio_meta.rate_hz)
    window_samples = int(round(window_sec * fs))

    # Determine prefix for naming
    if sensor_id is None:
        sensor_id = "sensor"
    if base_prefix is None:
        base_prefix = f"{audio_meta.audio_id}__{sensor_id}"

    out_paths: List[str] = []
    win_index = 0

    # Slide windows over the audio time range
    t = audio_start_utc
    while True:
        win_start = t
        win_end = win_start + window_sec

        # Stop if we don't have a full window at the tail
        if win_end > audio_end_utc:
            break

        # Map window times to audio sample indices
        start_idx = _safe_round((win_start - audio_start_utc) * fs)
        # constrain start_idx in [0, num_frames)
        start_idx = max(0, min(start_idx, audio_meta.num_frames - 1))

        # Read audio segment
        audio_seg = read_audio_segment_samples(audio_meta.wav_path, start_idx, window_samples)

        # Optional: ensure the length is exactly window_samples (pad with zeros at tail)
        if audio_seg.shape[0] < window_samples:
            pad_len = window_samples - audio_seg.shape[0]
            audio_seg = np.pad(audio_seg, (0, pad_len), mode="constant", constant_values=0)

        # Extract sensor rows within [win_start, win_end)
        mask = (sensor_df[sensor_time_col] >= win_start) & (sensor_df[sensor_time_col] < win_end)
        sensor_seg = sensor_df.loc[mask]

        # Skip if not enough sensor rows
        if len(sensor_seg) < min_sensor_rows:
            t += hop_sec
            continue

        # Prepare data to save
        if save_dtype_audio == "float32":
            audio_to_save = audio_seg.astype(np.float32)
        else:
            audio_to_save = audio_seg.astype(np.int16)

        if save_dtype_sensor == "float64":
            sensor_values = sensor_seg[sensor_value_cols].to_numpy(dtype=np.float64)
        else:
            sensor_values = sensor_seg[sensor_value_cols].to_numpy(dtype=np.float32)

        sensor_time_epoch = sensor_seg[sensor_time_col].to_numpy(dtype=float)

        if include_win_keyword:
            npz_name = f"{base_prefix}_win{win_index:05d}.npz"
        else:
            npz_name = f"{base_prefix}_{win_index:05d}.npz"
        npz_path = os.path.join(out_dir, npz_name)

        save_dict = dict(
            audio=audio_to_save,
            audio_rate_hz=int(audio_meta.rate_hz),
            audio_start_utc=float(win_start),
            audio_end_utc=float(win_end),
            audio_id=str(audio_meta.audio_id),
            audio_file_start_utc=float(audio_start_utc),     # 整个 wav 第 0 个 sample 的 UTC
            audio_sample_start_index=int(start_idx),         # 这个窗口在整个 wav 中的起始索引
            sensor_time_epoch=sensor_time_epoch,
            sensor_values=sensor_values,
            sensor_cols=np.array(sensor_value_cols),
            sensor_id=str(sensor_id),
        )

        if sensor_source_path is not None:
            save_dict["sensor_source_path"] = str(sensor_source_path)

        if label is not None:
            # Keep label as a simple string
            save_dict["label"] = np.array(label)

        np.savez(npz_path, **save_dict)
        out_paths.append(npz_path)

        win_index += 1
        t += hop_sec

    return out_paths


def slice_all_pairings(
    sensor_metas: List[SensorMeta],
    sensor_dict: Dict[str, pd.DataFrame],
    audio_metas: List[AudioMeta],
    out_root: str,
    group_by_parent_dir: bool = False,
    simple_naming: bool = False,
    window_sec: float = 10.0,
    hop_sec: Optional[float] = None,
    sensor_time_col: str = "Epoch_UTC",
    sensor_value_cols: Optional[List[str]] = None,
    min_sensor_rows: int = 1,
    pair_min_overlap_sec: float = 1.0,
) -> Dict[Tuple[str, str], List[str]]:
    """
    For all (sensor, audio) pairs that have time overlap, slice them into
    fixed-length segments and save .npz files under out_root.

    Parameters
    ----------
    group_by_parent_dir : bool, default False
        If True, place the generated segments inside a folder named after
        the parent directory of each WAV (e.g., source WAV in
        ".../abc/x.wav" saves to "<out_root>/abc"). If that parent name is
        the same as the basename of out_root, segments are written directly
        into out_root. If False, keep the previous "<sensor>__<audio>" folder
        structure.
    simple_naming : bool, default False
        If True, name output files as "<audio_stem>_<index>.npz" and omit the
        sensor id and "win" tag. Use with group_by_parent_dir=True to get
        paths like "./abc/x_00001.npz" for an input ".../abc/x.wav". If False,
        keep the original "<audio>__<sensor>_win<index>.npz" naming.

    Returns a dict mapping (sensor_id, audio_id) -> [list of .npz paths].
    """
    os.makedirs(out_root, exist_ok=True)

    pairings = compute_pairings(sensor_metas, audio_metas)
    results: Dict[Tuple[str, str], List[str]] = {}

    for _, row in pairings.iterrows():
        sensor_id = row["sensor_id"]
        audio_id = row["audio_id"]
        overlap_sec = float(row["overlap_sec"])
        if overlap_sec < pair_min_overlap_sec:
            continue

        sensor_file = row["sensor_file"]
        wav_path = row["wav_path"]

        sensor_df = sensor_dict[sensor_file]
        audio_meta = next(am for am in audio_metas if am.wav_path == wav_path)

        if group_by_parent_dir:
            audio_parent = os.path.basename(os.path.dirname(wav_path))
            out_root_base = os.path.basename(os.path.abspath(out_root))
            if audio_parent and audio_parent != out_root_base:
                pair_out_dir = os.path.join(out_root, audio_parent)
            else:
                pair_out_dir = out_root
        else:
            pair_out_dir = os.path.join(out_root, f"{sensor_id}__{audio_id}")

        if simple_naming:
            base_prefix = os.path.splitext(audio_meta.audio_id)[0]
            include_win_keyword = False
        else:
            base_prefix = None
            include_win_keyword = True

        paths = slice_wav_and_sensor_to_segments(
            audio_meta=audio_meta,
            sensor_df=sensor_df,
            out_dir=pair_out_dir,
            sensor_time_col=sensor_time_col,
            sensor_value_cols=sensor_value_cols,
            window_sec=window_sec,
            hop_sec=hop_sec,
            min_sensor_rows=min_sensor_rows,
            sensor_id=sensor_id,
            sensor_source_path=sensor_file,
            base_prefix=base_prefix,
            include_win_keyword=include_win_keyword,
        )
        results[(sensor_id, audio_id)] = paths

    return results
