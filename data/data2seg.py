'''遍历 MMDataset 下面所有一级子文件夹，
把每个子目录当作一个 session，
自动加载其中的 EDF/WAV 并按 10 秒窗口切成片段，
分别保存到对应的二级输出目录中。'''

import os
import sys

sys.path.append("../")

from util.multimodal_align import load_from_dir, slice_all_pairings

# 0. 总根目录：下面是一堆 session 子目录
parent_dir = r"../MMDataset"

# 顶层输出目录，例如： ./MMDataset_segments/
parent_name = os.path.basename(parent_dir.rstrip(r"\/"))
parent_out_root = os.path.join(".", f"{parent_name}_segments")
os.makedirs(parent_out_root, exist_ok=True)

all_results = {}

# 1. 遍历 MMDataset 下面 “一级子目录”
for entry in os.listdir(parent_dir):
    session_dir = os.path.join(parent_dir, entry)

    # 只处理目录，跳过文件
    if not os.path.isdir(session_dir):
        continue

    # 跳过隐藏/配置目录，例如 .vscode、.git 等
    if entry.startswith("."):
        print(f"跳过目录（看起来是配置目录）: {entry}")
        continue

    session_name = entry
    print(f"\n==== 开始处理 session: {session_name} ====")

    # 当前 session 的输出目录： ./MMDataset_segments/<session_name>/
    out_root = os.path.join(parent_out_root, session_name)
    os.makedirs(out_root, exist_ok=True)

    # 3. 自动扫描 + 读取所有 EDF / WAV (+ JSON)
    sensor_metas, sensor_dict, audio_metas = load_from_dir(session_dir)

    # 如果这个 session 根本没有数据，直接跳过
    if len(sensor_metas) == 0 and len(audio_metas) == 0:
        print(f"[{session_name}] 没找到 EDF / WAV，跳过。")
        continue

    # 4. 计算 EDF–WAV 时间重叠，并按 10s 切片保存
    results = slice_all_pairings(
        sensor_metas=sensor_metas,
        sensor_dict=sensor_dict,
        audio_metas=audio_metas,
        out_root=out_root,
        window_sec=10.0,        # 10 秒窗口
        hop_sec=10,             # 10 秒 hop = 不重叠（想要 50% 重叠改成 5.0）
        sensor_time_col="Epoch_UTC",
        # sensor_value_cols=None,
        min_sensor_rows=1,      # 窗口里至少要有 1 行传感器数据才会保存
        pair_min_overlap_sec=1.0,
    )

    all_results[session_name] = results

    total_segments = sum(len(v) for v in results.values())
    print(f"[{session_name}] Done. Total segments: {total_segments}")
    for (sensor_id, audio_id), paths in results.items():
        print(f"[{session_name}] {sensor_id}  <->  {audio_id}  ->  {len(paths)} segments")

# 5. 总结
print("\n===== 所有 session 统计汇总 =====")
grand_total = 0
for session_name, results in all_results.items():
    total_segments = sum(len(v) for v in results.values())
    print(f"Session {session_name}: {total_segments} segments")
    grand_total += total_segments

print(f"ALL sessions total segments: {grand_total}")

