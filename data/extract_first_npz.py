#!/usr/bin/env python3
from __future__ import annotations

import sys
import shutil
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python data/extract_first_npz.py <x>")
        return 2

    try:
        x = int(sys.argv[1])
    except ValueError:
        print("Error: x must be an integer.")
        return 2

    if x <= 0:
        print("Error: x must be a positive integer.")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "data" / "MMDataset_segments"
    dst_root = repo_root / "data" / f"MMDataset_segments_first{x}"

    if not src_root.is_dir():
        print(f"Error: source directory not found: {src_root}")
        return 1

    sessions = sorted(p for p in src_root.iterdir() if p.is_dir())
    if not sessions:
        print(f"No session folders found under {src_root}")
        return 0

    total_pairs = 0
    total_found = 0
    total_copied = 0
    total_skipped = 0

    for session_dir in sessions:
        pair_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir())
        for pair_dir in pair_dirs:
            total_pairs += 1
            npz_files = sorted(pair_dir.glob("*.npz"))
            if not npz_files:
                continue

            chosen = npz_files[:x]
            total_found += len(chosen)
            dst_dir = dst_root / session_dir.name / pair_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src_path in chosen:
                dst_path = dst_dir / src_path.name
                if dst_path.exists():
                    total_skipped += 1
                    continue
                shutil.copy2(src_path, dst_path)
                total_copied += 1

    print(
        "Done.",
        f"sessions={len(sessions)}",
        f"pair_dirs={total_pairs}",
        f"picked={total_found}",
        f"copied={total_copied}",
        f"skipped_existing={total_skipped}",
        f"output={dst_root}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
