"""
Utility helpers for fixing random seeds to make experiments reproducible.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Fix random seeds across common libs. Optionally toggles deterministic
    behavior for CuDNN to reduce run-to-run noise.
    """
    # Keep Python's hashing stable as well (affects set/dict iteration in rare cases)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Ensure deterministic algorithms (will raise if an op has no deterministic path)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_deterministic_debug_mode("error")
        except Exception:
            # Fallback for older torch versions
            torch.set_deterministic_debug_mode(True)
    else:
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker init fn to propagate deterministic seeds to numpy/random.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
