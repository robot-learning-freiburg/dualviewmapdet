# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch

def move_to_cpu(obj):
    """
    Recursively moves all torch.Tensor objects in a nested structure to CPU.
    Works for nested dicts, lists, tuples, and sets.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    elif isinstance(obj, set):
        return {move_to_cpu(v) for v in obj}
    else:
        return obj
