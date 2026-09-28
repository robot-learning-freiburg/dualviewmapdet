# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import logging
from collections import OrderedDict
from typing import Mapping, Optional

import torch


def _human_int(n: int) -> str:
    # 153288641 -> "153.29M", 12345 -> "12.35K"
    units = ["", "K", "M", "B", "T"]
    x = float(n)
    i = 0
    while x >= 1000.0 and i < len(units) - 1:
        x /= 1000.0
        i += 1
    return f"{x:.2f}{units[i]}" if i > 0 else f"{n:,}"


def log_params_per_child(
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None,
    trainable_only: bool = False,
    level: int = logging.INFO,
    title: str = "Parameter count per direct child",
    sort: str = "desc",  # "desc" | "asc" | "name" | "none"
) -> OrderedDict:
    """
    Logs a human-friendly table of parameter counts for *direct* children and total.

    Returns the OrderedDict (same keys as params_per_child), in the logged order.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
        params = module.parameters() if not trainable_only else (p for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    # Collect direct children
    rows = []
    for name, child in model.named_children():
        rows.append((name, count_params(child, trainable_only=trainable_only)))

    total = count_params(model, trainable_only=trainable_only)

    # Sort
    if sort == "desc":
        rows.sort(key=lambda x: x[1], reverse=True)
    elif sort == "asc":
        rows.sort(key=lambda x: x[1])
    elif sort == "name":
        rows.sort(key=lambda x: x[0].lower())
    elif sort == "none":
        pass
    else:
        raise ValueError(f"Unknown sort='{sort}'. Use 'desc', 'asc', 'name', or 'none'.")

    # Formatting widths
    name_w = max([len(n) for n, _ in rows] + [len("__total__")]) if rows else len("__total__")
    num_w = max([len(f"{c:,}") for _, c in rows] + [len(f"{total:,}")])
    hum_w = max([len(_human_int(c)) for _, c in rows] + [len(_human_int(total))])
    pct_w = 6  # "100.0%"

    # Header
    suffix = " (trainable only)" if trainable_only else ""
    logger.log(level, f"{title}{suffix}")
    logger.log(level, f"{'module':<{name_w}}  {'params':>{num_w}}  {'human':>{hum_w}}  {'share':>{pct_w}}")
    logger.log(level, f"{'-'*name_w}  {'-'*num_w}  {'-'*hum_w}  {'-'*pct_w}")

    out = OrderedDict()
    denom = max(total, 1)

    for name, cnt in rows:
        share = 100.0 * cnt / denom
        logger.log(
            level,
            f"{name:<{name_w}}  {cnt:>{num_w},}  {_human_int(cnt):>{hum_w}}  {share:5.1f}%"
        )
        out[name] = cnt

    logger.log(level, f"{'-'*name_w}  {'-'*num_w}  {'-'*hum_w}  {'-'*pct_w}")
    logger.log(level, f"{'__total__':<{name_w}}  {total:>{num_w},}  {_human_int(total):>{hum_w}}  {100.0:5.1f}%")
    out["__total__"] = total

    return out


# usage:
# logger = logging.getLogger(__name__)
# log_params_per_child(module, logger=logger, trainable_only=False, sort="desc")
# log_params_per_child(module, logger=logger, trainable_only=True,  sort="desc", title="Trainable params per child")
