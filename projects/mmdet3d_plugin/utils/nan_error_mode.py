# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
import itertools
import warnings

# adapted from https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py

class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def fmt(t: object, print_stats=False) -> str:
    if isinstance(t, torch.Tensor):
        s = f"torch.tensor(..., size={tuple(t.shape)}, dtype={t.dtype}, device='{t.device}')"
        if print_stats:
            s += f" [with stats min={t.min()}, max={t.max()}, mean={t.mean()}]"
        return Lit(s)
    else:
        return t


class NaNErrorMode(TorchDispatchMode):
    """
    To detect nan values (helpful for fp16 optimization)
    """
    def __init__(
        self, enabled=True, raise_error=False, print_stats=True, print_nan_index=False
    ):
        self.enabled = enabled
        # warning or error
        self.raise_error = raise_error
        # print min/max/mean stats
        self.print_stats = print_stats
        # print indices of invalid values in output
        self.print_nan_index = print_nan_index

    def __torch_dispatch__(self, func, types, args, kwargs):
        out = func(*args, **kwargs)
        if self.enabled:
            if isinstance(out, torch.Tensor):
                if not torch.isfinite(out).all():
                    # fmt_partial = partial(fmt, self.print_stats)
                    fmt_lambda = lambda t: fmt(t, self.print_stats)
                    fmt_args = ", ".join(
                        itertools.chain(
                            (repr(tree_map(fmt_lambda, a)) for a in args),
                            (
                                f"{k}={tree_map(fmt_lambda, v)}"
                                for k, v in kwargs.items()
                            ),
                        )
                    )
                    msg = f"NaN outputs in out = {func}({fmt_args})"
                    if self.print_nan_index:
                        msg += f"\nInvalid values detected at:\n{(~out.isfinite()).nonzero()}"
                    if self.raise_error:
                        raise RuntimeError(msg)
                    else:
                        warnings.warn(msg)

        return out