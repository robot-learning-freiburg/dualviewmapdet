# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class GradientNormHook(Hook):

    def __init__(self, top_k=10, **kwargs):
        super().__init__()
        self.top_k = top_k

    def after_train_iter(self, runner):
        """Called after each training iteration to log gradient norms."""
        grad_norms = {}
        for name, param in runner.model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        # Sort parameters by gradient norm (descending order).
        sorted_grad_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)

        # Create a readable string of sorted norms for logging.
        sorted_log = ", ".join([f"{name}: {norm:.4f}" for name, norm in sorted_grad_norms[:self.top_k]])

        # Log the gradient norms in descending order.
        runner.logger.info(f"Gradient Norms (Sorted Descending): {sorted_log}")

        # Optionally, add the gradient norms to the log buffer (for e.g. TensorBoard)
        # in the same sorted order.
        #for name, norm in sorted_grad_norms:
        #    runner.log_buffer.output[f"grad_norm/{name}"] = norm