# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFusion
#   https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/ops/spconv/ops.py
# Copyright (c) 2023 MIT HAN Lab licensed under the Apache License 2.0,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from . import sparse_conv_ext


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
            i
        ] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    # -----------------------------
    # Helper: build empty outputs
    # -----------------------------
    def _make_empty_outputs(indices_ref, out_shape, ksize_):
        # indices_ref: (N, ndim+1)
        ndim_ = indices_ref.shape[1] - 1
        # kernel volume
        kv = 1
        for v in ksize_:
            kv *= int(v)

        # outids: (0, ndim+1)
        outids = indices_ref.new_empty((0, indices_ref.shape[1]))
        # indice_pairs: (kv, 2, 0)  (this shape is commonly accepted downstream)
        indice_pairs = indices_ref.new_empty((kv, 2, 0))
        # indice_pair_num: (kv,) int32, typically CPU
        indice_pair_num = torch.zeros((kv,), dtype=torch.int32, device="cpu")
        #indice_pair_num = torch.zeros((kv,), dtype=torch.int32, device=indices_ref.device)
        return outids, indice_pairs, indice_pair_num

    # -----------------------------
    # Handle empty indices up-front
    # -----------------------------
    # Be careful: indices is expected to be a Tensor
    if indices is None or indices.numel() == 0 or indices.shape[0] == 0:
        # We still need ndim/ksize normalization to compute kv correctly.
        # If indices is None, fabricate a minimal empty indices tensor.
        if indices is None:
            # default to 4D indices shape (b,z,y,x); adjust if you use 2D/3D elsewhere
            indices = torch.empty((0, 4), dtype=torch.int32, device="cpu")
        ndim = indices.shape[1] - 1
        if not isinstance(ksize, (list, tuple)):
            ksize = [ksize] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(out_padding, (list, tuple)):
            out_padding = [out_padding] * ndim

        if not subm:
            if transpose:
                out_shape = get_deconv_output_size(
                    spatial_shape, ksize, stride, padding, dilation, out_padding
                )
            else:
                out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)
        else:
            out_shape = spatial_shape

        return _make_empty_outputs(indices, out_shape, ksize)

    # -----------------------------
    # Normalize args
    # -----------------------------
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    # -----------------------------
    # Compute output spatial shape
    # -----------------------------
    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)
    else:
        out_shape = spatial_shape

    # -----------------------------
    # Choose extension function
    # -----------------------------
    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_3d
        elif ndim == 4:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_4d
        else:
            raise NotImplementedError

        try:
            return get_indice_pairs_func(
                indices,
                batch_size,
                out_shape,
                spatial_shape,
                ksize,
                stride,
                padding,
                dilation,
                out_padding,
                int(subm),
                int(transpose),
            )
        except RuntimeError as e:
            msg = str(e)
            if ("CUDA kernel launch blocks must be positive" in msg) or ("N > 0 assert failed" in msg):
                return _make_empty_outputs(indices, out_shape, ksize)
            raise

    else:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError

        try:
            return get_indice_pairs_func(
                indices,
                grid,
                batch_size,
                out_shape,
                spatial_shape,
                ksize,
                stride,
                padding,
                dilation,
                out_padding,
                int(subm),
                int(transpose),
            )
        except RuntimeError as e:
            msg = str(e)
            if ("CUDA kernel launch blocks must be positive" in msg) or ("N > 0 assert failed" in msg):
                return _make_empty_outputs(indices, out_shape, ksize)
            raise



def indice_conv(
    features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_fp32(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_half(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    else:
        raise NotImplementedError


def fused_indice_conv(
    features, filters, bias, indice_pairs, indice_pair_num, num_activate_out, inverse, subm
):
    if features.dtype == torch.half:
        func = sparse_conv_ext.fused_indice_conv_half
    elif filters.dtype == torch.float32:
        func = sparse_conv_ext.fused_indice_conv_fp32
    else:
        raise NotImplementedError

    return func(
        features,
        filters,
        bias,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )


def indice_conv_backward(
    features, filters, out_bp, indice_pairs, indice_pair_num, inverse=False, subm=False
):
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_backward_fp32(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_backward_half(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    else:
        raise NotImplementedError


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32:
        return sparse_conv_ext.indice_maxpool_fp32(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
    elif features.dtype == torch.half:
        return sparse_conv_ext.indice_maxpool_half(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
    else:
        raise NotImplementedError


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
    if features.dtype == torch.float32:
        return sparse_conv_ext.indice_maxpool_backward_fp32(
            features, out_features, out_bp, indice_pairs, indice_pair_num
        )
    elif features.dtype == torch.half:
        return sparse_conv_ext.indice_maxpool_backward_half(
            features, out_features, out_bp, indice_pairs, indice_pair_num
        )
    else:
        raise NotImplementedError
