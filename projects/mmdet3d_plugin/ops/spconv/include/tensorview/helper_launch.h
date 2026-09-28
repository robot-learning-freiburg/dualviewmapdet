// This source code is from BEVFusion
//   https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/ops/spconv/include/tensorview/helper_launch.h
// Copyright (c) 2023 MIT HAN Lab licensed under the Apache License 2.0,
// cf. 3rd-party-licenses.txt file in the root directory of this source tree.

#pragma once
// from pytorch.aten
#include "tensorview.h"
namespace tv
{
namespace launch
{

template <typename T1, typename T2>
inline int DivUp(const T1 a, const T2 b) { return (a + b - 1) / b; }

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;
// CUDA: number of blocks for threads.
inline int getBlocks(const int N)
{
    TV_ASSERT_RT_ERR(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
    return DivUp(N, CUDA_NUM_THREADS);
}
} // namespace launch
} // namespace tv
