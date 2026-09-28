// This source code is from CMRNext
//   https://github.com/robot-learning-freiburg/CMRNext/blob/main/visibility_pkg/src/visibility.cpp
// Copyright (c) Robot Learning Lab, University of Freiburg, licensed under the GPL-3.0 license,
// cf. 3rd-party-licenses.txt file in the root directory of this source tree.

// visibility_pkg/src/visibility.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cmath>
#include <limits>

// --------------------------------------------------------
// CUDA declarations (visibility_kernel.cu)
// --------------------------------------------------------
at::Tensor depth_image_cuda(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output,
                           unsigned int size, unsigned int width, unsigned int height);

at::Tensor visibility_filter_cuda(at::Tensor input, at::Tensor output,
                                 unsigned int width, unsigned int height,
                                 unsigned int threshold);

at::Tensor visibility_filter_cuda2(at::Tensor input_depth, at::Tensor intrinsic4, at::Tensor output,
                                  unsigned int width, unsigned int height,
                                  float threshold, unsigned int radius);

// NEW
at::Tensor visibility_filter_cuda2_fullK(at::Tensor input_depth, at::Tensor K9, at::Tensor output,
                                        unsigned int width, unsigned int height,
                                        float threshold, unsigned int radius);

at::Tensor downsample_flow_cuda(at::Tensor input_uv, at::Tensor output,
                               unsigned int width_out, unsigned int height_out,
                               unsigned int kernel);

at::Tensor downsample_mask_cuda(at::Tensor input_mask, at::Tensor output,
                               unsigned int width_out, unsigned int height_out,
                               unsigned int kernel);

at::Tensor downsample_depth_cuda(at::Tensor input_depth, at::Tensor output,
                                unsigned int width_out, unsigned int height_out,
                                unsigned int kernel);

// --------------------------------------------------------
// Checks
// --------------------------------------------------------
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DEVICE(a, b) TORCH_CHECK(a.device() == b.device(), "tensors must be on the same device")
#define CHECK_CPU_OR_CUDA(x) TORCH_CHECK((x.is_cuda() || x.is_cpu()), #x " must be a CPU or CUDA tensor")

// --------------------------------------------------------
// CPU fallback: depth_image
// --------------------------------------------------------
template <typename scalar_t>
static inline void depth_image_cpu_impl(
    const int* uv_ptr,
    const scalar_t* depth_ptr,
    scalar_t* out_ptr,
    unsigned int size,
    unsigned int width,
    unsigned int height) {

    for (unsigned int i = 0; i < size; i++) {
        int u = uv_ptr[2 * i + 0];
        int v = uv_ptr[2 * i + 1];

        if (u < 0 || u >= (int)width || v < 0 || v >= (int)height)
            continue;

        unsigned int idx = (unsigned int)(v * (int)width + u);
        scalar_t val = depth_ptr[i];

        scalar_t old = out_ptr[idx];
        if (val < old) out_ptr[idx] = val;
    }
}

static at::Tensor depth_image_cpu(
    at::Tensor input_uv,
    at::Tensor input_depth,
    at::Tensor output,
    unsigned int size,
    unsigned int width,
    unsigned int height) {

    TORCH_CHECK(input_uv.is_cpu(), "depth_image_cpu: input_uv must be CPU");
    TORCH_CHECK(input_depth.is_cpu(), "depth_image_cpu: input_depth must be CPU");
    TORCH_CHECK(output.is_cpu(), "depth_image_cpu: output must be CPU");

    CHECK_CONTIGUOUS(input_uv);
    CHECK_CONTIGUOUS(input_depth);
    CHECK_CONTIGUOUS(output);

    TORCH_CHECK(input_uv.scalar_type() == at::kInt, "depth_image CPU expects input_uv int32");
    TORCH_CHECK(input_uv.numel() >= (int64_t)2 * (int64_t)size, "input_uv has too few elements");
    TORCH_CHECK(input_depth.numel() >= (int64_t)size, "input_depth too small");
    TORCH_CHECK(output.numel() >= (int64_t)width * (int64_t)height, "output too small");

    const int* uv_ptr = input_uv.data_ptr<int>();

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "depth_image_cpu", [&] {
        const scalar_t* depth_ptr = input_depth.data_ptr<scalar_t>();
        scalar_t* out_ptr = output.data_ptr<scalar_t>();
        depth_image_cpu_impl<scalar_t>(uv_ptr, depth_ptr, out_ptr, size, width, height);
    });

    return output;
}

// --------------------------------------------------------
// CPU fallback: visibility2 (supports intrinsic4 or K9)
// --------------------------------------------------------
template <typename scalar_t>
static inline scalar_t norm3(scalar_t x, scalar_t y, scalar_t z) {
    return x * x + y * y + z * z;
}

// Backproject with full 3x3 K (row-major) by computing K^{-1} once
template <typename scalar_t>
static void visibility2_cpu_fullK(
    const scalar_t* in_depth,
    const scalar_t* K9,
    scalar_t* out,
    unsigned int width,
    unsigned int height,
    float threshold,
    unsigned int radius) {

    // K row-major
    scalar_t k00 = K9[0], k01 = K9[1], k02 = K9[2];
    scalar_t k10 = K9[3], k11 = K9[4], k12 = K9[5];
    scalar_t k20 = K9[6], k21 = K9[7], k22 = K9[8];

    scalar_t det =
        k00 * (k11 * k22 - k12 * k21)
      - k01 * (k10 * k22 - k12 * k20)
      + k02 * (k10 * k21 - k11 * k20);

    if (det == (scalar_t)0) {
        // just copy input to output
        std::memcpy(out, in_depth, sizeof(scalar_t) * width * height);
        return;
    }

    scalar_t inv_det = (scalar_t)1.0 / det;

    scalar_t i00 =  (k11 * k22 - k12 * k21) * inv_det;
    scalar_t i01 = -(k01 * k22 - k02 * k21) * inv_det;
    scalar_t i02 =  (k01 * k12 - k02 * k11) * inv_det;

    scalar_t i10 = -(k10 * k22 - k12 * k20) * inv_det;
    scalar_t i11 =  (k00 * k22 - k02 * k20) * inv_det;
    scalar_t i12 = -(k00 * k12 - k02 * k10) * inv_det;

    scalar_t i20 =  (k10 * k21 - k11 * k20) * inv_det;
    scalar_t i21 = -(k00 * k21 - k01 * k20) * inv_det;
    scalar_t i22 =  (k00 * k11 - k01 * k10) * inv_det;

    const int W = (int)width;
    const int H = (int)height;
    const int r = (int)radius;

    for (int x = 0; x < H; x++) {
        for (int y = 0; y < W; y++) {
            unsigned int index = (unsigned int)(x * W + y);
            scalar_t pixel = in_depth[index];
            out[index] = pixel;
            if (pixel == (scalar_t)0) continue;

            scalar_t u = (scalar_t)y;
            scalar_t v = (scalar_t)x;

            scalar_t px = i00 * u + i01 * v + i02;
            scalar_t py = i10 * u + i11 * v + i12;
            scalar_t pz = i20 * u + i21 * v + i22;
            if (pz == (scalar_t)0) continue;
            px /= pz; py /= pz;

            scalar_t v_x = px * pixel;
            scalar_t v_y = py * pixel;
            scalar_t v_z = pixel;

            scalar_t v_norm = norm3(-v_x, -v_y, -v_z);
            if (v_norm <= (scalar_t)0) continue;
            v_norm = (scalar_t)std::sqrt((double)v_norm);

            scalar_t v2_x = -v_x / v_norm;
            scalar_t v2_y = -v_y / v_norm;
            scalar_t v2_z = -v_z / v_norm;

            scalar_t max_dot1 = (scalar_t)-1;
            scalar_t max_dot2 = (scalar_t)-1;
            scalar_t max_dot3 = (scalar_t)-1;
            scalar_t max_dot4 = (scalar_t)-1;

            auto update_maxdot = [&](int dx0, int dx1, int dy0, int dy1, scalar_t& maxdot) {
                for (int i = dx0; i <= dx1; i++) {
                    for (int j = dy0; j <= dy1; j++) {
                        int xx = x + i;
                        int yy = y + j;
                        if (xx < 0 || xx >= H || yy < 0 || yy >= W) continue;

                        unsigned int tidx = (unsigned int)(xx * W + yy);
                        scalar_t tpix = in_depth[tidx];
                        if (tpix == (scalar_t)0) continue;

                        scalar_t uu = (scalar_t)yy;
                        scalar_t vv = (scalar_t)xx;

                        scalar_t qx = i00 * uu + i01 * vv + i02;
                        scalar_t qy = i10 * uu + i11 * vv + i12;
                        scalar_t qz = i20 * uu + i21 * vv + i22;
                        if (qz == (scalar_t)0) continue;
                        qx /= qz; qy /= qz;

                        scalar_t c_x = qx * tpix;
                        scalar_t c_y = qy * tpix;
                        scalar_t c_z = tpix;

                        c_x -= v_x; c_y -= v_y; c_z -= v_z;

                        scalar_t c_norm = norm3(c_x, c_y, c_z);
                        if (c_norm <= (scalar_t)0) continue;
                        c_norm = (scalar_t)std::sqrt((double)c_norm);

                        c_x /= c_norm; c_y /= c_norm; c_z /= c_norm;

                        scalar_t dot = c_x * v2_x + c_y * v2_y + c_z * v2_z;
                        if (dot > maxdot) maxdot = dot;
                    }
                }
            };

            update_maxdot(-r, 0, -r, 0, max_dot1);
            update_maxdot(0,  r, -r, 0, max_dot2);
            update_maxdot(-r, 0,  0, r, max_dot3);
            update_maxdot(0,  r,  0, r, max_dot4);

            if ((float)(max_dot1 + max_dot2 + max_dot3 + max_dot4) >= threshold) {
                out[index] = (scalar_t)0;
            }
        }
    }
}

static at::Tensor visibility_filter_cpu2(
    at::Tensor input_depth,
    at::Tensor intrinsic,
    at::Tensor output,
    unsigned int width,
    unsigned int height,
    float threshold,
    unsigned int radius) {

    TORCH_CHECK(input_depth.is_cpu(), "visibility_filter_cpu2: input_depth must be CPU");
    TORCH_CHECK(intrinsic.is_cpu(), "visibility_filter_cpu2: intrinsic must be CPU");
    TORCH_CHECK(output.is_cpu(), "visibility_filter_cpu2: output must be CPU");

    CHECK_CONTIGUOUS(input_depth);
    CHECK_CONTIGUOUS(intrinsic);
    CHECK_CONTIGUOUS(output);

    TORCH_CHECK(input_depth.scalar_type() == output.scalar_type(), "input_depth/output dtype mismatch");
    TORCH_CHECK(intrinsic.scalar_type() == input_depth.scalar_type(), "intrinsic dtype mismatch");

    TORCH_CHECK(input_depth.numel() >= (int64_t)width * (int64_t)height, "input_depth too small");
    TORCH_CHECK(output.numel() >= (int64_t)width * (int64_t)height, "output too small");

    const int64_t nK = intrinsic.numel();
    TORCH_CHECK(nK == 4 || nK >= 9, "intrinsic must have 4 elems [fx,fy,cx,cy] or >=9 elems for 3x3 K");

    AT_DISPATCH_FLOATING_TYPES(input_depth.scalar_type(), "visibility_filter_cpu2_dispatch", [&] {
        const scalar_t* in_ptr = input_depth.data_ptr<scalar_t>();
        const scalar_t* k_ptr = intrinsic.data_ptr<scalar_t>();
        scalar_t* out_ptr = output.data_ptr<scalar_t>();

        if (nK == 4) {
            // old mode: fx,fy,cx,cy
            // we keep correctness identical to CUDA old path by calling the fullK path with diagonal K
            scalar_t K9[9];
            K9[0] = k_ptr[0]; K9[1] = (scalar_t)0; K9[2] = k_ptr[2];
            K9[3] = (scalar_t)0; K9[4] = k_ptr[1]; K9[5] = k_ptr[3];
            K9[6] = (scalar_t)0; K9[7] = (scalar_t)0; K9[8] = (scalar_t)1;
            visibility2_cpu_fullK<scalar_t>(in_ptr, K9, out_ptr, width, height, threshold, radius);
        } else {
            // new mode: 3x3 K
            visibility2_cpu_fullK<scalar_t>(in_ptr, k_ptr, out_ptr, width, height, threshold, radius);
        }
    });

    return output;
}

// --------------------------------------------------------
// Public API: runtime CPU/CUDA dispatch
// --------------------------------------------------------
at::Tensor depth_image(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output,
                       unsigned int size, unsigned int width, unsigned int height) {
    CHECK_CPU_OR_CUDA(input_uv);
    CHECK_CPU_OR_CUDA(input_depth);
    CHECK_CPU_OR_CUDA(output);

    CHECK_CONTIGUOUS(input_uv);
    CHECK_CONTIGUOUS(input_depth);
    CHECK_CONTIGUOUS(output);

    CHECK_SAME_DEVICE(input_uv, input_depth);
    CHECK_SAME_DEVICE(input_uv, output);

    if (input_uv.is_cuda()) {
        return depth_image_cuda(input_uv, input_depth, output, size, width, height);
    } else {
        return depth_image_cpu(input_uv, input_depth, output, size, width, height);
    }
}

at::Tensor visibility_filter(at::Tensor input, at::Tensor output,
                            unsigned int width, unsigned int height,
                            unsigned int threshold) {
    CHECK_CPU_OR_CUDA(input);
    CHECK_CPU_OR_CUDA(output);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_SAME_DEVICE(input, output);

    TORCH_CHECK(input.is_cuda(), "visibility (old kernel) is CUDA-only in this extension");
    return visibility_filter_cuda(input, output, width, height, threshold);
}

at::Tensor visibility_filter2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor output,
                             unsigned int width, unsigned int height,
                             float threshold, unsigned int radius) {
    CHECK_CPU_OR_CUDA(input_depth);
    CHECK_CPU_OR_CUDA(intrinsic);
    CHECK_CPU_OR_CUDA(output);

    CHECK_CONTIGUOUS(input_depth);
    CHECK_CONTIGUOUS(intrinsic);
    CHECK_CONTIGUOUS(output);

    CHECK_SAME_DEVICE(input_depth, intrinsic);
    CHECK_SAME_DEVICE(input_depth, output);

    const int64_t nK = intrinsic.numel();
    TORCH_CHECK(nK == 4 || nK >= 9, "intrinsic must have 4 elems [fx,fy,cx,cy] or >=9 elems for 3x3 K");

    if (input_depth.is_cuda()) {
        if (nK == 4) {
            return visibility_filter_cuda2(input_depth, intrinsic, output, width, height, threshold, radius);
        } else {
            // intrinsic is full 3x3 K
            return visibility_filter_cuda2_fullK(input_depth, intrinsic, output, width, height, threshold, radius);
        }
    } else {
        return visibility_filter_cpu2(input_depth, intrinsic, output, width, height, threshold, radius);
    }
}

at::Tensor downsample_flow(at::Tensor input_uv, at::Tensor output,
                          unsigned int width_out, unsigned int height_out,
                          unsigned int kernel) {
    CHECK_CPU_OR_CUDA(input_uv);
    CHECK_CPU_OR_CUDA(output);
    CHECK_CONTIGUOUS(input_uv);
    CHECK_CONTIGUOUS(output);
    CHECK_SAME_DEVICE(input_uv, output);

    TORCH_CHECK(input_uv.is_cuda(), "downsample_flow is CUDA-only in this extension");
    return downsample_flow_cuda(input_uv, output, width_out, height_out, kernel);
}

at::Tensor downsample_mask(at::Tensor input_mask, at::Tensor output,
                          unsigned int width_out, unsigned int height_out,
                          unsigned int kernel) {
    CHECK_CPU_OR_CUDA(input_mask);
    CHECK_CPU_OR_CUDA(output);
    CHECK_CONTIGUOUS(input_mask);
    CHECK_CONTIGUOUS(output);
    CHECK_SAME_DEVICE(input_mask, output);

    TORCH_CHECK(input_mask.is_cuda(), "downsample_mask is CUDA-only in this extension");
    return downsample_mask_cuda(input_mask, output, width_out, height_out, kernel);
}

at::Tensor downsample_depth(at::Tensor input_depth, at::Tensor output,
                           unsigned int width_out, unsigned int height_out,
                           unsigned int kernel) {
    CHECK_CPU_OR_CUDA(input_depth);
    CHECK_CPU_OR_CUDA(output);
    CHECK_CONTIGUOUS(input_depth);
    CHECK_CONTIGUOUS(output);
    CHECK_SAME_DEVICE(input_depth, output);

    TORCH_CHECK(input_depth.is_cuda(), "downsample_depth is CUDA-only in this extension");
    return downsample_depth_cuda(input_depth, output, width_out, height_out, kernel);
}

// --------------------------------------------------------
// PyBind
// --------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_image", &depth_image, "Generate Depth Image (CPU/CUDA)");
    m.def("visibility", &visibility_filter, "visibility_filter (CUDA only)");
    m.def("visibility2", &visibility_filter2, "visibility_filter2 (CPU/CUDA, intrinsic4 or full K)");
    m.def("downsample_flow", &downsample_flow, "downsample_flow (CUDA only)");
    m.def("downsample_mask", &downsample_mask, "downsample_mask (CUDA only)");
    m.def("downsample_depth", &downsample_depth, "downsample_depth (CUDA only)");
}
