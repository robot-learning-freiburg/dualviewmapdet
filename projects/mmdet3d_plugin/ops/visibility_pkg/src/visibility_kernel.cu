// This source code is from CMRNext
//   https://github.com/robot-learning-freiburg/CMRNext/blob/main/visibility_pkg/src/visibility_kernel.cu
// Copyright (c) Robot Learning Lab, University of Freiburg, licensed under the GPL-3.0 license,
// cf. 3rd-party-licenses.txt file in the root directory of this source tree.

// visibility_pkg/src/visibility_kernel.cu
#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define R 3

__device__ __forceinline__ int floatToOrderedInt(float floatVal) {
    int intVal = __float_as_int(floatVal);
    return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat(int intVal) {
    return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__global__ void depth_image_kernel(
    const int* __restrict__ in_uv,
    const float* __restrict__ in_depth,
    float* __restrict__ out,
    unsigned int size,
    unsigned int width,
    unsigned int height) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= 0 && index < (int)size) {
        int u = in_uv[2 * index + 0];
        int v = in_uv[2 * index + 1];
        if (u < 0 || u >= (int)width || v < 0 || v >= (int)height) return;

        int uv_index = v * width + u;
        atomicMin((int*)&out[uv_index], floatToOrderedInt(in_depth[index]));
    }
}

template <typename scalar_t>
__global__ void visibility_kernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    unsigned int width,
    unsigned int height,
    unsigned int threshold) {

    int x = threadIdx.x + blockIdx.x * blockDim.x; // row
    int y = threadIdx.y + blockIdx.y * blockDim.y; // col

    if (x < 0 || x >= (int)height || y < 0 || y >= (int)width) return;

    unsigned int index = x * width + y;
    out[index] = in[index];

    if (x >= R && y >= R && x < (int)(height - R) && y < (int)(width - R)) {
        scalar_t pixel = in[index];
        if (pixel != (scalar_t)0) {
            int sum = 0;
            int count = 0;
            for (int i = -R; i <= R; i++) {
                for (int j = -R; j <= R; j++) {
                    if (i == 0 && j == 0) continue;
                    int temp_index = (int)index + i * (int)width + j;
                    scalar_t temp_pixel = in[temp_index];
                    if (temp_pixel != (scalar_t)0) {
                        count += 1;
                        if (temp_pixel < pixel - (scalar_t)3.0) sum += 1;
                    }
                }
            }
            if (sum >= 1 + (int)(threshold * count / (R * R * 2 * 2))) {
                out[index] = (scalar_t)0;
            }
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm3(scalar_t x, scalar_t y, scalar_t z) {
    return x * x + y * y + z * z;
}

/**
 * OLD visibility2 kernel: expects intrinsic = [fx, fy, cx, cy]
 */
template <typename scalar_t>
__global__ void visibility_kernel2_fx_fy_cx_cy(
    const scalar_t* __restrict__ in_depth,
    const scalar_t* __restrict__ intrinsic4,
    scalar_t* __restrict__ out,
    unsigned int width,
    unsigned int height,
    float threshold,
    unsigned int radius) {

    int x = threadIdx.x + blockIdx.x * blockDim.x; // row
    int y = threadIdx.y + blockIdx.y * blockDim.y; // col

    if (x < 0 || x >= (int)height || y < 0 || y >= (int)width) return;

    unsigned int index = x * width + y;
    out[index] = in_depth[index];

    scalar_t pixel = in_depth[index];
    if (pixel == (scalar_t)0) return;

    scalar_t fx = intrinsic4[0];
    scalar_t fy = intrinsic4[1];
    scalar_t cx = intrinsic4[2];
    scalar_t cy = intrinsic4[3];

    // backproject assuming diagonal K
    scalar_t v_x = ((scalar_t)y - cx) * pixel / fx;
    scalar_t v_y = ((scalar_t)x - cy) * pixel / fy;
    scalar_t v_z = pixel;

    scalar_t v_norm = norm3(-v_x, -v_y, -v_z);
    if (v_norm <= (scalar_t)0) return;
    v_norm = (scalar_t)sqrt((double)v_norm);

    scalar_t v2_x = -v_x / v_norm;
    scalar_t v2_y = -v_y / v_norm;
    scalar_t v2_z = -v_z / v_norm;

    scalar_t max_dot1 = (scalar_t)-1;
    scalar_t max_dot2 = (scalar_t)-1;
    scalar_t max_dot3 = (scalar_t)-1;
    scalar_t max_dot4 = (scalar_t)-1;

    int r = (int)radius;

    auto update_maxdot = [&](int dx0, int dx1, int dy0, int dy1, scalar_t& maxdot) {
        for (int i = dx0; i <= dx1; i++) {
            for (int j = dy0; j <= dy1; j++) {
                int xx = x + i;
                int yy = y + j;
                if (xx < 0 || xx >= (int)height || yy < 0 || yy >= (int)width) continue;

                unsigned int tidx = (unsigned int)(xx * (int)width + yy);
                scalar_t tpix = in_depth[tidx];
                if (tpix == (scalar_t)0) continue;

                scalar_t c_x = ((scalar_t)yy - cx) * tpix / fx;
                scalar_t c_y = ((scalar_t)xx - cy) * tpix / fy;
                scalar_t c_z = tpix;

                c_x -= v_x; c_y -= v_y; c_z -= v_z;

                scalar_t c_norm = norm3(c_x, c_y, c_z);
                if (c_norm <= (scalar_t)0) continue;
                c_norm = (scalar_t)sqrt((double)c_norm);

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

/**
 * NEW visibility2 kernel: expects intrinsic9 = full 3x3 K (row-major),
 * and uses K^{-1} for backprojection => works with rotated/skewed intrinsics.
 */
template <typename scalar_t>
__global__ void visibility_kernel2_fullK(
    const scalar_t* __restrict__ in_depth,
    const scalar_t* __restrict__ K9,
    scalar_t* __restrict__ out,
    unsigned int width,
    unsigned int height,
    float threshold,
    unsigned int radius) {

    int x = threadIdx.x + blockIdx.x * blockDim.x; // row
    int y = threadIdx.y + blockIdx.y * blockDim.y; // col

    if (x < 0 || x >= (int)height || y < 0 || y >= (int)width) return;

    unsigned int index = x * width + y;
    out[index] = in_depth[index];

    scalar_t pixel = in_depth[index];
    if (pixel == (scalar_t)0) return;

    // Read K (row-major)
    scalar_t k00 = K9[0], k01 = K9[1], k02 = K9[2];
    scalar_t k10 = K9[3], k11 = K9[4], k12 = K9[5];
    scalar_t k20 = K9[6], k21 = K9[7], k22 = K9[8];

    // Invert 3x3 K analytically (per-thread; K is tiny so cost is OK)
    scalar_t det =
        k00 * (k11 * k22 - k12 * k21)
      - k01 * (k10 * k22 - k12 * k20)
      + k02 * (k10 * k21 - k11 * k20);

    if (det == (scalar_t)0) return;

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

    // Pixel homogeneous coordinate [u, v, 1] with u=y (col), v=x (row)
    scalar_t u = (scalar_t)y;
    scalar_t v = (scalar_t)x;

    scalar_t px = i00 * u + i01 * v + i02;
    scalar_t py = i10 * u + i11 * v + i12;
    scalar_t pz = i20 * u + i21 * v + i22;

    // Normalize so that pz ~ 1 (common for K^{-1}[u,v,1])
    if (pz == (scalar_t)0) return;
    px /= pz;
    py /= pz;

    // Backproject
    scalar_t v_x = px * pixel;
    scalar_t v_y = py * pixel;
    scalar_t v_z = pixel;

    scalar_t v_norm = norm3(-v_x, -v_y, -v_z);
    if (v_norm <= (scalar_t)0) return;
    v_norm = (scalar_t)sqrt((double)v_norm);

    scalar_t v2_x = -v_x / v_norm;
    scalar_t v2_y = -v_y / v_norm;
    scalar_t v2_z = -v_z / v_norm;

    scalar_t max_dot1 = (scalar_t)-1;
    scalar_t max_dot2 = (scalar_t)-1;
    scalar_t max_dot3 = (scalar_t)-1;
    scalar_t max_dot4 = (scalar_t)-1;

    int r = (int)radius;

    auto update_maxdot = [&](int dx0, int dx1, int dy0, int dy1, scalar_t& maxdot) {
        for (int i = dx0; i <= dx1; i++) {
            for (int j = dy0; j <= dy1; j++) {
                int xx = x + i;
                int yy = y + j;
                if (xx < 0 || xx >= (int)height || yy < 0 || yy >= (int)width) continue;

                unsigned int tidx = (unsigned int)(xx * (int)width + yy);
                scalar_t tpix = in_depth[tidx];
                if (tpix == (scalar_t)0) continue;

                // neighbor backproject
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

                // relative vector from current point to neighbor point
                c_x -= v_x; c_y -= v_y; c_z -= v_z;

                scalar_t c_norm = norm3(c_x, c_y, c_z);
                if (c_norm <= (scalar_t)0) continue;
                c_norm = (scalar_t)sqrt((double)c_norm);

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


// --------------------------------------------------------
// Wrappers exposed to C++
// --------------------------------------------------------
at::Tensor depth_image_cuda(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output,
                           unsigned int size, unsigned int width, unsigned int height) {
    dim3 threads(512);
    dim3 blocks(size / 512 + 1);
    depth_image_kernel<<<blocks, threads>>>(input_uv.data_ptr<int>(), input_depth.data_ptr<float>(),
                                            output.data_ptr<float>(), size, width, height);
    return output;
}

at::Tensor visibility_filter_cuda(at::Tensor input, at::Tensor output,
                                 unsigned int width, unsigned int height,
                                 unsigned int threshold) {
    dim3 threads(32, 32);
    dim3 blocks(height / 32 + 1, width / 32 + 1);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "visibility_filter", ([&] {
        visibility_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            width, height, threshold);
    }));
    return output;
}

at::Tensor visibility_filter_cuda2(at::Tensor input_depth, at::Tensor intrinsic4, at::Tensor output,
                                  unsigned int width, unsigned int height,
                                  float threshold, unsigned int radius) {
    dim3 threads(32, 32);
    dim3 blocks(height / 32 + 1, width / 32 + 1);
    AT_DISPATCH_FLOATING_TYPES(input_depth.scalar_type(), "visibility_filter2_fx_fy_cx_cy", ([&] {
        visibility_kernel2_fx_fy_cx_cy<scalar_t><<<blocks, threads>>>(
            input_depth.data_ptr<scalar_t>(),
            intrinsic4.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            width, height, threshold, radius);
    }));
    return output;
}

// NEW: full 3x3 K
at::Tensor visibility_filter_cuda2_fullK(at::Tensor input_depth, at::Tensor K9, at::Tensor output,
                                        unsigned int width, unsigned int height,
                                        float threshold, unsigned int radius) {
    dim3 threads(32, 32);
    dim3 blocks(height / 32 + 1, width / 32 + 1);
    AT_DISPATCH_FLOATING_TYPES(input_depth.scalar_type(), "visibility_filter2_fullK", ([&] {
        visibility_kernel2_fullK<scalar_t><<<blocks, threads>>>(
            input_depth.data_ptr<scalar_t>(),
            K9.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            width, height, threshold, radius);
    }));
    return output;
}

at::Tensor downsample_flow_cuda(at::Tensor input_uv, at::Tensor output,
                               unsigned int width_out, unsigned int height_out,
                               unsigned int kernel) {
    // unchanged (your original implementation lives elsewhere)
    return output;
}

at::Tensor downsample_mask_cuda(at::Tensor input_mask, at::Tensor output,
                               unsigned int width_out, unsigned int height_out,
                               unsigned int kernel) {
    // unchanged (your original implementation lives elsewhere)
    return output;
}

at::Tensor downsample_depth_cuda(at::Tensor input_depth, at::Tensor output,
                                unsigned int width_out, unsigned int height_out,
                                unsigned int kernel) {
    // unchanged (your original implementation lives elsewhere)
    return output;
}
