# This source code is from SparseDrive
# Copyright (c) 2024 Horizon Robotics
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",

        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        #include_package_data=True,
        #package_data={".": ["*/*.so"]},
        ext_modules=[
            make_cuda_ext(
                "deformable_aggregation_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation.cpp",
                    f"src/deformable_aggregation_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module=".roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name='bev_pool_v2_ext',
                module='.bev_pool_v2',
                sources=[
                    'src/bev_pool.cpp',
                ],
                sources_cuda=[
                    'src/bev_pool_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='iou3d_cuda',
                module='.iou3d',
                sources=[
                    'src/iou3d.cpp',
                ],
                sources_cuda=[
                    'src/iou3d_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='visibility',
                module='.visibility_pkg',
                sources=[
                    'src/visibility.cpp',
                ],
                sources_cuda=[
                    'src/visibility_kernel.cu',
                ],
                extra_args=['-std=c++14'],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/voxelization_cpu.cpp",
                ],
                sources_cuda=[
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cuda.cu",
                ]
            ),
            make_cuda_ext(
                name="sparse_conv_ext",
                module="spconv",
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*"spconv".split("."), "include/")
                    )
                ],
                sources=[
                    "src/all.cc",
                    "src/reordering_cpu.cc",
                    "src/reordering_cuda.cu",
                    "src/indice_cpu.cc",
                    "src/indice_cuda.cu",
                    "src/maxpool_cpu.cc",
                    "src/maxpool_cuda.cu",
                ],
                extra_args=["-w", "-std=c++17"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
