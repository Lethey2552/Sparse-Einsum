import numpy as np
import os
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

extra_compile_args = []
op_name = platform.system()

# TODO: try compiling with linux subsystem gcc

# Determine the absolute path to ips4o/include dynamically
ips4o_include_dir = os.path.join(os.getcwd(), "libs", "ips4o", "include")

intel_tbb_include = "C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\2021.13\\include"
intel_tbb_lib = "C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\2021.13\\lib"

if op_name == "Linux":
    extra_compile_args = ['-fopenmp', '-O2', '-mavx2', '-std=c++17']
    extra_link_args = ['-ltbb']
elif op_name == "Windows":
    extra_compile_args = ['/openmp:experimental',
                          '/O2', '/arch:AVX2', '/std:c++17']
    extra_link_args = [os.path.join(intel_tbb_lib, 'tbb.lib')]

extensions = [
    Extension(
        "einsum.backends.BMM.cpp_methods.coo_methods_lib",
        sources=["./einsum/backends/BMM/cpp_methods/coo_methods_lib.pyx",
                 "./einsum/backends/BMM/cpp_methods/coo_methods.cpp"],
        include_dirs=[np.get_include(), intel_tbb_include, ips4o_include_dir],
        library_dirs=[intel_tbb_lib],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="coo_methods_lib",
    ext_modules=cythonize(extensions),
    options={
        'build_ext': {
            'build_lib': './einsum/backends/BMM/cpp_methods',  # Specify the build directory
        }
    }
)
