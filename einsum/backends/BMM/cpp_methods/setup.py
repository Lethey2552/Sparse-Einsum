import numpy as np
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

extra_compile_args = []
op_name = platform.system()

# TODO: try compiling with linux subsystem gcc

if op_name == "Linux":
    extra_compile_args = ['-fopenmp', '-O2', '-mavx2', '-std=c++17']
elif op_name == "Windows":
    extra_compile_args = ['/openmp:experimental',
                          '/O2', '/arch:AVX2', '/std:c++17']

extensions = [
    Extension(
        "einsum.backends.BMM.cpp_methods.coo_methods_lib",
        sources=["./einsum/backends/BMM/cpp_methods/coo_methods_lib.pyx",
                 "./einsum/backends/BMM/cpp_methods/coo_methods.cpp"],
        include_dirs=[np.get_include(), "./libs/ips4o/include"],
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
