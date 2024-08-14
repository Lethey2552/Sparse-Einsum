import numpy as np
import os
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from distutils.command.clean import clean

use_mingw = False
MINGW_DIR = "C:/msys64/mingw64"
INCLUDE_DIRS = [np.get_include()]
LIB_DIRS = []
EXTRA_COMPILE_ARGS = []
EXTRA_LINK_ARGS = []
op_name = platform.system()


# Determine the absolute path to ips4o/include dynamically
ips4o_include_dir = os.path.join(os.getcwd(), "libs", "ips4o", "include")
intel_tbb_include = "C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\2021.13\\include"
intel_tbb_lib = "C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\2021.13\\lib"

INCLUDE_DIRS = INCLUDE_DIRS + [intel_tbb_include] + [ips4o_include_dir]
LIB_DIRS = LIB_DIRS + [intel_tbb_lib]

if op_name == "Linux" or use_mingw:
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

    if use_mingw:
        INCLUDE_DIRS = INCLUDE_DIRS + [os.path.join(MINGW_DIR, "include")]
        LIB_DIRS = LIB_DIRS + [os.path.join(MINGW_DIR, "lib")]

    EXTRA_COMPILE_ARGS = ['-fopenmp', '-O2', '-mavx2', '-std=c++17']
    EXTRA_LINK_ARGS = ['-static']
elif op_name == "Windows":
    EXTRA_COMPILE_ARGS = ['/openmp:experimental',
                          '/O2', '/arch:AVX2', '/std:c++17']
    EXTRA_LINK_ARGS = [os.path.join(intel_tbb_lib, 'tbb.lib')]

extensions = [
    Extension(
        "einsum.backends.BMM.cpp_methods.coo_methods_lib",
        sources=["./einsum/backends/BMM/cpp_methods/coo_methods_lib.pyx",
                 "./einsum/backends/BMM/cpp_methods/coo_methods.cpp"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIB_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
    )
]

setup(
    name="coo_methods_lib",
    cmdclass={'clean': clean, 'build_ext': build_ext},
    ext_modules=cythonize(extensions, force=True),
    options={
        'build_ext': {
            'build_lib': './einsum/backends/BMM/cpp_methods',  # Specify the build directory
        }
    }
)
