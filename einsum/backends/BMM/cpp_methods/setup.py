import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "einsum.backends.BMM.cpp_methods.coo_methods_lib",
        sources=["./einsum/backends/BMM/cpp_methods/coo_methods_lib.pyx",
                 "./einsum/backends/BMM/cpp_methods/coo_methods.cpp"],
        include_dirs=[np.get_include()],  # include the numpy headers
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
