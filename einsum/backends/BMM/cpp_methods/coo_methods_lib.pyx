# windows version (compiler directives for cython compiler)
# cython: language_level = 3
# distutils: language = c++
# distutils: sources = coo_methods.cpp
# distutils: extra_compile_args = -std=c++14 -fopenmp -O2 -march=native
# cython: cplus = 14

import numpy as np
cimport numpy as np
from libc.stdint cimport *

cdef extern from "coo_methods.h":
    void coo_matmul(double* A_data, int A_rows, int A_cols,
                    double* B_data, int B_rows, int B_cols,
                    double** C_data, int* C_rows, int* C_cols);

def c_coo_matmul(double[:] A_data, int A_rows, int A_cols,
                 double[:] B_data, int B_rows, int B_cols) -> tuple:
    cdef double* A_data_ptr = &A_data[0]
    cdef double* B_data_ptr = &B_data[0]
    cdef double* C_data_ptr
    cdef int C_rows, C_cols

    coo_matmul(A_data_ptr, A_rows, A_cols, B_data_ptr, B_rows, B_cols, &C_data_ptr, &C_rows, &C_cols)

    # Convert C_data_ptr to a numpy array
    cdef np.ndarray[np.double_t, ndim=2] C_data = np.zeros((C_rows, C_cols), dtype=np.float64)

    # Copy data from C_data_ptr to C_data
    for i in range(C_rows):
        for j in range(C_cols):
            C_data[i, j] = C_data_ptr[i * C_cols + j]

    # Free C_data_ptr if needed (depending on ownership)
    # delete [] C_data_ptr  # Uncomment if ownership of C_data_ptr is in Cython's hands

    return C_data
    