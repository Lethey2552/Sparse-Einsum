# windows version (compiler directives for cython compiler)
# cython: language_level = 3
# distutils: language = c++
# distutils: extra_compile_args = /openmp:experimental /O2 /arch:AVX2 /std:c++17
# cython: cplus = 14

import numpy as np
from timeit import default_timer as timer
cimport numpy as np
from libc.stdint cimport *

cdef extern from "coo_methods.h":
    void coo_matmul(double* A_data, int A_rows, int A_cols,
                    double* B_data, int B_rows, int B_cols,
                    double** C_data, int* C_rows, int* C_cols);
    void coo_bmm(const double* A_data, int A_rows, int A_cols,
                     const double* B_data, int B_rows, int B_cols,
                     double** C_data, int* C_rows, int* C_cols);

def c_coo_matmul(double[:] A_data, int A_rows, int A_cols,
                 double[:] B_data, int B_rows, int B_cols) -> np.ndarray:
    cdef double* A_data_ptr = &A_data[0]
    cdef double* B_data_ptr = &B_data[0]
    cdef double* C_data_ptr
    cdef int C_rows, C_cols

    coo_matmul(A_data_ptr, A_rows, A_cols, B_data_ptr, B_rows, B_cols, &C_data_ptr, &C_rows, &C_cols)

    # Convert C_data_ptr to a numpy array
    cdef np.ndarray[np.double_t, ndim=2] C_data = np.zeros((C_rows, C_cols), dtype=np.float64)

    for i in range(C_rows):
        for j in range(C_cols):
            C_data[i, j] = C_data_ptr[i * C_cols + j]

    return C_data

def c_coo_bmm(double[:] A_data, int A_rows, int A_cols,
              double[:] B_data, int B_rows, int B_cols) -> np.ndarray:
    cdef const double* A_data_ptr = &A_data[0]
    cdef const double* B_data_ptr = &B_data[0]
    cdef double* C_data_ptr
    cdef int C_rows, C_cols

    time = 0
    tic = timer()
    coo_bmm(A_data_ptr, A_rows, A_cols, B_data_ptr, B_rows, B_cols, &C_data_ptr, &C_rows, &C_cols)
    toc = timer()
    time += toc - tic
    print(f"Measured result: {time}s")

    # Convert C_data_ptr to a numpy array
    cdef np.ndarray[np.double_t, ndim=2] C_data = np.zeros((C_rows, C_cols), dtype=np.float64)

    for i in range(C_rows):
        for j in range(C_cols):
            C_data[i, j] = C_data_ptr[i * C_cols + j]

    return C_data
    