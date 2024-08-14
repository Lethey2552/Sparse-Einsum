# windows version (compiler directives for cython compiler)
# cython: language_level = 3
# distutils: language = c++

import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound
from libc.stdint cimport uint32_t, int32_t, uint64_t, int64_t
from libc.stdlib cimport malloc, free
import ctypes

cdef extern from "coo_methods.h":
    void coo_bmm(const double* A_data, int A_rows, int A_cols,
                 const double* B_data, int B_rows, int B_cols,
                 double** C_data, int* C_rows, int* C_cols);
    void single_einsum(const double *data, int rows, int cols, const char *notation, const int *shape,
                       double **result_data, int *result_rows, int *result_cols,
                       int **new_shape, int *new_shape_size);
    void reshape(const double* data, int data_rows, int data_cols, 
                 const int* shape, const int shape_length,
                 const int* new_shape, const int new_shape_length,
                 double** result_data, int* result_rows, int* result_cols);
    void einsum_dim_2(
        uint32_t* in_out_flat,
        int32_t* in_out_sizes,
        int n_tensors,
        int n_map_items,
        uint32_t* keys_sizes,
        uint64_t* values_sizes,
        int32_t* path,
        double* arrays
    );

@boundscheck(False)
@wraparound(False)
def c_single_einsum(double[:] data, int rows, int cols, int[:] shape, bytes notation):
    cdef const double* data_ptr = &data[0]
    cdef const int* shape_ptr = &shape[0]
    cdef double *result_data
    cdef int result_rows, result_cols
    cdef int *new_shape = NULL
    cdef int new_shape_size

    single_einsum(data_ptr, rows, cols, notation, shape_ptr, &result_data, &result_rows, &result_cols, &new_shape, &new_shape_size)

    # Use the NumPy C-API to create an array from the C pointer
    cdef np.npy_intp dims[2]
    dims[0] = result_rows
    dims[1] = result_cols

    result = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, result_data)

    # Convert shape array to a Python tuple
    shape_tuple = tuple(new_shape[i] for i in range(new_shape_size))

    return result, shape_tuple

@boundscheck(False)
@wraparound(False)
def c_coo_bmm(double[:] A_data, int A_rows, int A_cols,
              double[:] B_data, int B_rows, int B_cols) -> np.ndarray:
    cdef const double* A_data_ptr = &A_data[0]
    cdef const double* B_data_ptr = &B_data[0]
    cdef double* C_data_ptr
    cdef int C_rows, C_cols

    coo_bmm(A_data_ptr, A_rows, A_cols, B_data_ptr, B_rows, B_cols, &C_data_ptr, &C_rows, &C_cols)

    # Use the NumPy C-API to create an array from the C pointer
    cdef np.npy_intp dims[2]
    dims[0] = C_rows
    dims[1] = C_cols
    result = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, C_data_ptr)

    return result
    
@boundscheck(False)
@wraparound(False)
def c_reshape(double[:] data, int data_rows, int data_cols, int[:] shape, int shape_len, int[:] new_shape, int new_shape_len) -> np.ndarray:
    # Cast pointers to the beginning of each array
    cdef double* data_ptr = &data[0]
    cdef int* shape_ptr = &shape[0]
    cdef int shape_length = shape_len
    cdef int new_shape_length = new_shape_len
    cdef int* new_shape_ptr = &new_shape[0]
    cdef double *result_data
    cdef int result_rows, result_cols

    # Call the C++ reshape function
    reshape(data_ptr, data_rows, data_cols, shape_ptr, shape_length, new_shape_ptr, new_shape_length, &result_data, &result_rows, &result_cols)

    # Use the NumPy C-API to create an array from the C pointer
    cdef np.npy_intp dims[2]
    dims[0] = result_rows
    dims[1] = result_cols
    result = np.PyArray_SimpleNewFromData(2, dims, np.NPY_DOUBLE, result_data)

    return result

@boundscheck(False)
@wraparound(False)
def c_einsum_dim_2(np.ndarray[np.uint32_t, ndim=1] in_out_flat,
                   np.ndarray[np.int32_t, ndim=1] in_out_sizes,
                   int n_tensors,
                   int n_map_items,
                   np.ndarray[np.uint32_t, ndim=1] keys_sizes,
                   np.ndarray[np.uint64_t, ndim=1] values_sizes,
                   np.ndarray[np.int32_t, ndim=1] path,
                   double[:] arrays):
    cdef uint32_t *in_out_flat_ptr = <uint32_t*>&in_out_flat[0]
    cdef int32_t *in_out_sizes_ptr = <int32_t*>&in_out_sizes[0]
    cdef int num_tensors = n_tensors
    cdef int num_map_items = n_map_items
    cdef uint32_t *keys_sizes_ptr = <uint32_t*>&keys_sizes[0]
    cdef uint64_t *values_sizes_ptr = <uint64_t*>&values_sizes[0]
    cdef int32_t *path_ptr = <int32_t*>&path[0]
    cdef double *data_ptr = &arrays[0]

    einsum_dim_2(
        in_out_flat_ptr,
        in_out_sizes_ptr,
        num_tensors,
        num_map_items,
        keys_sizes_ptr,
        values_sizes_ptr,
        path_ptr,
        data_ptr
    )