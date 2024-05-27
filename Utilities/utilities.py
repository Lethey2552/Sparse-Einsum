import numpy as np
from operator import itemgetter


def coo_to_standard(coo_mat: np.ndarray) -> np.ndarray:
    """
    Converts a sparse matrix in COO (Coordinate) format to a dense standard NumPy array.

    The input `coo_mat` is expected to be an ndarray where each row represents a non-zero 
    entry in the sparse matrix. The last element of each row is the value, and the preceding 
    elements are the coordinates of the value.

    Parameters:
    coo_mat (np.ndarray): A 2D NumPy array of shape (n, m+1), where `n` is the number of 
                          non-zero entries and `m` is the number of dimensions of the 
                          resulting dense matrix. Each row gives the coordinates for the
                          dimensions and the corresponding value.

    Returns:
    np.ndarray: A dense NumPy array with the same dimensionality as the coordinate 
                representation, populated with the values from `coo_mat` and zeros elsewhere.

    Example:
    >>> coo_mat = np.array([[0, 0, 1],
                            [1, 2, 3],
                            [2, 1, 4]])
    >>> coo_to_standard(coo_mat)
    array([[1, 0, 0],
           [0, 0, 3],
           [0, 4, 0]])
    """

    max_dim = tuple(max(coo_mat[:, i]) + 1 for i in range(coo_mat.shape[1] - 1))
        
    mat = np.zeros(max_dim, dtype=int)

    for entry in coo_mat:
        mat[tuple(entry[:-1])] = entry[-1]

    return mat


def compare_matrices(mat_a: np.ndarray, mat_b: np.ndarray):
    mat_a = coo_to_standard(mat_a)
    
    return(np.allclose(mat_a, mat_b))