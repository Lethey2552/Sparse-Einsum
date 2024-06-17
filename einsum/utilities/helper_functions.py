import numpy as np
import math
from einsum.utilities.classes.coo_matrix import Coo_matrix

def find_idc_types(input_idc, output_idc, shape_left, shape_right):
    batch_idc = []          # A, B, O
    con_idc = []            # A, B, .
    keep_left = []          # A, . , O
    keep_right = []         # ., B, O
    sizes = {}

    # Left input
    seen = set()
    for i, d in zip(input_idc[0], shape_left):

        if sizes.setdefault(i, d) != d:
            raise ValueError("Error with indices.")

        if i in seen:
            continue
        seen.add(i)

        # Batch dimension, contraction or keep left indices
        if i in input_idc[1]:
            if i in output_idc:
                batch_idc.append(i)
            else:
                con_idc.append(i)
        elif i in output_idc:
            keep_left.append(i)

    # Right input
    seen = set()
    for i, d in zip(input_idc[1], shape_right):

        if sizes.setdefault(i, d) != d:
            raise ValueError("Error with indices.")

        if i in seen:
            continue
        seen.add(i)

        # Keep right indices
        if i not in input_idc[0]:
            if i in output_idc:
                keep_right.append(i)

    # Remove trivial axis and transpose if necessary
    order_left = "".join((*batch_idc, *keep_left, *con_idc))
    if input_idc[0] != order_left:
        eq_left = f"{input_idc[0]}->{order_left}"
    else:
        eq_left = None

    order_right = "".join((*batch_idc, *con_idc, *keep_right))
    if input_idc[1] != order_right:
        eq_right = f"{input_idc[1]}->{order_right}"
    else:
        eq_right = None

    # Get permutation for output
    order_output = "".join((*batch_idc, *keep_left, *keep_right))
    perm_AB = tuple(order_output.index(i) for i in output_idc)
    if perm_AB == tuple(range(len(perm_AB))):
        perm_AB = None

    # Get new shapes for left, right and output
    if batch_idc:
        groups_left = (batch_idc, keep_left, con_idc)
        groups_right = (batch_idc, con_idc, keep_right)
        groups_out = (batch_idc, keep_left, keep_right)

    if any(len(group) != 1 for group in groups_left):
        shape_left = tuple(
            math.prod(sizes[i] for i in i_group) for i_group in groups_left
        )
    else:
        shape_left = None

    if any(len(group) != 1 for group in groups_right):
        shape_right = tuple(
            math.prod(sizes[i] for i in i_group) for i_group in groups_right
        )
    else:
        shape_right = None

    if any(len(group) != 1 for group in groups_out):
        shape_out = tuple(
            sizes[i] for i_group in groups_out for i in i_group
        )
    else:
        shape_out = None

    return eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB

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


def compare_matrices(mat_a: Coo_matrix, mat_b: np.ndarray):
    mat_a_standard = mat_a.coo_to_standard()

    return(np.allclose(mat_a_standard, mat_b))