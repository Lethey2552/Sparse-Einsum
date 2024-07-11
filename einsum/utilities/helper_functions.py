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
    if batch_idc or keep_left or keep_right or con_idc:
        groups_left = (batch_idc, keep_left, con_idc)
        groups_right = (batch_idc, con_idc, keep_right)
        groups_out = (batch_idc, keep_left, keep_right)
    else:
        groups_left = ()
        groups_right = ()
        groups_out = ()

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
        if len(shape_out) == 0:
            shape_out = (1,)
    else:
        shape_out = None

    return eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB


def compare_matrices(mat_a: Coo_matrix, mat_b: np.ndarray):
    mat_a_standard = mat_a.coo_to_standard()

    return (np.allclose(mat_a_standard, mat_b))


def get_sizes(input_idc, shapes):
    index_sizes = {}
    for einsum_index, shape in zip(input_idc, shapes):
        shape = list(shape)
        for index, dimension in zip(list(einsum_index), shape):
            if not index in index_sizes:
                index_sizes[index] = dimension
            else:
                if index_sizes[index] != dimension:
                    raise Exception(f"Dimension error for index '{index}'.")
    return index_sizes
