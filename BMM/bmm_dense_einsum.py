import numpy as np
import math


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


def single_einsum(eq: str, tensor: np.ndarray):
    return np.einsum(eq, tensor)


def bmm_contraction(tensors: dict, results: tuple):
    a = tensors["A"]
    b = tensors["B"]
    eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB = results

    # Left side
    if eq_left is not None:
        a = single_einsum(eq_left, tensors["A"])
    if shape_left is not None:
        a = np.reshape(a, shape_left)

    # Right side
    if eq_right is not None:
        b = single_einsum(eq_right, tensors["B"])
    if shape_right is not None:
        b = np.reshape(b, shape_right)

    ab = a @ b

    # Output reshape
    if shape_out is not None:
        ab = np.reshape(ab, shape_out)
    if perm_AB is not None:
        ab = np.transpose(ab, perm_AB)

    return ab


def determine_transpose(order, target_order):
    # Find the transpose required to match the target order
    return [order.index(i) for i in target_order if i in order]


if __name__ == "__main__":
    einsum_notation = "kbi,bkj->bij"

    tensor_names = ["A", "B"]
    tensors = {
        "A": np.random.rand(2, 3, 4),
        "B": np.random.rand(3, 2, 5),
    }
    arrays = [tensors["A"], tensors["B"]]

    # sql_einsum_query_opt(einsum_notation, tensor_names, tensors, arrays)

    # Get index lists and sets
    einsum_notation = einsum_notation.replace(" ", "")
    input_idc = einsum_notation.split("->")[0].split(",")
    shape_left = np.shape(tensors["A"])
    shape_right = np.shape(tensors["B"])
    output_idc = einsum_notation.split("->")[1]

    results = find_idc_types(
        input_idc,
        output_idc,
        shape_left,
        shape_right
    )

    ab = bmm_contraction(tensors, results)

    print(ab)

    # Get reference result
    np_einsum = np.einsum(
        einsum_notation, tensors["A"], tensors["B"])

    print(f"\n\n{np.allclose(ab, np_einsum)}")
