import numpy as np
from einsum.utilities.helper_functions import find_idc_types


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
    einsum_notation = "ckbi,cbkj->cbij"

    tensor_names = ["A", "B"]
    tensors = {
        "A": np.random.rand(2, 2, 3, 4),  # kbi -> bik (3, 4, 2)
        "B": np.random.rand(2, 3, 2, 5),  # bkj -> bkj (3, 2, 5)
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
