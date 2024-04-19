import numpy as np


def find_idc_types(input_set, input_idc, output_set, output_idc):
    triv_sum_idc = []
    contracted_idc = []
    contraction_idc = []
    batch_idc = []

    for i in input_set:
        count = 0
        output = True

        if i not in output_set:
            output = False

        for list in input_idc:
            if i in list:
                count += 1

        # Trivial sum or contracted indices
        if not output:

            if count < len(input_idc) or len(input_idc) == 1:
                triv_sum_idc.append(i)
            else:
                contracted_idc.append(i)

        # Batch dimension or contraction indices
        else:

            if count == len(input_idc):
                batch_idc.append(i)
            else:
                contraction_idc.append(i)

    order_left = batch_idc + \
        [idx for idx in contraction_idc if idx in input_idc[0]] + contracted_idc
    order_right = batch_idc + contracted_idc + \
        [idx for idx in contraction_idc if idx in input_idc[1]]
    order_output = batch_idc + \
        [idx for idx in output_idc if idx not in contracted_idc]

    perm_AB = tuple(order_output.index(i) for i in output_idc)
    if perm_AB == tuple(range(len(perm_AB))):
        perm_AB = None

    return order_left, order_right, order_output, perm_AB


def single_einsum(eq: str, tensor: np.ndarray):
    return np.einsum(eq, tensor)


def bmm_contraction(tensors: dict, input_idc: list, results: tuple):
    eq_a, eq_b = input_idc
    target_order_A, target_order_B, target_order_out, perm_AB = results

    # Left side
    if eq_a is not None:
        a = single_einsum(eq_a, tensors["A"])
        print(eq_a)
        print(a)
    # if target_order_A is not None:
    #     transpose_A = [current_order_A.index(dim) for dim in target_order_A]
    #     a = np.transpose(tensors["A"], transpose_A)
    print("\n")
    # Right side
    if eq_b is not None:
        b = single_einsum(eq_b, tensors["B"])
        print(eq_b)
        print(b)
    # if target_order_B is not None:
    #     transpose_B = [current_order_B.index(dim) for dim in target_order_B]
    #     b = np.transpose(tensors["B"], transpose_B)

    ab = a @ b

    # Output reshape
    # if target_order_out is not None:
    #     ab =
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
    output_idc = einsum_notation.split("->")[1]
    input_sets = set().union(*input_idc)
    output_sets = set(output_idc)

    results = find_idc_types(input_sets, input_idc, output_sets, output_idc)
    target_order_A, target_order_B, target_order_out, _ = results
    current_order_A, current_order_B = input_idc
    current_order_out = output_idc

    transpose_out = [current_order_out.index(dim) for dim in target_order_out]

    ab = bmm_contraction(tensors, input_idc, results)

    print(ab)

    # Get reference result
    np_einsum = np.einsum(
        einsum_notation, tensors["A"], tensors["B"])

    print(f"\n\n{np.allclose(ab, np_einsum)}")
