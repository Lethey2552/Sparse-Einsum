import numpy as np

# from .SQL.sql_sparse_einsum import sql_einsum_query_opt


def find_idc_types(input_set, input_idc, output_set):
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

    return order_left, order_right


def determine_transpose(order, target_order):
    # Find the transpose required to match the target order
    return [order.index(i) for i in target_order if i in order]


if __name__ == "__main__":
    einsum_notation = "kbi,bkj->bij"

    tensor_names = ["A", "B"]
    tensors = {
        "A": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        "B": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    }
    arrays = [tensors["A"], tensors["B"]]

    # sql_einsum_query_opt(einsum_notation, tensor_names, tensors, arrays)

    """
    [[[ 2  3]
      [ 6 11]]

    [[46 55]
     [66 79]]]
  """

    # Get index lists and sets
    einsum_notation = einsum_notation.replace(" ", "")
    input_idc = einsum_notation.split("->")[0].split(",")
    output_idc = einsum_notation.split("->")[1]
    input_sets = set().union(*input_idc)
    output_sets = set(output_idc)

    results = find_idc_types(input_sets, input_idc, output_sets)
    target_order_A, target_order_B = results
    current_order_A, current_order_B = input_idc

    transpose_A = [current_order_A.index(dim) for dim in target_order_A]
    transpose_B = [current_order_B.index(dim) for dim in target_order_B]

    print(transpose_A)
    print(transpose_B)
