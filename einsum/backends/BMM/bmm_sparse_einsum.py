import numpy as np
import sesum.sr as sr
from einsum.utilities.helper_functions import find_idc_types
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer

handle_idx_time = 0
fit_tensor_time = 0
bmm_time = 0
shape_out_time = 0
permute_time = 0


def fit_tensor_to_bmm(mat: Coo_matrix, eq: str | None, shape: tuple | None):
    if eq:
        mat.single_einsum(eq)
    if shape:
        mat.reshape(shape)

    return mat


def calculate_contractions(cl: list, arrays: np.ndarray):
    global handle_idx_time
    global fit_tensor_time
    global bmm_time
    global shape_out_time
    global permute_time

    for contraction in cl:
        current_arrays = [arrays[idx] for idx in contraction[0]]

        for id in contraction[0]:
            arrays.pop(id)

        tic = timer()
        # Get index lists and sets
        input_idc, output_idc = clean_einsum_notation(contraction[2])
        shape_left = current_arrays[1].shape
        shape_right = current_arrays[0].shape

        results = find_idc_types(
            input_idc,
            output_idc,
            shape_left,
            shape_right
        )
        toc = timer()
        handle_idx_time += toc - tic

        eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB = results

        tic = timer()

        # Fit both input tensors to match contraction
        current_arrays[1] = fit_tensor_to_bmm(
            current_arrays[1], eq_left, shape_left)
        current_arrays[0] = fit_tensor_to_bmm(
            current_arrays[0], eq_right, shape_right)
        toc = timer()
        fit_tensor_time += toc - tic

        tic = timer()
        arrays.append(Coo_matrix.coo_bmm(current_arrays[1], current_arrays[0]))
        toc = timer()
        bmm_time += toc - tic

        tic = timer()

        # Output reshape
        if shape_out is not None:
            arrays[-1].reshape(shape_out)
        toc = timer()
        shape_out_time += toc - tic

        tic = timer()
        if perm_AB is not None:
            arrays[-1].swap_cols(perm_AB)
        toc = timer()
        permute_time += toc - tic

        if type(arrays[-1].shape) != tuple:
            arrays[-1].shape = tuple(arrays[-1].shape)

    print("handle_idx_time TIME:", handle_idx_time)
    print("fit_tensor_time TIME:", fit_tensor_time)
    print("bmm_time TIME:", bmm_time)
    print("shape_out_time TIME:", shape_out_time)
    print("permute_time TIME:", permute_time)

    print()
    print("TOAL TIME:", handle_idx_time + fit_tensor_time + bmm_time)

    return arrays[0]


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    idc_removed = idc_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idc_removed, idc_contract


def generate_contraction_list(in_out_idc: str, path):
    cl = []

    input_idc, output_idc = in_out_idc
    input_sets = [set(indices) for indices in input_idc]
    output_set = set(output_idc)

    ##### INFO ######

    #   2           GEMM                k,kj->j                               ij,j->i
    #   2           GEMM                j,ij->i                                  i->i
    # [((2, 1), {'k'}, 'k,kj->j', ('ij', 'j'), 'GEMM'), ((1, 0), {'j'}, 'j,ij->i', ('i',), 'GEMM')]

    ##### INFO END #####

    # Create contraction list with (contract_idc, idc_removed, current_formula, remaining_formula)
    for cnum, contract_idc in enumerate(path):
        contract_idc = tuple(sorted(list(contract_idc), reverse=True))

        out_idc, input_sets, idc_removed, idc_contract = find_contraction(
            contract_idc, input_sets, output_set)

        tmp_inputs = [input_idc.pop(x) for x in contract_idc]

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_idc
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_idc, key=all_input_inds.find))

        einsum_str = ",".join(reversed(tmp_inputs)) + "->" + idx_result

        remaining_formula = tuple(["".join(i) for i in input_sets])
        cl.append(tuple([contract_idc, idc_removed,
                         einsum_str, remaining_formula]))

        input_idc.append(idx_result)

    return cl


def clean_einsum_notation(einsum_notation: str):
    einsum_notation = einsum_notation.replace(" ", "")
    input_idc = einsum_notation.split("->")[0].split(",")
    output_idc = einsum_notation.split("->")[1]

    return input_idc, output_idc


def arrays_to_coo(arrays):
    all_coo = True
    for array in arrays:
        if not isinstance(array, Coo_matrix):
            all_coo = False
            break

    if not all_coo:
        tmp = []
        for array in arrays:
            if not isinstance(array, Coo_matrix):
                tmp.append(Coo_matrix.coo_from_standard(array))

        arrays = tmp

    return arrays


def is_dim_size_two(list: list):
    for t in list:
        if not all(element == 2 for element in t.shape):
            return False
    return True


def sparse_einsum(einsum_notation: str, arrays: list, path=None):
    dim_size_two = is_dim_size_two(arrays)
    in_out_idc = clean_einsum_notation(einsum_notation)

    if not dim_size_two:
        arrays = arrays_to_coo(arrays)

        if len(arrays) == 1:
            arrays[0].single_einsum(einsum_notation)

            return arrays[0]

    if path is None:
        # Get Sesum contraction path
        path, flops_log10, size_log2 = sr.compute_path(
            einsum_notation,
            *arrays,
            seed=0,
            minimize='size',
            algorithm="greedy",
            max_repeats=8,
            max_time=0.0,
            progbar=False,
            is_outer_optimal=False,
            threshold_optimal=12
        )

    # Run specialized einsum for dim size 2 problems
    if dim_size_two:
        res = Coo_matrix.coo_einsum_dim_2(arrays, in_out_idc, path)
        return res

    cl = generate_contraction_list(in_out_idc, path)
    res = calculate_contractions(cl, arrays)

    return res
