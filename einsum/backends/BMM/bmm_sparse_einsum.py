import copy
import numpy as np
import opt_einsum as oe
import sys
from cgreedy import compute_path
from einsum.utilities.helper_functions import find_idc_types
from einsum.utilities.classes.coo_matrix import Coo_tensor
from timeit import default_timer as timer

handle_idx_time = 0
fit_tensor_time = 0
bmm_time = 0
shape_out_time = 0
permute_time = 0
dense_time = 0


def fit_tensor_to_bmm(mat: Coo_tensor, eq: str | None, shape: tuple | None):
    if eq:
        mat.single_einsum(eq)
    if shape:
        mat.reshape(shape)

    return mat


def calculate_contractions(cl: list, arrays: list, show_progress: bool):
    global handle_idx_time
    global fit_tensor_time
    global bmm_time
    global shape_out_time
    global permute_time
    global dense_time

    num_contractions_tenth = len(cl) // 10
    progress = 0

    for i, contraction in enumerate(cl):
        if show_progress and num_contractions_tenth != 0 and i % num_contractions_tenth == 0:
            sys.stdout.write(f"\rProgress: {progress}% done!")
            sys.stdout.flush()
            progress += 10

        current_arrays = [arrays[idx] for idx in contraction[0]]

        for id in contraction[0]:
            del arrays[id]

        is_coo_1 = isinstance(current_arrays[1], Coo_tensor)
        is_coo_0 = isinstance(current_arrays[0], Coo_tensor)
        is_sparse_1 = False
        is_sparse_0 = False

        if not is_coo_1:
            is_sparse_1 = (np.count_nonzero(
                current_arrays[1]) / np.prod(current_arrays[1].shape)) <= 0.9
        if not is_coo_0:
            is_sparse_0 = (np.count_nonzero(
                current_arrays[0]) / np.prod(current_arrays[0].shape)) <= 0.9

        if (is_coo_1 or is_coo_0) or (is_sparse_1 and is_sparse_0):
            if not isinstance(current_arrays[1], Coo_tensor):
                current_arrays[1] = Coo_tensor.from_numpy(current_arrays[1])
            if not isinstance(current_arrays[0], Coo_tensor):
                current_arrays[0] = Coo_tensor.from_numpy(current_arrays[0])

            # tic = timer()
            # Get index lists and sets
            input_idc, output_idc = clean_einsum_notation(contraction[1])
            shape_left = current_arrays[1].shape
            shape_right = current_arrays[0].shape

            results = find_idc_types(
                input_idc,
                output_idc,
                shape_left,
                shape_right
            )
            # toc = timer()
            # handle_idx_time += toc - tic

            eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB = results

            # tic = timer()
            # Fit both input tensors to match contraction
            current_arrays[1] = fit_tensor_to_bmm(
                current_arrays[1], eq_left, shape_left)
            current_arrays[0] = fit_tensor_to_bmm(
                current_arrays[0], eq_right, shape_right)
            # toc = timer()
            # fit_tensor_time += toc - tic

            scalar_mul = True if (len(current_arrays[1].shape) == 1 and
                                  len(current_arrays[0].shape) == 1 and
                                  current_arrays[1].shape[0] == 1 and
                                  current_arrays[0].shape[0] == 1) else False

            # tic = timer()
            if scalar_mul:
                AB = np.array([[0.0, current_arrays[1].data[0][1]
                                * current_arrays[0].data[0][1]]])
                arrays.append(Coo_tensor(AB, current_arrays[1].shape))
            else:
                arrays.append(Coo_tensor.coo_bmm(
                    current_arrays[1], current_arrays[0]))
            # toc = timer()
            # bmm_time += toc - tic

            # tic = timer()

            # Output reshape
            if shape_out is not None:
                arrays[-1].reshape(shape_out)
            # toc = timer()
            # shape_out_time += toc - tic

            # tic = timer()
            if perm_AB is not None:
                arrays[-1].swap_cols(perm_AB)
            # toc = timer()
            # permute_time += toc - tic
        else:
            # tic = timer()
            res = oe.contract(
                contraction[1], current_arrays[1], current_arrays[0])
            arrays.append(Coo_tensor.from_numpy(res))
            # toc = timer()
            # dense_time += toc - tic

        if type(arrays[-1].shape) != tuple:
            arrays[-1].shape = tuple(arrays[-1].shape)

    # print("\nhandle_idx_time TIME:", handle_idx_time)
    # print("fit_tensor_time TIME:", fit_tensor_time)
    # print("bmm_time TIME:", bmm_time)
    # print("shape_out_time TIME:", shape_out_time)
    # print("permute_time TIME:", permute_time)
    # print("dense_time TIME:", permute_time)
    # print()

    return arrays[0]


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    remaining.append(new_result)

    return new_result, remaining


def generate_contraction_list_with_code_points(in_out_idc: str, path):
    cl = []

    # Convert characters to code points
    input_idc, output_idc = in_out_idc
    input_sets = [set(map(ord, indices)) for indices in input_idc]
    output_set = set(map(ord, output_idc))

    for cnum, contract_idc in enumerate(path):
        contract_idc = sorted(contract_idc, reverse=True)
        out_idc, input_sets = find_contraction(
            contract_idc, input_sets, output_set)

        # Pop inputs and convert back to characters
        tmp_inputs = [input_idc.pop(idx) for idx in contract_idc]

        if cnum == len(path) - 1:
            idx_result = output_idc
        else:
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(
                sorted(map(chr, out_idc), key=all_input_inds.index))

        einsum_str = f"{','.join(reversed(tmp_inputs))}->{idx_result}"
        cl.append((contract_idc, einsum_str))

        input_idc.append(idx_result)

    return cl


def clean_einsum_notation(einsum_notation: str):
    einsum_notation = einsum_notation.replace(" ", "")
    input_idc = einsum_notation.split("->")[0].split(",")
    output_idc = einsum_notation.split("->")[1]

    return input_idc, output_idc


def arrays_to_coo(arrays):
    if not isinstance(arrays[0], Coo_tensor):
        tmp = []
        for array in arrays:
            if not isinstance(array, Coo_tensor):
                tmp.append(Coo_tensor.from_numpy(array))

        arrays = tmp

    return arrays


def is_dim_size_two(list: list):
    for t in list:
        if not all(element == 2 for element in t.shape):
            return False
    return True


def sparse_einsum(einsum_notation: str, arrays: list, path=None, show_progress=True, allow_alter_input=False):
    """
    Perform sparse tensor contraction using the Einstein summation convention.

    This function computes the result of an Einstein summation operation on a list of input arrays
    based on the provided einsum notation. It can optionally compute an optimized contraction path
    if one is not provided. The resulting tensor is returned as a NumPy array.

    Parameters
    ----------
    einsum_notation : str
        The Einstein summation notation specifying the tensor contraction operation. 
        For example, 'abc,cd->abd' describes a contraction between tensors 'abc' and 'cd' 
        resulting in a tensor 'abd'.

    arrays : list
        A list of NumPy arrays or sparse Coo_tensors to be contracted. The number of arrays should
        match the number of tensors specified in the einsum notation.

    path : list of tuples, optional
        A list of tuples specifying a path. Each tuple represents a pair of tensors to be 
        contracted in the path. If not provided, the an optimized contraction path is computed
        via `Cgreedy.compute_path`.

    show_progress : bool, optional, default=True
        If True, progress information will be displayed during the computation of the contractions.

    Returns
    --------
    numpy.ndarray
        The result of the Einstein summation operation, returned as a NumPy array.

    """
    # dim_size_two = False

    in_out_idc = clean_einsum_notation(einsum_notation)
    if allow_alter_input:
        tensors = arrays
    else:
        tensors = copy.deepcopy(arrays)

    if path is None:
        # Get Sesum contraction path
        path, flops_log10, size_log2 = compute_path(
            einsum_notation,
            *tensors,
            seed=0,
            minimize='size',
            max_repeats=1024,
            max_time=1.0,
            progbar=False,
            is_outer_optimal=False,
            threshold_optimal=12
        )

    if len(tensors) == 1:
        return np.einsum(einsum_notation, tensors[0])

    # Run specialized einsum for dim size 2 problems
    # if dim_size_two:
    #     res = Coo_matrix.coo_einsum_dim_2(arrays, in_out_idc, path)
    #     return res

    # tic = timer()
    cl = generate_contraction_list_with_code_points(in_out_idc, path)
    # toc = timer()
    # generate_contraction_list_time = toc - tic

    # tic = timer()
    res = calculate_contractions(cl, tensors, show_progress)
    # toc = timer()
    # calculate_contractions_time = toc - tic

    # print(f"GENERATE CONTRACTION LIST TIME: {generate_contraction_list_time}s")
    # print(f"CALCULATE CONTRACTIONS TIME: {calculate_contractions_time}s")
    # print()

    return res.to_numpy()
