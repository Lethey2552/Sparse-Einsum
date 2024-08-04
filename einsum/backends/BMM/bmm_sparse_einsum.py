import numpy as np
import opt_einsum as oe
import sys
import torch
from cgreedy import compute_path
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


def calculate_contractions(cl: list, arrays: list, show_progress: bool):
    global handle_idx_time
    global fit_tensor_time
    global bmm_time
    global shape_out_time
    global permute_time

    num_contractions_tenth = len(cl) // 10
    progress = 0

    # print(cl[2])
    # print(arrays[2].shape)

    for i, contraction in enumerate(cl):
        if show_progress and num_contractions_tenth != 0 and i % num_contractions_tenth == 0:
            sys.stdout.write(f"\rProgress: {progress}% done!")
            sys.stdout.flush()
            progress += 10

        current_arrays = [arrays[idx] for idx in contraction[0]]

        for id in contraction[0]:
            del arrays[id]

        tic = timer()
        if contraction[2] == False:
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
        else:
            shape_out = contraction[3]
            perm_AB = contraction[4]

        scalar_mul = True if (len(current_arrays[1].shape) == 1 and
                              len(current_arrays[0].shape) == 1 and
                              current_arrays[1].shape[0] == 1 and
                              current_arrays[0].shape[0] == 1) else False

        tic = timer()
        if scalar_mul:
            AB = np.array([[0.0, current_arrays[1].data[0][1]
                          * current_arrays[0].data[0][1]]])
            arrays.append(Coo_matrix(AB, current_arrays[1].shape))
        else:
            arrays.append(Coo_matrix.coo_bmm(
                current_arrays[1], current_arrays[0]))
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

    print("\nhandle_idx_time TIME:", handle_idx_time)
    print("fit_tensor_time TIME:", fit_tensor_time)
    print("bmm_time TIME:", bmm_time)
    print("shape_out_time TIME:", shape_out_time)
    print("permute_time TIME:", permute_time)
    print()

    return arrays[0]


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    remaining.append(new_result)

    return new_result, remaining


def generate_contraction_list(in_out_idc: str, path):
    cl = []

    input_idc, output_idc = in_out_idc
    input_sets = [set(indices) for indices in input_idc]
    output_set = set(output_idc)

    ##### INFO ######
    # [((2, 1), 'k,kj->j'), ((1, 0), 'j,ij->i')]
    ##### INFO END #####

    # Create contraction list with (contract_idc, current_formula)
    for cnum, contract_idc in enumerate(path):
        contract_idc = tuple(sorted(list(contract_idc), reverse=True))
        out_idc, input_sets = find_contraction(
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
        cl.append(tuple([contract_idc, einsum_str]))

        input_idc.append(idx_result)

    return cl


def clean_einsum_notation(einsum_notation: str):
    einsum_notation = einsum_notation.replace(" ", "")
    input_idc = einsum_notation.split("->")[0].split(",")
    output_idc = einsum_notation.split("->")[1]

    return input_idc, output_idc


def arrays_to_coo(arrays):
    if not isinstance(arrays[0], Coo_matrix):
        tmp = []
        for array in arrays:
            if not isinstance(array, Coo_matrix):
                tmp.append(Coo_matrix.from_numpy(array))

        arrays = tmp

    return arrays


def preprocess_input_tensors(cl, tensors):
    # # Copy the original tensors list to keep track of active tensors
    # active_tensors = list(tensors)
    # original_indices = list(range(len(tensors)))

    # for i, contraction in enumerate(cl):
    #     contraction_indices = contraction[0]
    #     current_arrays = [active_tensors[idx] for idx in contraction[0]]

    #     # TODO: Solve original indice problem
    #     original_index_1 = tensors.index(current_arrays[1])
    #     original_index_0 = np.where(tensors == current_arrays[0])

    #     tmp_con = list(cl[i] + (False,))

    #     for id in contraction[0]:
    #         del active_tensors[id]
    #         del original_indices[id]

    #     if current_arrays[1] is not None and current_arrays[0] is not None:
    #         input_idc, output_idc = clean_einsum_notation(contraction[1])

    #         tensor_A_idc = input_idc[0]
    #         tensor_B_idc = input_idc[1]
    #         if len(tensor_A_idc) != len(current_arrays[1].shape):
    #             print(
    #                 f"{i}: {current_arrays[1].shape}        {input_idc[1]}        {len(tensor_A_idc)}")
    #         if len(tensor_B_idc) != len(current_arrays[0].shape):
    #             print(
    #                 f"{i}: {current_arrays[0].shape}        {input_idc[0]}        {len(tensor_B_idc)}")

    #         input_idc, output_idc = clean_einsum_notation(contraction[1])
    #         shape_left = current_arrays[1].shape
    #         shape_right = current_arrays[0].shape

    #         results = find_idc_types(
    #             input_idc,
    #             output_idc,
    #             shape_left,
    #             shape_right
    #         )

    #         eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB = results

    #         current_arrays[1] = torch.from_numpy(current_arrays[1])
    #         current_arrays[0] = torch.from_numpy(current_arrays[0])
    #         if eq_left:
    #             current_arrays[1] = oe.contract(
    #                 eq_left, current_arrays[1], backend="torch")
    #         if eq_right:
    #             current_arrays[0] = oe.contract(
    #                 eq_right, current_arrays[0], backend="torch")
    #         if shape_left:
    #             current_arrays[1] = torch.reshape(
    #                 current_arrays[1], shape_left)
    #         if shape_right:
    #             current_arrays[0] = torch.reshape(
    #                 current_arrays[0], shape_right)

    #         tmp_con[2] = True
    #         tmp_con.append(shape_out)
    #         tmp_con.append(perm_AB)

    #         # Write the modified tensors back to the original array
    #         for idx, original_idx in zip(contraction_indices, original_indices):
    #             tensors[original_idx] = current_arrays[idx]

    #     cl[i] = tuple(tmp_con)
    #     active_tensors.append(None)

    # print(active_tensors)

    # for i, contraction in enumerate(cl):
    #     idx1, idx2 = contraction[0]

    #     # Ensure indices are within the range of active tensors
    #     if idx1 < len(active_tensors) and idx2 < len(active_tensors):
    #         tensor1 = active_tensors[idx2]
    #         tensor2 = active_tensors[idx1]

    #         # Get index lists
    #         input_idc, output_idc = clean_einsum_notation(contraction[1])
    #         # shape_left = tensor_A.shape
    #         # shape_right = tensor_B.shape

    #         tensor_A_idc = input_idc[1]
    #         tensor_B_idc = input_idc[0]
    #         if len(tensor_A_idc) != len(tensor1.shape) and i <= 30:
    #             print(
    #                 f"{i}: {tensor1.shape}        {input_idc[1]}        {len(tensor_A_idc)}        {idx2} and {idx1}")
    #         if len(tensor_B_idc) != len(tensor2.shape) and i <= 30:
    #             print(
    #                 f"{i}: {tensor2.shape}        {input_idc[0]}        {len(tensor_B_idc)}        {idx1} and {idx2}")

    #         active_tensors.pop(idx1)
    #         active_tensors.pop(idx2)

    active_tensor_indices = list(range(len(tensors)))

    for i, contraction in enumerate(cl):
        idx0, idx1 = contraction[0]
        tmp_con = list(cl[i] + (False,))

        # Get the actual indices from the active tensor list
        actual_idx0 = active_tensor_indices[idx0]
        actual_idx1 = active_tensor_indices[idx1]
        # print(contraction)
        # print()
        # print("ITERATION:", contraction[1])
        # print(active_tensor_indices)

        # Preprocess only if both are from the original tensors list
        if actual_idx0 is not None and actual_idx1 is not None:
            current_arrays = [tensors[active_tensor_indices[idx]]
                              for idx in contraction[0]]

            # Get index lists
            input_idc, output_idc = clean_einsum_notation(contraction[1])
            shape_left = current_arrays[1].shape
            shape_right = current_arrays[0].shape

            tensor_A_idc = input_idc[0]
            tensor_B_idc = input_idc[1]
            if len(tensor_A_idc) != len(current_arrays[1].shape):
                print(
                    f"{i}: {current_arrays[1].shape}        {input_idc[1]}        {len(tensor_A_idc)}")
            if len(tensor_B_idc) != len(current_arrays[0].shape):
                print(
                    f"{i}: {current_arrays[0].shape}        {input_idc[0]}        {len(tensor_B_idc)}")

            results = find_idc_types(
                input_idc,
                output_idc,
                shape_left,
                shape_right
            )
            eq_left, eq_right, shape_left, shape_right, shape_out, perm_AB = results

            # current_arrays[1] = torch.from_numpy(current_arrays[1])
            # current_arrays[0] = torch.from_numpy(current_arrays[0])
            if eq_left:
                current_arrays[1] = oe.contract(
                    eq_left, current_arrays[1], use_blas=True)
            if eq_right:
                current_arrays[0] = oe.contract(
                    eq_right, current_arrays[0], use_blas=True)
            if shape_left:
                current_arrays[1] = np.reshape(
                    current_arrays[1], shape_left)
            if shape_right:
                current_arrays[0] = np.reshape(
                    current_arrays[0], shape_right)

            tmp_con[2] = True
            tmp_con.append(shape_out)
            tmp_con.append(perm_AB)
            for j, idx in enumerate(contraction[0]):
                # test_A = tensors[active_tensor_indices[idx]]
                # test_B = current_arrays[j].numpy()

                # if not np.allclose(test_A, test_B):
                #     print(test_A.shape)
                #     print(test_B.shape)
                #     print(test_A)
                #     print(test_B)
                tensors[active_tensor_indices[idx]] = current_arrays[j]

        cl[i] = tuple(tmp_con)
        if idx0 > idx1:
            active_tensor_indices.pop(idx0)
            active_tensor_indices.pop(idx1)
        else:
            active_tensor_indices.pop(idx1)
            active_tensor_indices.pop(idx0)

        # Append a new index to represent the result of the contraction
        active_tensor_indices.append(None)

    print(len(cl))

    count = 0
    for i in range(len(cl)):
        if cl[i][2] == True:
            count += 1
    print(count)

    return tensors


def is_dim_size_two(list: list):
    for t in list:
        if not all(element == 2 for element in t.shape):
            return False
    return True


def sparse_einsum(einsum_notation: str, arrays: list, path=None, show_progress=True):
    dim_size_two = False

    in_out_idc = clean_einsum_notation(einsum_notation)

    if path is None:
        # Get Sesum contraction path
        path, flops_log10, size_log2 = compute_path(
            einsum_notation,
            *arrays,
            seed=0,
            minimize='size',
            max_repeats=1024,
            max_time=1.0,
            progbar=False,
            is_outer_optimal=False,
            threshold_optimal=12
        )

    if len(arrays) == 1:
        return np.einsum(einsum_notation, arrays[0])

    # Run specialized einsum for dim size 2 problems
    # if dim_size_two:
    #     res = Coo_matrix.coo_einsum_dim_2(arrays, in_out_idc, path)
    #     return res

    tic = timer()
    cl = generate_contraction_list(in_out_idc, path)
    toc = timer()
    generate_contraction_list_time = toc - tic

    arrays = preprocess_input_tensors(cl, arrays)

    # if not dim_size_two:
    arrays = arrays_to_coo(arrays)

    tic = timer()
    res = calculate_contractions(cl, arrays, show_progress)
    toc = timer()
    calculate_contractions_time = toc - tic

    print(f"GENERATE CONTRACTION LIST TIME: {generate_contraction_list_time}s")
    print(f"CALCULATE CONTRACTIONS TIME: {calculate_contractions_time}s")
    print()

    return res.to_numpy()
