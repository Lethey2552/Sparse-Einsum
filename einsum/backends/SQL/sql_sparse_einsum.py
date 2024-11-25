import numpy as np
import sqlite3 as sql
from cgreedy import compute_path
from string import ascii_letters
from einsum.utilities.helper_functions import get_sizes, clean_einsum_notation
from itertools import product

DB_CONNECTION = sql.connect("SQL_einsum.db")
DB = DB_CONNECTION.cursor()

ASCII = list(ascii_letters)


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    idc_removed = idc_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idc_removed, idc_contract

# TODO: Handle SQL results being zero and shaping of output


def sql_einsum_values(tensors: dict):
    """
    Creates the tensors in COO format as SQL compatible structures
    and returns the appropriate query.
    """
    query_for_single_tensors = []
    for tensor_name, tensor in tensors.items():
        if isinstance(tensor, float) or isinstance(tensor, int):
            query = f" {tensor_name}(val) AS ( VALUES ("
            query += "("
            query += f"CAST({tensor} AS DOUBLE PRECISION))))\n"
        else:
            query = f" {tensor_name}({', '.join(ASCII[:len(tensor.shape)])}, val) AS (\n"
            # create value tuples
            values = []
            cast = False
            for row, indices in enumerate(product(*[range(i) for i in tensor.shape])):
                # skip zero values
                if tensor[indices] == 0:
                    continue
                # if we add the first value we have to give the data type
                if not cast:
                    type_definition = "("
                    for index in indices:
                        type_definition += f"CAST({index} AS INTEGER), "
                    type_definition += f"CAST({tensor[indices]} AS DOUBLE PRECISION))"
                    values.append(type_definition)
                    cast = True
                else:
                    values.append(f"{indices + (tensor[indices],)}")

            # Handle the case where all values are zero
            if not values:
                # Insert a placeholder value (e.g., with indices 0, 0, 0,... and val 0.0)
                type_definition = "("
                type_definition += ", ".join([f"CAST(0 AS INTEGER)"]
                                             * len(tensor.shape))
                type_definition += ", CAST(0.0 AS DOUBLE PRECISION))"
                values.append(type_definition)

            query += f"  VALUES {', '.join(values)}\n)"
        query_for_single_tensors.append(query)
    query = f"WITH {', '.join(query_for_single_tensors)}"
    return query


def sql_einsum_contraction(einsum_notation: str, tensor_names: list):
    # get einsum-notation indices
    formula = einsum_notation.replace(" ", "")
    tensorindices, outindices = formula.replace(" ", "").split("->")
    tensorindices = tensorindices.split(",")

    # get tensor names and set constants
    arrays = tensor_names
    array_aliases = []
    names = set()
    i = 1

    # generate einsum-summation
    from_clause = "FROM "
    for arr in arrays:
        alias = arr
        while alias in names:
            alias = "T" + str(i)
            i += 1
        names.add(alias)
        array_aliases.append(alias)
        from_clause += arr
        if arr != alias:
            from_clause += " " + alias
        from_clause += ", "
    from_clause = from_clause[:-2]

    group_by_clause = "GROUP BY "
    select_clause = "SELECT "

    # Die Indizes die bleiben, kommen in GROUP BY, diese Indice kommen auch aufgezählt in Ausgabereihenfolge in SELECT
    used_indices = set()
    idx = 0
    for i in range(len(outindices)):
        for t in range(len(tensorindices)):
            for j in range(len(tensorindices[t])):
                if tensorindices[t][j] == outindices[i] and outindices[i] not in used_indices:
                    used_indices.add(outindices[i])
                    varSQL = array_aliases[t] + "." + ASCII[j]
                    select_clause += varSQL + " AS " + ASCII[idx] + ", "
                    group_by_clause += varSQL + ", "
                    idx += 1

    group_by_clause = group_by_clause[:-2]

    # neue val ist immer SUM von allen val aufmultipliziert
    select_clause += "SUM("
    for t in array_aliases:
        select_clause += t + ".val * "
    select_clause = select_clause[:-3] + ") AS val"

    # Indices die gleich sind zwischen den Eingabetensoren kommen in die WHERE Klausel in transitiver Beziehung zueinander
    unique_indices = ""
    for t in range(len(tensorindices)):
        for j in range(len(tensorindices[t])):
            if tensorindices[t][j] not in unique_indices:
                unique_indices += tensorindices[t][j]

    related_tensors_per_index = []
    for i in range(len(unique_indices)):
        related_tensors_per_index.append([])
        for t in range(len(tensorindices)):
            for j in range(len(tensorindices[t])):
                if unique_indices[i] == tensorindices[t][j]:
                    related_tensors_per_index[i].append((t, j))

    where_clause = "WHERE "
    for i in range(len(related_tensors_per_index)):
        if len(related_tensors_per_index[i]) > 1:
            t, j = related_tensors_per_index[i][0]
            firstvarSQL = array_aliases[t] + "." + ASCII[j]
            for j in range(1, len(related_tensors_per_index[i])):
                t, j = related_tensors_per_index[i][j]
                varSQL = array_aliases[t] + "." + ASCII[j]
                where_clause += firstvarSQL + "=" + varSQL + " AND "

    where_clause = where_clause[:-5]

    # use an order by for
    order_by_clause = "ORDER BY "
    for c in ASCII[:len(outindices)]:
        order_by_clause += c + ", "
    order_by_clause = order_by_clause[:-2]

    if len(where_clause) < 5:
        where_clause = ""
    else:
        where_clause = " " + where_clause

    gb_and_ob = ""
    if outindices:
        gb_and_ob = " " + group_by_clause + \
            (" " + order_by_clause if True else "")

    # combine everything
    query = select_clause + " " + from_clause + where_clause + gb_and_ob
    return query


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    idc_removed = idc_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idc_removed, idc_contract


def sql_einsum_with_path(einsum_notation: str, tensor_names: list, path_info):
    # Contraction list
    path = path_info
    cl = []

    # Generating SQL query
    i = 1
    arrays = tensor_names.copy()
    names = set(arrays)
    query = ", "
    c = 0

    # Build contraction tuple (positions, einsum_not, remaining_einsum_not)
    input_idc, output_idc = clean_einsum_notation(einsum_notation)
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

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        remaining_formula = tuple(["".join(i) for i in input_sets])
        cl.append(tuple([contract_idc, idc_removed,
                         einsum_str, remaining_formula]))

        input_idc.append(idx_result)

    for contraction in cl:
        current_arrays = [arrays[idx] for idx in contraction[0]]

        for id in contraction[0]:
            arrays.pop(id)

        current_formula = contraction[2]
        current_removed = contraction[1]

        name = f"K{i}"
        while name in names:
            i += 1
            name = f"K{i}"
        names.add(name)
        arrays.append(name)
        i += 1
        c += 1

        # Generate SQL query for Einsum
        if c < len(cl):
            query += name + " AS (\n "
            query += sql_einsum_contraction(current_formula,
                                            current_arrays) + "\n), "
        else:

            query = query[:-2] + " " + \
                sql_einsum_contraction(
                    current_formula, current_arrays) + "\n"

    return query


def sql_einsum_query(einsum_notation: str, tensor_names: list, tensors: dict, path_info=None):
    query = ""
    values_query = sql_einsum_values(tensors)

    tensor_shapes = [t.shape for t in tensors.values()]

    input_idc, output_idc = clean_einsum_notation(einsum_notation)
    shapes_dict = get_sizes(input_idc, tensor_shapes)
    out_shape = tuple([shapes_dict[i] for i in output_idc])

    if path_info is None:
        path_info, _, _ = compute_path(
            einsum_notation,
            *tensor_shapes,
            seed=0,
            minimize='size',
            max_repeats=8,
            max_time=0.0,
            progbar=False,
            is_outer_optimal=False,
            threshold_optimal=12
        )

    contraction_query = sql_einsum_with_path(
        einsum_notation, tensor_names, path_info)

    query += values_query + contraction_query

    return query, out_shape


def sql_einsum_execute(query, res_shape):
    sql_result = DB.execute(query)
    sql_result = get_matrix_from_sql_response(sql_result.fetchall(), res_shape)

    return sql_result


def get_matrix_from_sql_response(coo_mat: np.ndarray, res_shape: tuple):
    if len(coo_mat) == 0 or coo_mat[0][0] is None:
        return np.zeros(res_shape)

    mat = np.zeros(res_shape)

    # Populate the matrix with non-zero entries
    for entry in coo_mat:
        coords = tuple([int(i) for i in entry[:-1]])
        mat[coords] = entry[-1]

    return mat
