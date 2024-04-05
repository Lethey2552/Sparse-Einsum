import numpy as np
import sesum.sr as sr
import sqlite3 as sql
import opt_einsum as oe
from operator import itemgetter

ASCII = [
    "i", "j", "x", "y", "z",
    "a", "b", "c", "d", "e",
    "f", "g", "h", "k", "l",
    "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v",
    "w"
]


def sql_einsum_values(tensors: dict):
    """
    Creates the tensors in COO format as SQL compatible structures
    and returns the appropriate query.
    """
    query = ""

    for tensor_name, tensor in tensors.items():
        cast = True
        query += f"{tensor_name}({', '.join(ASCII[:len(tensor.shape)])}, val) AS (\n"

        it = np.nditer(tensor, flags=['multi_index'])
        for x in it:
            # skip zero values
            if tensor[it.multi_index] == 0:
                continue

            # give data type for first entry with COO format
            if cast:
                type_casts = ""

                for i in it.multi_index:
                    type_casts += f"CAST({i} AS INTEGER), "
                type_casts += f"CAST({x} AS INTEGER)"

                query += f"VALUES({type_casts}),\n"

                cast = False
            else:
                item_coo = "("

                for i in it.multi_index:
                    item_coo += f"{i}, "

                query += f"{item_coo}{x}), "
        query = query[:-2]
        query += "\n), "

    return query[:-2] + "\n"


def sql_einsum_contraction(einsum_notation: str, tensor_names: list, current_removed: set):
    # get einsum-notation indices
    formula = einsum_notation.replace(" ", "")
    tensorindices, outindices = formula.replace(" ", "").split("->")
    tensorindices = tensorindices.split(",")

    # get tensor names and set constants
    arrays = tensor_names
    len_arrays = len(arrays)
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
    if False:
        if len_arrays == 1:
            select_clause += "SUM("
            for t in array_aliases:
                select_clause += t + ".re * "
            select_clause = select_clause[:-3] + ") AS re"

            select_clause += ", SUM("
            for t in array_aliases:
                select_clause += t + ".im * "
            select_clause = select_clause[:-3] + ") AS im"
        else:
            # (a+bi)(c+di) = (ac−bd) + (ad+bc)i
            a = array_aliases[0] + ".re"
            b = array_aliases[0] + ".im"
            c = array_aliases[1] + ".re"
            d = array_aliases[1] + ".im"
            select_clause += "SUM(" + a + " * " + c + \
                " - " + b + " * " + d + ") AS re"
            select_clause += ", SUM(" + a + " * " + \
                d + " + " + b + " * " + c + ") AS im"
    else:
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


def sql_einsum_with_path(einsum_notation: str, tensor_names: list, tensors: dict, path_info):
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
    einsum_notation = einsum_notation.replace(" ", "")
    einsum_notation = einsum_notation.split("->")

    input_idc = einsum_notation[0].split(",")
    output_idc = einsum_notation[1]
    input_sets = [set(indices) for indices in einsum_notation[0].split(",")]
    output_set = set(einsum_notation[1])

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

    with open("tmp.txt", "w", encoding="utf-8") as file:

        for inum, l in enumerate(cl):

            file.write(f"{inum}.      ")

            for k in range(4):
                if type(l[k]) is tuple:
                    file.write("(")
                    for j in l[k]:
                        file.write(str(j))
                        file.write(", ")
                    file.write("), ")
                else:
                    file.write(str(l[k]))
                    file.write(", ")

            file.write("\n\n\n")

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
                                            current_arrays, current_removed) + "\n), "
        else:

            query = query[:-2] + " " + \
                sql_einsum_contraction(
                    current_formula, current_arrays, current_removed) + "\n"

    return query


def sql_einsum_query(einsum_notation: str, tensor_names: list, tensors: dict, path_info=None):
    query = "WITH "
    values_query = sql_einsum_values(tensors)

    if path_info:
        contraction_query = sql_einsum_with_path(
            einsum_notation, tensor_names, tensors, path_info)
    else:
        contraction_query = sql_einsum_contraction(
            einsum_notation, tensor_names)

    query += values_query + contraction_query

    return query


def sql_einsum_query_opt(einsum_notation: str, tensor_names: list, tensors: dict, arrays: list):
    tensor_shapes = []
    for tensor in tensors.values():
        tensor_shapes.append(tensor.shape)

    # Get Sesum contraction path
    path, flops_log10, size_log2 = sr.compute_path(einsum_notation, *arrays, seed=0, minimize='size', algorithm="greedy", max_repeats=8,
                                                   max_time=0.0, progbar=False, is_outer_optimal=False,
                                                   threshold_optimal=12)

    query = sql_einsum_query(
        einsum_notation, tensor_names, tensors, path_info=path)
    return query


def get_2d_coo_matrix(mat: np.ndarray):
    coo_mat = [[], [], []]

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 0:
                continue

            coo_mat[0].append(i)
            coo_mat[1].append(j)
            coo_mat[2].append(mat[i][j])

    return coo_mat


def get_matrix_from_sql_response(coo_mat: np.ndarray):
    max_dim = ()
    for i in range(len(coo_mat[0]) - 1):
        max_dim = max_dim + (max(coo_mat, key=itemgetter(i))[i] + 1,)

    mat = np.zeros(max_dim, dtype=int)

    for entry in coo_mat:
        mat[entry[:-1]] = entry[-1]

    return mat


def _get_sizes(einsum_notation, tensor_names, tensors):
    index_sizes = {}
    for einsum_index, tensor_name in zip(einsum_notation.split("->")[0].split(","), tensor_names):
        for index, dimension in zip(list(einsum_index), list(np.array(tensors[tensor_name]).shape)):
            if not index in index_sizes:
                index_sizes[index] = dimension
            else:
                if index_sizes[index] != dimension:
                    raise Exception(f"Dimension error for index '{index}'.")
    return index_sizes


if __name__ == "__main__":
    einsum_notation = "ij,kj,k->i"

    tensor_names = ["A", "B", "v"]
    tensors = {
        "A": np.array([[0, 1, 0, 6], [19, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]]),
        "B": np.array([[0, 0, 5, 0], [0, 1, 0, 0], [0, 0, 18, 0], [0, 0, 0, 8]]),
        "v": np.array([1, 0, 9, 11])
    }
    arrays = [tensors["A"], tensors["B"], tensors["v"]]

    query = sql_einsum_query_opt(
        einsum_notation, tensor_names, tensors, arrays)
    # with open("SQL/test_query.sql", "w") as file:
    #     file.write(query)

    print(
        f"--------SQL EINSUM QUERY--------\n\n{query}\n\n--------SQL EINSUM QUERY END--------\n\n")

    # Implicitly create database if not present, run sql query and format result
    db_connection = sql.connect("SQL/test.db")
    db = db_connection.cursor()
    res = db.execute(query)
    mat = get_matrix_from_sql_response(res.fetchall())

    # Get reference result
    np_einsum = np.einsum(
        einsum_notation, tensors["A"], tensors["B"], tensors["v"])

    print(
        f"--------SQL EINSUM RESULT--------\n\n{mat}\n\n--------NUMPY EINSUM RESULT--------\n\n{np_einsum}")
