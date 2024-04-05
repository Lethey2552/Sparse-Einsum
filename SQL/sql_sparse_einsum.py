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


def sql_einsum_contraction(einsum_notation: str, tensor_names: list):
    einsum_notation = einsum_notation.replace(" ", "")
    einsum_notation = einsum_notation.split("->")

    input_indices = einsum_notation[0].split(",")
    output_indices = einsum_notation[1]

    # <R1> FROM clause
    from_clause = f"FROM {', '.join(tensor_names)} "

    # <R2> SELECT and GROUP BY clause
    select_clause = "SELECT "
    group_clause = "GROUP BY "
    id = 0
    for out_index in output_indices:
        tensor_name_index, ascii_index = [
                (i, tensor_index.index(out_index)) 
                for i, tensor_index in enumerate(input_indices) 
                if out_index in tensor_index
            ][-1]
        
        select_clause += f"{tensor_names[tensor_name_index]}.{ASCII[ascii_index]} AS {ASCII[id]}, "
        group_clause += f"{tensor_names[tensor_name_index]}.{ASCII[ascii_index]}, "
        id += 1

    # <R3> Calculation by summing the products
    sum_clause = f"SUM({'.val * '.join(tensor_names)}.val) AS val "

    # <R4> WHERE clause
    where_clause = "WHERE "
    input_index_iterator = [i for list in input_indices for i in list]
    
    where_equations = []
    for index in input_index_iterator:
        equate_list = [
                (i, tensor_index.index(index)) 
                for i, tensor_index in enumerate(input_indices) 
                if index in tensor_index
            ]
        
        for i in equate_list[1:]:
            where_equations.append(f"{tensor_names[equate_list[0][0]]}.{ASCII[equate_list[0][1]]}={tensor_names[i[0]]}.{ASCII[i[1]]}")

    if len(output_indices) > 0:
        where_clause += f"{' AND '.join(set(where_equations))} "
    else:
        where_clause += "TRUE\n"

    return select_clause + sum_clause + from_clause + where_clause + (group_clause[:-2] if len(output_indices) > 0 else "")


def find_contraction(positions, input_sets, output_set):
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idc_contract = set.union(*inputs)
    idc_remain = output_set.union(*remaining)

    new_result = idc_remain & idc_contract
    idc_removed = idc_contract- new_result
    remaining.append(new_result)

    return new_result, remaining, idc_removed, idc_contract


def sql_einsum_with_path(einsum_notation: str, tensor_names: list, tensors: dict, path_info):
    # Contraction list
    cl = path_info

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
    input_sets = [set(indices) for indices in einsum_notation[0].split(",")]
    output_set = set(einsum_notation[1])
    
    ##### INFO ######

    #   2           GEMM                k,kj->j                               ij,j->i
    #   2           GEMM                j,ij->i                                  i->i
    #[((2, 1), {'k'}, 'k,kj->j', ('ij', 'j'), 'GEMM'), ((1, 0), {'j'}, 'j,ij->i', ('i',), 'GEMM')]

    ##### INFO END #####

    for cnum, contract_idc in enumerate(cl):
        contract_idc = tuple(sorted(list(contract_idc)))

        output_idc, input_sets, idc_removed, idc_contract = find_contraction(contract_idc, input_sets, output_set)

        current_formula = f"{''.join(input_idc[contract_idc[0]])},{''.join(input_idc[contract_idc[1]])}->{''.join(output_idc)}"
        remaining_formula = tuple(["".join(i) for i in input_sets])
        cl[cnum] = tuple([contract_idc, idc_removed, current_formula, remaining_formula])

        del input_idc[contract_idc[1]]
        del input_idc[contract_idc[0]]

        input_idc.append(output_idc)

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

        for id in reversed(contraction[0]):
            arrays.pop(id)
        
        current_formula = contraction[2]

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
            query += sql_einsum_contraction(current_formula, current_arrays) + "\n), "
        else:

            query = query[:-2] + " " + sql_einsum_contraction(current_formula, current_arrays) + "\n"

    return query


def sql_einsum_query(einsum_notation: str, tensor_names: list, tensors: dict, path_info=None):
    query = "WITH "
    values_query = sql_einsum_values(tensors)

    if path_info:
        contraction_query = sql_einsum_with_path(einsum_notation, tensor_names, tensors, path_info)
    else:
        contraction_query = sql_einsum_contraction(einsum_notation, tensor_names)

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

    query = sql_einsum_query(einsum_notation, tensor_names, tensors, path_info=path)
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

    query = sql_einsum_query_opt(einsum_notation, tensor_names, tensors, arrays)
    # with open("SQL/test_query.sql", "w") as file:
    #     file.write(query)

    print(f"--------SQL EINSUM QUERY--------\n\n{query}\n\n--------SQL EINSUM QUERY END--------\n\n")

    # Implicitly create database if not present, run sql query and format result
    db_connection = sql.connect("SQL/test.db")
    db = db_connection.cursor()
    res = db.execute(query)
    mat = get_matrix_from_sql_response(res.fetchall())

    # Get reference result
    np_einsum = np.einsum(einsum_notation, tensors["A"], tensors["B"], tensors["v"])

    print(f"--------SQL EINSUM RESULT--------\n\n{mat}\n\n--------NUMPY EINSUM RESULT--------\n\n{np_einsum}")