import numpy as np
import sesum.sr as sr
import sqlite3 as sql
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
    from_clause = f"FROM {', '.join(tensor_names)}\n"

    # <R2> SELECT and GROUP BY clause
    select_clause = "SELECT "
    group_clause = "GROUP BY "
    for out_index in output_indices:
        tensor_name_index, ascii_index = [
                (i, tensor_index.index(out_index)) 
                for i, tensor_index in enumerate(input_indices) 
                if out_index in tensor_index
            ][0]
        
        select_clause += f"{tensor_names[tensor_name_index]}.{ASCII[ascii_index]} AS {ASCII[ascii_index]}, "
        group_clause += f"{tensor_names[tensor_name_index]}.{ASCII[ascii_index]}, "

    # <R3> Calculation by summing the products
    sum_clause = f"SUM({'.val * '.join(tensor_names)}.val) AS val\n"

    # <R4> WHERE clause
    where_clause = "WHERE "
    input_index_iterator = [i for list in input_indices for i in list]
    
    where_equations = []
    for index in input_index_iterator:
        test = [
                (i, tensor_index.index(index)) 
                for i, tensor_index in enumerate(input_indices) 
                if index in tensor_index
            ]
        
        for i in test[1:]:
            where_equations.append(f"{tensor_names[test[0][0]]}.{ASCII[test[0][1]]}={tensor_names[i[0]]}.{ASCII[i[1]]}")

    where_clause += f"{' AND '.join(set(where_equations))}\n"

    return select_clause + "\n" + sum_clause + from_clause + where_clause + group_clause[:-2]


def sql_einsum_query(einsum_notation: str, tensor_names: list, tensors: dict):
    query = "WITH "
    values_query = sql_einsum_values(tensors)
    contraction_query = sql_einsum_contraction(einsum_notation, tensor_names)

    query += values_query + contraction_query

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


if __name__ == "__main__":
    einsum_notation = "ij,kj,k->i"

    tensor_names = ["A", "B", "v"]
    tensors = {
        "A": np.array([[0, 1, 0, 6], [19, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]]),
        "B": np.array([[0, 0, 5, 0], [0, 1, 0, 0], [0, 0, 18, 0], [0, 0, 0, 8]]),
        "v": np.array([1, 0, 9, 11])
    }

    tensor_shapes = []
    for tensor in tensors.values():
        tensor_shapes.append(tensor.shape)

    # Get Sesum contraction path
    path, flops_log10, size_log2 = sr.compute_path(einsum_notation, *tensor_shapes, seed=0, minimize='size', algorithm="greedy", max_repeats=8,
                                               max_time=0.0, progbar=False, is_outer_optimal=False,
                                               threshold_optimal=12)
    
    print(path)

    query = sql_einsum_query(einsum_notation, tensor_names, tensors)
    with open("SQL/test_query.sql", "w") as file:
        file.write(query)

    print(f"--------SQL EINSUM QUERY--------\n\n{query}\n\n--------SQL EINSUM QUERY END--------\n\n")

    # Implicitly create database if not present, run sql query and format result
    db_connection = sql.connect("SQL/test.db")
    db = db_connection.cursor()
    res = db.execute(query)
    mat = get_matrix_from_sql_response(res.fetchall())

    # Get reference result
    np_einsum = np.einsum(einsum_notation, tensors["A"], tensors["B"], tensors["v"])

    print(f"--------SQL EINSUM RESULT--------\n\n{mat}\n\n--------NUMPY EINSUM RESULT--------\n\n{np_einsum}")