import numpy as np
import sqlite3 as sql

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

    return query[:-2]


def sql_einsum_query(einstein_notation: str, tensor_names: list, tensors: dict):
    query = "WITH "
    values_query = sql_einsum_values(tensors)

    query += values_query

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


def get_matrix_from_sql_response(SIZE: int, coo_mat: np.ndarray):
    mat = np.zeros((SIZE, SIZE), dtype=int)

    for entry in coo_mat:
         mat[entry[0]][entry[1]] = entry[2]

    return mat


if __name__ == "__main__":
    einstein_notation = "ij,jk->ik"

    tensor_names = ["A", "B"]
    tensors = {
        "A": np.array([[0, 1, 0, 6], [19, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]]),
        "B": np.array([[0, 0, 5, 0], [0, 1, 0, 0], [0, 0, 18, 0], [0, 0, 0, 8]])
    }

    query = sql_einsum_query(einstein_notation, tensor_names, tensors)
    print(query)

    # print(np.einsum("ij,jk->ik", mat_A, mat_B))
    
    # with_sql = get_with_clause(coo_matrices)

    # db_connection = sql.connect("test.db")
    # db = db_connection.cursor()

    # res = db.execute(
    #     """
    #     WITH A(i, j, val) AS (
    #     VALUES (CAST(0 AS INTEGER), CAST(1 AS INTEGER), CAST(1 AS INTEGER)), 
    #             (0, 3, 6), (1, 0, 19), (2, 2, 5), (3, 3, 4)
    #     ), B(i, j, val) AS (
    #     VALUES (CAST(0 AS INTEGER), CAST(2 AS INTEGER), CAST(5 AS INTEGER)),
    #             (1, 1, 1), (2, 2, 18), (3, 3, 8)
    #     ) SELECT A.i AS i, B.j AS j,
    #              SUM(B.val * A.val) AS val 
    #       FROM A, B 
    #       WHERE B.i=A.j 
    #       GROUP BY A.i, B.j 
    #       ORDER BY i,j
    #     """
    # )

    # mat = get_matrix_from_sql_response(SIZE, res.fetchall())
    # print(mat)