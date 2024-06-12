import logging
import numpy as np
import sys
from einsum.utilities.helper_functions import find_idc_types, compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from itertools import product

def get_2d_coo_matrix(mat: np.ndarray):
    coo_mat = [[], [], []]

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 0:
                continue

            coo_mat[0].append(i)
            coo_mat[1].append(j)
            coo_mat[2].append(mat[i][j])

    return np.transpose(np.array(coo_mat))


def coo_matmul(A: Coo_matrix, B: Coo_matrix, debug=False):
    B_T = B.coo_transpose()
    C_dict = {}

    for i in range(len(A)):
        for j in range(len(B_T)):

            if A[i, 1] == B_T[j, 1]:
                key = (A[i, 0], B_T[j, 0])

                if key not in C_dict:
                    C_dict[key] = 0
                C_dict[key] += A[i, 2] * B_T[j, 2]

    C = [[], [], []]
    for (i, j), v in C_dict.items():
        C[0].append(i)
        C[1].append(j)
        C[2].append(v)

    AB_shape = tuple([A.shape[0], B.shape[1]])
    AB = Coo_matrix(np.transpose(np.array(C)), AB_shape)

    if debug:
        log_message = f"""
            \nMatrix A {A.shape}:\n{A}\n
            \nMatrix B {B.shape}:\n{B}\n
            \nMatrix B^T {B_T.shape}:\n{B_T}\n
            \nMatrix AxB {AB.shape}:\n{AB}\n
        """
        logging.debug(log_message)

    return AB


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    np.random.seed(0)
    A = np.random.randint(0, 10, (2, 2, 2))
    np.random.seed(1)
    B = np.random.randint(0, 10, (2, 2, 2))

    A_coo = Coo_matrix.coo_from_standard(A)
    B_coo = Coo_matrix.coo_from_standard(B)

    cols_to_consider = A_coo[:, :-3]
    unique_values = [np.unique(cols_to_consider[:, i]) for i in range(cols_to_consider.shape[1])]
    combinations = list(product(*unique_values))

    AB_coo = None
    for comb in combinations:
        A_test = A_coo[np.all(cols_to_consider == comb, axis=1), :]
        B_test = B_coo[np.all(B_coo[:, :-3] == comb, axis=1), :]

        A_test = Coo_matrix(A_test[:, -3:], A_coo.shape[-2:])
        B_test = Coo_matrix(B_test[:, -3:], B_coo.shape[-2:])

        AB = Coo_matrix.coo_matmul(A_test, B_test)
        insert_shape = np.zeros((len(comb)), dtype=int)

        if AB_coo is not None:
            AB_coo.data = np.vstack([AB_coo.data, np.insert(AB.data, insert_shape, list(comb), axis=1)])
        else:
            AB_coo = AB
            AB_coo.data = np.insert(AB_coo.data, insert_shape, list(comb), axis=1)

            new_shape = list(AB_coo.shape)
            new_shape = np.insert(new_shape, insert_shape, list(A_coo.shape[:len(comb)]))
            AB_coo.shape = tuple(new_shape)

    AB = A @ B

    print(f"""True Matrix AB {AB.shape}:\n{AB}\n""")

    print("Coo matmul result:")
    print(AB_coo.coo_to_standard())
    print()

    print(
        f"""Comparing coo matmul result to standard python matmul:
        \n{'Passed' if compare_matrices(AB_coo, AB) else 'Not passed'}"""
    )
