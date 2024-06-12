import logging
import numpy as np
import sys
from einsum.utilities.helper_functions import find_idc_types, compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix

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
    A = np.random.randint(0, 10, (4, 2, 5, 2))
    np.random.seed(1)
    B = np.random.randint(0, 10, (4, 2, 2, 3))

    A_coo = Coo_matrix.coo_from_standard(A)
    B_coo = Coo_matrix.coo_from_standard(B)

    AB_coo = Coo_matrix.coo_bmm(A_coo, B_coo)

    AB = A @ B

    print(f"""True Matrix AB {AB.shape}:\n{AB}\n""")

    print("Coo matmul result:")
    print(AB_coo.coo_to_standard())
    print()

    print(
        f"""Comparing coo matmul result to standard python matmul:
        \n{'Passed' if compare_matrices(AB_coo, AB) else 'Not passed'}"""
    )
