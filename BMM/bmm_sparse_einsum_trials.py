import numpy as np
import sys

sys.path.append("./Utilities")
import utilities as util    # type: ignore


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


def coo_transpose(M, sort=True):
    M[:, [1, 0]] = M[:, [0, 1]]

    if sort:
        M = M[np.lexsort((M[:, 1], M[:, 0]))]

    return M


def coo_matmul(A, B, debug=False):
    B_T = coo_transpose(B.copy())

    if debug:
        print("A:")
        print(A)
        print()

        print("B:")
        print(B)
        print()

        print("B':")
        print(B_T)
        print()

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

    return np.transpose(np.array(C))


if __name__ == "__main__":
    A = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 10, 4, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 4, 2, 0, 0]
    ])

    B = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0],
        [0, 0, 0, 0, 3],
        [0, 0, 2, 9, 0],
        [0, 2, 7, 0, 0]
    ])

    A_coo = get_2d_coo_matrix(A)
    B_coo = get_2d_coo_matrix(B)

    AB_coo = coo_matmul(A_coo, B_coo)
    AB = A @ B

    print(AB_coo)
    print()
    print(AB)
    print(util.compare_matrices(AB_coo, AB))
