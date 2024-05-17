from operator import itemgetter
import numpy as np


def get_2d_coo_matrix(mat: np.ndarray):
    coo_mat = [[], [], []]

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 0:
                continue

            coo_mat[0].append(i)
            coo_mat[1].append(j)
            coo_mat[2].append(mat[i][j])

    return np.transpose(np.matrix(coo_mat))


if __name__ == "__main__":
    A = np.array([
        [0, 0, 0, 0, 9, 0],
        [0, 8, 0, 0, 0, 0],
        [4, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [0, 0, 2, 0, 0, 0]
    ])

    B = np.array([
        [0, 0, 4, 0 ,0],
        [0, 8, 0, 0 ,0],
        [0, 0, 0, 0 ,2],
        [0, 0, 2, 0 ,0],
        [9, 0, 0, 0 ,0],
        [0, 0, 0, 5 ,0],
    ])

    A_coo = get_2d_coo_matrix(A)
    B_coo = get_2d_coo_matrix(B)

    B_coo_swap = B_coo.copy()
    B_coo_swap[:, [1, 0]] = B_coo_swap[:, [0, 1]]
    B_coo_transp = sorted(B_coo_swap, key=itemgetter(0, 1))

    print("B:\n")
    print(B_coo)
    print(B_coo_swap)
    print(B_coo_transp)
    print()

    print(A @ B)