import numpy as np
import sqlite3 as sql

def get_coo_matrix(mat: np.ndarray):
    row = []
    col = []
    data = []

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 0:
                continue

            row.append(i)
            col.append(j)
            data.append(mat[i][j])

    print(mat)
    print()
    print(row)
    print(col)
    print(data)

if __name__ == "__main__":
    # mat_A = np.array([[0, 1, 0, 6], [19, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]])
    # mat_B = np.array([[0, 0, 5, 0], [0, 1, 0, 0], [0, 0, 18, 0], [0, 0, 0, 8]])

    # get_coo_matrix(mat_A)
    # get_coo_matrix(mat_B)

    # print(np.einsum("ij,jk->ik", mat_A, mat_B))

    db_connection = sql.connect("test.db")
    db = db_connection.cursor()

    res = db.execute(
        """
        WITH A(i, j, val) AS (
        VALUES (CAST(0 AS INTEGER), CAST(0 AS INTEGER), CAST(0.7056014072212418 AS DOUBLE PRECISION)), (0, 1, 0.45971589315238937), 
                (1, 0, 0.35489758282488826), (1, 1, 0.29359389730739716)
        ), B(i, j, val) AS (
        VALUES (CAST(0 AS INTEGER), CAST(0 AS INTEGER), CAST(0.22951224078373988 AS DOUBLE PRECISION)), (0, 1, 0.5675223248600516), 
                (0, 2, 0.40707012918777563), (0, 3, 0.9604683213986135), (0, 4, 0.4737084865351948), (1, 0, 0.9452823353846342), 
                (1, 1, 0.29468652270266804), (1, 2, 0.7796228987151677), (1, 3, 0.9852058160883493), (1, 4, 0.1183786073900277)
        ), C(i, val) AS (
        VALUES (CAST(0 AS INTEGER), CAST(0.18847560525837315 AS DOUBLE PRECISION)), (1, 0.5453807869796465), 
                (2, 0.33211095396533385), (3, 0.6498585676438938), (4, 0.8004577043178819)
        ),  K1 AS (
        SELECT B.i AS i, SUM(C.val * B.val) AS val FROM C, B WHERE C.i=B.j GROUP BY B.i
        ) SELECT A.i AS i, SUM(K1.val * A.val) AS val FROM K1, A WHERE K1.i=A.j GROUP BY A.i ORDER BY i
        """
    )

    print(res.fetchall())