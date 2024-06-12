import logging
import numpy as np

class Coo_matrix:
    def __init__(self, data: np.ndarray, shape: np.array):
        self.data = data
        self.shape = shape


    @classmethod
    def coo_from_standard(cls, mat: np.ndarray):
        coo_mat = [[] for _ in range(len(mat.shape) + 1)]
        
        for idx, value in np.ndenumerate(mat):
            if mat[idx] == 0:
                continue

            for i, id in enumerate(idx):
                coo_mat[i].append(id)
            
            coo_mat[-1].append(value)

        return cls(np.transpose(np.array(coo_mat)), mat.shape)


    @classmethod
    def coo_matmul(cls, A: "Coo_matrix", B: "Coo_matrix", debug=False):
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

        return cls(np.transpose(np.array(C)), AB_shape)
    

    def __getitem__(self, items):
        return self.data[tuple(items)]


    def __setitem__(self, key, value):
        # Find the index where this key exists or should be inserted
        for i, row in enumerate(self.data):
            if tuple(row[:-1]) == tuple(key):
                self.data[i, -1] = value
                return
        new_entry = np.append(key, value)
        self.data = np.vstack([self.data, new_entry])


    def __len__(self):
        return len(self.data)


    def __str__(self):
        return np.array2string(self.data)


    def coo_transpose(self, sort=True):
        M = np.array(self.data)

        M[:, [1, 0]] = M[:, [0, 1]]

        if sort:
            M = M[np.lexsort((M[:, 1], M[:, 0]))]

        transposed_shape = (self.shape[1], self.shape[0])
        return Coo_matrix(M, transposed_shape)


    def coo_to_standard(self) -> np.ndarray:
        """
        Converts a sparse matrix in COO (Coordinate) format to a dense standard NumPy array.

        The `self.data` is expected to be an ndarray where each row represents a non-zero 
        entry in the sparse matrix. The last element of each row is the value, and the preceding 
        elements are the coordinates of the value.

        Returns:
        np.ndarray: A dense NumPy array with the same dimensionality as `self.shape`, 
                    populated with the values from `self.data` and zeros elsewhere.

        Example:
        >>> coo_mat = np.array([[0, 0, 1],
                                [1, 2, 3],
                                [2, 1, 4]])
        >>> coo_matrix = Coo_matrix(coo_mat, (3, 3))
        >>> coo_matrix.coo_to_standard()
        array([[1, 0, 0],
               [0, 0, 3],
               [0, 4, 0]])
        """

        mat = np.zeros(self.shape, dtype=int)
        
        for entry in self.data:
            mat[tuple(entry[:-1])] = entry[-1]

        return mat