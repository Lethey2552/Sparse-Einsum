import numpy as np

class Coo_matrix:
    def __init__(self, data: np.ndarray, shape: np.array):
        self.data = data
        self.shape = shape

    @classmethod
    def coo_from_standard(cls, mat: np.ndarray):
        coo_mat = [[], [], []]

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i][j] == 0:
                    continue

                coo_mat[0].append(i)
                coo_mat[1].append(j)
                coo_mat[2].append(mat[i][j])

        return cls(np.transpose(np.array(coo_mat)), mat.shape)


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