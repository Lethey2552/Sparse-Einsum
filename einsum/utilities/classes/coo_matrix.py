import numpy as np
from timeit import default_timer as timer
from einsum.backends.BMM.cpp_methods.coo_methods_lib import (
    c_coo_bmm,
    c_single_einsum,
    c_reshape,
)

time = 0
time_bmm = 0


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

        return cls(np.transpose(np.array(coo_mat, dtype=np.float64)), mat.shape)

    @classmethod
    def coo_matmul(cls, A: "Coo_matrix", B: "Coo_matrix"):
        if A.data.size == 0 or B.data.size == 0:
            C_data = np.transpose(np.array([[], [], []]))
        else:
            # Call the Cython function
            C_data = c_coo_bmm(A.data.flatten(), A.data.shape[0], A.data.shape[1],
                               B.data.flatten(), B.data.shape[0], B.data.shape[1])

        AB_shape = (A.shape[0], B.shape[1])

        return cls(C_data, AB_shape)

    # @classmethod
    # def coo_matmul_test(cls, A: "Coo_matrix", B: "Coo_matrix"):
    #     B_T = B.coo_transpose()
    #     C_dict = {}

    #     for i in range(len(A)):
    #         for j in range(len(B_T)):

    #             if A[i, 1] == B_T[j, 1]:
    #                 key = (A[i, 0], B_T[j, 0])

    #                 if key not in C_dict:
    #                     C_dict[key] = 0
    #                 C_dict[key] += A[i, 2] * B_T[j, 2]

    #     C = [[], [], []]
    #     for (i, j), v in C_dict.items():
    #         C[0].append(i)
    #         C[1].append(j)
    #         C[2].append(v)

    #     AB_shape = tuple([A.shape[0], B.shape[1]])

    #     return cls(np.transpose(np.array(C)), AB_shape)

    # @classmethod
    # def coo_bmm_test(cls, A: "Coo_matrix", B: "Coo_matrix"):
    #     cols_to_consider_A = A[:, :-3]
    #     cols_to_consider_B = B[:, :-3]
    #     combinations = np.unique(cols_to_consider_A[:, 0])

    #     # time = 0
    #     # tic = timer()
    #     # toc = timer()
    #     # time += toc - tic
    #     # print(f"Measured result: {time}s")

    #     AB_data = []
    #     for comb in combinations:
    #         A_masked = A.data[A[:, 0] == comb]
    #         B_masked = B.data[B[:, 0] == comb]

    #         A_test = Coo_matrix(A_masked[:, -3:], A.shape[-2:])
    #         B_test = Coo_matrix(B_masked[:, -3:], B.shape[-2:])

    #         AB = Coo_matrix.coo_matmul(A_test, B_test)

    #         # Append combination to the results
    #         comb_with_values = np.hstack(
    #             [np.full((AB.data.shape[0], 1), comb), AB.data])

    #         AB_data.append(comb_with_values)

    #     if AB_data:
    #         AB_data = np.vstack(AB_data)
    #         new_shape = tuple(A.shape[:-2]) + AB.shape[-2:]
    #         return cls(AB_data, new_shape)
    #     else:
    #         return cls(np.array([]), A.shape[:-1] + B.shape[-1:])

    @classmethod
    def coo_bmm(cls, A: "Coo_matrix", B: "Coo_matrix"):
        # global time_bmm
        # tic = timer()

        # Ensure the input matrices have compatible dimensions
        if A.shape[-1] != B.shape[-2]:
            raise ValueError(
                "Inner dimensions of A and B must match for multiplication.")

        A_rows, A_cols = A.data.shape
        B_rows, B_cols = B.data.shape

        C_data = c_coo_bmm(A.data.flatten(), A_rows, A_cols,
                           B.data.flatten(), B_rows, B_cols)

        new_shape = tuple(list(A.shape[:-1]) + list(B.shape[-1:]))

        # toc = timer()
        # time_bmm += toc - tic
        # print("BMM TIME: ")
        # print(time_bmm)

        return cls(C_data, new_shape)

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

    def __size__(self):
        return self.data.size

    def __str__(self):
        return np.array2string(self.data)

    # def single_einsum_test(self, notation: str):
    #     input_notation, output_notation = notation.split('->')

    #     # Ensure that the notation is for a single input tensor
    #     assert ',' not in input_notation

    #     indices = {k: i for i, k in enumerate(input_notation)}
    #     output_indices = [indices[idx]
    #                       for idx in output_notation if idx in indices]

    #     result_dict = {}

    #     for row in self.data:
    #         key = tuple(row[idx] for idx in output_indices)
    #         if key in result_dict:
    #             result_dict[key] += row[-1]
    #         else:
    #             result_dict[key] = row[-1]

    #     self.data = np.array([list(key) + [value]
    #                          for key, value in result_dict.items() if value != 0])

    #     # Adjust shape
    #     if self.data.size == 0:
    #         self.shape = (0,) * len(output_notation)
    #     else:
    #         self.shape = tuple(int(max(self.data[:, i]) + 1)
    #                            for i in range(self.data.shape[1] - 1))

    def single_einsum(self, notation: str):
        self.data, self.shape = c_single_einsum(self.data.flatten(),
                                                self.data.shape[0],
                                                self.data.shape[1],
                                                np.array(self.shape),
                                                notation.encode('utf-8')
                                                )

    # def reshape(self, new_shape):
    #     if np.prod(self.shape) != np.prod(new_shape):
    #         raise ValueError(
    #             "The total number of elements must remain the same for reshaping.")

    #     integer_indices = self.data[:, :-1].astype(int)
    #     float_values = self.data[:, -1]

    #     # Flatten, calculate new indices and create new data array
    #     original_flat_indices = np.ravel_multi_index(
    #         integer_indices.T, self.shape)

    #     new_indices = np.unravel_index(original_flat_indices, new_shape)

    #     self.data = np.column_stack(
    #         [np.array(new_indices).T, float_values.reshape(-1, 1)])
    #     self.shape = new_shape

    def reshape(self, new_shape):
        self.data = c_reshape(self.data.flatten(),
                              self.data.shape[0],
                              self.data.shape[1],
                              np.array(self.shape),
                              len(self.shape),
                              np.array(new_shape),
                              len(new_shape)
                              )
        self.shape = new_shape

    def swap_cols(self, idc: tuple | list):
        if type(idc) == tuple:
            idc = list(idc)

        # Swap columns given by idc list
        max_id = max(idc)
        idc.append(max_id + 1)
        self.data = self.data[:, idc]

        # Adjust shape
        new_shape = np.array(self.shape)[idc[:-1]]
        self.shape = new_shape

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

        mat = np.zeros(self.shape)

        for entry in self.data:
            mat[tuple([int(i) for i in entry[:-1]])] = entry[-1]

        return mat
