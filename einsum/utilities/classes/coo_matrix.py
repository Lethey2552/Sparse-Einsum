import numpy as np
from timeit import default_timer as timer
import einsum.utilities.helper_functions as helper_functions
from einsum.backends.BMM.cpp_methods.coo_methods_lib import (
    c_coo_bmm,
    c_single_einsum,
    c_reshape,
    c_einsum_dim_2,
)

time = 0
time_bmm = 0


class Coo_tensor:
    """
    A class representing a sparse tensor in COO (Coordinate) format.

    The COO format is a format for representing sparse tensors, where the 
    tensor is stored as a list of non-zero elements along with their coordinates 
    (indices).

    Attributes
    -----------
    data : np.ndarray
        A 2D NumPy array where each row represents a non-zero element in the 
        tensor. The columns are the indices for each dimension of the non-zero 
        elements, with the last column containing the values of these elements.
    shape : tuple of int
        The shape of the original dense tensor from which the COO tensor was created.
    nnz : int
        The number of non-zero elements in the tensor.
    sparsity : float
        The fraction of non-zero elements relative to the total number of elements 
        in the tensor, computed as `nnz / np.prod(shape)`.

    Parameters
    -----------
    data : np.ndarray
        A 2D array containing the COO format data. Expected shape is (nnz, 3) 
        where nnz is the number of non-zero elements.
    shape : np.array
        A 1D array specifying the dimensions of the original matrix.
    """

    def __init__(self, data: np.ndarray, shape: np.array):
        """
        Initialize a COO tensor with the given data and shape.

        Parameters
        -----------
        data : np.ndarray
            A 2D NumPy array where each row represents a non-zero element in the 
            tensor. The columns are the indices for each dimension of the non-zero 
            elements, with the last column containing the values of these elements.
        shape : np.array
            A 1D array indicating the dimensions of the original dense tensor. 
            This should match the shape of the tensor from which the COO tensor 
            was derived.

        Notes
        ------
        The data array is expected to have non-zero elements in the format of 
        [index 0, index 1, ..., index n, value]. The class computes additional 
        attributes such as the total number of non-zero elements (`nnz`) and the 
        sparsity of the tensor based on the provided shape.
        """
        # sorted_indices = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
        # data = data[sorted_indices]
        self.data = data
        self.shape = shape
        self.nnz = data.shape[0]
        self.sparsity = self.nnz / np.prod(shape)

    @classmethod
    def from_numpy(cls, mat: np.ndarray):
        """
            Create a COO (Coordinate) tensor representation from a NumPy array.

            This class method converts a dense NumPy array into a COO tensor format. 
            The resulting COO tensor includes the coordinates of non-zero elements 
            and their corresponding values, stacked into a format suitable for sparse 
            matrix operations.

            Parameters:
            -----------
            mat : np.ndarray
                A dense NumPy array to be converted into COO tensor format.

            Returns:
            --------
            Coo_tensor
                An instance of the `Coo_tensor` class initialized with the COO format data 
                and the original shape of the input array.
        """
        non_zero_indices = np.nonzero(mat)

        if len(mat.shape) == 0:
            non_zero_values = [mat]
        else:
            non_zero_values = mat[non_zero_indices]

        # Stack indices and append the values as the last row
        coo_mat = np.vstack(non_zero_indices +
                            (non_zero_values,), dtype=np.double)

        return cls(coo_mat.T, mat.shape)

    @classmethod
    def coo_matmul(cls, A: "Coo_tensor", B: "Coo_tensor"):
        if A.data.size == 0 or B.data.size == 0:
            C_data = np.transpose(np.array([[], [], []]))
        else:
            # Call the Cython function
            C_data = c_coo_bmm(A.data.flatten(), A.data.shape[0], A.data.shape[1],
                               B.data.flatten(), B.data.shape[0], B.data.shape[1])

        AB_shape = (A.shape[0], B.shape[1])

        return cls(C_data, AB_shape)

    @classmethod
    def coo_bmm(cls, A: "Coo_tensor", B: "Coo_tensor"):
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

    @classmethod
    def coo_einsum_dim_2(cls, arrays: list, in_out_idc: tuple, path: list):
        input_idc, output_idc = in_out_idc
        shapes = [array.shape for array in arrays]
        sizes = helper_functions.get_sizes(input_idc, shapes)
        out_shape = [sizes[i] for i in output_idc]
        arrays.append(np.empty(shape=out_shape, dtype=np.double, order='C'))
        arrays = np.concatenate(arrays).ravel()

        l_flat = [ord(char) for s in input_idc for char in s]
        l_flat += [ord(char) for s in output_idc for char in s]
        l_sizes = [len(s) for s in input_idc]
        l_sizes.append(len(output_idc))

        in_out_flat = np.array(l_flat, dtype=np.uint32)
        in_out_sizes = np.array(l_sizes, dtype=np.int32)
        n_tensors = len(in_out_sizes)
        n_map_items = len(sizes)
        keys_sizes = np.array(
            [ord(k) for k in sizes.keys()], dtype=np.uint32, order='C')
        values_sizes = np.array(list(sizes.values()),
                                dtype=np.uint64, order='C')
        path_flat = np.array(
            [i for tuple in path for i in tuple], dtype=np.int32)

        c_einsum_dim_2(
            in_out_flat,
            in_out_sizes,
            n_tensors,
            n_map_items,
            keys_sizes,
            values_sizes,
            path_flat,
            arrays
        )
        return cls(arrays[-1], arrays[-1].shape)

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

    def single_einsum(self, notation: str):
        self.data, self.shape = c_single_einsum(self.data.flatten(),
                                                self.data.shape[0],
                                                self.data.shape[1],
                                                np.array(
                                                    self.shape, dtype=np.int32),
                                                notation.encode('utf-8')
                                                )

    def reshape(self, new_shape):
        self.data = c_reshape(self.data.flatten(),
                              self.data.shape[0],
                              self.data.shape[1],
                              np.array(self.shape, dtype=np.int32),
                              len(self.shape),
                              np.array(new_shape, dtype=np.int32),
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
        return Coo_tensor(M, transposed_shape)

    def to_numpy(self) -> np.ndarray:
        """
        Converts a sparse matrix in COO (Coordinate) format to a dense standard NumPy array.

        The `self.data` is expected to be an ndarray where each row represents a non-zero 
        entry in the sparse matrix. The last element of each row is the value, and the preceding 
        elements are the coordinates of the value.

        Returns:
        np.ndarray: A dense NumPy array with the same dimensionality as `self.shape`, 
                    populated with the values from `self.data` and zeros elsewhere.
        """
        mat = np.zeros(self.shape, dtype=self.data.dtype)
        indices = tuple(self.data[:, i].astype(int)
                        for i in range(self.data.shape[1] - 1))

        if len(self.shape) == 0 or (len(self.shape) == 1 and self.shape[0] == 1):
            mat = self.data[:, -1][0]
        else:
            mat[indices] = self.data[:, -1]

        return mat


######## Legacy functions kept for validation ########
"""
    @classmethod
    def coo_matmul(cls, A: "Coo_matrix", B: "Coo_matrix"):
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

        return cls(np.transpose(np.array(C)), AB_shape)

    @classmethod
    def coo_bmm(cls, A: "Coo_matrix", B: "Coo_matrix"):
        cols_to_consider_A = A[:, :-3]
        cols_to_consider_B = B[:, :-3]
        combinations = np.unique(cols_to_consider_A[:, 0])

        # time = 0
        # tic = timer()
        # toc = timer()
        # time += toc - tic
        # print(f"Measured result: {time}s")

        AB_data = []
        for comb in combinations:
            A_masked = A.data[A[:, 0] == comb]
            B_masked = B.data[B[:, 0] == comb]

            A_test = Coo_matrix(A_masked[:, -3:], A.shape[-2:])
            B_test = Coo_matrix(B_masked[:, -3:], B.shape[-2:])

            AB = Coo_matrix.coo_matmul(A_test, B_test)

            # Append combination to the results
            comb_with_values = np.hstack(
                [np.full((AB.data.shape[0], 1), comb), AB.data])

            AB_data.append(comb_with_values)

        if AB_data:
            AB_data = np.vstack(AB_data)
            new_shape = tuple(A.shape[:-2]) + AB.shape[-2:]
            return cls(AB_data, new_shape)
        else:
            return cls(np.array([]), A.shape[:-1] + B.shape[-1:])

    def single_einsum(self, notation: str):
        input_notation, output_notation = notation.split('->')

        # Ensure that the notation is for a single input tensor
        assert ',' not in input_notation

        indices = {k: i for i, k in enumerate(input_notation)}
        output_indices = [indices[idx]
                          for idx in output_notation if idx in indices]

        result_dict = {}

        for row in self.data:
            key = tuple(row[idx] for idx in output_indices)
            if key in result_dict:
                result_dict[key] += row[-1]
            else:
                result_dict[key] = row[-1]

        self.data = np.array([list(key) + [value]
                             for key, value in result_dict.items() if value != 0])

        # Adjust shape
        if self.data.size == 0:
            self.shape = (0,) * len(output_notation)
        else:
            self.shape = tuple(int(max(self.data[:, i]) + 1)
                               for i in range(self.data.shape[1] - 1))

    def reshape(self, new_shape):
        if np.prod(self.shape) != np.prod(new_shape):
            raise ValueError(
                "The total number of elements must remain the same for reshaping.")

        integer_indices = self.data[:, :-1].astype(int)
        float_values = self.data[:, -1]

        # Flatten, calculate new indices and create new data array
        original_flat_indices = np.ravel_multi_index(
            integer_indices.T, self.shape)

        new_indices = np.unravel_index(original_flat_indices, new_shape)

        self.data = np.column_stack(
            [np.array(new_indices).T, float_values.reshape(-1, 1)])
        self.shape = new_shape
"""
