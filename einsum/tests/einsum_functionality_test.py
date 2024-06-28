import unittest
import sparse
import numpy as np
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.utilities.classes.coo_matrix import Coo_matrix
from einsum.utilities.helper_functions import compare_matrices


def get_dense(sparse_arrays):
    dense_arrays = []

    for i in sparse_arrays:
        dense_arrays.append(sparse.asnumpy(i))

    return dense_arrays


def get_sparse_einsum(dense_arrays):
    sparse_einsum_arrays = []

    for i in dense_arrays:
        sparse_einsum_arrays.append(Coo_matrix.coo_from_standard(i))

    return sparse_einsum_arrays


def run_einsum(einsum_notation, dense_arrays, sparse_einsum_arrays):
    numpy_res = np.einsum(einsum_notation, *dense_arrays)
    sparse_einsum_res = sparse_einsum(einsum_notation, sparse_einsum_arrays)

    return numpy_res, sparse_einsum_res


def sparse_einsum_equals_numpy(einsum_notation, sparse_arrays):
    dense_arrays = get_dense(sparse_arrays)
    sparse_einsum_arrays = get_sparse_einsum(dense_arrays)

    numpy_res, sparse_einsum_res = run_einsum(
        einsum_notation, dense_arrays, sparse_einsum_arrays)

    return compare_matrices(sparse_einsum_res, numpy_res)

##### TODO: ADDING A BATCH DIMENSION OF SIZE 1 TO THE FAILING PROBLEMS SOLVES THEM #####


class TestEinsumFunctions(unittest.TestCase):

    def test_diagonal(self):
        einsum_notation = "ii->i"

        A = sparse.random((2, 2), density=1.0, idx_dtype=int)

        sparse_arrays = [A]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'ii->i' failed.")

    def test_outer_product(self):
        einsum_notation = "i,j->ij"

        A = sparse.random((2,), density=1.0, idx_dtype=int)
        B = sparse.random((3,), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'i,j->ij' failed.")

    def test_mahalanobis_distance(self):
        einsum_notation = "i,ij,j->"

        A = sparse.random((2,), density=1.0, idx_dtype=int)
        B = sparse.random((2, 3), density=1.0, idx_dtype=int)
        C = sparse.random((3,), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B, C]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'i,ij,j->' failed.")

    def test_marginalization(self):
        einsum_notation = "ijklmno->i"

        A = sparse.random((2, 3, 4, 7, 3, 8, 2), density=1.0, idx_dtype=int)

        sparse_arrays = [A]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'ijklmno->i' failed.")

    def test_bmm(self):
        einsum_notation = "bik,bkj->bij"

        A = sparse.random((2, 3, 4), density=1.0, idx_dtype=int)
        B = sparse.random((2, 4, 3), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'bik,bkj->bij' failed.")

    def test_bilinear_transformation(self):
        einsum_notation = "ik,klj,il->ij"

        A = sparse.random((2, 3), density=1.0, idx_dtype=int)
        B = sparse.random((3, 4, 5), density=1.0, idx_dtype=int)
        C = sparse.random((2, 4), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B, C]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(equal_output, "Calculation of 'ik,klj,il->ij' failed.")

    def test_element_wise_product(self):
        einsum_notation = "ijkl,ijkl->ijkl"

        A = sparse.random((2, 3, 4, 5), density=1.0, idx_dtype=int)
        B = sparse.random((2, 3, 4, 5), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(
            equal_output, "Calculation of 'ijkl,ijkl->ijkl' failed.")

    def test_matrix_chain_multiplication(self):
        einsum_notation = "ik,kl,lm,mn,nj->ij"

        A = sparse.random((2, 3), density=1.0, idx_dtype=int)
        B = sparse.random((3, 4), density=1.0, idx_dtype=int)
        C = sparse.random((4, 6), density=1.0, idx_dtype=int)
        D = sparse.random((6, 5), density=1.0, idx_dtype=int)
        E = sparse.random((5, 2), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B, C, D, E]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(
            equal_output, "Calculation of 'ik,kl,lm,mn,nj->ij' failed.")

    def test_2x3_tensor_network(self):
        einsum_notation = "ij,iml,lo,jk,kmn,no->"

        A = sparse.random((2, 3), density=1.0, idx_dtype=int)
        B = sparse.random((2, 4, 5), density=1.0, idx_dtype=int)
        C = sparse.random((5, 6), density=1.0, idx_dtype=int)
        D = sparse.random((3, 7), density=1.0, idx_dtype=int)
        E = sparse.random((7, 4, 8), density=1.0, idx_dtype=int)
        F = sparse.random((8, 6), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B, C, D, E, F]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(
            equal_output, "Calculation of 'ij,iml,lo,jk,kmn,no->' failed.")

    def test_tucker_decomposition(self):
        einsum_notation = "ijkl,ai,bj,ck,dl->abcd"

        A = sparse.random((2, 3, 4, 5), density=1.0, idx_dtype=int)
        B = sparse.random((6, 2), density=1.0, idx_dtype=int)
        C = sparse.random((7, 3), density=1.0, idx_dtype=int)
        D = sparse.random((8, 4), density=1.0, idx_dtype=int)
        E = sparse.random((9, 5), density=1.0, idx_dtype=int)

        sparse_arrays = [A, B, C, D, E]
        equal_output = sparse_einsum_equals_numpy(
            einsum_notation, sparse_arrays)

        self.assertTrue(
            equal_output, "Calculation of 'ijkl,ai,bj,ck,dl->abcd' failed.")

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
