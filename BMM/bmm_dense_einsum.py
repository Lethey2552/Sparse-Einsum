import numpy as np

from SQL.sql_sparse_einsum import sql_einsum_query_opt

if __name__ == "__main__":
    einsum_notation = "bik,bkj->bij"

    tensor_names = ["A", "B"]
    tensors = {
        "A": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        "B": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    }
    arrays = [tensors["A"], tensors["B"]]

    sql_einsum_query_opt(einsum_notation, tensor_names, tensors, arrays)

    """
    [[[ 2  3]
      [ 6 11]]

    [[46 55]
     [66 79]]]
  """