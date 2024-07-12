import opt_einsum as oe
import pickle
import numpy as np
import torch
import sesum.sr as sr
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from timeit import default_timer as timer

if __name__ == "__main__":

    with open(
        "./einsum/benchmarks/instances/mc_2022_079.pkl", "rb"
    ) as file:
        format_string, tensors, path_meta, sum_output = pickle.load(file)

    # path optimized for minimal intermediate size
    path, size_log2, flops_log10, min_density, avg_density = path_meta[0]
    # print("Size optimized path")
    # print("log10[FLOPS]:", round(flops_log10, 2))
    # print("log2[SIZE]:", round(size_log2, 2))
    # result = oe.contract(format_string, *tensors, optimize=path)
    # print("sum[OUTPUT]:", np.sum(result), sum_output)

    # # path optimized for minimal total flops is stored und the key 1
    # print("Flops optimized path")
    # print("log10[FLOPS]:", round(flops_log10, 2))
    # print("log2[SIZE]:", round(size_log2, 2))
    # path, size_log2, flops_log10, min_density, avg_density = path_meta[1]
    # try:
    #     tic = timer()
    #     result = oe.contract(format_string, *tensors,
    #                          optimize=path, backend='sparse')
    #     toc = timer()
    #     print("sum[OUTPUT]:", np.sum(result), sum_output)
    #     print(f"opt_einsum time: {toc - tic}s\n")
    # except Exception as e:
    #     print(e)

    tic = timer()
    result = sr.sesum(format_string, *tensors, path=path, dtype=None, debug=False, safe_convert=False,
                      backend="sparse", semiring=sr.standard)
    toc = timer()
    print("sum[OUTPUT]:", np.sum(np.squeeze(result.data)), sum_output)
    print(f"Sesum time: {toc - tic}s\n")

    tic = timer()
    result = sparse_einsum(format_string, tensors, path=path)
    toc = timer()
    print("sum[OUTPUT]:", np.sum(np.squeeze(result.data)), sum_output)
    print(f"sparse_einsum time: {toc - tic}s\n")

    tic = timer()
    torch_tensors = [torch.from_numpy(i) for i in tensors]
    result = oe.contract(format_string, *torch_tensors,
                         optimize=path, backend='torch')
    toc = timer()
    print("sum[OUTPUT]:", result, sum_output)
    print(f"Torch time: {toc - tic}s")
