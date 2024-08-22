import pathlib
from util import (get_sql_query, get_sql_performance, get_sparse_performance,
                  get_torch_performance, get_sparse_einsum_performance)


def run_instance_experiment(instance_name, iterations=10, run_sparse=True, run_sql_einsum=True, run_torch=False, run_legacy=False):
    import einsum_benchmark

    instance = einsum_benchmark.instances[instance_name]
    format_string = instance.format_string
    tensors = instance.tensors
    path_meta = instance.paths.opt_size
    sum_output = instance.result_sum

    # path optimized for minimal intermediate size
    path = path_meta.path
    # print("Size optimized path")
    # print("log10[FLOPS]:", round(path_meta.flops, 2))
    # print("log2[SIZE]:", round(path_meta.size, 2))

    sparse_result = False
    # Time opt_einsum sparse backend average execution
    if run_sparse:
        sparse_time, sparse_result = get_sparse_performance(
            iterations, format_string, tensors, path)

    # Generate SQL einsum query and time average execution
    if run_sql_einsum:
        file = pathlib.Path(__file__).parent.parent.resolve().joinpath(
            "benchmarks", f"{instance_name}_sql_query.sql")

        if file.is_file():
            with open(file, "r") as f:
                query = f.read()
        else:
            with open(file, "w") as f:
                query = get_sql_query(format_string, tensors, f)

        sql_time = get_sql_performance(iterations, query, sparse_result)

    # Time opt_einsum torch backend average execution
    if run_torch:
        torch_time = get_torch_performance(
            iterations, format_string, tensors, path, sparse_result)

    sparse_einsum_time = get_sparse_einsum_performance(
        iterations, format_string, tensors, path, sparse_result,  run_density=True)

    # Time our sparse_einsum function
    if run_legacy:
        legacy_sparse_einsum_time = get_sparse_einsum_performance(
            iterations, format_string, tensors, path, sparse_result, run_parallel=True, run_density=True)

    output = f"{f'{instance_name}':<30}"

    if run_sparse:
        output += f"{f'{sparse_time:.3f} it/s':<15}"

    if run_sql_einsum:
        output += f"{f'{sql_time:.3f} it/s':<18}"

    if run_torch:
        output += f"{f'{torch_time:.3f} it/s':<15}"

    output += f"{f'{sparse_einsum_time:.3f} it/s':<21}"

    if run_legacy:
        output += f"{f'{legacy_sparse_einsum_time:.3f} it/s'}"

    print(output)
