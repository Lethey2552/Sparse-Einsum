from instance_experiment import run_instance_experiment
from hypernetwork_experiment import run_hypernetwork_experiment

if __name__ == "__main__":
    NUMBER_OF_RUNS = 10
    RUN_SPARSE = True
    RUN_SQL_EINSUM = False
    RUN_TORCH = False
    INSTANCE_NAME = "mc_2022_087"

    number_of_tensors = [30, 35, 40, 45, 50]
    sparsities = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    dim_sizes = [2, 4, 6, 8, 10]
    number_of_dims = [6, 8, 10, 12, 14]

    random_hypernetwork_params = {
        "number_of_tensors": 40,
        "regularity": 2.5,
        "max_tensor_order": 10,
        "max_edge_order": 5,
        "number_of_output_indices": 5,
        "number_of_single_summation_indices": 15,
        "min_axis_size": 2,
        "max_axis_size": 8,
        "return_size_dict": True,
        "seed": 12345
    }

    # run_instance_experiment(INSTANCE_NAME, NUMBER_OF_RUNS,
    #                         RUN_SPARSE, RUN_SQL_EINSUM)

    # # Run varying number of tensors experiment
    # for i in number_of_tensors:
    #     random_hypernetwork_params = {
    #         "number_of_tensors": i,
    #         "regularity": 4.0,
    #         "max_tensor_order": 10,
    #         "max_edge_order": 5,
    #         "number_of_output_indices": 5,
    #         "number_of_single_summation_indices": 15,
    #         "min_axis_size": 2,
    #         "max_axis_size": 2,
    #         "return_size_dict": True,
    #         "seed": 12345
    #     }

    #     run_hypernetwork_experiment(NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params
    #                                 )

    # Run varying sparsity experiment
    for i in sparsities:
        random_hypernetwork_params = {
            "number_of_tensors": 40,
            "regularity": 4.0,
            "max_tensor_order": 10,
            "max_edge_order": 5,
            "number_of_output_indices": 5,
            "number_of_single_summation_indices": 15,
            "min_axis_size": 2,
            "max_axis_size": 2,
            "return_size_dict": True,
            "seed": 12345,
            "global_dim": True
        }

        run_hypernetwork_experiment(NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    random_hypernetwork_params,
                                    sparsity=i,
                                    )

    # # Run varying max dimension sizes experiment
    # for i in dim_sizes:
    #     random_hypernetwork_params = {
    #         "number_of_tensors": 40,
    #         "regularity": 2.5,
    #         "max_tensor_order": 10,
    #         "max_edge_order": 5,
    #         "number_of_output_indices": 5,
    #         "number_of_single_summation_indices": 15,
    #         "min_axis_size": 2,
    #         "max_axis_size": i,
    #         "return_size_dict": True,
    #         "seed": 12345,
    #         "global_dim": True
    #     }

    #     run_hypernetwork_experiment(NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params,
    #                                 )

    # # Run varying max number of dimensions experiment
    # for i in number_of_dims:
    #     random_hypernetwork_params = {
    #         "number_of_tensors": 50,
    #         "regularity": 2.5,
    #         "max_tensor_order": i,
    #         "max_edge_order": 5,
    #         "number_of_output_indices": 5,
    #         "number_of_single_summation_indices": 10,
    #         "min_axis_size": 2,
    #         "max_axis_size": 8,
    #         "return_size_dict": True,
    #         "seed": 12345,
    #         "global_dim": True
    #     }

    #     run_hypernetwork_experiment(NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params,
    #                                 )

########    RESULTS     ########
"""
Instance        Sparse Time     Sparse Einsum Time
mc_2022_087     189.314s        5.929s
"""
