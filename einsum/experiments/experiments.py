from instance_experiment import run_instance_experiment
from hypernetwork_experiment import run_hypernetwork_experiment

if __name__ == "__main__":
    NUMBER_OF_RUNS = 10
    ITERATIONS_PER_NETWORK = 10
    RUN_SPARSE = True
    RUN_SQL_EINSUM = True
    RUN_TORCH = True
    INSTANCE_NAME = "mc_2022_087"

    number_of_tensors = [50, 75, 100, 125, 150, 175, 200, 225]
    densities = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
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
    # print("Experiment - Number of Tensors:")
    # print("Tensors    Sparse time     SQL Einsum time    Torch time      Sparse_Einsum time")
    # for i in number_of_tensors:
    #     random_hypernetwork_params = {
    #         "number_of_tensors": i,
    #         "regularity": 2.5,
    #         "max_tensor_order": 5,
    #         "max_edge_order": 5,
    #         "number_of_output_indices": 0,
    #         "number_of_single_summation_indices": 0,
    #         "min_axis_size": 2,
    #         "max_axis_size": 2,
    #         "return_size_dict": True
    #     }

    #     run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
    #                                 NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params,
    #                                 density=0.0001,
    #                                 change=f"{i}        ")
    # print()

    # # Run varying density experiment
    # print("Experiment - Density:")
    # print(f"{f'Density':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse_Einsum time':<15}")
    # for i in densities:
    #     random_hypernetwork_params = {
    #         "number_of_tensors": 6,
    #         "regularity": 2.5,
    #         "max_tensor_order": 15,
    #         "max_edge_order": 3,
    #         "number_of_output_indices": 0,
    #         "number_of_single_summation_indices": 11,
    #         "min_axis_size": 2,
    #         "max_axis_size": 15,
    #         "return_size_dict": True
    #     }

    #     run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
    #                                 NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params,
    #                                 density=i,
    #                                 change=i)
    # print()

    # # Run varying max dimension sizes experiment
    # print("Experiment - Max Dimension Size:")
    # print(f"{f'Max dim size':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse_Einsum time':<15}")
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
    #         "global_dim": True
    #     }

    #     run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
    #                                 NUMBER_OF_RUNS,
    #                                 RUN_SPARSE,
    #                                 RUN_SQL_EINSUM,
    #                                 RUN_TORCH,
    #                                 random_hypernetwork_params,
    #                                 change=i)
    # print()

    # Run varying max number of dimensions experiment
    print("Experiment - Max Number of Dimensions:")
    print(f"{f'Max Num Dim':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse_Einsum time':<15}")
    for i in number_of_dims:
        random_hypernetwork_params = {
            "number_of_tensors": 50,
            "regularity": 2.5,
            "max_tensor_order": i,
            "max_edge_order": 5,
            "number_of_output_indices": 5,
            "number_of_single_summation_indices": 10,
            "min_axis_size": 2,
            "max_axis_size": 8,
            "return_size_dict": True
        }

        run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
                                    NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    random_hypernetwork_params,
                                    change=i)
    print()

    print(f"{f'Max Num Dim':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse_Einsum time':<15}")
    for i in number_of_dims:
        random_hypernetwork_params = {
            "number_of_tensors": 50,
            "regularity": 2.5,
            "max_tensor_order": i,
            "max_edge_order": 5,
            "number_of_output_indices": 5,
            "number_of_single_summation_indices": 10,
            "min_axis_size": 2,
            "max_axis_size": 8,
            "return_size_dict": True,
            "global_dim": True
        }

        run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
                                    NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    random_hypernetwork_params,
                                    change=i)
    print()

########    RESULTS     ########
"""
Instance        Sparse Time     Sparse Einsum Time
mc_2022_087     189.314s        5.929s

Experiment - Number of Tensors:

Number of Tensors   Sparse Time     SQL Einsum Time    Torch Time      Sparse Einsum Time    Sparse Einsum Legacy Time
50                  421.397 it/s    306.391 it/s       252.421 it/s    116.182 it/s             
75                  243.002 it/s    174.003 it/s       170.495 it/s    79.501  it/s             
100                 163.534 it/s    99.646  it/s       127.919 it/s    60.205  it/s             
125                 98.735  it/s    58.171  it/s       87.753  it/s    44.611  it/s             
150                 51.171  it/s    16.880  it/s       62.042  it/s    31.703  it/s             
175                 27.372  it/s    10.975  it/s       42.147  it/s    23.280  it/s             
200                 8.103   it/s    1.923   it/s       25.338  it/s    15.195  it/s
225                 2.608   it/s    0.431   it/s       11.279  it/s    7.810   it/s             


Experiment - Density:

Density        Sparse time    SQL Einsum time   Torch time     Sparse_Einsum time                   
0.1            50.654 it/s    170.941 it/s      57.177 it/s    31.197 it/s                              0.5            1861.184 it/s  28.514 it/s       965.630 it/s   186.834 it/s
0.01           51.067 it/s    26.587  it/s      57.256 it/s    29.747 it/s                              0.1            2099.729 it/s  306.577 it/s      999.673 it/s   268.900 it/s
0.001          49.171 it/s    25.168  it/s      56.582 it/s    29.646 it/s                              0.01           2058.218 it/s  3897.225 it/s     1045.695 it/s  461.444 it/s
0.0001         49.626 it/s    23.980  it/s      56.319 it/s    28.825 it/s                              0.001          2073.565 it/s  10740.302 it/s    994.059 it/s   562.939 it/s
1e-05          49.284 it/s    22.177  it/s      57.707 it/s    30.156 it/s                              0.0001         2086.577 it/s  10216.584 it/s    1070.339 it/s  609.744 it/s
1e-06          50.306 it/s    30.444  it/s      56.356 it/s    29.194 it/s                              1e-05          2090.805 it/s  4496.742 it/s     999.965 it/s   571.347 it/s


Experiment - Max Dim Size:

Max dim size   Sparse time     SQL Einsum time    Torch time      Sparse_Einsum time
2              373.610 it/s    376.019 it/s       194.520 it/s    97.629 it/s
4              175.790 it/s    681.750 it/s       153.421 it/s    81.697 it/s
6              50.254  it/s    557.416 it/s       92.871  it/s    59.370 it/s
8              10.481  it/s    278.805 it/s       40.257  it/s    31.935 it/s


Experiment - Max Num Dim:

Max Num Dim    Sparse time    SQL Einsum time    Torch time      Sparse_Einsum time
6              35.815 it/s    303.658 it/s       100.942 it/s    75.436 it/s
8              8.490  it/s    419.351 it/s       62.617  it/s    56.345 it/s
10             3.460  it/s    287.978 it/s       39.712  it/s    41.610 it/s
12             3.123  it/s    410.620 it/s       38.471  it/s    38.495 it/s
14             3.139  it/s    450.974 it/s       35.973  it/s    37.363 it/s

"""
