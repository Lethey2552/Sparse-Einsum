from instance_experiment import run_instance_experiment
from hypernetwork_experiment import run_hypernetwork_experiment, random_hypernetwork_benchmark_density

if __name__ == "__main__":
    NUMBER_OF_RUNS = 10
    ITERATIONS_PER_NETWORK = 10
    RUN_SPARSE = True
    RUN_SQL_EINSUM = True
    RUN_TORCH = True
    RUN_LEGACY = True

    instance_names = ["mc_2021_036", "mc_2022_087", "mc_2021_027"]
    # 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    number_of_tensors = [34, 38, 42]
    densities = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    dim_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    number_of_dims = [6, 7, 8, 9, 10, 11, 12, 13, 14]

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

    # Run instance of einsum benchmark
    print("Experiment - Instance:")
    print(f"{f'Instance Name':<30}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse Einsum time':<21}{f'Legacy Sparse Einsum time':<25}")
    for i in instance_names:
        run_instance_experiment(i,
                                ITERATIONS_PER_NETWORK,
                                RUN_SPARSE,
                                RUN_SQL_EINSUM,
                                RUN_TORCH,
                                RUN_LEGACY)

    # Run varying number of tensors experiment
    print("Experiment - Number of Tensors:")
    print(f"{f'Tensors':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse Einsum time':<21}{f'Legacy Sparse Einsum time':<25}")
    for i in number_of_tensors:
        random_hypernetwork_params = {
            "number_of_tensors": i,
            "regularity": 3.0,
            "max_tensor_order": 9,
            "max_edge_order": 3,
            "number_of_output_indices": 0,
            "number_of_single_summation_indices": 15,
            "min_axis_size": 2,
            "max_axis_size": 9,
            "return_size_dict": True
        }

        run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
                                    NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    RUN_LEGACY,
                                    random_hypernetwork_params,
                                    density=0.0001,
                                    change=f"{i}")
    print()

    # Run varying density experiment
    print("Experiment - Density:")
    print(f"{f'Density':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse Einsum time':<21}{f'Legacy Sparse Einsum time':<25}")

    random_hypernetwork_params = {
        "number_of_tensors": 6,
        "regularity": 3.0,
        "max_tensor_order": 15,
        "max_edge_order": 3,
        "number_of_output_indices": 0,
        "number_of_single_summation_indices": 15,
        "min_axis_size": 2,
        "max_axis_size": 15,
        "return_size_dict": True
    }

    random_hypernetwork_benchmark_density(ITERATIONS_PER_NETWORK,
                                          NUMBER_OF_RUNS,
                                          RUN_SPARSE,
                                          RUN_SQL_EINSUM,
                                          RUN_TORCH,
                                          RUN_LEGACY,
                                          random_hypernetwork_params,
                                          densities=densities)
    print()

    # Run varying max dimension sizes experiment
    print("Experiment - Max Dimension Size:")
    print(f"{f'Max dim size':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse Einsum time':<21}{f'Legacy Sparse Einsum time':<25}")
    for i in dim_sizes:
        random_hypernetwork_params = {
            "number_of_tensors": 6,
            "regularity": 3.0,
            "max_tensor_order": 15,
            "max_edge_order": 3,
            "number_of_output_indices": 0,
            "number_of_single_summation_indices": 15,
            "min_axis_size": 2,
            "max_axis_size": i,
            "return_size_dict": True
        }

        run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
                                    NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    RUN_LEGACY,
                                    random_hypernetwork_params,
                                    change=i)
    print()

    # Run varying max number of dimensions experiment
    print("Experiment - Max Number of Dimensions:")
    print(f"{f'Max Num Dim':<15}{f'Sparse time':<15}{f'SQL Einsum time':<18}{f'Torch time':<15}{f'Sparse Einsum time':<21}{f'Legacy Sparse Einsum time':<25}")
    for i in number_of_dims:
        random_hypernetwork_params = {
            "number_of_tensors": 6,
            "regularity": 3.0,
            "max_tensor_order": i,
            "max_edge_order": 3,
            "number_of_output_indices": 0,
            "number_of_single_summation_indices": 15,
            "min_axis_size": 2,
            "max_axis_size": 15,
            "return_size_dict": True
        }

        run_hypernetwork_experiment(ITERATIONS_PER_NETWORK,
                                    NUMBER_OF_RUNS,
                                    RUN_SPARSE,
                                    RUN_SQL_EINSUM,
                                    RUN_TORCH,
                                    RUN_LEGACY,
                                    random_hypernetwork_params,
                                    change=i)
    print()

########    RESULTS     ########
"""
Experiment - Instance:
Instance Name                 Sparse time    SQL Einsum time   Torch time     Sparse Einsum time   Legacy Sparse Einsum time
mc_2021_027                   0.613 it/s     0.632 it/s        3.258 it/s     10.052 it/s          7.339 it/s
mc_2021_036                   ----- it/s     ----- it/s        0.009 it/s     0.074 it/s           0.056 it/
mc_2022_087                   0.005 it/s     ----- it/s        0.056 it/s     0.167 it/s           0.154 it/s


Experiment - Number of Tensors:
Tensors        Sparse time    SQL Einsum time   Torch time     Sparse Einsum time   Legacy Sparse Einsum time
10             476.545 it/s   2284.108 it/s     404.718 it/s   492.369 it/s         266.300 it/s
12             263.609 it/s   1294.497 it/s     236.199 it/s   382.741 it/s         214.103 it/s
14             213.613 it/s   731.881 it/s      195.213 it/s   326.109 it/s         179.091 it/s
16             89.564 it/s    1061.477 it/s     167.394 it/s   268.279 it/s         151.105 it/s
18             74.276 it/s    548.178 it/s      142.753 it/s   249.704 it/s         143.827 it/s
20             76.383 it/s    772.391 it/s      171.697 it/s   257.161 it/s         145.156 it/s
22             61.785 it/s    597.661 it/s      116.004 it/s   213.647 it/s         120.983 it/s
24             53.520 it/s    533.237 it/s      135.279 it/s   200.779 it/s         122.016 it/s
26             41.633 it/s    669.754 it/s      91.784 it/s    210.283 it/s         114.738 it/s
28             7.829 it/s     344.313 it/s      69.143 it/s    173.847 it/s         94.909 it/s
30             4.485 it/s     569.061 it/s      61.248 it/s    171.929 it/s         94.717 it/s


Experiment - Density:
Density        Sparse time    SQL Einsum time   Torch time     Sparse_Einsum time   Sparse Einsum Legacy Time
0.1            103.350 it/s   0.046 it/s        67.884 it/s    7.789 it/s           8.646 it/s      
0.01           101.109 it/s   0.929     it/s    70.186 it/s    40.1406 it/s         35.231  it/s
0.001          108.362 it/s   108.230   it/s    68.328 it/s    156.571 it/s         174.506 it/s
0.0001         106.939 it/s   3952.256  it/s    59.668 it/s    560.104 it/s         298.441 it/s
1e-05          104.600 it/s   23169.600 it/s    67.295 it/s    706.623 it/s         385.398 it/s


Experiment - Max Dim Size:
Max dim size   Sparse time    SQL Einsum time   Torch time     Sparse Einsum time   Legacy Sparse Einsum time
2              2949.141 it/s  2954.971 it/s     1438.503 it/s  1221.739 it/s        558.216 it/s
3              2728.197 it/s  2861.648 it/s     1422.954 it/s  1165.713 it/s        511.409 it/s
4              2514.223 it/s  6282.668 it/s     1385.463 it/s  1349.421 it/s        584.589 it/s
5              2130.835 it/s  7085.857 it/s     1145.090 it/s  1298.878 it/s        594.116 it/s
6              1600.128 it/s  7246.113 it/s     921.999 it/s   1199.998 it/s        536.521 it/s
7              1193.220 it/s  4068.557 it/s     880.472 it/s   1027.373 it/s        531.115 it/s
8              833.779 it/s   3514.551 it/s     705.581 it/s   924.336 it/s         454.884 it/s
9              647.050 it/s   2631.675 it/s     600.319 it/s   758.730 it/s         409.967 it/s
10             419.688 it/s   1632.327 it/s     383.697 it/s   625.246 it/s         346.686 it/s
11             286.894 it/s   765.026 it/s      298.314 it/s   515.746 it/s         300.110 it/s
12             190.042 it/s   482.367 it/s      194.233 it/s   426.570 it/s         270.774 it/s
13             132.532 it/s   236.127 it/s      129.782 it/s   341.911 it/s         234.450 it/s
14             92.850 it/s    184.713 it/s      96.937 it/s    270.924 it/s         202.283 it/s
15             67.926 it/s    113.928 it/s      69.999 it/s    217.710 it/s         164.314 it/s


Experiment - Max Number of Dimensions:
Max Num Dim    Sparse time    SQL Einsum time   Torch time     Sparse Einsum time   Legacy Sparse Einsum time
6              477.020 it/s   1136.583 it/s     361.691 it/s   502.425 it/s         327.207 it/s
7              146.725 it/s   250.692 it/s      127.232 it/s   317.017 it/s         214.516 it/s
8              116.174 it/s   266.833 it/s      107.009 it/s   225.428 it/s         158.049 it/s
9              67.275 it/s    114.571 it/s      62.497 it/s    208.049 it/s         159.632 it/s
10             66.349 it/s    102.055 it/s      61.353 it/s    214.421 it/s         160.321 it/s
11             65.991 it/s    105.546 it/s      65.995 it/s    212.171 it/s         161.135 it/s
12             62.998 it/s    103.656 it/s      65.120 it/s    219.846 it/s         163.633 it/s
13             49.544 it/s    123.441 it/s      69.146 it/s    209.382 it/s         163.505 it/s
14             67.529 it/s    117.988 it/s      65.189 it/s    202.798 it/s         160.476 it/s

"""
