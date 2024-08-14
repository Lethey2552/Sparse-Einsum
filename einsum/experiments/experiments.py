from instance_experiment import run_instance_experiment
from hypernetwork_experiment import run_hypernetwork_experiment

if __name__ == "__main__":
    NUMBER_OF_RUNS = 10
    RUN_SPARSE = True
    RUN_SQL_EINSUM = True
    RUN_TORCH = True
    INSTANCE_NAME = "mc_2021_027"

    # run_instance_experiment(INSTANCE_NAME, NUMBER_OF_RUNS,
    #                         RUN_SPARSE, RUN_SQL_EINSUM)
    run_hypernetwork_experiment(NUMBER_OF_RUNS,
                                RUN_SPARSE, RUN_SQL_EINSUM, RUN_TORCH)

########    RESULTS     ########
"""
Instance        Sparse Time     Sparse Einsum Time
mc_2022_087     189.314s        5.929s
"""
