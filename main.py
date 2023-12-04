from exp.experiment import run_fixed
from exp.experiment import run_random
from exp.experiment import run_real
from exp.examples import num_samps_graph
from exp.examples import num_samps_real
import os


def main_graph(parent_path, run_func, run):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    parent_path: parent path for the experiments
    run_func: running function
    run: run number of the experiment
    """

    # argument for estimation
    data_type = "ordinary"
    heuristic = True
    method = "dcov_fast"
    alpha = 0.05

    # argument for architecture
    dof = None
    dof_method = "uniform"

    # real dataset
    exp_path = os.path.join(parent_path, f"experiment_{run}")
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    # fixed and random dataset
    run_func(num_samps_graph, data_type, heuristic, method, alpha, dof, dof_method, exp_path, seed=run)


def main_fixed(parent_path, run):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    parent_path: parent path for the experiments
    run: run number of the experiment
    """

    run_func = run_fixed
    main_graph(parent_path, run_func, run)


def main_random(parent_path, run):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    parent_path: parent path for the experiments
    run: run number of the experiment
    """

    run_func = run_random
    main_graph(parent_path, run_func, run)


def main_real(dataset_name, parent_path):
    """ Run MeDIL on the real dataset
    Parameters
    ----------
    dataset_name: name of the dataset
    parent_path: parent path for the experiments
    """

    # argument for estimation
    alpha = 0.05
    heuristic = True
    method = "dcov_fast"

    # argument for architecture
    dof = 3560
    dof_method = "uniform"

    # real dataset
    exp_path = os.path.join(parent_path, f"{dataset_name}")
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    run_real(dataset_name, num_samps_real, heuristic, method, alpha, dof, dof_method, exp_path, seed=0)


# if __name__ == "__main__":
#     parent_path = "/Volumes/SanDisk_2T/MeDIL/data/dataset"
#     dataset_name = "tumors"
#     main_real(dataset_name, parent_path)


if __name__ == "__main__":
    parent_path = "/Volumes/SanDisk_2T/MeDIL/data/experiments"
    main_random(parent_path, run=0)
