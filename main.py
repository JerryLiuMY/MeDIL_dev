from exp.experiment import run_fixed
from exp.experiment import run_random
from exp.experiment import run_real_full
from exp.experiment import run_real
from exp.examples import num_samps_graph
from exp.examples import num_samps_real
import os


def main(dataset_name, parent_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    dataset_name: name of the dataset
    parent_path: parent path for the experiments
    """

    # argument for estimation
    alpha = 0.05
    heuristic = True
    method = "xicor"

    # argument for architecture
    dof = None
    dof_method = "uniform"

    run = 0
    # real dataset
    exp_path = os.path.join(parent_path, f"experiment_{run}")
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    # fixed and random dataset
    run_fixed(num_samps_graph, heuristic, method, alpha, dof, dof_method, exp_path, seed=run)
    run_random(num_samps_graph, heuristic, method, alpha, dof, dof_method, exp_path, seed=run)
    # run_real(dataset_name, num_samps_real, heuristic, method, alpha, dof, dof_method, exp_path, seed=run)
    # run_real_full(dataset_name, num_samps_real, heuristic, method, alpha, dof, dof_method, exp_path, seed=run)


if __name__ == "__main__":
    parent_path = "../data/experiments"
    dataset_name = "gene"
    main(dataset_name, parent_path)
