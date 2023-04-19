from exp.experiment import run_fixed
from exp.experiment import run_random
from exp.experiment import run_real_full
from exp.experiment import run_real
from exp.examples import linspace_graph
from exp.examples import linspace_real
import os


def main(dataset_name, parent_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    dataset_name: name of the dataset
    parent_path: parent path for the experiments
    """

    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    heuristic = True
    method = "dcov_fast"

    for run in range(10):
        # real dataset
        exp_path = os.path.join(parent_path, f"experiment_{run}")
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)

        # fixed and random dataset
        run_fixed(linspace_graph, heuristic, method, alphas, exp_path, seed=run)
        run_random(linspace_graph, heuristic, method, alphas, exp_path, seed=run)
        run_real(dataset_name, linspace_real, heuristic, method, alphas, exp_path, seed=run)
        run_real_full(dataset_name, linspace_real, heuristic, method, alphas, exp_path, seed=run)


if __name__ == "__main__":
    parent_path = "../data/experiments"
    dataset_name = "gene"
    main(dataset_name, parent_path)
