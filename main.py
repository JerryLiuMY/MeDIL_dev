from exp.experiment import run_fixed
from exp.experiment import run_random
from exp.experiment import run_real
import numpy as np
import os


def main(dataset_name, parent_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    dataset_name: name of the dataset
    parent_path: parent path for the experiments
    """

    run = 0
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    exp_path = os.path.join(parent_path, f"experiment_{run}")
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
        # fixed and random dataset

    # linspace = np.exp(np.linspace(np.log(100), np.log(2500), 10))
    # linspace = np.array(sorted(set(np.round(linspace)))).astype(int)
    # run_fixed(linspace, alphas, exp_path)
    # run_random(linspace, alphas, exp_path)

    # real dataset
    linspace = np.exp(np.linspace(np.log(100), np.log(632), 10))
    linspace = np.array(sorted(set(np.round(linspace)))).astype(int)
    run_real(dataset_name, linspace, alphas, exp_path)


if __name__ == "__main__":
    parent_path = "../data/experiments"
    dataset_name = "tcga"
    main(dataset_name, parent_path)
