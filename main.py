from exp.experiment import run_defined, run_random
import numpy as np
import os


def main(linspace, alphas, parent_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    linspace: linspace for the
    alphas: list of alphas
    parent_path: parent path for the experiments
    """

    for run in range(10):
        exp_path = os.path.join(parent_path, f"experiment_{run}")
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)
            run_defined(linspace, alphas, exp_path)
            run_random(8, 0.5, linspace, alphas, exp_path)


if __name__ == "__main__":
    linspace = np.exp(np.linspace(np.log(100), np.log(2500), 10))
    linspace = np.array(sorted(set(np.round(linspace)))).astype(int)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    parent_path = "../data/experiments"
    main(linspace, alphas, parent_path)
