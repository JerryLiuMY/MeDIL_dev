from exp.experiment import run_fixed, run_random
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

    run = 0
    exp_path = os.path.join(parent_path, f"experiment_{run}")
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
        # run_fixed(linspace, alphas, exp_path)
    run_random(linspace, alphas, exp_path)


if __name__ == "__main__":
    linspace = np.exp(np.linspace(np.log(100), np.log(2500), 10))
    linspace = np.array(sorted(set(np.round(linspace)))).astype(int)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    parent_path = "../data/experiments"
    main(linspace, alphas, parent_path)
