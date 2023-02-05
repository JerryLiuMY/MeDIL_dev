from exp.examples import biadj_mat_list
from exp.examples import conversion_dict
from exp.pipeline import pipeline
from datetime import datetime
import numpy as np
import os


def run_defined(linspace, alphas, exp_path):
    """ Run MeDIL on the define graphs
    Parameters
    ----------
    linspace: linspace for the
    alphas: list of alphas
    exp_path: path for the experiment
    """

    for idx, biadj_mat in enumerate(biadj_mat_list):
        graph_idx = conversion_dict[idx]
        graph_path = os.path.join(exp_path, f"Graph_{graph_idx}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {graph_idx} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                pipeline(biadj_mat, num_samps, alpha, folder_path, seed=0)


def run_random(linspace, alphas, exp_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    linspace: linspace for the
    alphas: list of alphas
    exp_path: path for the experiment
    """

    for i in range(10):

        graph_path = os.path.join(exp_path, f"Graph_{i}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {graph_idx} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                pipeline(biadj_mat, num_samps, alpha, folder_path, seed=0)

    pass


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
        run_defined(linspace, alphas, exp_path)
        run_random(linspace, alphas, exp_path)


if __name__ == "__main__":
    linspace = np.exp(np.linspace(0, np.log(2500), 10))
    linspace = np.array(sorted(set(np.round(linspace)))).astype(int)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    parent_path = "../data/experiments"
    main(linspace, alphas, parent_path)
