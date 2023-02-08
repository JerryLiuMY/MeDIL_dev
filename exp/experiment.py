from datetime import datetime
from exp.pipeline import pipeline
from exp.examples import biadj_mat_list, conversion_dict
from medil.functional_MCM import rand_biadj_mat
import os


def run_defined(linspace, alphas, exp_path):
    """ Run MeDIL on the define graphs
    Parameters
    ----------
    linspace: linspace for the number of samples
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


def run_random(num_obs, edge_prob, linspace, alphas, exp_path):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    num_obs: number of observations
    edge_prob: edge probability
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    """

    for i in range(10):
        biadj_mat = rand_biadj_mat(num_obs=num_obs, edge_prob=edge_prob)
        graph_path = os.path.join(exp_path, f"Graph_{i}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {i} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                    pipeline(biadj_mat, num_samps, alpha, folder_path, seed=0)


def run_real(linspace, alphas, exp_path):
    """ Run MeDIL on real dataset
    Parameters
    ----------
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    """

    for i in range(10):
        graph_path = os.path.join(exp_path, f"Real_{i}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on real data {i} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
