from datetime import datetime
from exp.pipeline import pipeline_graph
from exp.pipeline import pipeline_real
from exp.examples import fixed_biadj_mat_list, conversion_dict
from exp.examples import rand_biadj_mat_list, tcga_key_list, tcga_subsize
from sklearn.preprocessing import StandardScaler
from gloabl_settings import DATA_PATH
import pandas as pd
import os


def run_fixed(linspace, alphas, exp_path, seed):
    """ Run MeDIL on the fixed graphs
    Parameters
    ----------
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    for idx, biadj_mat in enumerate(fixed_biadj_mat_list):
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
                    pipeline_graph(biadj_mat, num_samps, alpha, folder_path, seed=seed)


def run_random(linspace, alphas, exp_path, seed):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    for idx, biadj_mat in enumerate(rand_biadj_mat_list):
        graph_path = os.path.join(exp_path, f"Graph_{idx}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {idx} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                    pipeline_graph(biadj_mat, num_samps, alpha, folder_path, seed=seed)


def run_real(dataset_name, linspace, alphas, exp_path, seed):
    """ Run MeDIL on real dataset
    Parameters
    ----------
    dataset_name: name of dataset
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    dataset_path = os.path.join(DATA_PATH, "dataset")
    sc = StandardScaler()
    dataset_train = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_train.csv"))
    dataset_valid = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_valid.csv"))
    dataset_train = pd.DataFrame(sc.fit_transform(dataset_train), dataset_train.index, dataset_train.columns).values
    dataset_valid = pd.DataFrame(sc.fit_transform(dataset_valid), dataset_valid.index, dataset_valid.columns).values

    if dataset_name == "tcga":
        dataset_key_list = tcga_key_list
    else:
        raise ValueError("Invalid dataset name")

    for idx, dataset_key in enumerate(dataset_key_list):
        graph_path = os.path.join(exp_path, f"Real_{idx}")
        dataset_train_sub = dataset_train[:, dataset_key]
        dataset_valid_sub = dataset_valid[:, dataset_key]
        dataset = [dataset_train_sub, dataset_valid_sub]

        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)
        for num_samps in linspace:
            for alpha in alphas:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on real data {idx} with "
                      f"num_samps={num_samps} and alpha={alpha}")
                folder_name = f"num_samps={num_samps}_alpha={alpha}"
                folder_path = os.path.join(graph_path, folder_name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                    pipeline_real(dataset, alpha, folder_path, seed=seed)


def run_real_full(dataset_name, linspace, alphas, exp_path, seed):
    """ Run MeDIL on real dataset
    Parameters
    ----------
    dataset_name: name of dataset
    linspace: linspace for the number of samples
    alphas: list of alphas
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    dataset_path = os.path.join(DATA_PATH, "dataset")
    sc = StandardScaler()
    dataset_train = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_train.csv"))
    dataset_valid = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_valid.csv"))
    dataset_train = pd.DataFrame(sc.fit_transform(dataset_train), dataset_train.index, dataset_train.columns).values
    dataset_valid = pd.DataFrame(sc.fit_transform(dataset_valid), dataset_valid.index, dataset_valid.columns).values

    graph_path = os.path.join(exp_path, f"Real_Full")
    dataset_train = dataset_train[:, :tcga_subsize]
    dataset_valid = dataset_valid[:, :tcga_subsize]
    dataset = [dataset_train, dataset_valid]

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    for num_samps in linspace:
        for alpha in alphas:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on real data full with "
                  f"num_samps={num_samps} and alpha={alpha}")
            folder_name = f"num_samps={num_samps}_alpha={alpha}"
            folder_path = os.path.join(graph_path, folder_name)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                pipeline_real(dataset, alpha, folder_path, seed=seed)
