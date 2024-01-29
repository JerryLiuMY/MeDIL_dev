#!python3
import os
import warnings
from importlib.metadata import version
from datetime import datetime

import pycurl
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exp.pipeline import pipeline_real


def run_ncfa(data_path):
    # (dataset_name, num_samps_real, heuristic, method, alpha, dof, dof_method, exp_path, seed):
    """Run MeDIL on real dataset
    Parameters
    ----------
    dataset_name: name of dataset
    num_samps_real: number of samples for real dataset
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: alpha value
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    exp_path: path for the experiment
    seed: random seed for the experiments
    """
    raw_df = pd.read_csv(data_path, sep="\t ", header=None).T
    valid_df = raw_df.sample(frac=0.3)
    train_df = raw_df.drop(valid_df)

    sc = StandardScaler()

    dataset_train = pd.DataFrame(
        sc.fit_transform(train_df), train_df.index, train_df.columns
    ).values
    dataset_valid = pd.DataFrame(
        sc.fit_transform(valid_df), valid_df.index, valid_df.columns
    ).values

    dataset_key = "factor_analysis"

    dataset_train_sub = dataset_train[:, dataset_key]
    dataset_valid_sub = dataset_valid[:, dataset_key]
    dataset = [dataset_train_sub, dataset_valid_sub]

    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on real data with "
        f"num_samps={len(train_df)}"
    )
    pipeline_real(
        dataset, heuristic, method, alpha, dof, dof_method, exp_path, seed=seed
    )


# script logic for CLI
if __name__ == "__main__":
    # check versions to ensure accurate reproduction
    if version("medil") != "0.7.0":
        warnings.warn(f"Current `medil` version unsupported.")

    # download data unless it's already been downloaded
    path = "reproduce_ncfa_results/"
    data_path = path + "dataset.txt"
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading data set...")
        url = "http://www2.stat.duke.edu/~mw/mwsoftware/BFRM/Example1/dataset.txt"
        with open(data_path, "wb") as f:
            c = pycurl.Curl()
            c.setopt(c.URL, url)
            c.setopt(c.WRITEDATA, f)
            c.perform()
            c.close()

    run_ncfa(data_path)
