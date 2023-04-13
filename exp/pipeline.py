from exp.run_funcs import run_vae_oracle, run_vae_suite
from medil.functional_MCM import sample_from_minMCM
from learning.data_loader import load_dataset, load_dataset_real
from graph_est.estimation import estimation
from learning.params import params_dict
from datetime import datetime
import numpy as np
import pickle
import time
import os


def pipeline_graph(biadj_mat, num_samps, heuristic, method, alpha, path, seed):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: adjacency matrix of the bipartite graph
    num_samps: number of samples
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: significance level
    path: path for saving the files
    seed: random seed for the experiments
    """

    # load parameters
    np.random.seed(seed)
    batch_size, num_valid = params_dict["batch_size"], params_dict["num_valid"]

    # create biadj_mat and samples
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sampling from biadj_mat")
    time.sleep(1)
    samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)

    # learn MeDIL model and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    num_latent = biadj_mat.shape[0]
    ud_graph, biadj_mat_recon = estimation(samples, heuristic=heuristic, method=method, alpha=alpha)
    info = {"heuristic": heuristic, "method": method, "alpha": alpha}
    np.save(os.path.join(path, "biadj_mat.npy"), biadj_mat)
    np.save(os.path.join(path, "biadj_mat_recon.npy"), biadj_mat_recon)
    with open("info.pkl", "wb") as f:
        pickle.dump(info, f)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset(samples, num_latent, batch_size)
    cov_train = cov[num_latent:, num_latent:]

    valid_samples, cov_valid = sample_from_minMCM(biadj_mat, num_samps=num_valid)
    valid_loader = load_dataset(valid_samples, num_latent, batch_size)
    cov_valid = cov_valid[num_latent:, num_latent:]

    # perform vae training
    run_vae_oracle(biadj_mat, train_loader, valid_loader, cov_train, cov_valid, path, seed)
    run_vae_suite(biadj_mat_recon, train_loader, valid_loader, cov_train, cov_valid, path, seed)


def pipeline(dataset, heuristic, method, alpha, path, seed):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    dataset: dataset for real experiments
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: significance level
    path: path for saving the files
    seed: random seed
    """

    # load parameters
    np.random.seed(seed)
    batch_size = params_dict["batch_size"]
    samples, valid_samples = dataset

    # learn MeDIL model and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    ud_graph, biadj_mat_recon = estimation(samples, heuristic=heuristic, method=method, alpha=alpha)
    info = {"heuristic": heuristic, "method": method, "alpha": alpha}
    np.save(os.path.join(path, "ud_graph.npy"), ud_graph)
    np.save(os.path.join(path, "biadj_mat_recon.npy"), biadj_mat_recon)
    with open("info.pkl", "wb") as f:
        pickle.dump(info, f)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset_real(samples, batch_size)
    valid_loader = load_dataset_real(valid_samples, batch_size)

    # perform vae training
    cov_train, cov_valid = np.eye(samples.shape[1]), np.eye(samples.shape[1])
    run_vae_suite(biadj_mat_recon, train_loader, valid_loader, cov_train, cov_valid, path, seed)
