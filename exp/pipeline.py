from medil.functional_MCM import sample_from_minMCM
from learning.data_loader import load_dataset, load_dataset_real
from graph_est.estimation import estimation, estimation_real
from learning.params import params_dict
from learning.train import train_vae
from datetime import datetime
import pickle
import numpy as np
import time
import os


def pipeline_graph(biadj_mat, num_samps, alpha, path, seed=0):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: adjacency matrix of the bipartite graph
    num_samps: number of samples used for adjacency matrix
    alpha: significance level
    path: path for saving the files
    seed: random seed
    """

    # load parameters
    np.random.seed(seed)
    batch_size, num_valid = params_dict["batch_size"], params_dict["num_valid"]

    # create biadj_mat and samples
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sampling from biadj_mat")
    time.sleep(1)
    dim_obs = biadj_mat.shape[1]
    samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)

    # learn MeDIL model and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    num_latent = biadj_mat.shape[0]
    biadj_mat_exact, _, _, _ = estimation(biadj_mat, dim_obs, num_latent, samples, heuristic=False, alpha=alpha)
    biadj_mat_hrstc, _, _, _ = estimation(biadj_mat, dim_obs, num_latent, samples, heuristic=True, alpha=alpha)
    np.save(os.path.join(path, "biadj_mat.npy"), biadj_mat)
    np.save(os.path.join(path, "biadj_mat_exact.npy"), biadj_mat_exact)
    np.save(os.path.join(path, "biadj_mat_hrstc.npy"), biadj_mat_hrstc)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset(samples, num_latent, batch_size)
    cov_train = cov[num_latent:, num_latent:]

    valid_samples, cov_valid = sample_from_minMCM(biadj_mat, num_samps=num_valid)
    valid_loader = load_dataset(valid_samples, num_latent, batch_size)
    cov_valid = cov_valid[num_latent:, num_latent:]

    # perform vae training
    run_vae_oracle(biadj_mat, train_loader, valid_loader, cov_train, cov_valid, path)
    run_vae_suite(biadj_mat_exact, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, path)


def pipeline_real(dataset, alpha, path, seed=0):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    dataset: dataset for real experiments
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

    biadj_mat_exact = estimation_real(samples, heuristic=False, alpha=alpha)
    biadj_mat_hrstc = estimation_real(samples, heuristic=True, alpha=alpha)
    np.save(os.path.join(path, "biadj_mat_exact.npy"), biadj_mat_exact)
    np.save(os.path.join(path, "biadj_mat_hrstc.npy"), biadj_mat_hrstc)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset_real(samples, batch_size)
    valid_loader = load_dataset_real(valid_samples, batch_size)

    cov_train, cov_valid = np.eye(samples.shape[1]), np.eye(samples.shape[1])
    run_vae_suite(biadj_mat_exact, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, path)


def run_vae_oracle(biadj_mat, train_loader, valid_loader, cov_train, cov_valid, path):
    """ Run training loop for oracle VAE
    Parameters
    ----------
    biadj_mat: ground truth adjacency matrix
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    """

    # train & validate oracle VAE
    m, n = biadj_mat.shape
    loss_true, error_true = train_vae(m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid)
    with open(os.path.join(path, "loss_true.pkl"), "wb") as handle:
        pickle.dump(loss_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_true.pkl"), "wb") as handle:
        pickle.dump(error_true, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_vae_suite(biadj_mat_exact, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, path):
    """ Run training loop for exact VAE
    Parameters
    ----------
    biadj_mat_exact: adjacency matrix for exact graph
    biadj_mat_hrstc: adjacency matrix for heuristic graph
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    """

    # train & validate exact MeDIL VAE
    me, ne = biadj_mat_exact.shape
    loss_exact, error_exact = train_vae(me, ne, biadj_mat_exact, train_loader, valid_loader, cov_train, cov_valid)
    with open(os.path.join(path, "loss_exact.pkl"), "wb") as handle:
        pickle.dump(loss_exact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_exact.pkl"), "wb") as handle:
        pickle.dump(error_exact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate heuristic MeDIL VAE
    mh, nh = biadj_mat_hrstc.shape
    loss_hrstc, error_hrstc = train_vae(mh, nh, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid)
    with open(os.path.join(path, "loss_hrstc.pkl"), "wb") as handle:
        pickle.dump(loss_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_hrstc.pkl"), "wb") as handle:
        pickle.dump(error_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate vanilla VAE
    biadj_mat_vanilla = np.ones((me, ne))
    loss_vanilla, error_vanilla = train_vae(me, ne, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid)
    with open(os.path.join(path, "loss_vanilla.pkl"), "wb") as handle:
        pickle.dump(loss_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_vanilla.pkl"), "wb") as handle:
        pickle.dump(error_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
