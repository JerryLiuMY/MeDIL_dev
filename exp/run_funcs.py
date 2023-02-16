from learning.train import train_vae
import numpy as np
import pickle
import os


def run_vae_oracle(biadj_mat, train_loader, valid_loader, cov_train, cov_valid, path, seed):
    """ Run training loop for oracle VAE
    Parameters
    ----------
    biadj_mat: ground truth adjacency matrix
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    seed: random seed for the experiments
    """

    # train & validate oracle VAE
    m, n = biadj_mat.shape
    loss_true, error_true = train_vae(m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid, seed)
    with open(os.path.join(path, "loss_true.pkl"), "wb") as handle:
        pickle.dump(loss_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_true.pkl"), "wb") as handle:
        pickle.dump(error_true, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_vae_hrstc(biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, path, seed):
    """ Run training loop for exact VAE
    Parameters
    ----------
    biadj_mat_hrstc: adjacency matrix for heuristic graph
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    seed: random seed for the experiments
    """

    # train & validate heuristic MeDIL VAE
    mh, nh = biadj_mat_hrstc.shape
    loss_hrstc, error_hrstc = train_vae(
        mh, nh, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    with open(os.path.join(path, "loss_hrstc.pkl"), "wb") as handle:
        pickle.dump(loss_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_hrstc.pkl"), "wb") as handle:
        pickle.dump(error_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_vae_suite(biadj_mat_exact, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, path, seed):
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
    seed: random seed for the experiments
    """

    # train & validate exact MeDIL VAE
    me, ne = biadj_mat_exact.shape
    loss_exact, error_exact = train_vae(
        me, ne, biadj_mat_exact, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    with open(os.path.join(path, "loss_exact.pkl"), "wb") as handle:
        pickle.dump(loss_exact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_exact.pkl"), "wb") as handle:
        pickle.dump(error_exact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate heuristic MeDIL VAE
    mh, nh = biadj_mat_hrstc.shape
    loss_hrstc, error_hrstc = train_vae(
        mh, nh, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    with open(os.path.join(path, "loss_hrstc.pkl"), "wb") as handle:
        pickle.dump(loss_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_hrstc.pkl"), "wb") as handle:
        pickle.dump(error_hrstc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate vanilla VAE
    biadj_mat_vanilla = np.ones((me, ne))
    loss_vanilla, error_vanilla = train_vae(
        me, ne, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    with open(os.path.join(path, "loss_vanilla.pkl"), "wb") as handle:
        pickle.dump(loss_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_vanilla.pkl"), "wb") as handle:
        pickle.dump(error_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
