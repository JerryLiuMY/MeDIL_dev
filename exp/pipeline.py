from medil.functional_MCM import sample_from_minMCM
from exp.estimation import estimation
from learning.train import train_vae
from learning.params import params_dict
from learning.data_loader import load_dataset
from datetime import datetime
import numpy as np
import time


def pipeline(biadj_mat, num_samps, alpha, path, seed=0):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: adjacency matrix of the bipartite graph
    num_samps: number of samples used for adjacency matrix
    alpha: significance level
    path: path for saving the files
    seed: random seed

    Returns
    -------
    num_obs: number of observed variables
    edge_prob: edge probability
    num_samps: number of samples
    num_latent: number of latent variables
    shd: structural hamming distance
    num_latent_recon: number of reconstructed latent variables
    """

    # load parameters
    np.random.seed(seed)
    batch_size = params_dict["batch_size"]
    num_train, num_valid = params_dict["num_train"], params_dict["num_valid"]

    # create biadj_mat and samples
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sampling from biadj_mat")
    time.sleep(1)
    dim_obs = biadj_mat.shape[1]
    samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)

    # learn MeDIL model
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    num_latent = biadj_mat.shape[0]
    biadj_mat_exact, _, _, _ = estimation(biadj_mat, dim_obs, num_latent, samples, heuristic=False)
    biadj_mat_hrstc, _, _, _ = estimation(biadj_mat, dim_obs, num_latent, samples, heuristic=True)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset(samples, num_latent, batch_size)
    cov_train = cov[num_latent:, num_latent:]

    valid_samples, cov_valid = sample_from_minMCM(biadj_mat, num_samps=num_valid)
    valid_loader = load_dataset(valid_samples, num_latent, batch_size)
    cov_valid = cov_valid[num_latent:, num_latent:]

    # train & validate oracle VAE
    m, n = biadj_mat.shape
    loss_true, error_true = train_vae(m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid)

    # train & validate Vanilla VAE
    biadj_mat_vanilla = np.ones((m, n))
    loss_vanilla, error_vanilla = train_vae(m, n, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid)

    # train & validate exact MeDIL VAE
    loss_exact, error_exact = train_vae(m, n, biadj_mat_exact, train_loader, valid_loader, cov_train, cov_valid)

    # train & validate heuristic MeDIL VAE
    loss_hrstc, error_hrstc = train_vae(m, n, biadj_mat_hrstc, train_loader, valid_loader, cov_train, cov_valid)
