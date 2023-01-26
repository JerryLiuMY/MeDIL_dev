from medil.functional_MCM import sample_from_minMCM
from exp.estimation import estimation
from learning.train import train_vae
from learning.params import params_dict
from learning.data_loader import load_dataset
from datetime import datetime
import numpy as np
import time


def pipeline(biadj_mat, seed=0):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: adjacency matrix of the bipartite graph
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
    num_samps, batch_size = params_dict["num_samps"], params_dict["batch_size"]
    num_train, num_valid = params_dict["num_train"], params_dict["num_valid"]

    # create biadj_mat and samples
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sampling from biadj_mat")
    time.sleep(1)
    dim_obs = biadj_mat.shape[1]
    samples, _ = sample_from_minMCM(biadj_mat, num_samps=num_samps)

    # learn MeDIL model
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    num_latent = biadj_mat.shape[0]
    biadj_mat_medil, _, _, _ = estimation(biadj_mat, dim_obs, num_latent, samples)

    # define training sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader, cov_train = load_dataset(biadj_mat, num_train, batch_size)
    valid_loader, cov_valid = load_dataset(biadj_mat, num_valid, batch_size)
    cov_train = cov_train[num_latent:, num_latent:]
    cov_valid = cov_valid[num_latent:, num_latent:]

    # train & validate MeDIL VAE generative models
    m, n = biadj_mat_medil.shape
    # medil_output = train_vae(m, n, biadj_mat_medil, train_loader, valid_loader, cov_train, cov_valid)
    medil_output = train_vae(m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid)
    model_medil, train_loss_medil, valid_loss_medil = medil_output
    loss_medil = [train_loss_medil, valid_loss_medil]

    # train & validate Vanilla VAE generative models
    biadj_mat_vanilla = np.ones((m, n))
    vanilla_output = train_vae(m, n, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid)
    model_vanilla, train_loss_vanilla, valid_loss_vanilla = vanilla_output
    loss_vanilla = [train_loss_vanilla, valid_loss_vanilla]

    return loss_medil, loss_vanilla, biadj_mat_medil
