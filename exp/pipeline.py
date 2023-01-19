from medil.functional_MCM import rand_biadj_mat
from medil.functional_MCM import sample_from_minMCM
from exp.estimation import estimation
from learning.train import train_vae, valid_vae
from learning.params import params_dict
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np


def pipeline(num_obs, edge_prob, seed=0):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    num_obs: number of observed variables
    edge_prob: edge probability
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
    num_samps, batch_size = params_dict["num_samps"], params_dict["batch_size"]
    num_train, num_valid = params_dict["num_train"], params_dict["num_valid"]

    # create biadj_mat and samples
    np.random.seed(seed)
    biadj_mat = rand_biadj_mat(num_obs, edge_prob)
    samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)

    # learn MeDIL model
    num_latent = biadj_mat.shape[0]
    biadj_mat_learned, shd, ushd, num_latent_recon = estimation(biadj_mat, num_obs, num_latent, samples)
    m, n = biadj_mat_learned.shape

    # train generative models
    train_samples, _ = sample_from_minMCM(biadj_mat, num_samps=num_train)
    train_dataset = TensorDataset(torch.tensor(train_samples))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    model, train_loss = train_vae(m, n, train_loader, biadj_mat_learned)

    # perform validation on generative models
    valid_samples, _ = sample_from_minMCM(biadj_mat, num_samps=num_valid)
    valid_dataset = TensorDataset(torch.tensor(valid_samples))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    valid_loss = valid_vae(model, valid_loader)

    return valid_loss
