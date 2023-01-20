from medil.functional_MCM import sample_from_minMCM
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def load_dataset(biadj_mat, num_samps, batch_size):
    """ Generate dataset given the adjacency matrix, number of samples and batch size
    Parameters
    ----------
    biadj_mat: adjacency matrix
    num_samps: number of samples
    batch_size: batch size

    Returns
    -------
    data_loader: data loader
    cov: covariance matrix
    """

    num_latent = biadj_mat.shape[0]
    samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)
    samples_x = samples[:, num_latent:].astype(np.float32)
    samples_z = samples[:, :num_latent].astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x), torch.tensor(samples_z))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, cov
