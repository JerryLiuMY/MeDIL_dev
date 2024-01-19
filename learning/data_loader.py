from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from medil.functional_MCM import sample_from_minMCM
from rpy2.robjects.packages import STAP
from graph_est.utils import biadj_to_adj
import numpy as np
import torch
from rpy2.robjects import numpy2ri
numpy2ri.activate()


def load_dataset(samples, num_latent, batch_size):
    """ Generate dataset given the adjacency matrix, number of samples and batch size
    Parameters
    ----------
    samples: samples from the MCM
    num_latent: number of latent variables
    batch_size: batch size

    Returns
    -------
    data_loader: data loader
    """

    samples_x = samples[:, num_latent:].astype(np.float32)
    samples_z = samples[:, :num_latent].astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x), torch.tensor(samples_z))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def load_dataset_real(samples, batch_size):
    """ Generate dataset given the adjacency matrix, number of samples and batch size
    Parameters
    ----------
    samples: samples from the MCM
    batch_size: batch size

    Returns
    -------
    data_loader: data loader
    """

    samples_x = samples.astype(np.float32)
    samples_z = np.empty(shape=(samples_x.shape[0], 0)).astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x), torch.tensor(samples_z))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def sample_from_graph(biadj_mat, num_samps, data_type):
    """
    Parameters
    ----------
    biadj_mat: adjacency matrix
    num_samps: number of samples
    data_type: type of the data generated

    Returns
    -------

    """

    if data_type == "ordinary":
        samples, _ = sample_from_minMCM(biadj_mat, num_samps=num_samps)
    elif data_type == "GP":
        with open("learning/simulateGP.R", "r") as f:
            string = f.read()
        sampleDataFromG = STAP(string, "sampleDataFromG")
        adj_mat = biadj_to_adj(biadj_mat)
        samples = sampleDataFromG.sampleDataFromG(num_samps, adj_mat)
    else:
        raise ValueError("Invalid data type")

    return samples
