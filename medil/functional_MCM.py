"""Randomly sample from and generate functional MeDIL Causal Models."""
from numpy.random import default_rng
from .ecc_algorithms import find_clique_min_cover as find_cm
import numpy as np


def rand_biadj_mat(num_obs, edge_prob, rng=default_rng(0)):
    """ Generate random undirected graph over observed variables
    Parameters
    ----------
    num_obs: dimension of the observed space
    edge_prob: edge probability
    rng: type of random generator

    Returns
    -------
    biadj_mat: adjacency matrix of the directed graph
    """

    udg = np.zeros((num_obs, num_obs), bool)
    max_edges = (num_obs * (num_obs - 1)) // 2
    udg[np.triu_indices(num_obs, k=1)] = rng.choice(
        (True, False), max_edges, (edge_prob, 1 - edge_prob)
    )
    udg += udg.T
    np.fill_diagonal(udg, True)

    # find latent connections (minimum edge clique cover)
    biadj_mat = find_cm(udg)
    biadj_mat = biadj_mat.astype(bool)

    return biadj_mat


def sample_from_minMCM(minMCM, num_samps=1000, rng=default_rng(0)):
    """ Sample from the minMCM graph: minMCM should either be the binary bi-adjacency matrix or covariance matrix
    Parameters
    ----------
    minMCM: adjacency matrix of the minMCM
    num_samps
    rng

    Returns
    -------

    """

    if minMCM.dtype == bool:
        biadj_mat = minMCM

        # generate random weights in [-2, 2] \ {0}
        num_edges = biadj_mat.sum()
        num_latent, num_obs = biadj_mat.shape
        idcs = np.argwhere(biadj_mat)
        idcs[:, 1] += num_latent

        weights = (rng.random(num_edges) - 1) * 2
        weights[rng.choice((True, False), num_edges)] *= -1

        precision = np.eye(num_latent + num_obs, dtype=float)
        precision[idcs[:, 0], idcs[:, 1]] = weights
        precision = precision.dot(precision.T)

        cov = np.linalg.inv(precision)

    else:
        cov = minMCM

    samples = rng.multivariate_normal(np.zeros(len(cov)), cov, num_samps)

    return samples, minMCM
