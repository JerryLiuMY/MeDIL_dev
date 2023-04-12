"""Randomly sample from and generate functional MeDIL Causal Models."""
from .ecc_algorithms import find_heuristic_clique_cover as find_cm
from numpy.random import default_rng
import numpy as np


def rand_biadj_mat(num_obs, edge_prob, rng=default_rng(0)):
    """Generate random undirected graph over observed variables
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
        a=(True, False), size=max_edges, p=(edge_prob, 1 - edge_prob)
    )
    udg += udg.T
    np.fill_diagonal(udg, True)

    # find latent connections (minimum edge clique cover)
    biadj_mat = find_cm(udg)
    biadj_mat = biadj_mat.astype(bool)

    return biadj_mat


def sample_from_minMCM(minMCM, num_samps=1000, rng=default_rng(0)):
    """Sample from the minMCM graph: minMCM should either be the binary bi-adjacency matrix or covariance matrix
    Parameters
    ----------
    minMCM: adjacency matrix of the minMCM
    num_samps: number of samples
    rng: type of random generator

    Returns
    -------
    samples: samples
    cov: covariance matrix
    """

    if minMCM.dtype == bool:
        biadj_mat = minMCM

        # generate random weights in +-[0.5, 2]
        num_edges = biadj_mat.sum()
        num_latent, num_obs = biadj_mat.shape
        idcs = np.argwhere(biadj_mat)
        idcs[:, 1] += num_latent

        weights = (rng.random(num_edges) * 1.5) + 0.5
        weights[rng.choice((True, False), num_edges)] *= -1

        precision = np.eye(num_latent + num_obs, dtype=float)
        precision[idcs[:, 0], idcs[:, 1]] = weights
        precision = precision.dot(precision.T)

        cov = np.linalg.inv(precision)

    else:
        cov = minMCM

    samples = rng.multivariate_normal(np.zeros(len(cov)), cov, num_samps)

    return samples, cov


def assign_DoF(biadj_mat, deg_of_freedom, method="uniform", variances=None):
    num_cliques = len(biadj_mat)
    if deg_of_freedom < num_cliques:
        print(
            f"Input `deg_of_freedom={deg_of_freedom}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
        )
        deg_of_freedom = num_cliques

    if method == "uniform":
        latents_per_clique = np.ones(num_cliques, int) * (deg_of_freedom // num_cliques)
        remainder = deg_of_freedom % num_cliques
        latents_per_clique[0:remainder] += 1
    elif method == "clique_size":
        pass
    elif method == "tot_var" or method == "avg_var":
        clique_variances = biadj_mat @ variances
        clique_variances /= clique_variances.sum()
        num_extra = np.round(clique_variances * (num_latents - len(variances)))
        latent_per_clique = num_extra + 1
    elif method == "avg_var":
        clique_variances = (biadj_mat / biadj_mat.sum(1)) @ variances
        clique_variances /= clique_variances.sum()
        num_extra = np.round(clique_variances * (num_latents - len(variances)))
        latent_per_clique = num_extra + 1

    redundant_biadj_mat = np.repeat(biadj_mat, latents_per_clique, axis=0)

    return redundant_biadj_mat
