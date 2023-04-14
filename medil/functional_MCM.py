"""Randomly sample from and generate functional MeDIL Causal Models."""
from .ecc_algorithms import find_heuristic_clique_cover as find_cm
from numpy.random import default_rng
import numpy as np
import warnings


def rand_biadj_mat(num_obs, edge_prob, rng=default_rng(0)):
    """Generate random undirected graph over observed variables
    Parameters
    ----------
    num_obs: dimension of the observed space
    edge_prob: edge probability
    rng: random generator

    Returns
    -------
    biadj_mat: biadjacency matrix of the directed graph, with entry (i,j) indicating an edge from latent variable L_i to measurement variable M_j
    """

    udg = np.zeros((num_obs, num_obs), bool)
    max_edges = (num_obs * (num_obs - 1)) // 2
    num_edges = np.round(edge_prob * max_edges).astype(int)
    edges = np.ones(max_edges)
    edges[num_edges:] = 0
    udg[np.triu_indices(num_obs, k=1)] = rng.permutation(edges)
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
    minMCM: covariance matrix over full MCM or boolean biadjacency matrix of MCM (for which random cov matrix will be generated)
    num_samps: number of samples
    rng: random generator

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
    """Assign degrees of freedom (latent variables) of VAE to latent factors from causal structure learning
    Parameters
    ----------
    biadj_mat: biadjacency matrix of MCM
    deg_of_freedom: desired size of latent space of VAE
    method: how to distribute excess degrees of freedom to latent causal factors
    variances: diag of covariance matrix over measurement variables

    Returns
    -------
    redundant_biadj_mat: biadjacency matrix specifing VAE structure from latent space to decoder
    """
    num_cliques = len(biadj_mat)
    if deg_of_freedom < num_cliques:
        warnings.warn(
            f"Input `deg_of_freedom={deg_of_freedom}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
        )
        deg_of_freedom = num_cliques

    if method == "uniform":
        latents_per_clique = np.ones(num_cliques, int) * (deg_of_freedom // num_cliques)
    elif method == "clique_size":
        latents_per_clique = np.round(
            (biadj_mat.sum(1) / biadj_mat.sum()) * (deg_of_freedom - num_cliques)
        ).astype(int)
    elif method == "tot_var" or method == "avg_var":
        clique_variances = biadj_mat @ variances
        if method == "avg_var":
            clique_variances /= biadj_mat.sum(1)
        clique_variances /= clique_variances.sum()
        latents_per_clique = np.round(
            clique_variances * (deg_of_freedom - num_cliques)
        ).astype(int)

    for _ in range(2):
        remainder = deg_of_freedom - latents_per_clique.sum()
        latents_per_clique[np.argsort(latents_per_clique)[0:remainder]] += 1

    redundant_biadj_mat = np.repeat(biadj_mat, latents_per_clique, axis=0)

    return redundant_biadj_mat
