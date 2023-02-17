from graph_est.utils import expand_recon, contract_recon, permute_graph, shd_func
from medil.ecc_algorithms import find_clique_min_cover
from medil.ecc_algorithms import find_heuristic_clique_cover
from medil.independence_testing import estimate_UDG
from itertools import permutations, combinations
import numpy as np


def estimation(biadj_mat, num_obs, num_latent, samples, heuristic, alpha, seed):
    """ Perform estimations of the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: input directed graph
    num_obs: number of observed variables
    num_latent: number of latent variables
    samples: number of samples
    heuristic: whether to use the heuristic solver
    alpha: significance level
    seed: random seed for the experiments

    Returns
    -------
    biadj_mat_learned: learned directed graph in the form of adjacency matrix
    shd: structural hamming distance (directed graph)
    ushd: structural hamming distance (undirected graph)
    num_latent_recon: number of reconstructed latent variables
    """

    # step 1: estimate UDG
    np.random.seed(seed)
    samples_in, samples_out = samples[:, :num_latent], samples[:, num_latent:]
    ud_graph = estimate_UDG(samples_out, method="dcov_fast", significance_level=alpha)
    np.fill_diagonal(ud_graph, val=True)

    # step 2: learn graphical MCM
    if heuristic:
        biadj_mat_recon = find_heuristic_clique_cover(ud_graph)
    else:
        biadj_mat_recon = find_clique_min_cover(ud_graph)
    num_latent_recon = biadj_mat_recon.shape[0]

    # step 3: change the matrix to int
    biadj_mat = biadj_mat.astype(int)
    biadj_mat_recon = biadj_mat_recon.astype(int)

    # learned graphs with permutations taken into consideration
    if num_latent < num_latent_recon:
        biadj_mat_recon_list = [
            contract_recon(biadj_mat_recon, comb) for comb in combinations(np.arange(num_latent_recon), num_latent)
        ]
        shd_learned_list = [find_learned(biadj_mat, biadj_mat_recon) for biadj_mat_recon in biadj_mat_recon_list]
        shd_list = [_[0] for _ in shd_learned_list]
        ushd_list = [_[1] for _ in shd_learned_list]
        learned_list = [_[2] for _ in shd_learned_list]
        idx = np.argmin(shd_list)
        shd, ushd, biadj_mat_learned = shd_list[idx], ushd_list[idx], learned_list[idx]
    elif num_latent > num_latent_recon:
        biadj_mat_recon = expand_recon(biadj_mat_recon, num_obs, num_latent)
        shd, ushd, biadj_mat_learned = find_learned(biadj_mat, biadj_mat_recon)
    else:
        shd, ushd, biadj_mat_learned = find_learned(biadj_mat, biadj_mat_recon)

    return biadj_mat_learned, shd, ushd, num_latent_recon


def estimation_real(samples_out, heuristic=False, alpha=0.05):
    """ Perform estimations of the shd and number of reconstructed latent
    Parameters
    ----------
    samples_out: output samples
    heuristic: whether to use the heuristic solver
    alpha: significance level
    """

    # step 1: estimate UDG
    ud_graph = estimate_UDG(samples_out, method="dcov_fast", significance_level=alpha)
    np.fill_diagonal(ud_graph, val=True)

    # step 2: learn graphical MCM
    if heuristic:
        biadj_mat_recon = find_heuristic_clique_cover(ud_graph)
    else:
        biadj_mat_recon = find_clique_min_cover(ud_graph)

    return biadj_mat_recon


def find_learned(biadj_mat, biadj_mat_recon):
    """ Find the learned directed graph that minimizes the SHD
    Parameters
    ----------
    biadj_mat: original graph
    biadj_mat_recon: reconstructed graph

    Returns
    -------
    shd: minimal structural hamming distance for all permutations (directed graph)
    ushd: minimal structural hamming distance for all permutations (undirected graph)
    biadj_mat_learned: learned graph that minimizes the SHD
    """

    # find the number of latent variables and shd
    num_latent = biadj_mat.shape[0]
    shd_perm_list = [(shd_func(biadj_mat, permute_graph(biadj_mat_recon, perm)), perm) for perm
                     in permutations(np.arange(num_latent))]

    shd_list, perm_list = [_[0] for _ in shd_perm_list], [_[1] for _ in shd_perm_list]
    idx = np.argmin(shd_list)
    shd = shd_list[idx]

    # find the directed graph recovered
    biadj_mat_learned = biadj_mat_recon[perm_list[idx], :]

    # find the undirected graph recovered
    ug_mat = recover_ug(biadj_mat)
    ug_mat_recon = recover_ug(biadj_mat_learned)
    ushd = shd_func(ug_mat, ug_mat_recon)

    return shd, ushd, biadj_mat_learned


def recover_ug(biadj_mat):
    """ Recover the undirected graph from the directed graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """

    # get the undirected graph from the directed graph
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, 0.)

    return ug
