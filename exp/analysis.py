from gloabl_settings import DATA_PATH
from exp.examples import paths_list
from exp.examples import num_samps_graph
from exp.examples import num_samps_real
from graph_est.utils import expand_recon, contract_recon
from graph_est.utils import permute_graph, shd_func
from itertools import permutations, combinations
import seaborn as sns
import pandas as pd
import numpy as np
import os
sns.set()


def analysis(biadj_mat, biadj_mat_recon):
    """Perform analysis of the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: input directed graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix

    Returns
    -------
    shd: structural hamming distance (directed graph)
    ushd: structural hamming distance (undirected graph)
    """

    # change the matrix to int
    num_latent, num_obs = biadj_mat.shape
    num_latent_recon = biadj_mat_recon.shape[0]
    biadj_mat = biadj_mat.astype(int)
    biadj_mat_recon = biadj_mat_recon.astype(int)

    # learned graphs with permutations taken into consideration
    if num_latent < num_latent_recon:
        biadj_mat_recon_list = [
            contract_recon(biadj_mat_recon, comb)
            for comb in combinations(np.arange(num_latent_recon), num_latent)
        ]
        shd_learned_list = [
            find_learned(biadj_mat, biadj_mat_recon)
            for biadj_mat_recon in biadj_mat_recon_list
        ]
        shd_list = [_[0] for _ in shd_learned_list]
        ushd_list = [_[1] for _ in shd_learned_list]
        idx = np.argmin(shd_list)
        shd, ushd = shd_list[idx], ushd_list[idx]
    elif num_latent > num_latent_recon:
        biadj_mat_recon = expand_recon(biadj_mat_recon, num_obs, num_latent)
        shd, ushd = find_learned(biadj_mat, biadj_mat_recon)
    else:
        shd, ushd = find_learned(biadj_mat, biadj_mat_recon)

    return shd, ushd


def find_learned(biadj_mat, biadj_mat_recon):
    """Find the learned directed graph that minimizes the SHD
    Parameters
    ----------
    biadj_mat: original graph
    biadj_mat_recon: reconstructed graph

    Returns
    -------
    shd: minimal structural hamming distance for all permutations (directed graph)
    ushd: minimal structural hamming distance for all permutations (undirected graph)
    """

    # find the number of latent variables and shd
    num_latent = biadj_mat.shape[0]
    shd_perm_list = [
        (shd_func(biadj_mat, permute_graph(biadj_mat_recon, perm)), perm)
        for perm in permutations(np.arange(num_latent))
    ]

    shd_list, perm_list = [_[0] for _ in shd_perm_list], [_[1] for _ in shd_perm_list]
    idx = np.argmin(shd_list)
    shd = shd_list[idx]

    # find the directed graph recovered
    biadj_mat_learned = biadj_mat_recon[perm_list[idx], :]

    # find the undirected graph recovered
    ug_mat = recover_ug(biadj_mat)
    ug_mat_recon = recover_ug(biadj_mat_learned)
    ushd = shd_func(ug_mat, ug_mat_recon)

    return shd, ushd


def recover_ug(biadj_mat):
    """Recover the undirected graph from the directed graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """

    # get the undirected graph from the directed graph
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, 0.0)

    return ug


def build_table(alpha):
    """Build table for SHD, ELBO, and losses
    Parameters
    ----------
    alpha: alpha for the hypothesis test

    Returns
    -------
    table: table for summarizing the results
    """

    exp_path = os.path.join(DATA_PATH, "experiments")
    columns = [
        "loss_true_train",
        "loss_true_valid",
        "error_true_train",
        "error_true_valid",
        "loss_exact_train",
        "loss_exact_valid",
        "error_exact_train",
        "error_exact_valid",
        "loss_recon_train",
        "loss_recon_valid",
        "error_recon_train",
        "error_recon_valid",
        "loss_vanilla_train",
        "loss_vanilla_valid",
        "error_vanilla_train",
        "error_vanilla_valid",
        "shd_recon",
    ] + ["run"]

    table = pd.DataFrame(columns=columns)

    for idx in range(5):
        sub_table = pd.DataFrame(pd.NA, index=paths_list, columns=columns)

        for path in paths_list:
            graph_path = os.path.join(exp_path, f"experiment_{idx}", path)
            n = num_samps_graph if "Graph" in path else num_samps_real
            result_path = os.path.join(graph_path, f"num_samps={n}_alpha={alpha}")

            # hrstc graph
            loss_recon = pd.read_pickle(os.path.join(result_path, "loss_recon.pkl"))
            error_recon = pd.read_pickle(os.path.join(result_path, "error_recon.pkl"))
            sub_table.loc[path, "loss_recon_train"] = loss_recon[0][-1]
            sub_table.loc[path, "loss_recon_valid"] = loss_recon[1][-1]
            sub_table.loc[path, "error_recon_train"] = error_recon[0][-1]
            sub_table.loc[path, "error_recon_valid"] = error_recon[1][-1]

            # vanilla graph
            loss_vanilla = pd.read_pickle(os.path.join(result_path, "loss_vanilla.pkl"))
            error_vanilla = pd.read_pickle(
                os.path.join(result_path, "error_vanilla.pkl")
            )
            sub_table.loc[path, "loss_vanilla_train"] = loss_vanilla[0][-1]
            sub_table.loc[path, "loss_vanilla_valid"] = loss_vanilla[1][-1]
            sub_table.loc[path, "error_vanilla_train"] = error_vanilla[0][-1]
            sub_table.loc[path, "error_vanilla_valid"] = error_vanilla[1][-1]

            if "Graph" in path:
                # true graph
                loss_true = pd.read_pickle(os.path.join(result_path, "loss_true.pkl"))
                error_true = pd.read_pickle(os.path.join(result_path, "error_true.pkl"))
                sub_table.loc[path, "loss_true_train"] = loss_true[0][-1]
                sub_table.loc[path, "loss_true_valid"] = loss_true[1][-1]
                sub_table.loc[path, "error_true_train"] = error_true[0][-1]
                sub_table.loc[path, "error_true_valid"] = error_true[1][-1]

                # SHD for reconstruction
                biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
                biadj_mat_recon = np.load(
                    os.path.join(result_path, "biadj_mat_recon.npy")
                )
                shd, ushd = analysis(biadj_mat, biadj_mat_recon)
                sub_table.loc[path, "shd_recon"] = shd

                # comparing SHD(true, exact), SHD(true, heur), SHD(exact, huer)
                # so maybe "biadj_mat_exact.npy" and "biadj_mat_heur.npy" to distinguish the two

            sub_table["run"] = idx

        table = pd.concat([table, sub_table])

    return table
