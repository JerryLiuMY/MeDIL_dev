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
    """Perform analysis of the distances between true and reconstructed structures
    Parameters
    ----------
    biadj_mat: input directed graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix

    Returns
    -------
    sfd: squared Frobenius distance (bipartite graph)
    ushd: structural hamming distance (undirected graph)
    """

    # ushd = shd_func(recover_ug(biadj_mat), recover_ug(biadj_mat_recon))
    ug = recover_ug(biadj_mat)
    ug_recon = recover_ug(biadj_mat_recon)

    ushd = np.triu(np.logical_xor(ug, ug_recon), 1).sum()

    biadj_mat = biadj_mat.astype(int)
    biadj_mat_recon = biadj_mat_recon.astype(int)

    wtd_ug = biadj_mat.T @ biadj_mat
    wtd_ug_recon = biadj_mat_recon.T @ biadj_mat_recon

    sfd = ((wtd_ug - wtd_ug_recon) ** 2).sum()

    return sfd, ushd


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
    np.fill_diagonal(ug, False)

    return ug


def build_table(n, p):
    """Build table for SHD, ELBO, and losses
    Parameters
    ----------
    n: number of observed variables
    p: edge probability

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
        "loss_recon_train",
        "loss_recon_valid",
        "error_recon_train",
        "error_recon_valid",
        "loss_vanilla_train",
        "loss_vanilla_valid",
        "error_vanilla_train",
        "error_vanilla_valid",
        "shd_recon"
    ] + ["flag_train", "flag_valid"] + ["dof", "run"]

    table = pd.DataFrame(columns=columns)

    for idx in range(5):
        sub_table = pd.DataFrame(pd.NA, index=paths_list, columns=columns)

        for path in paths_list:
            graph_path = os.path.join(exp_path, f"experiment_{idx}", path)
            num = path.split("_")[1]
            if num in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                result_path = os.path.join(graph_path, f"n={n}_p={p}")
            else:
                result_path = os.path.join(graph_path)

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
                biadj_mat_recon = np.load(os.path.join(result_path, "biadj_mat_recon.npy"))
                shd, ushd = analysis(biadj_mat, biadj_mat_recon)
                sub_table.loc[path, "shd_recon"] = shd

            # performance information
            train_flag = table["loss_recon_train"] < table["loss_vanilla_train"]
            valid_flag = table["loss_recon_valid"] < table["loss_vanilla_valid"]
            boolean_dictionary = {True: "recon", False: "vanilla"}
            table["train_flag"] = train_flag.map(boolean_dictionary)
            table["valid_flag"] = valid_flag.map(boolean_dictionary)

            # other information
            info = pd.read_pickle(os.path.join(result_path, "info.pkl"))
            biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
            _, num_obs = biadj_mat.shape
            if info["dof"] is None:
                dof = num_obs**2 // 4
            else:
                dof = info["dof"]
            sub_table["dof"] = dof
            sub_table["run"] = idx

        table = pd.concat([table, sub_table])

    return table
