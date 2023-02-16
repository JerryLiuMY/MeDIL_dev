from gloabl_settings import DATA_PATH
from exp.examples import paths_list
from exp.examples import linspace_graph
from exp.examples import linspace_real
from cdt.metrics import SHD
import seaborn as sns
import pandas as pd
import numpy as np
import os
sns.set()


def build_table(order_n, alpha):
    """ Build table for SHD, ELBO, and losses
    Parameters
    ----------
    order_n: order of the number of samples
    alpha: alpha for the hypothesis test

    Returns
    -------
    table: table for summarizing the results
    """

    exp_path = os.path.join(DATA_PATH, "experiments")
    columns = ["loss_true_train", "loss_true_valid", "error_true_train", "error_true_valid",
               "loss_exact_train", "loss_exact_valid", "error_exact_train", "error_exact_valid",
               "loss_hrstc_train", "loss_hrstc_valid", "error_hrstc_train", "error_hrstc_valid",
               "loss_vanilla_train", "loss_vanilla_valid", "error_vanilla_train", "error_vanilla_valid",
               "shd_exact", "shd_hrstc"] + ["run"]

    table = pd.DataFrame(columns=columns)

    for idx in range(10):
        sub_table = pd.DataFrame(pd.NA, index=paths_list, columns=columns)

        for path in paths_list:
            graph_path = os.path.join(exp_path, f"experiment_{idx}", path)
            n = linspace_graph[order_n] if "Graph" in path else linspace_real[order_n]
            result_path = os.path.join(graph_path, f"num_samps={n}_alpha={alpha}")

            # hrstc graph
            loss_hrstc = pd.read_pickle(os.path.join(result_path, "loss_hrstc.pkl"))
            error_hrstc = pd.read_pickle(os.path.join(result_path, "error_hrstc.pkl"))
            sub_table.loc[path, "loss_hrstc_train"] = loss_hrstc[0][-1]
            sub_table.loc[path, "loss_hrstc_valid"] = loss_hrstc[1][-1]
            sub_table.loc[path, "error_hrstc_train"] = error_hrstc[0][-1]
            sub_table.loc[path, "error_hrstc_valid"] = error_hrstc[1][-1]

            # vanilla graph
            loss_vanilla = pd.read_pickle(os.path.join(result_path, "loss_vanilla.pkl"))
            error_vanilla = pd.read_pickle(os.path.join(result_path, "error_vanilla.pkl"))
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

                # exact graph
                loss_exact = pd.read_pickle(os.path.join(result_path, "loss_exact.pkl"))
                error_exact = pd.read_pickle(os.path.join(result_path, "error_exact.pkl"))
                sub_table.loc[path, "loss_exact_train"] = loss_exact[0][-1]
                sub_table.loc[path, "loss_exact_valid"] = loss_exact[1][-1]
                sub_table.loc[path, "error_exact_train"] = error_exact[0][-1]
                sub_table.loc[path, "error_exact_valid"] = error_exact[1][-1]

                # SHD for exact
                biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
                biadj_mat_exact = np.load(os.path.join(result_path, "biadj_mat_exact.npy"))
                shd_exact = SHD(biadj_mat, biadj_mat_exact)
                sub_table.loc[path, "shd_exact"] = shd_exact

                # SHD for hrstc
                biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
                biadj_mat_hrstc = np.load(os.path.join(result_path, "biadj_mat_hrstc.npy"))
                shd_hrstc = SHD(biadj_mat, biadj_mat_hrstc)
                sub_table.loc[path, "shd_hrstc"] = shd_hrstc

            sub_table["run"] = idx

        table = pd.concat([table, sub_table])

    return table
