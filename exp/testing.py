from medil.functional_MCM import sample_from_minMCM
from est.estimation import estimation
import numpy as np
import pandas as pd
import time


def testing(biadj_mat, num_runs, num_samps):
    """ Experiment for verifying the SHD, USHD, diff and time
    Parameters
    ----------
    biadj_mat: adjacency matrix of the generated graph
    num_runs: number of runs
    num_samps: number of samples in each run

    Returns
    -------
    shd_df: dataframe of SHD
    diff_df: dataframe of mis-specification in latent space
    """

    num_latent, num_obs = biadj_mat.shape
    linspace = np.array(sorted(set([int(_) for _ in np.exp(np.linspace(2.5, np.log(num_samps), 10))])))
    linspace = np.append(linspace, num_samps)
    shd_df = pd.DataFrame(0., index=range(num_runs), columns=linspace)
    ushd_df = pd.DataFrame(0., index=range(num_runs), columns=linspace)
    diff_df = pd.DataFrame(0., index=range(num_runs), columns=linspace)
    time_df = pd.DataFrame(0., index=range(num_runs), columns=linspace)

    for i in range(num_runs):
        samples, cov = sample_from_minMCM(biadj_mat, num_samps=num_samps)
        for j in linspace:
            print(f"Working on num_run={i} and num_sample={j}")
            t0 = time.time()
            biadj_mat_learned, shd, ushd, num_latent_recon = estimation(biadj_mat, num_obs, num_latent, samples[:j, :])
            t1 = time.time()

            shd_df.loc[i, j] = shd
            ushd_df.loc[i, j] = ushd
            diff_df.loc[i, j] = abs(num_latent - num_latent_recon)
            time_df.loc[i, j] = t1 - t0

    return shd_df, ushd_df, diff_df, time_df
