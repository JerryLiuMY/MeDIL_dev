from exp.run_funcs import run_vae_oracle, run_vae_suite
from learning.data_loader import sample_from_graph
from learning.data_loader import load_dataset, load_dataset_real
from exp.analysis import recover_ug
from graph_est.estimation import estimation
from medil.functional_MCM import assign_DoF
from learning.params import params_dict
from datetime import datetime
from project_udgs.project_udgs import rm_project
from medil.ecc_algorithms import find_heuristic_1pc
import numpy as np
import pickle
import time
import os


def pipeline_graph(biadj_mat, num_samps, data_type, heuristic, method, alpha, dof, dof_method, path, seed):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    biadj_mat: adjacency matrix of the bipartite graph
    num_samps: number of samples
    data_type: type of the data to be generated
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: significance level
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    path: path for saving the files
    seed: random seed for the experiments
    """

    # load parameters
    np.random.seed(seed)
    batch_size, num_valid = params_dict["batch_size"], params_dict["num_valid"]

    # create biadj_mat and samples
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sampling from biadj_mat")
    time.sleep(1)
    samples = sample_from_graph(biadj_mat, num_samps=num_samps, data_type=data_type)

    # learn MeDIL model and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    num_latent = biadj_mat.shape[0]
    biadj_mat_recon = estimation(samples[:, num_latent:], heuristic=heuristic, method=method, alpha=alpha)

    np.save(os.path.join(path, "biadj_mat.npy"), biadj_mat)
    np.save(os.path.join(path, "biadj_mat_recon.npy"), biadj_mat_recon)

    ud_graph = recover_ug(biadj_mat)
    ud_graph_recon = recover_ug(biadj_mat_recon)
    np.save(os.path.join(path, "ud_graph.npy"), ud_graph)
    np.save(os.path.join(path, "ud_graph_recon.npy"), ud_graph_recon)

    info = {"heuristic": heuristic, "method": method, "alpha": alpha, "dof": dof, "dof_method": dof_method}
    with open(os.path.join(path, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset(samples, num_latent, batch_size)

    valid_samples = sample_from_graph(biadj_mat, num_samps=num_valid, data_type=data_type)
    valid_loader = load_dataset(valid_samples, num_latent, batch_size)

    # perform vae training
    run_vae_oracle(biadj_mat, train_loader, valid_loader, path, seed)

    redundant_path = os.path.join(path, "redundant")
    if not os.path.isdir(redundant_path):
        os.mkdir(redundant_path)
    biadj_mat_redundant = assign_DoF(biadj_mat_recon, deg_of_freedom=dof, method=dof_method)
    np.save(os.path.join(redundant_path, "biadj_mat_redundant.npy"), biadj_mat_redundant)
    run_vae_suite(biadj_mat_redundant, train_loader, valid_loader, redundant_path, seed)

    random_path = os.path.join(path, "random")
    if not os.path.isdir(random_path):
        os.mkdir(random_path)
    biadj_mat_random = np.random.choice(a=[False, True], size=biadj_mat_redundant.shape, p=[0.5, 0.5])
    np.save(os.path.join(random_path, "biadj_mat_random.npy"), biadj_mat_random)
    run_vae_suite(biadj_mat_random, train_loader, valid_loader, random_path, seed)


def pipeline_real(dataset, heuristic, method, alpha, dof, dof_method, path, seed):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    dataset: dataset for real experiments
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: significance level
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    path: path for saving the files
    seed: random seed
    """

    # define paths
    path_1pc = path + "_1pc"
    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isdir(path_1pc):
        os.mkdir(path_1pc)

    # load parameters
    np.random.seed(seed)
    batch_size = params_dict["batch_size"]
    samples, valid_samples = dataset

    # learn MeDIL model and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    biadj_mat_recon = estimation(samples, heuristic=heuristic, method=method, alpha=alpha)
    biadj_mat_redundant = assign_DoF(biadj_mat_recon, deg_of_freedom=dof, method=dof_method)
    np.save(os.path.join(path, "biadj_mat_recon.npy"), biadj_mat_recon)
    np.save(os.path.join(path, "biadj_mat_redundant.npy"), biadj_mat_redundant)

    ud_graph_recon = recover_ug(biadj_mat_recon)
    np.save(os.path.join(path, "ud_graph_recon.npy"), ud_graph_recon)

    info = {"heuristic": heuristic, "method": method, "alpha": alpha, "dof": dof, "dof_method": dof_method}
    with open(os.path.join(path, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    biadj_mat_1pc = find_heuristic_1pc(ud_graph_recon)
    biadj_mat_1pc_redundant = assign_DoF(biadj_mat_1pc, deg_of_freedom=dof, method=dof_method)
    np.save(os.path.join(path_1pc, "biadj_mat_1pc.npy"), biadj_mat_1pc)
    np.save(os.path.join(path_1pc, "biadj_mat_1pc_redundant.npy"), biadj_mat_1pc_redundant)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset_real(samples, batch_size)
    valid_loader = load_dataset_real(valid_samples, batch_size)

    # perform vae training
    biadj_mat_redundant = np.load(os.path.join(path, "biadj_mat_redundant.npy"))
    run_vae_suite(biadj_mat_redundant, train_loader, valid_loader, path, seed, shuffle=True)

    biadj_mat_1pc_redundant = np.load(os.path.join(path_1pc, "biadj_mat_1pc_redundant.npy"))
    run_vae_suite(biadj_mat_1pc_redundant, train_loader, valid_loader, path_1pc, seed, shuffle=True)
