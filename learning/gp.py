import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal


def compute_gauss_kernel(x, sigmay, sigmax):

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[0]

    print(f"x={x}")
    print(f"pdist={pdist(x, 'euclidean')}")
    xnorm = squareform(pdist(x, 'euclidean')) ** 2
    kx = sigmay * np.exp(-xnorm / (2 * sigmax ** 2))

    return kx


def compute_caus_order(G):

    p = G.shape[1]
    remaining = list(range(p))
    caus_order = [None] * p
    for i in range(p - 1):
        root = remaining.index(min([j for j, col_sum in enumerate(np.sum(G, axis=0)) if col_sum == 0]))
        caus_order[i] = remaining[root]
        remaining.pop(root)
        G = np.delete(np.delete(G, root, axis=0), root, axis=1)
    caus_order[p - 1] = remaining[0]

    return caus_order


def random_B(G, lB=0.1, uB=0.9, two_intervals=1):
    num_coeff = np.sum(G)
    B = G.T.copy()
    if num_coeff == 1:
        coeffs = (np.random.choice([-1, 1], size=num_coeff, p=[0.5, 0.5]) ** two_intervals) * np.random.uniform(lB, uB)
    else:
        coeffs = np.diag(
            (np.random.choice([-1, 1], size=num_coeff, p=[0.5, 0.5]) ** two_intervals)) @ \
                 np.random.uniform(lB, uB, size=num_coeff)
    B[B == 1] = coeffs

    return B


def sample_data_from_G(n, G):

    func_type = "GAM"
    pars_func_type = {"B": random_B(G), "kap": 1, "sigmax": 1, "sigmay": 1, "output": False}
    pars_noise = {"noiseExp": 1, "varMin": 1, "varMax": 2, "noiseExpVarMin": 2, "noiseExpVarMax": 4,
                  "bound": [1] * G.shape[1]}
    noise_type = "normalRandomVariances"

    if func_type == "GAM":
        pars_func_type["kap"] = 1
        return sample_data_from_GAM_GP(n, G, pars_func_type, noise_type, pars_noise)
    elif func_type == "GP":
        pars_func_type["kap"] = 0
        return sample_data_from_GAM_GP(n, G, pars_func_type, noise_type, pars_noise)
    elif func_type == "GAMGP":
        return sample_data_from_GAM_GP(n, G, pars_func_type, noise_type, pars_noise)
    else:
        raise ValueError("This function type does not exist!")


def sample_data_from_GAM_GP(n, G, pars_func_type, noise_type, pars_noise):
    p = G.shape[1]
    X = np.empty((n, p))
    caus_order = compute_caus_order(G)

    if pars_func_type["output"]:
        print(caus_order)

    noise_var = np.random.uniform(pars_noise["varMin"], pars_noise["varMax"], p)

    for node in caus_order:
        if pars_func_type["output"]:
            print(f"generating GP for node {node}")

        pa_of_node = np.where(G[:, node] == 1)[0]

        if len(pa_of_node) == 0:
            if noise_type == "normalRandomVariances" or noise_type == "normalRandomVariancesFixedExp":
                ran = np.random.randn(n)
                noisetmp = (np.sqrt(noise_var[node]) * np.abs(ran)) ** (pars_noise["noiseExp"]) * np.sign(ran)
            else:
                raise NotImplementedError("This noiseType is not implemented yet.")
            X[:, node] = noisetmp
        else:
            nu_pa = len(pa_of_node)
            X[:, node] = np.zeros(n)

            if pars_func_type["kap"] > 0:
                for pa in pa_of_node:
                    kern_pa = compute_gauss_kernel(X[:, pa], pars_func_type["sigmay"], pars_func_type["sigmax"])
                    fpa = multivariate_normal(mean=np.zeros(n), cov=kern_pa).rvs()
                    X[:, node] = X[:, node] + pars_func_type["kap"] * fpa

            if pars_func_type["kap"] < 1:
                kern_all_pa = compute_gauss_kernel(X[:, pa_of_node], pars_func_type["sigmay"], pars_func_type["sigmax"])
                fall_pa = multivariate_normal(mean=np.zeros(n), cov=kern_all_pa).rvs()
                X[:, node] = X[:, node] + (1 - pars_func_type["kap"]) * fall_pa

            if noise_type == "normalRandomVariances" or noise_type == "normalRandomVariancesFixedExp":
                ran = np.random.randn(n)
                noisetmp = (0.2 * np.sqrt(noise_var[node]) * np.abs(ran)) ** (pars_noise["noiseExp"]) * np.sign(ran)
            else:
                raise NotImplementedError("This noiseType is not implemented yet.")
            X[:, node] = X[:, node] + noisetmp

    return X

# Example usage:
# n = 100
# p = 5
# G = np.random.choice([0, 1], size=(p, p))
# X = sample_data_from_G(n, G, func_type="GAM", noise_type="normalRandomVariances")
