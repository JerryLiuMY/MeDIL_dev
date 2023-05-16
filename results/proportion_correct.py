import numpy as np


proportions = np.empty(9)
for idx, p in enumerate(np.arange(0.1, 0.91, 0.1)):
    p = np.round(p, decimals=1)
    path = f"results/table/table_n=10_p={p}.csv"
    all_sfds = np.loadtxt(path, skiprows=1, usecols=[13], delimiter=",")
    base = np.tile(np.arange(9, 19), 10)
    mult = np.repeat(np.arange(1, 11), 10)
    idcs = base * mult
    relevant_sfds = all_sfds[idcs]
    proportions[idx] = np.logical_not(relevant_sfds).mean()
