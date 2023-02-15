import numpy as np
from medil.independence_testing import estimate_UDG
from medil.ecc_algorithms import find_heuristic_clique_cover, find_clique_min_cover

path = "e_coli_application/"
# load data
data = np.loadtxt(
    path + "gene_expression.csv", delimiter=",", usecols=list(range(1, 24)), skiprows=1
).T
gene_names = np.loadtxt(
    path + "gene_expression.csv", dtype=str, delimiter=",", usecols=[0], skiprows=1
)

## estimated UDG and cover
udg = estimate_UDG(
    data, significance_level=0.05
)  # sig_level has substantial effect on estimation; alpa = 0.16089 gives same number of edges as true, but only 0.72 accuracy
np.fill_diagonal(udg, True)
heuristic_cover = find_heuristic_clique_cover(udg)
exact_cover = find_clique_min_cover(udg)
# 14 and 15 latents


## expert knowledge UDG and cover, estimated cover with expert knowledge UDG
true_cover = np.loadtxt(
    path + "latents.csv",
    dtype=bool,
    delimiter=",",
    usecols=list(range(1, 17)),
    skiprows=1,
).T  # has 16 latents

true_udg = true_latents.T @ true_latents
# np.fill_diagonal(true_udg, False)

expert_exact_cover = find_clique_min_cover(true_udg)
expert_heuristic_cover = find_heuristic_clique_cover(true_udg)
# both have 16 latents
