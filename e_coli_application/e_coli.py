import numpy as np
from medil.independence_testing import estimate_UDG
from medil.ecc_algorithms import find_heuristic_clique_cover, find_clique_min_cover
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


## load data
path = "/home/alex/projects/cfa_code/e_coli_application/"
# load data
data = np.loadtxt(
    path + "gene_expression.csv", delimiter=",", usecols=list(range(1, 24)), skiprows=1
).T
gene_names = np.loadtxt(
    path + "gene_expression.csv", dtype=str, delimiter=",", usecols=[0], skiprows=1
)

## estimated UDG and cover
# udg = estimate_UDG(
#     data, significance_level=0.05
# )  # sig_level has substantial effect on estimation; alpa = 0.16089 gives same number of edges as true, but only 0.72 accuracy
# np.fill_diagonal(udg, True)
# heuristic_cover = find_heuristic_clique_cover(udg)
# exact_cover = find_clique_min_cover(udg)
# 14 and 15 latents


## expert knowledge UDG and cover, estimated cover with expert knowledge UDG
true_cover = np.loadtxt(
    path + "latents.csv",
    dtype=bool,
    delimiter=",",
    usecols=list(range(1, 17)),
    skiprows=1,
).T  # has 16 latents

true_udg = true_cover.T @ true_cover
# np.fill_diagonal(true_udg, False)

# expert_exact_cover = find_clique_min_cover(true_udg)
# expert_heuristic_cover = find_heuristic_clique_cover(true_udg)
# both have 16 latents


## ROC curves for UDG estimation
matplotlib.use("qtagg")
alphas = np.round(np.arange(0.01, 1.001, 0.005), 2)
tprs = np.empty_like(alphas, dtype=float)
fprs = np.empty_like(alphas, dtype=float)

num_vars = data.shape[1]
total_p = np.triu(true_udg, 1).sum()
total_n = np.triu(~true_udg, 1).sum()
corr = np.corrcoef(data, rowvar=False)
for idx, alpha in enumerate(alphas):
    # est_udg, p_vals = estimate_UDG(data, "dcov_big", alpha)
    est_udg = np.abs(corr) >= (1 - alpha)
    tprs[idx] = np.triu(np.logical_and(est_udg, true_udg), 1).sum() / total_p
    fprs[idx] = np.triu(np.logical_and(est_udg, ~true_udg), 1).sum() / total_n

g = sns.scatterplot(
    x=fprs,
    y=tprs
    # x="proportion of false positives",
    # y="proportion of true positives",
    # hue="algorithm:",
    # style="algorithm:",
    # data=results_dict,
    # s=60,
)
plt.plot([0, 1], [0, 1])

# for idx, alpha in enumerate(results_dict["alpha"]):
#     plt.text(
#         results_dict["number of false positives (skeleton)"][idx] - 0.5,
#         results_dict["number of true positives (skeleton)"][idx] + 0.15,
#         str(alpha),
#         size="xx-small",
#     )
# g.set_yticks(range(3, 14, 2))
# g.set_xticks(range(3, 21, 4))
# sns.scatterplot(
#     [14], [14], marker="^", color=["red"], label="GES", s=75
# )  # fp: 14; tp: 14
# # y = np.linspace(3, 12)
# # sns.lineplot(y, y, color="grey", linestyle="--")
# plt.legend(loc="lower right")
plt.savefig(path + "udg_roc.png", dpi=200, bbox_inches="tight")
plt.clf()
