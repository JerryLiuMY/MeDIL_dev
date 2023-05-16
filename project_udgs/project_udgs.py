import numpy as np

from gues.grues import InputData as rand_walker

rng = np.random.default_rng(0)

mnist_udg = np.load("project_udgs/mnist_udg-0.001.npy")
tcga_udg = np.load("project_udgs/tcga_udg-0.05.npy")
tumor_udg = np.load("project_udgs/tumor_udg-0.05.npy")


def project(udg):
    num_obs = len(udg)
    np.fill_diagonal(udg, False)

    dummy_samp = rng.random(udg.shape)
    rw = rand_walker(dummy_samp, rng)

    # # find ECC in polynomial time
    rw.init_uec(udg)
    rw.get_max_cpdag()
    max_cpdag = rw.cpdag
    sinks = np.flatnonzero(np.logical_and(max_cpdag.sum(1) == 0, max_cpdag.sum(0)))
    nonsinks = np.delete(np.arange(num_obs), sinks)
    order = np.append(nonsinks, sinks)
    dag = np.triu(max_cpdag[:, order][order, :])
    sources = np.flatnonzero(dag.sum(0) == 0)
    dag[sources, sources] = True  # take de(sources) as cliques
    biadj_mat = dag[sources, :]
    return biadj_mat


# mnist_projected = project(mnist_udg)
# tcga_projected = project(tcga_udg)
# tumor_projected = project(tumor_udg)

# np.save("project_udgs/mnist_projected.npy", mnist_projected)
# np.save("project_udgs/tcga_projected.npy", tcga_projected)
# np.save("project_udgs/tumor_projected.npy", tumor_projected)

mnist_projected = np.load("project_udgs/mnist_projected.npy")
tcga_projected = np.load("project_udgs/tcga_projected.npy")
tumor_projected = np.load("project_udgs/tumor_projected.npy")

# np.savetxt(
#     "project_udgs/mnist_projected.csv", mnist_projected.astype(int), delimiter=","
# )
# np.savetxt("project_udgs/tcga_projected.csv", tcga_projected.astype(int), delimiter=",")
# np.savetxt(
#     "project_udgs/tumor_projected.csv", tumor_projected.astype(int), delimiter=","
# )
