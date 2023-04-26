import numpy as np
import string
from medil.functional_MCM import rand_biadj_mat

# fixed graph
conversion_dict = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i"}
fixed_biadj_mat_list = [
    np.array([[True, True]]),
    np.array([[True, True, True]]),
    np.array([[True, True, True, True]]),
    np.array([[True, True, False], [False, True, True]]),
    np.array([[True, True, True, False, False],
              [False, False, True, True, True]]),
    np.array([[True, True, True, False, False, False, False],
              [False, False, True, True, True, False, False],
              [False, False, False, False, True, True, True]]),
    np.array([[True, True, True, False, False, False, False, False, False],
              [False, False, True, True, True, False, False, False, False],
              [False, False, False, False, True, True, True, False, False],
              [False, False, False, False, False, False, True, True, True]]),
    np.array([[True, True, True, False, False, False, False, False, False, False, False],
              [False, False, True, True, True, False, False, False, False, False, False],
              [False, False, False, False, True, True, True, False, False, False, False],
              [False, False, False, False, False, False, True, True, True, False, False],
              [False, False, False, False, False, False, False, False, True, True, True]]),
    np.array(
        [[1, 1, 1, 0, 0, 0],
         [1, 1, 0, 1, 0, 0],
         [1, 1, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 1]], dtype=bool
    )
]


# random graph
num_obs = [8, 15, ]
rand_biadj_mat_list = []
for i, num_obs in zip(range(10), num_obs):
    np.random.seed(i)
    rand_biadj_mat_list.append(rand_biadj_mat(num_obs=num_obs, edge_prob=np.random.uniform(size=1)[0]))


# tcga dataset
tcga_size, num_obs = 17440, 8
tcga_key_list = []
tcga_subsize = 1000
for i in range(10):
    np.random.seed(i)
    tcga_key_list.append(np.random.choice(tcga_size, size=num_obs))


# mnist dataset
mnist_size, num_obs = 784, 8
mnist_key_list = []
mnist_subsize = 784
for i in range(10):
    np.random.seed(i)
    mnist_key_list.append(np.random.choice(mnist_size, size=num_obs))


# gene dataset
gene_size, num_obs = 23, 8
gene_key_list = []
gene_subsize = 23
for i in range(10):
    np.random.seed(i)
    gene_key_list.append(np.random.choice(gene_size, size=num_obs))


# # sub paths
# fixed_paths_list = [f"Graph_{i}" for i in string.ascii_lowercase[:9]]
# rand_paths_list = [f"Graph_{i}" for i in range(10)]
# real_paths_list = [f"Real_{i}" for i in range(10)]
# paths_list = fixed_paths_list + rand_paths_list + real_paths_list

# sub paths
fixed_paths_list = [f"Graph_{i}" for i in string.ascii_lowercase[:9]]
rand_paths_list = [f"Graph_{i}" for i in range(10)]
paths_list = fixed_paths_list + rand_paths_list


# linspace for the simulated graphs and the real dataset
linspace_graph = np.exp(np.linspace(np.log(100), np.log(2500), 10))
linspace_graph = np.array(sorted(set(np.round(linspace_graph)))).astype(int)
linspace_real = np.exp(np.linspace(np.log(100), np.log(632), 10))
linspace_real = np.array(sorted(set(np.round(linspace_real)))).astype(int)
