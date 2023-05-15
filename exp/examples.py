from medil.functional_MCM import rand_biadj_mat
import numpy as np
import string
import random

# fixed graph
conversion_dict = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h",
    8: "i",
}

fixed_biadj_mat_list = [
    np.array([[True, True]]),
    np.array([[True, True, True]]),
    np.array([[True, True, True, True]]),
    np.array([[True, True, False], [False, True, True]]),
    np.array([[True, True, True, False, False], [False, False, True, True, True]]),
    np.array(
        [
            [True, True, True, False, False, False, False],
            [False, False, True, True, True, False, False],
            [False, False, False, False, True, True, True],
        ]
    ),
    np.array(
        [
            [True, True, True, False, False, False, False, False, False],
            [False, False, True, True, True, False, False, False, False],
            [False, False, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, True, True, True],
        ]
    ),
    np.array(
        [
            [True, True, True, False, False, False, False, False, False, False, False],
            [False, False, True, True, True, False, False, False, False, False, False],
            [False, False, False, False, True, True, True, False, False, False, False],
            [False, False, False, False, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, False, False, True, True, True],
        ]
    ),
    np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


# random graph -- setting 1
num_runs = 10
num_obs_list = [10, 50, 100]
edge_prob_list1 = np.round(np.arange(0.1, 0.91, 0.1), 2)
rand_biadj_mat_list = {}

for idx in range(num_runs):
    for n in num_obs_list:
        for p in edge_prob_list1:
            print(f"Generating sample with idx={idx}, num_obs={n}, and edge_prob={p}")
            random.seed(idx)
            rand_biadj_mat_list[f"{idx}_{n}_{p}"] = rand_biadj_mat(num_obs=n, edge_prob=p)


# random graph -- setting 2
num_runs = 10
num_obs = 10
num_latent_list2 = np.range(9) + 1
rand_biadj_mat_list = {}

for idx in range(num_runs):
    for num_latent in num_latent_list2:
        flag = False
        print(f"Generating sample with idx={idx} and edge_prob={num_latent}")
        random.seed(idx)
        while flag is False:
            biadj_mat = rand_biadj_mat(num_obs=num_obs, edge_prob=np.random.uniform(size=1)[0])
            if biadj_mat.shape[0] == num_latent:
                rand_biadj_mat_list[f"{idx}_{num_latent}"] = biadj_mat
                flag = True


# tcga dataset
tcga_size, num_obs = 17440, 1000
np.random.seed(0)
tcga_key = np.random.choice(tcga_size, size=num_obs)


# mnist dataset
mnist_size, num_obs = 784, 784
np.random.seed(0)
mnist_key = np.random.choice(mnist_size, size=num_obs)


# mnist dataset
tumors_size, num_obs = 356, 356
np.random.seed(0)
tumors_key = np.random.choice(tumors_size, size=num_obs)


# # gene dataset
# gene_size, num_obs = 23, 8
# gene_key_list = []
# gene_subsize = 23
# for i in range(10):
#     np.random.seed(i)
#     gene_key_list.append(np.random.choice(gene_size, size=num_obs))


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
num_samps_graph = 1000
num_samps_real = 632
