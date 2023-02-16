import numpy as np
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
num_obs = 8
rand_biadj_mat_list = []
for i in range(10):
    np.random.seed(i)
    rand_biadj_mat_list.append(rand_biadj_mat(num_obs=num_obs, edge_prob=np.random.uniform(size=1)[0]))


# tcga dataset
tcga_size, num_obs = 17440, 8
tcga_key_list = []
tcga_subsize = 1000
for i in range(10):
    np.random.seed(i)
    tcga_key_list.append(np.random.choice(tcga_size, size=num_obs))
