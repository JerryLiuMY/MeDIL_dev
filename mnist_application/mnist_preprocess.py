# from keras.datasets import mnist
from medil.independence_testing import estimate_UDG
from medil.ecc_algorithms import find_heuristic_clique_cover
import numpy as np


path = "mnist_application/"
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
# data = train_X.reshape(60000, 28 * 28)
# np.save(path + "mnist_data", data)
data = np.load(path + "mnist_data.npy").astype(float)

# udg = estimate_UDG(data, "dcov_big", 0.001)
# np.save(path + "mnist_udg", udg)
udg = np.load(path + "mnist_udg.npy")

# the_cover = find_heuristic_clique_cover(udg)
# np.save(path + "mnist_cover", the_cover)
the_cover = np.load(path + "mnist_cover.npy")