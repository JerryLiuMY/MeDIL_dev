import numpy as np
conversion_dict = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i"}

biadj_mat_list = [
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
