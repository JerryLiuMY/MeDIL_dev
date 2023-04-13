import numpy as np
from medil.functional_MCM import rand_biadj_mat, sample_from_minMCM, assign_DoF
import warnings


def test_rand_biadj_mat():
    pass


def test_sample_from_minMCM():
    pass


def test_assign_DoF():
    biadj_mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])

    warnings.filterwarnings("error")
    try:
        test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
        assert False
    except UserWarning:
        warnings.resetwarnings()
        warnings.simplefilter("ignore")
        test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
        assert (test_insufficient == biadj_mat).all()

    test_uniform = assign_DoF(biadj_mat, 8, "uniform")
    unique_uniform, counts_uniform = np.unique(test_uniform, axis=0, return_counts=True)
    assert (biadj_mat == unique_uniform).all()
    assert min(counts_uniform) == 2
    assert max(counts_uniform) == 3
    assert counts_uniform.sum() == 8

    test_clique = assign_DoF(biadj_mat, 11, "clique_size")
    unique_clique, counts_clique = np.unique(test_clique, axis=0, return_counts=True)
    assert (biadj_mat == unique_clique).all()
    assert min(counts_clique) == 3
    assert max(counts_clique) == 4
    assert counts_clique.sum() == 11

    # need to test total and avg var still

    # make sure there are no rounding errors, regardless of method
    # for dof in range(3, 12):
    #     for method in ("uniform", "clique_size", "total_var", "avg_var"):
    #         pass
