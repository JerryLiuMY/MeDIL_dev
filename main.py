from exp.examples import biadj_mat_list
from exp.examples import conversion_dict
from exp.pipeline import pipeline
from datetime import datetime
import numpy as np
import os

exp_path = "../data/experiments"
linspace = np.exp(np.linspace(0, np.log(2500), 10))
linspace = np.array(sorted(set(np.round(linspace)))).astype(int)


# random graph
for idx, biadj_mat in enumerate(biadj_mat_list):
    graph_idx = conversion_dict[idx]
    for num_samps in linspace:
        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {graph_idx} with "
                  f"num_samps={num_samps} and alpha={alpha}")
            folder_name = f"num_samps={num_samps}_alpha={alpha}"
            folder_path = os.path.join(exp_path, folder_name)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            pipeline(biadj_mat, num_samps, alpha, folder_path, seed=0)
