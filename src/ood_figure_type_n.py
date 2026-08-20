import os
os.chdir('/scratch/project_2012831/naht-dev')
print (f"This is the current directory address: {os.getcwd()}")

import matplotlib.pyplot as plt
import yaml
import numpy as np
from scipy import stats
import itertools
import shutil

from utils.tb_vis_utils import gen_log_data, gen_expt_dict
# from utils.ed_vis_utils import load_data, vis_within_episode_performance
from utils.vis_utils import plot_learning_curves, plot_learning_curves_all, \
    get_ood_gen_scores1, plot_ood_gen_scores3, \
    plot_mm_selfplay_scores, get_selfplay_scores, vis_nk_curve_all1, \
    generate_summary, generate_summary1


BASE_RES_PATH = "/scratch/project_2012831/naht-dev/naht_results/"

# sc2 ood generalization result
exp_namemap = { # format is alg log name: display name
        "poam_shapley1_eps0.2": "SM-POAM",
        "poam_banzhaf_eps0.2": "BM-POAM",
        "poam_type": "POAM",
        # "ippo_shapley2v_eps0.2": "SM-IPPO",
        # "ippo_banzhaf_eps0.2": "BM-IPPO",
        # "ippo_naht1v_type": "IPPO-NAHT",
}

target_algs = [        
    "vdn", 
    "ippo", 
]

ood_results = {}
id_baselines = {}
ood_raw_data = {}  # Store raw data for t-test

n_eval_alg_seeds = 5
n_target_alg_seeds = 5


for task, algs_to_eval in {
    "mpe-pp": ["poam_shapley1_eps0.2", "poam_type", "poam_banzhaf_eps0.2"], 
    "3sv5z": ["poam_shapley1_eps0.2", "poam_type", "poam_banzhaf_eps0.2"], # all done
    "5v6": ["poam_shapley1_eps0.2", "poam_type", "poam_banzhaf_eps0.2"], # all done
    "8v9": ["poam_shapley1_eps0.2", "poam_type"], # all done
    "10v11": ["poam_shapley1_eps0.2", "poam_type"] # all done
    # "mpe-pp": ["ippo_shapley2v_eps0.2", "ippo_naht1v_type", "ippo_banzhaf_eps0.2"], 
    # "3sv5z": ["ippo_shapley2v_eps0.2", "ippo_naht1v_type", "ippo_banzhaf_eps0.2"], # all done
    # "5v6": ["ippo_shapley2v_eps0.2", "ippo_naht1v_type", "ippo_banzhaf_eps0.2"], # all done
    # "8v9": ["ippo_shapley2v_eps0.2", "ippo_naht1v_type"], # all done
    # "10v11": ["ippo_shapley2v_eps0.2", "ippo_naht1v_type"], # all done
}.items():
    print("Setting: ", task)
    # determine the number of agents from the task name. take the 
    # first part of task name until the first letter
    if task == "mpe-pp":
        n_agents = 3
    else:
        n_agents = int(task.split('v')[0].split('s')[0])
    ood_gen_means, ood_gen_cis = get_ood_gen_scores1(log_path=os.path.join(BASE_RES_PATH, task, "type_generalization_poam"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * n_target_alg_seeds * (n_agents - 1),
                                    remove_empty_evals=False,
                                    remove_duplicates=False
                                    )
    ood_results[task] = (ood_gen_means, ood_gen_cis)

plot_ood_gen_scores3(ood_results, exp_namemap, 
                    plot_suptitle=f"OOD Evaluation with Train/Test Set: MAPPO, QMIX, IQL / IPPO, VDN",
                    plot_subplot_titles=True,
                    save=True, 
                    savedir="plots-naht/5trials-95ci", 
                    savename="type-ood-poam.pdf"
)




target_algs = {  # log_name: display_name      
    "vdn": "vdn", 
    # "qmix": "qmix",
    # "iql": "iql",
    "ippo": "ippo", 
    # "mappo": "mappo", 
}

eval_algs = {
    # log_name: display_name
    "poam_shapley1_eps0.2": "SM-POAM",
    "poam_banzhaf_eps0.2": "BM-POAM",
    "poam_type": "POAM",
    # "ippo_shapley2v_eps0.2": "SM-IPPO",
    # "ippo_banzhaf_eps0.2": "BM-IPPO",
    # "ippo_naht1v_type": "IPPO-NAHT",
}

tasks = {
    "mpe-pp": 3,
    "5v6": 5,
    "8v9": 8,
    "10v11": 10,
    "3sv5z": 3,
}

print("Please note that the error bars are 95% CI's computed across the mean scores against each opponent type (VDN, QMIX, IPPO, MAPPO).")
print("The highest value of N reflects the self-play setting and is displayed as a horizontal line")
vis_nk_curve_all1(tasks=tasks, base_path=BASE_RES_PATH,
                 eval_algs=eval_algs, target_algs=target_algs, 
                 plot_n_controlled=True,
                 save=True, 
                 savedir="plots-naht/5trials-95ci",
                 savename="nk_mean_ood_type_curves.pdf")