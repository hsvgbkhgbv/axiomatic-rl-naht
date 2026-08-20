import os
os.chdir('/scratch/project_2012831/naht-dev')
print (f"This is the current directory address: {os.getcwd()}")

import matplotlib.pyplot as plt
import yaml

from utils.tb_vis_utils import gen_log_data, gen_expt_dict
# from utils.ed_vis_utils import load_data, vis_within_episode_performance
from utils.vis_utils import plot_learning_curves, plot_learning_curves_all, \
    get_ood_gen_scores1, plot_ood_gen_scores1, \
    plot_mm_selfplay_scores, get_selfplay_scores, vis_nk_curve_all


BASE_RES_PATH = "/scratch/project_2012831/naht-dev/naht_results"

# sc2 ood generalization result
exp_namemap = { # format is alg log name: display name
        "poam_shapley1_eps0.2": "Shapley Machine",
        "poam-pqvmq_open": "POAM",
}

target_algs = [        
    "vdn", 
    "qmix",
    "iql",
    "ippo", 
    "mappo", 
]


ood_results = {}
id_results = {}
id_baselines = {}

n_eval_alg_seeds = 5
n_target_alg_seeds = 4

for task, algs_to_eval in {
    "3sv5z": ["poam_shapley1_eps0.2", "poam-pqvmq_open"], # all done
    "5v6": ["poam_shapley1_eps0.2", "poam-pqvmq_open"], # all done
    "8v9": ["poam_shapley1_eps0.2", "poam-pqvmq_open"], # all done
    "10v11": ["poam_shapley1_eps0.2", "poam-pqvmq_open"] # all done
}.items():
    print("Setting: ", task)
    # determine the number of agents from the task name. take the 
    # first part of task name until the first letter
    n_agents = int(task.split('v')[0].split('s')[0])
    ood_gen_means, ood_gen_cis = get_ood_gen_scores1(log_path=os.path.join(BASE_RES_PATH, task, "ood_generalization_main"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * n_target_alg_seeds * (n_agents - 1),
                                    remove_empty_evals=True,
                                    remove_duplicates=True
                                    )
    ood_results[task] = (ood_gen_means, ood_gen_cis)

    # get in-distribution performances 
    id_means, id_cis = get_ood_gen_scores1(log_path=os.path.join(BASE_RES_PATH, task, "in_distr_eval_main"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * 1 * (n_agents - 1),
                                    remove_empty_evals=True,
                                    remove_duplicates=True
                                    )
    id_results[task] = (id_means, id_cis)
    id_baselines[task] = id_means

plot_ood_gen_scores1(ood_results, exp_namemap, id_baselines, 
                    plot_suptitle="OOD Evaluation: SC2",
                    plot_subplot_titles=True,
                    save=True, 
                    savedir="figures/5trials-95ci", 
                    savename="sc2_open-pqvmq_main_gen-id-baseline.pdf"
)

print ("This is the in distribution result")
for task in ["3sv5z", "5v6", "8v9", "10v11"]:
    print (f"This is the task: {task}")
    target_id = id_results[task]
    print (target_id)

print ("\n\n")

print ("This is the out of distribution result")
for task in ["3sv5z", "5v6", "8v9", "10v11"]:
    print (f"This is the task: {task}")
    target_ood = ood_results[task]
    print (target_ood)