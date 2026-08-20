import os
os.chdir('/scratch/project_2012831/naht-dev')
print (f"This is the current directory address: {os.getcwd()}")

import matplotlib.pyplot as plt
import yaml

from utils.tb_vis_utils import gen_log_data, gen_expt_dict
# from utils.ed_vis_utils import load_data, vis_within_episode_performance
from utils.vis_utils import plot_learning_curves, plot_learning_curves_all, \
    get_ood_gen_scores1, plot_ood_gen_scores3, \
    plot_mm_selfplay_scores, get_selfplay_scores, vis_nk_curve_all, plot_ood_gen_scores4


BASE_RES_PATH = "/scratch/project_2012831/naht-dev/naht_results"

# sc2 ood generalization result
exp_namemap = { # format is alg log name: display name
        # "poam_shapley1_eps0.2": "SM-POAM",
        # "poam_banzhaf_eps0.2": "BM-POAM",
        # "poam-pqvmq_open": "POAM",
        # "ippo_shapley2v_eps0.2": "SM-IPPO",
        # "ippo_banzhaf_eps0.2": "BM-IPPO",
        # "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
        # "poam_shapley1_eps0.2": "SM-POAM",
        # "ippo_shapley2v_eps0.2": "SM-IPPO",
        # "poam_banzhaf_eps0.2": "BM-POAM",
        # "ippo_banzhaf_eps0.2": "BM-IPPO",
        "poam-pqvmq_open": "POAM",
        "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
}

target_algs = [        
    "vdn", 
    "qmix",
    "iql",
    "ippo", 
    "mappo", 
]


ood_results = {}
id_baselines = {}

n_eval_alg_seeds = 5
n_target_alg_seeds = 4

ood_results = {}
for task, algs_to_eval in {
    # "mpe-pp": ["poam_shapley1_eps0.2", "poam_banzhaf_eps0.2", "poam-pqvmq_open"], 
    # "3sv5z": ["poam_shapley1_eps0.2", "poam_banzhaf_eps0.2", "poam-pqvmq_open"], # all done
    # "5v6": ["poam_shapley1_eps0.2", "poam_banzhaf_eps0.2", "poam-pqvmq_open"], # all done
    # "8v9": ["poam_shapley1_eps0.2", "poam-pqvmq_open"], # all done
    # "10v11": ["poam_shapley1_eps0.2", "poam-pqvmq_open"] # all done
    # "mpe-pp": ["ippo_shapley2v_eps0.2", "ippo_banzhaf_eps0.2", "ippo_naht1v-pqvmq_open"], 
    # "3sv5z": ["ippo_shapley2v_eps0.2", "ippo_banzhaf_eps0.2", "ippo_naht1v-pqvmq_open"], # all done
    # "5v6": ["ippo_shapley2v_eps0.2", "ippo_banzhaf_eps0.2", "ippo_naht1v-pqvmq_open"], # all done
    # "8v9": ["ippo_shapley2v_eps0.2", "ippo_naht1v-pqvmq_open"], # all done
    # "10v11": ["ippo_shapley2v_eps0.2", "ippo_naht1v-pqvmq_open"] # all done
    # "mpe-pp": ["poam_shapley1_eps0.2", "ippo_shapley2v_eps0.2"], 
    # "3sv5z": ["poam_shapley1_eps0.2", "ippo_shapley2v_eps0.2"], # all done
    # "5v6": ["poam_shapley1_eps0.2", "ippo_shapley2v_eps0.2"], # all done
    # "8v9": ["poam_shapley1_eps0.2", "ippo_shapley2v_eps0.2"], # all done
    # "10v11": ["poam_shapley1_eps0.2", "ippo_shapley2v_eps0.2"] # all done
    # "mpe-pp": ["poam_banzhaf_eps0.2", "ippo_banzhaf_eps0.2"], 
    # "3sv5z": ["poam_banzhaf_eps0.2", "ippo_banzhaf_eps0.2"], # all done
    # "5v6": ["poam_banzhaf_eps0.2", "ippo_banzhaf_eps0.2"] # all done
    "mpe-pp": ["poam-pqvmq_open", "ippo_naht1v-pqvmq_open"], 
    # "3sv5z": ["poam-pqvmq_open", "ippo_naht1v-pqvmq_open"], # all done
    # "5v6": ["poam-pqvmq_open", "ippo_naht1v-pqvmq_open"], # all done
    # "8v9": ["poam-pqvmq_open", "ippo_naht1v-pqvmq_open"], # all done
    # "10v11": ["poam-pqvmq_open", "ippo_naht1v-pqvmq_open"] # all done
}.items():
    print("Setting: ", task)
    # determine the number of agents from the task name. take the 
    # first part of task name until the first letter
    if task == "mpe-pp":
        n_agents = 3
    else:
        n_agents = int(task.split('v')[0].split('s')[0])
    ood_gen_means, ood_gen_cis = get_ood_gen_scores1(log_path=os.path.join(BASE_RES_PATH, task, "ood_generalization_baseline"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * n_target_alg_seeds * (n_agents - 1),
                                    remove_empty_evals=True,
                                    remove_duplicates=True
                                    )
    ood_results[task] = (ood_gen_means, ood_gen_cis)

    # get in-distribution performances 
    id_means, _ = get_ood_gen_scores1(log_path=os.path.join(BASE_RES_PATH, task, "in_distr_eval_baseline"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * 1 * (n_agents - 1),
                                    remove_empty_evals=True,
                                    remove_duplicates=True
                                    )
    id_baselines[task] = id_means

plot_ood_gen_scores3(ood_results, exp_namemap, id_baselines, 
                    plot_suptitle="OOD Evaluation: Unseen Seeds",
                    plot_subplot_titles=True,
                    save=True, 
                    savedir="plots-naht/5trials-95ci", 
                    savename="main-ood-baseline-mpe.pdf"
)

# plot_ood_gen_scores4(ood_results, exp_namemap, id_baselines, 
#                     # plot_suptitle="OOD Evaluation: Unseen Seeds",
#                     # plot_subplot_titles=True,
#                     save=True, 
#                     savedir="plots-naht/5trials-95ci", 
#                     savename="main-ood-baseline-avg.pdf"
# )