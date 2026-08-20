import os

import matplotlib.pyplot as plt
import yaml

from utils.tb_vis_utils import gen_log_data, gen_expt_dict
# from utils.ed_vis_utils import load_data, vis_within_episode_performance
from utils.vis_utils import plot_learning_curves, plot_learning_curves_all, \
    get_ood_gen_scores, plot_ood_gen_scores, \
    plot_mm_selfplay_scores, get_selfplay_scores, vis_nk_curve_all


BASE_RES_PATH = ""

# mpe-pp ood generalization result
exp_namemap = { # format is alg log name: display name
        "poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7": "Shapley Machine",
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
id_baselines = {}
for task, algs_to_eval in {
    "mpe-pp": ["poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7", "poam-pqvmq_open"], 
    # "mpe-pp": ["poam_shapley_ood_eps0.2_m7", "poam-ood_open"], 
}.items():
    print("Setting: ", task)
    n_agents = 3
    n_eval_alg_seeds = 5
    n_target_alg_seeds = 4
    ood_gen_means, ood_gen_cis = get_ood_gen_scores(log_path=os.path.join(BASE_RES_PATH, task, "ood_generalization_main"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * n_target_alg_seeds  * (n_agents - 1), 
                                    remove_empty_evals=False,
                                    remove_duplicates=False
                                    )
    ood_results[task] = (ood_gen_means, ood_gen_cis)
    # get in-distribution performances 
    id_means, id_cis = get_ood_gen_scores(log_path=os.path.join(BASE_RES_PATH, task, "in_distr_eval_main"), 
                                    algs_to_eval=algs_to_eval,
                                    target_algs=target_algs,
                                    n_expected_evals=n_eval_alg_seeds * 1 * (n_agents - 1),
                                    remove_empty_evals=False,
                                    remove_duplicates=False
                                    )
    id_baselines[task] = id_means

plot_ood_gen_scores(ood_results, exp_namemap, 
                    in_distr_baseline=id_baselines,
                    plot_suptitle="Predator Prey",
                    plot_subplot_titles=False,
                    save=True, 
                    savedir="figures/5trials-95ci", 
                    savename="mpe-pp_open-pqvmq_main_gen-id-baseline.pdf"
)

print (f"This is the ood means: \n {ood_gen_means}")
print (f"This is the ood cis: \n {ood_gen_cis}")
print (f"This is the id means: \n {id_means}")
print (f"This is the id cis: \n {id_cis}")