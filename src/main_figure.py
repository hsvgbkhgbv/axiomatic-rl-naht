import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from scipy.interpolate import interp1d


configs = [
    ({
    "mpe-pp": [
        "naht_results/mpe-pp/open_train/poam-pqvmq_open",
        "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
        "naht_results/mpe-pp/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
        "naht_results/mpe-pp/open_train/ippo_naht1v-pqvmq_open",
        "naht_results/mpe-pp/open_train/ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
        "naht_results/mpe-pp/open_train/ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
    ],
    "5v6":[
        "naht_results/5v6/open_train/poam-pqvmq_open",
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31",
        "naht_results/5v6/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31",
        "naht_results/5v6/open_train/ippo_naht1v-pqvmq_open",
        "naht_results/5v6/open_train/ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31",
        "naht_results/5v6/open_train/ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31",
    ],
    "3sv5z":[
        "naht_results/3sv5z/open_train/poam-pqvmq_open",
        "naht_results/3sv5z/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
        "naht_results/3sv5z/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
        "naht_results/3sv5z/open_train/ippo_naht1v-pqvmq_open",
        "naht_results/3sv5z/open_train/ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
        "naht_results/3sv5z/open_train/ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
    ],
    "8v9":[
        "naht_results/8v9/open_train/poam-pqvmq_open",
        "naht_results/8v9/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m120",
        "naht_results/8v9/open_train/ippo_naht1v-pqvmq_open",
        "naht_results/8v9/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m120",
    ],
    "10v11":[
        "naht_results/10v11/open_train/poam-pqvmq_open",
        "naht_results/10v11/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
        "naht_results/10v11/open_train/ippo_naht1v-pqvmq_open",
        "naht_results/10v11/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
    ]
}, "./plots-naht"),
]

scalar_data_mode = [
    ({
    "test_return": "test_return_mean",
    "test_winrate": "test_battle_won_mean",
    }, "test"),
    # ({
    # "train_return": "train_return_mean",
    # "train_winrate": "train_battle_won_mean",
    # }, "train"), 
    # ({
    # "entropy": "entropy",
    # }, "entropy"),
    # ({
    # "critic_loss": "critic_loss",
    # }, "critic_loss"),
]

ema_alpha = 1 - 0.95

def smooth_dataframe(df, y_col="mean", ema_alpha=0.3):
    """
    Apply EMA smoothing to the specified column for each method.
    A smaller ema_alpha produces stronger smoothing.
    """
    return (
        df.sort_values("step")
          .groupby(["method", "seed"])
          .apply(lambda g: g.assign(
              smoothed=g[y_col].ewm(alpha=ema_alpha, adjust=False).mean()
          ))
          .reset_index(drop=True)
    )

# --------- Utility functions ---------
def find_all_event_files(tb_log_dir):
    if not os.path.isdir(tb_log_dir):
        return []
    event_files = []
    for subdir in os.listdir(tb_log_dir):
        subdir_path = os.path.join(tb_log_dir, subdir)
        if os.path.isdir(subdir_path):
            found = glob.glob(os.path.join(subdir_path, "events.out.tfevents.*"))
            if found:
                event_files.append(found[0])
    return event_files

def extract_scalar_data(event_file, tags):
    ea = event_accumulator.EventAccumulator(event_file)
    try:
        ea.Reload()
    except Exception:
        return {tag: [] for tag in tags}
    data = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            data[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
        else:
            data[tag] = []
    return data

def interpolate_seeds(df, step_range, kind="nearest"):
    """
    Interpolate each method-and-seed curve onto a common step range.
    kind: 'linear', 'nearest', 'cubic', 'quadratic'...
    """
    all_interp = []

    for (method, seed), group in df.groupby(["method", "seed"]):
        group = group.sort_values("step")
        if len(group) < 2:
            continue  # Insufficient data for interpolation

        try:
            interp_func = interp1d(
                group["step"],
                group["value"],
                kind=kind,
                bounds_error=False,
                fill_value=np.nan  # Avoid extrapolation
            )
            interp_vals = interp_func(step_range)
        except ValueError:
            # Cubic interpolation, for example, fails with too few points.
            continue

        interp_df = pd.DataFrame({
            "step": step_range,
            "value": interp_vals,
            "method": method,
            "seed": seed
        })
        all_interp.append(interp_df)

    return pd.concat(all_interp, ignore_index=True)

sns.set(style="darkgrid")

for scalar_tags, mode in scalar_data_mode:
    for experiment_config,output_dir in configs:
        
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5)) 
        
        # Collect the algorithms present across all panels.
        all_existing_methods = set()
        for map_name, paths in experiment_config.items():
            for path in paths:
                method = os.path.basename(path)
                # Apply the same algorithm-name mapping used for plotting.
                if 'poam-pqvmq' in method:
                    method = 'POAM'
                elif 'ippo_naht1v-pqvmq' in method or 'ippo_naht-pqvmq' in method:
                    method = 'IPPO-NAHT'
                elif 'poam_shapley' in method:
                    method = 'SM-POAM'
                elif 'ippo_shapley' in method:
                    method = 'SM-IPPO'
                elif 'ippo_banzhaf' in method:
                    method = 'BM-IPPO'
                elif 'poam_banzhaf' in method:
                    method = 'BM-POAM'
                all_existing_methods.add(method)
        
        i = 0
        for map_name, paths in experiment_config.items():
            ax = axes[i]  # Use one subplot for each environment
            print(f"\n📍 Processing environment: {map_name}")
            data_by_plot = {key: [] for key in scalar_tags}

            for path in paths:
                method = os.path.basename(path)
                tb_logs_path = os.path.join(path, "tb_logs")
                event_files = find_all_event_files(tb_logs_path)

                if not event_files:
                    print(f"⚠️  No event files found in {path}")
                    continue
                print(f"load {len(event_files)} event_files....")
                # print(event_files)
                for ef in event_files:
                    scalar_data = extract_scalar_data(ef, list(scalar_tags.values()))
                    for key, tag in scalar_tags.items():
                        for step, value in scalar_data.get(tag, []):
                            data_by_plot[key].append({
                                "step": step,
                                "value": value,
                                "method": method,
                                "seed": ef
                            })

            # Select the scalar appropriate for this environment.
            key_to_plot = None
            if map_name == "mpe-pp":
                if "test_return" in data_by_plot and data_by_plot["test_return"]:
                    key_to_plot = "test_return"
                elif "train_return" in data_by_plot and data_by_plot["train_return"]:
                    key_to_plot = "train_return"
                elif "entropy" in data_by_plot and data_by_plot["entropy"]:
                    key_to_plot = "entropy"
                elif "critic_loss" in data_by_plot and data_by_plot["critic_loss"]:
                    key_to_plot = "critic_loss"
            else:
                if "test_winrate" in data_by_plot and data_by_plot["test_winrate"]:
                    key_to_plot = "test_winrate"
                elif "train_winrate" in data_by_plot and data_by_plot["train_winrate"]:
                    key_to_plot = "train_winrate"
                elif "entropy" in data_by_plot and data_by_plot["entropy"]:
                    key_to_plot = "entropy"
                elif "critic_loss" in data_by_plot and data_by_plot["critic_loss"]:
                    key_to_plot = "critic_loss"
            
            if key_to_plot is None:
                print(f"❌ No suitable data found for {map_name}")
                i += 1
                continue
                
            print(map_name, key_to_plot)
            df = pd.DataFrame(data_by_plot[key_to_plot])

            if df.empty:
                print(f"❌ No data found for {key_to_plot} in {map_name}")
                i += 1
                continue
            
            # Align all curves onto a common timestep grid.
            step_min = int(df["step"].min())
            step_max = int(df["step"].max())
            step_interval = int((step_max + 1 - step_min) / 400)
            step_range = np.arange(step_min, step_max + 1, step_interval)
            df_interp = interpolate_seeds(df, step_range)

            # Apply smoothing.
            df_interp_smooth = smooth_dataframe(df_interp, y_col="value", ema_alpha=ema_alpha)
            
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam-pqvmq'), 'method'] = 'POAM'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_naht1v-pqvmq'), 'method'] = 'IPPO-NAHT'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_naht-pqvmq'), 'method'] = 'IPPO-NAHT'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_shapley'), 'method'] = 'SM-POAM'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley'), 'method'] = 'SM-IPPO'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_banzhaf'), 'method'] = 'BM-IPPO'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_banzhaf'), 'method'] = 'BM-POAM'
            
            # Use a fixed method list and palette for consistent colors.
            all_methods = ['POAM', 'IPPO-NAHT', 'SM-POAM', 'SM-IPPO', 'BM-POAM', 'BM-IPPO']
            method_colors = {
                'POAM': '#1f77b4',       # Blue
                'IPPO-NAHT': '#ff7f0e',  # Orange
                'SM-POAM': '#2ca02c',    # Green
                'SM-IPPO': '#d62728',    # Red
                'BM-IPPO': '#9467bd',    # Purple
                'BM-POAM': '#8c564b'     # Brown
            }
            
            # Add placeholders for absent methods to preserve color assignments.
            existing_methods = list(df_interp_smooth['method'].unique())
            
            # The final panel includes only methods present in at least one panel.
            methods_to_add = all_existing_methods if map_name == "10v11" else all_methods
            for method in methods_to_add:
                if method not in existing_methods:
                    # A NaN placeholder adds the method to the legend without drawing data.
                    empty_row = pd.DataFrame({
                        'step': [0],
                        'value': [np.nan],
                        'method': [method],
                        'seed': ['dummy'],
                        'smoothed': [np.nan]
                    })
                    df_interp_smooth = pd.concat([df_interp_smooth, empty_row], ignore_index=True)
            
            # Sort methods in the predefined order.
            df_interp_smooth['method_order'] = df_interp_smooth['method'].map({method: i for i, method in enumerate(all_methods)})
            df_interp_smooth = df_interp_smooth.sort_values('method_order')
            
            print(f"Scene: {map_name}, Methods found: {existing_methods}") 

            sns.lineplot(
                data=df_interp_smooth,
                x="step",
                y="smoothed",
                hue="method",
                hue_order=all_methods,  # Use the predefined method order
                palette=method_colors,  # Use the fixed color mapping
                errorbar=('ci', 95),     # std deviation
                estimator="mean",  # Compute the mean
                err_style="band",  # Display uncertainty as a shaded band
                linewidth=2,
                ax=ax
            )
            labelsize = 14
            fontsize = 18
            ax.tick_params(axis='both', labelsize=labelsize)
            ax.set_xlim([0, 2e7])
            if 'return' in key_to_plot:
                if map_name == "mpe-pp":
                    if key_to_plot == "train_return":
                        ax.set_ylabel('Mean Train Return', fontsize=fontsize, color='black')
                    else:
                        ax.set_ylabel('Mean Test Return', fontsize=fontsize, color='black')
                elif map_name == "5v6":  
                    if key_to_plot == "train_winrate":
                        ax.set_ylabel('Mean Train Winrate', fontsize=fontsize, color='black')
                    else:
                        ax.set_ylabel('Mean Test Winrate', fontsize=fontsize, color='black') 
                else:
                    ax.set_ylabel(None)
                if map_name == "10v11":
                    # Show all methods present across the figure in the final legend.
                    handles, labels = ax.get_legend_handles_labels()
                    filtered_handles = []
                    filtered_labels = []
                    for handle, label in zip(handles, labels):
                        if label in all_existing_methods:
                            filtered_handles.append(handle)
                            filtered_labels.append(label)
                    ax.legend(filtered_handles, filtered_labels, title=None, loc="lower right", prop={'size': fontsize})
                else:
                    ax.legend().remove()
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
            elif 'winrate' in key_to_plot:
                if map_name == "5v6":
                    if key_to_plot == "train_winrate":
                        ax.set_ylabel('Mean Train Winrate', fontsize=fontsize, color='black')
                    else:
                        ax.set_ylabel('Mean Test Winrate', fontsize=fontsize, color='black')
                else:   
                    ax.set_ylabel(None)
                if map_name == "10v11":
                    # Show all methods present across the figure in the final legend.
                    handles, labels = ax.get_legend_handles_labels()
                    filtered_handles = []
                    filtered_labels = []
                    for handle, label in zip(handles, labels):
                        if label in all_existing_methods:
                            filtered_handles.append(handle)
                            filtered_labels.append(label)
                    ax.legend(filtered_handles, filtered_labels, title=None, loc="lower right", prop={'size': fontsize})
                else:
                    ax.legend().remove()
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
            elif 'entropy' in key_to_plot:
                if map_name == "mpe-pp":
                    ax.set_ylabel('Entropy', fontsize=fontsize, color='black')
                else:   
                    ax.set_ylabel(None)
                if map_name == "10v11":
                    # Show all methods present across the figure in the final legend.
                    handles, labels = ax.get_legend_handles_labels()
                    filtered_handles = []
                    filtered_labels = []
                    for handle, label in zip(handles, labels):
                        if label in all_existing_methods:
                            filtered_handles.append(handle)
                            filtered_labels.append(label)
                    ax.legend(filtered_handles, filtered_labels, title=None, loc="upper right", prop={'size': fontsize})
                else:
                    ax.legend().remove()
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
            elif 'critic_loss' in key_to_plot:
                if map_name == "mpe-pp":
                    ax.set_ylabel('Critic Loss', fontsize=fontsize, color='black')
                else:   
                    ax.set_ylabel(None)
                if map_name == "10v11":
                    # Show all methods present across the figure in the final legend.
                    handles, labels = ax.get_legend_handles_labels()
                    filtered_handles = []
                    filtered_labels = []
                    for handle, label in zip(handles, labels):
                        if label in all_existing_methods:
                            filtered_handles.append(handle)
                            filtered_labels.append(label)
                    ax.legend(filtered_handles, filtered_labels, title=None, loc="upper right", prop={'size': fontsize})
                else:
                    ax.legend().remove()
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
            
            i += 1  # Advance after processing one environment.

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"main_{mode}.pdf")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
                
