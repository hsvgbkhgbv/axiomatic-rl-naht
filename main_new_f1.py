import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import rcParams
# ========= 参数 =========

configs = [
    ({
    "8v9": [
        "/scratch/project_2012831/naht_results/8v9/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.3_avg_m50",
        "naht_results/8v9/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m120"
    ],
    "10v11": [
        "/scratch/project_2012831/naht_results/10v11/open_train/poam_shapley1_truncate_revLam0.98_vl0.5_sr0.01_sl0.0001_eps0.1_m70",
        "naht_results/10v11/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
    ], 
    }, "./plots-naht")
]
     
# mode = "newf1_test_10v11" 
# mode = "newf1_test_8v9"  
mode = "new-f1-fig1"
# scalar tags

ema_alpha = 1 - 0.95

def smooth_dataframe(df, y_col="mean", ema_alpha=0.3):
    """
    对每个 method 的指定列进行 EMA 平滑。
    ema_alpha 越小，平滑效果越强。
    """
    return (
        df.sort_values("step")
          .groupby(["method", "seed"])
          .apply(lambda g: g.assign(
              smoothed=g[y_col].ewm(alpha=ema_alpha, adjust=False).mean()
          ))
          .reset_index(drop=True)
    )

# --------- 工具函数 ---------
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
    except:
        return {tag: [] for tag in tags}
    data = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            data[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
        else:
            data[tag] = []
    return data

from scipy.interpolate import interp1d
def interpolate_seeds(df, step_range, kind="nearest"):
    """
    对每个 method + seed 曲线插值到统一 step
    kind: 'linear', 'nearest', 'cubic', 'quadratic'...
    """
    all_interp = []

    for (method, seed), group in df.groupby(["method", "seed"]):
        group = group.sort_values("step")
        if len(group) < 2:
            continue  # 无法插值

        try:
            interp_func = interp1d(
                group["step"],
                group["value"],
                kind=kind,
                bounds_error=False,
                fill_value=np.nan  # 避免外推
            )
            interp_vals = interp_func(step_range)
        except ValueError:
            # 比如 cubic 插值点太少会失败
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

scalar_tags = {
    "Mean Test Return": "test_return_mean",
    # "Mean Shaped Reward": "shaped_rewards_mean",
    # "Mean Test Winrate": "test_battle_won_mean"
}


for experiment_config,output_dir in configs:
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
    i = 0
    for map_name, paths in experiment_config.items():
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
        if mode == "newf1_test_8v9" or mode == "newf1_test_10v11":
            for key in data_by_plot.keys():
                ax = axes[i]  # 每次使用一个子图
                i += 1
                print(map_name, key)
                df = pd.DataFrame(data_by_plot[key])

                if df.empty:
                    print(f"❌ No data found for {key} in {map_name}")
                    continue

                # === 对齐所有曲线 ===
                step_min = int(df["step"].min())
                step_max = int(df["step"].max())
                step_interval = int((step_max + 1 - step_min) / 400)
                step_range = np.arange(step_min, step_max + 1, step_interval)

                df_interp = interpolate_seeds(df, step_range)
                # df_interp.to_csv("interpolated_data.csv", index=False)

                # === 平滑处理 ===
                df_interp_smooth = smooth_dataframe(df_interp, y_col="value", ema_alpha=ema_alpha)
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m70'), 'method'] = 'w/o value loss clip'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m150'), 'method'] = 'w/ value loss clip'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m50'), 'method'] = 'w/o value loss clip'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m120'), 'method'] = 'w/ value loss clip'
                
                
                sns.lineplot(
                    data=df_interp_smooth,
                    x="step",
                    y="smoothed",
                    hue="method",
                    errorbar=('ci', 95),     # std deviation
                    estimator="mean",      # 求 mean
                    err_style="band",  # 默认就是带状阴影
                    linewidth=2,
                    ax=ax
                )
            
                
                labelsize = 14
                fontsize = 18

                if "10v11" in map_name:
                    ax.axvline(x=0.75e7, color='red', linestyle='--', linewidth=2)
                    ax.axvline(x=1e7, color='red', linestyle='--', linewidth=2)
                else:
                    ax.axvline(x=1.5e7, color='red', linestyle='--', linewidth=2)
                    ax.axvline(x=1.75e7, color='red', linestyle='--', linewidth=2)
                
                ax.tick_params(axis='both', labelsize=labelsize)
                ax.set_xlim([0, 2e7])
                ax.legend(title=None, loc="lower right", prop={'size': fontsize})

                ax.get_legend().remove()
                ax.yaxis.label.set_visible(False)
                ax.set_title(key, fontsize=fontsize, color='black')
                
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
        else:
            # for key in data_by_plot.keys():
            ax = axes[i]  # 每次使用一个子图
            i += 1
            print(map_name, key)
            df = pd.DataFrame(data_by_plot[key])

            if df.empty:
                print(f"❌ No data found for {key} in {map_name}")
                continue

            # === 对齐所有曲线 ===
            step_min = int(df["step"].min())
            step_max = int(df["step"].max())
            step_interval = int((step_max + 1 - step_min) / 400)
            step_range = np.arange(step_min, step_max + 1, step_interval)

            df_interp = interpolate_seeds(df, step_range)
            # df_interp.to_csv("interpolated_data.csv", index=False)

            # === 平滑处理 ===
            df_interp_smooth = smooth_dataframe(df_interp, y_col="value", ema_alpha=ema_alpha)
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m70'), 'method'] = 'w/o value loss clip'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m150'), 'method'] = 'w/ value loss clip'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m50'), 'method'] = 'w/o value loss clip'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m120'), 'method'] = 'w/ value loss clip'
            
            
            sns.lineplot(
                data=df_interp_smooth,
                x="step",
                y="smoothed",
                hue="method",
                errorbar=('ci', 95),     # std deviation
                estimator="mean",      # 求 mean
                err_style="band",  # 默认就是带状阴影
                linewidth=2,
                ax=ax
            )
        
            
            labelsize = 14
            fontsize = 18
            
            ax.tick_params(axis='both', labelsize=labelsize)
            ax.set_xlim([0, 2e7])
            ax.legend(title=None, loc="lower right", prop={'size': fontsize})
            if map_name == "8v9":
                ax.set_ylabel('Mean Test Return', fontsize=fontsize, color='black')
            else:   
                ax.set_ylabel(None)
            ax.set_title(map_name, fontsize=fontsize, color='black')
            ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"main_{mode}.pdf")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {save_path}")
            