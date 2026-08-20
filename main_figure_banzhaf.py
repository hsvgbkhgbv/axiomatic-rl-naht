import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import rcParams
# ========= 参数 =========

# mpe-pp/open_train/poam_shapley1_truncate_revLam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m8
# 3sv5z/open_train/poam_shapley1_truncate_revLam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m8
# 5v6/open_train/poam_shapley1_truncate_revLam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m20
# 8v9/open_train/poam_shapley1_truncate_revLam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m30
# 10v11/open_train/poam_shapley1_truncate_revLam0.98_vl0.5_sr0.01_sl0.0001_eps0.2_m70
# 10v11/open_train/poam_shapley1_truncate_revLam0.98_vl0.5_sr0.01_sl0.001_eps0.2_m70
# 3sv5z/open_train/poam_shapley1_truncate_revLam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m8

configs = [
    ({
    "mpe-pp": [
        "naht_results/mpe-pp/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
        "naht_results/mpe-pp/open_train/poam-pqvmq_open",
    ],
    "5v6":[
        "naht_results/5v6/open_train/poam-pqvmq_open",
        "naht_results/5v6/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31",
    ]}, "./plots")
]

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
rcParams['font.weight'] = 'bold'

scalar_tags = {
    "test_return": "test_return_mean",
    "test_winrate": "test_battle_won_mean",
}

# scalar_tags = {
#     "train_return": "train_return_mean",
#     "train_winrate": "train_battle_won_mean",
# }

# scalar_tags = {
#     "entropy": "entropy",
# }

# scalar_tags = {
#     "critic_loss": "critic_loss",
# }

mode = "banzhaf"

for experiment_config,output_dir in configs:
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    i = 0
    for map_name, paths in experiment_config.items():
        ax = axes[i]  # 每次使用一个子图
        i += 1
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
                # if "banzhaf" in ef:
                #     print(f"⚠️  Skip event file: {ef}")
                #     continue
                scalar_data = extract_scalar_data(ef, list(scalar_tags.values()))
                for key, tag in scalar_tags.items():
                    for step, value in scalar_data.get(tag, []):
                        data_by_plot[key].append({
                            "step": step,
                            "value": value,
                            "method": method,
                            "seed": ef
                        })

        for key in data_by_plot.keys():
            if map_name == "mpe-pp":
                if "return" in key or "entropy" in key or "critic_loss" in key:
                    pass
                else:
                    continue
            else:
                if "winrate" in key or "entropy" in key or "critic_loss" in key:
                    pass
                else:
                    continue
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
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('pqvmq_open'), 'method'] = 'POAM'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('banzhaf'), 'method'] = 'Banzhaf Machine'
            
            print(df_interp_smooth)
            sns.lineplot(
                data=df_interp_smooth,
                x="step",
                y="smoothed",
                hue="method",
                errorbar=('ci', 95),     # std deviation
                estimator="mean",  # 求 mean
                err_style="band",  # 默认就是带状阴影
                linewidth=2,
                ax=ax
            )
            labelsize = 14
            fontsize = 18
            ax.tick_params(axis='both', labelsize=labelsize)
            ax.set_xlim([0, 2e7])
            if 'return' in key:
                if map_name == "mpe-pp":
                    ax.set_ylabel('Mean Test Return', fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                elif map_name == "5v6":  
                    ax.set_ylabel('Mean Win Rate', fontsize=fontsize, fontdict={'weight': 'bold'}, color='black') 
                else:
                    ax.set_ylabel(None)
                ax.legend(title=None, loc="lower right", prop={'size': fontsize})
                ax.set_title(map_name, fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
            elif 'winrate' in key:
                if map_name == "5v6":
                    ax.set_ylabel('Mean Win Rate', fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                else:   
                    ax.set_ylabel(None)
                ax.legend(title=None, loc="lower right", prop={'size': fontsize})
                ax.set_title(map_name, fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"main_{mode}.pdf")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {save_path}")
            