import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import rcParams
from scipy.stats import t     # NEW
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
    # "mpe-pp": [
    #     "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
    #     "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.99_vl0.5_sr0.01_sl0.01_eps0.2_m100",
    "mpe-pp": [
        "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
        "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m3",
        "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m14",
        "naht_results/mpe-pp/open_train/poam_shapley1_truncate_Lam0.99_vl0.5_sr0.01_sl0.01_eps0.2_m100",
    ],
    "5v6":[
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31",
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.1_avg_m20",
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m70",
    ],
    "8v9": [
        "naht_results/8v9/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m120",
        "naht_results/8v9/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m20",
    ],
    # "10v11": [
    #     "naht_results/10v11/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m20",
    #     "naht_results/10v11/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150"
    # ], 
    }, "./plots-naht")
]

# mode = "f5"
mode = "m"
# mode = "rebutal-m"
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
        # NOTE: FOR REBUTTAL
        # m=7: 112358, 112360, 112361
        # if 'm7' in method:
        #     if ('112362' in seed) or ('112359' in seed):
        #         continue
        # else:
        #     if ('112358' in seed) or ('112359' in seed):
        #         continue
        # if ('112362' in seed) or ('112361' in seed):
        #     continue
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
    "test_return": "test_return_mean",
    "test_winrate": "test_battle_won_mean",
}

def print_table(df):
    BIN = 5_000_000           # 0.5 × 1e7

    # ── 0. Load the frame ─────────────────────────────────────────────
    # df = pd.read_csv("your_table.csv")

    # ── 1. Extract a clean seed ID (optional) ─────────────────────────
    df["seed_id"] = (
        df["seed"].str.extract(r"_t(\d+)", expand=False).astype("Int64")
    )

    metric = "smoothed"

    # ── 2. Aggregate by (method, step) across seeds ───────────────────
    agg = (
        df.groupby(["method", "step"])[metric]
        .agg(["count", "mean", "std"])      # std = sample std (ddof=1)
        .reset_index()
    )

    # 2‑b) 95 % CI using Student‑t if n < 30, z ≈ 1.96 otherwise
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg["t_mult"] = np.where(
        agg["count"] < 30,
        t.ppf(0.975, df=agg["count"] - 1),
        1.96
    )
    agg["ci95_lo"] = agg["mean"] - agg["t_mult"] * agg["sem"]
    agg["ci95_hi"] = agg["mean"] + agg["t_mult"] * agg["sem"]

    # ── 3. Down‑sample to nearest 5 M steps per method ────────────────
    checkpoints = pd.DataFrame({"target_step": np.arange(0, agg["step"].max()+BIN, BIN)})

    out = []
    for method, sub in agg.groupby("method", sort=False):
        sub = sub.sort_values("step")
        merged = pd.merge_asof(
            checkpoints, sub,
            left_on="target_step", right_on="step",
            direction="nearest", tolerance=BIN//2
        ).dropna(subset=["step"])              # discard targets with no close log
        merged["method"] = method
        out.append(merged)

    final = pd.concat(out, ignore_index=True)

    # ── 4. Final tidy frame ───────────────────────────────────────────
    cols = ["method", "target_step", "step", "mean", "ci95_lo", "ci95_hi", "count"]
    final = final[cols].rename(columns={"mean": "mean_"+metric})

    # ── 1.  Build a symmetric ± value ------------------------------------
    final["ci_half"] = final["ci95_hi"] - final["mean_smoothed"]    # half‑width

    # nice, compact string — tweak the .round(2) precision if you prefer
    final["mean ± CI"] = (
        final["mean_smoothed"].round(2).astype(str)
        + " ± "
        + final["ci_half"].round(2).astype(str)
    )

    # ── 2.  Long (tidy) table  -------------------------------------------
    tidy = (
        final.assign(step_M = final["target_step"] / 1e6)            # human units
            .loc[:, ["method", "step_M", "mean ± CI"]]
            .sort_values(["method", "step_M"])
    )

    # print(tidy)

    # ── 3.  Wide pivot (rows = step, cols = method) ----------------------
    wide = tidy.pivot(index="step_M", columns="method", values="mean ± CI").sort_index()

    # optional pretty print
    with pd.option_context("display.max_columns", None):
        print(wide)

for experiment_config,output_dir in configs:
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
    # fig, axes = plt.subplots(1, 1, figsize=(5, 5)) 
    i = 0
    for map_name, paths in experiment_config.items():
        ax = axes[i]  # 每次使用一个子图
        # ax = axes
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
                if "banzhaf" in ef:
                    print(f"⚠️  Skip event file: {ef}")
                    continue
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
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m20'), 'method'] = 'm=20'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m70'), 'method'] = 'm=70'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m31'), 'method'] = 'm=31'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m7'), 'method'] = 'm=7'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m3'), 'method'] = 'm=3'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m14'), 'method'] = 'm=14'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m100'), 'method'] = 'm=100'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m150'), 'method'] = 'm=150'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('_m120'), 'method'] = 'm=120'
            
            # === 根据配置路径顺序设置图例顺序 ===
            # 从配置路径中提取m值，保持原始顺序
            ordered_methods = []
            for path in paths:
                method_name = os.path.basename(path)
                if '_m20' in method_name:
                    ordered_methods.append('m=20')
                elif '_m70' in method_name:
                    ordered_methods.append('m=70')
                elif '_m31' in method_name:
                    ordered_methods.append('m=31')
                elif '_m7' in method_name:
                    ordered_methods.append('m=7')
                elif '_m3' in method_name:
                    ordered_methods.append('m=3')
                elif '_m14' in method_name:
                    ordered_methods.append('m=14')
                elif '_m100' in method_name:
                    ordered_methods.append('m=100')
                elif '_m150' in method_name:
                    ordered_methods.append('m=150')
                elif '_m120' in method_name:
                    ordered_methods.append('m=120')
            
            # 去重但保持顺序
            hue_order = []
            for method in ordered_methods:
                if method not in hue_order:
                    hue_order.append(method)
            
            
            
            # print(df_interp_smooth)
            print_table(df_interp_smooth)
            sns.lineplot(
                data=df_interp_smooth,
                x="step",
                y="smoothed",
                hue="method",
                hue_order=hue_order,  # 使用配置中定义的顺序
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
                    ax.set_ylabel('Mean Test Return', fontsize=fontsize, color='black')
                # elif map_name == "5v6":  
                #     ax.set_ylabel('Mean Test Winrate', fontsize=fontsize, color='black') 
                else:
                    ax.set_ylabel(None)
                ax.legend(title=None, loc="lower right", prop={'size': fontsize})
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')
            elif 'winrate' in key:
                # if map_name == "5v6" or map_name == "8v9":
                if map_name == "5v6":
                    ax.set_ylabel('Mean Test Winrate', fontsize=fontsize, color='black')
                else:   
                    ax.set_ylabel(None)
                ax.legend(title=None, loc="lower right", prop={'size': fontsize})
                ax.set_title(map_name, fontsize=fontsize, color='black')
                ax.set_xlabel("Timesteps", fontsize=fontsize, color='black')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"main_{mode}.pdf")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {save_path}")
            