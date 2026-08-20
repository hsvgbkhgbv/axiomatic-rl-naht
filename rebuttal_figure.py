import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import rcParams
from scipy.interpolate import interp1d
from scipy.stats import t     # NEW
# ========= 参数 =========

configs = [
    ({
    # "mpe-pp": [
    #     # "naht_results/mpe-pp/open_train/ippo_poam-pqvmq_open",
    #     # "naht_results/mpe-pp/open_train/ippo_shapley_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7"
    #     "naht_results/mpe-pp/open_train/ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7",
    #     "naht_results/mpe-pp/open_train/ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7"
    # ],
    "5v6":[
        # "naht_results/5v6/open_train/ippo_poam-pqvmq_open",
        "naht_results/5v6/open_train/ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31",
        "naht_results/5v6/open_train/ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31",
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31",
        "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31",
        "naht_results/5v6/open_train/ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31",
        "naht_results/5v6/open_train/ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31",
        "naht_results/5v6/open_train/ippo_naht-pqvmq_open",
    ]
    # "8v9":[
    #     # "naht_results/8v9/open_train/ippo_naht-pqvmq_open",
    #     # "naht_results/8v9/open_train/ippo_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m120"
    #     "naht_results/8v9/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m120",
    #     # "naht_results/8v9/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m120",
    #     # "naht_results/8v9/open_train/ippo_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m120",
    #     "naht_results/8v9/open_train/ippo_naht1v-pqvmq_open",
    # ]
    # "3sv5z":[
    #     "naht_results/3sv5z/open_train/ippo_naht-pqvmq_open",
    #     "naht_results/3sv5z/open_train/ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.01_eps0.2_m7"
    # ]
    # "10v11":[
    #     # "naht_results/10v11/open_train/ippo_naht-pqvmq_open",
    #     # "naht_results/10v11/open_train/ippo_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
    #     # "naht_results/10v11/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
    #     # "naht_results/10v11/open_train/ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_m150",
    #     # "naht_results/10v11/open_train/ippo_naht1v-pqvmq_open",
    #     "naht_results/10v11/open_train/poam_shapley1_episode_length_Lam0.98_vc_m200",
    #     "naht_results/10v11/open_train/poam_shapley1_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150",
    # ]
}, "./rebuttal_plots"),
]

scalar_data_mode = [
    ({
    # "test_return": "test_return_mean",
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

# scalar tags

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

def select(interp_df):
    max_values = interp_df.groupby(['method', 'seed'])['smoothed'].max().reset_index()

    # 分别处理方法 a 和 方法 b
    top5_a = (
        max_values[max_values['method'] == 'Shapley Machine']
        .sort_values('smoothed', ascending=False)
        .head(5)
    )

    bottom5_b = (
        max_values[max_values['method'] == 'POAM']
        .sort_values('smoothed', ascending=True)
        .head(5)
    )

    # 合并两个方法的 top/bottom 结果
    selected_seeds = pd.concat([top5_a, bottom5_b])

    # 用筛选出的 method 和 seed 过滤原始数据
    filtered_df = interp_df.merge(selected_seeds[['method', 'seed']], on=['method', 'seed'])
    return filtered_df

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

sns.set(style="darkgrid")
rcParams['font.weight'] = 'bold'

ema_alpha = 1 - 0.95

for scalar_tags, mode in scalar_data_mode:
    for experiment_config,output_dir in configs:
        
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5)) 

        for map_name, paths in experiment_config.items():

            # Loading data
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
            
            for key in data_by_plot.keys():
                print(map_name, key)
                df = pd.DataFrame(data_by_plot[key])

                if df.empty:
                    print(f"❌ No data found for {key} in {map_name}")
                    continue
                
                # 统计每个 (method, seed) 的最大 step
                seed_max_steps = df.groupby(['method', 'seed'])['step'].max().reset_index()
                seed_max_steps.columns = ['method', 'seed', 'max_step']
                
                # === 对齐所有曲线 ===
                step_min = int(df["step"].min())
                step_max = int(df["step"].max())
                step_interval = int((step_max + 1 - step_min) / 400)
                step_range = np.arange(step_min, step_max + 1, step_interval)

                df_interp = interpolate_seeds(df, step_range)

                # === 平滑处理 ===
                df_interp_smooth = smooth_dataframe(df_interp, y_col="value", ema_alpha=ema_alpha)
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31'), 'method'] = 'SM'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31'), 'method'] = 'SM AVG'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley2v_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31'), 'method'] = 'SM2V AVG'

                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31'), 'method'] = 'SM POAM'
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31'), 'method'] = 'SM POAM AVG'

                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31'), 'method'] = 'Banzhaf Machine'
                
                df_interp_smooth.loc[df_interp_smooth['method'].str.contains('pqvmq_open'), 'method'] = 'IPPO'
                # df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley2v_truncate_Lam0.95_vl0.5_sr0.01_sl0.001_eps0.2_vc_m150'), 'method'] = 'Shapley Machine 2'
                # df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_avg_m31'), 'method'] = 'Shapley Machine 1'
                # df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m31'), 'method'] = 'Banzhaf Machine'
                
                # df_interp_smooth = select(df_interp_smooth)

                print_table(df_interp_smooth)

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

                # Legend
                if 'return' in key:
                    ax.set_ylabel('Mean Test Return', fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                    ax.set_title(map_name, fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                    ax.set_xlabel("Timesteps", fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                elif 'winrate' in key:
                    ax.set_ylabel('Mean Win Rate', fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                    ax.set_title(map_name, fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')
                    ax.set_xlabel("Timesteps", fontsize=fontsize, fontdict={'weight': 'bold'}, color='black')

                ax.legend(title=None, loc="lower right", prop={'size': fontsize})

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"rebuttal_{mode}_ippo_compare_5v6.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
                
