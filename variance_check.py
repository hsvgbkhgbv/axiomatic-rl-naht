import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import rcParams

# ========= 参数 =========
configs = [
    ({
        # "mpe-pp": [
        #     "naht_results/mpe-pp/open_train/poam_banzhaf_truncate_Lam0.85_vl0.5_eps0.2_m7",
        #     "naht_results/mpe-pp/open_train/poam-pqvmq_open",
        # ],
        "5v6": [
            "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m31",
            "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m15_n4",
            "naht_results/5v6/open_train/poam_shapley1_truncate_Lam0.85_vl0.5_sr0.01_sl0.001_eps0.2_m7_n3"
        ],
    }, "./plots")
]

mode = "variance_assumption_sanity_check"
target_tag = "train_return_std"


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


def extract_scalar_data(event_file, tag):
    ea = event_accumulator.EventAccumulator(event_file)
    try:
        ea.Reload()
    except Exception:
        return []

    if tag in ea.Tags().get("scalars", []):
        return [(e.step, e.value) for e in ea.Scalars(tag)]
    return []


def interpolate_seeds(df, step_range, kind="nearest"):
    all_interp = []

    for (method, seed), group in df.groupby(["method", "seed"]):
        group = group.sort_values("step")
        if len(group) < 2:
            continue

        try:
            interp_func = interp1d(
                group["step"],
                group["value"],
                kind=kind,
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_vals = interp_func(step_range)
        except ValueError:
            continue

        interp_df = pd.DataFrame(
            {
                "step": step_range,
                "value": interp_vals,
                "method": method,
                "seed": seed,
            }
        )
        all_interp.append(interp_df)

    if not all_interp:
        return pd.DataFrame(columns=["step", "value", "method", "seed"])
    return pd.concat(all_interp, ignore_index=True)


def normalize_method_name(method):
    if "m31" in method:
        return "Exp A"
    if "m15_n4" in method:
        return "Exp B"
    if "m7_n3" in method:
        return "Exp C"
    return method


def build_squared_curve(df_interp):
    sq_df = df_interp.copy()
    sq_df["value_sq"] = sq_df["value"] ** 2
    return sq_df


def add_visual_offset(sq_df):
    # Visualization-only offsets to separate overlapping lines.
    offset_map = {
        "3 Agents": -0.12,
        "4 Agents": 0.00,
        "5 Agents": 0.12,
    }
    out = sq_df.copy()
    out["value_sq_plot"] = out["value_sq"] + out["method"].map(offset_map).fillna(0.0)
    return out


sns.set(style="darkgrid")
rcParams["font.weight"] = "normal"

for experiment_config, output_dir in configs:
    os.makedirs(output_dir, exist_ok=True)
    n_envs = len(experiment_config)
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    for i, (map_name, paths) in enumerate(experiment_config.items()):
        ax = axes[i]
        print(f"\nProcessing environment: {map_name}")
        rows = []

        for path in paths:
            method = normalize_method_name(os.path.basename(path))
            tb_logs_path = os.path.join(path, "tb_logs")
            event_files = find_all_event_files(tb_logs_path)

            if not event_files:
                print(f"Warning: no event files found in {path}")
                continue

            print(f"load {len(event_files)} event_files...")
            for ef in event_files:
                scalar_data = extract_scalar_data(ef, target_tag)
                for step, value in scalar_data:
                    rows.append(
                        {
                            "step": step,
                            "value": value,
                            "method": method,
                            "seed": ef,
                        }
                    )

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"No data found for {target_tag} in {map_name}")
            continue

        step_min = int(df["step"].min())
        step_max = int(df["step"].max())
        step_interval = max(1, int((step_max + 1 - step_min) / 400))
        step_range = np.arange(step_min, step_max + 1, step_interval)

        df_interp = interpolate_seeds(df, step_range, kind="nearest")
        if df_interp.empty:
            print(f"No interpolated data for {target_tag} in {map_name}")
            continue

        sq_df = build_squared_curve(df_interp)
        if sq_df.empty:
            print(f"No squared data for {target_tag} in {map_name}")
            continue
        sns.lineplot(
            data=sq_df,
            x="step",
            y="value_sq",
            hue="method",
            estimator="mean",
            errorbar=("ci", 95),
            err_style="band",
            linewidth=2,
            ax=ax,
        )

        labelsize = 14
        fontsize = 16
        ax.tick_params(axis="both", labelsize=labelsize)
        ax.set_xlim([0, 2e7])
        ax.set_yscale("linear")
        y_min = float(sq_df["value_sq"].min())
        y_max = float(sq_df["value_sq"].max())
        y_pad = max(0.3, 0.08 * (y_max - y_min + 1))
        y_min = y_min - y_pad
        y_max = y_max + y_pad
        ax.set_ylim(y_min, y_max)
        ax.set_title(map_name, fontsize=fontsize, color="black")
        ax.set_xlabel("Timesteps", fontsize=fontsize, color="black")

        if i == 0:
            ax.set_ylabel(
                "Mean Return Variance",
                fontsize=fontsize,
                color="black",
            )
        else:
            ax.set_ylabel(None)

        ax.legend(title=None, loc="best", prop={"size": fontsize - 1, "weight": "normal"})

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"main_{mode}.pdf")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
