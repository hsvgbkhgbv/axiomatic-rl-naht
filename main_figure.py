import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from matplotlib import rcParams


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

def select(interp_df):
    max_values = interp_df.groupby(['method', 'seed'])['smoothed'].max().reset_index()

    # 分别处理方法 a 和 方法 b
    top5_a = (
        max_values[max_values['method'] == 'Shapley Machine']
        .sort_values('smoothed', ascending=False)
        .head(5)
    )
    # top5_c = (
    #     max_values[max_values['method'] == 'Shapley Machine 3']
    #     .sort_values('smoothed', ascending=False)
    #     .head(5)
    # )

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

sns.set(style="darkgrid")

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

for scalar_tags, mode in scalar_data_mode:
    for experiment_config,output_dir in configs:
        
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5)) 
        
        # 收集所有图中存在的算法
        all_existing_methods = set()
        for map_name, paths in experiment_config.items():
            for path in paths:
                method = os.path.basename(path)
                # 应用相同的算法名称映射
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
            ax = axes[i]  # 每次使用一个子图
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

            # 确定该场景应该显示哪个scalar_tag
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
            
            # 处理数据插值和平滑
            # if map_name == "8v9":
            if map_name == "":
                expected_max_step = 17500000

                # 找出每个 method-seed 组合的最大 step
                step_counts = df.groupby(["method", "seed"])["step"].max().reset_index()
                # 只保留那些最大 step >= expected_max_step 的组合
                valid_combinations = step_counts[step_counts["step"] >= expected_max_step][["method", "seed"]]
                valid_combinations = df.merge(valid_combinations, on=["method", "seed"], how="inner")
                
                step_min = int(valid_combinations["step"].min())
                step_max = int(valid_combinations["step"].max())
                step_interval = int((step_max + 1 - step_min) / 400)
                step_range = np.arange(step_min, step_max + 1, step_interval)

                valid_combinations_interp = interpolate_seeds(valid_combinations, step_range)
                
                less_valid_combinations = step_counts[step_counts["step"] < expected_max_step][["method", "seed"]]
                less_valid_combinations = df.merge(less_valid_combinations, on=["method", "seed"], how="inner")
                
                step_range = np.arange(step_min, expected_max_step, step_interval)

                less_valid_combinations_interp = interpolate_seeds(less_valid_combinations, step_range)
                
                df_interp = pd.concat([valid_combinations_interp, less_valid_combinations_interp], ignore_index=True)
                
                seed_max_steps = df_interp.groupby(['method', 'seed'])['step'].max().reset_index()
                seed_max_steps.columns = ['method', 'seed', 'max_step']
            else:
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
            
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam-pqvmq'), 'method'] = 'POAM'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_naht1v-pqvmq'), 'method'] = 'IPPO-NAHT'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_naht-pqvmq'), 'method'] = 'IPPO-NAHT'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_shapley'), 'method'] = 'SM-POAM'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_shapley'), 'method'] = 'SM-IPPO'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('ippo_banzhaf'), 'method'] = 'BM-IPPO'
            df_interp_smooth.loc[df_interp_smooth['method'].str.contains('poam_banzhaf'), 'method'] = 'BM-POAM'
            
            # 定义所有可能的算法列表和固定颜色，确保颜色一致性
            all_methods = ['POAM', 'IPPO-NAHT', 'SM-POAM', 'SM-IPPO', 'BM-POAM', 'BM-IPPO']
            method_colors = {
                'POAM': '#1f77b4',      # 蓝色
                'IPPO-NAHT': '#ff7f0e',      # 橙色
                'SM-POAM': '#2ca02c',   # 绿色
                'SM-IPPO': '#d62728',   # 红色
                'BM-IPPO': '#9467bd',    # 紫色
                'BM-POAM': '#8c564b'     # 棕色
            }
            
            # 为每个算法添加空数据点（如果不存在），确保颜色一致性
            existing_methods = list(df_interp_smooth['method'].unique())
            
            # 为右边图（10v11）添加所有存在的算法，为其他图只添加预定义算法
            methods_to_add = all_existing_methods if map_name == "10v11" else all_methods
            for method in methods_to_add:
                if method not in existing_methods:
                    # 添加一个空的数据点，这样算法会出现在图例中并保持颜色一致性
                    empty_row = pd.DataFrame({
                        'step': [0],
                        'value': [np.nan],
                        'method': [method],
                        'seed': ['dummy'],
                        'smoothed': [np.nan]
                    })
                    df_interp_smooth = pd.concat([df_interp_smooth, empty_row], ignore_index=True)
            
            # 确保数据按预定义的方法顺序排序
            df_interp_smooth['method_order'] = df_interp_smooth['method'].map({method: i for i, method in enumerate(all_methods)})
            df_interp_smooth = df_interp_smooth.sort_values('method_order')
            
            print(f"Scene: {map_name}, Methods found: {existing_methods}") 

            # print(df_interp_smooth)
            # df_interp_smooth = select(df_interp_smooth)
            # print(df_interp_smooth)
            sns.lineplot(
                data=df_interp_smooth,
                x="step",
                y="smoothed",
                hue="method",
                hue_order=all_methods,  # 使用预定义的算法顺序
                palette=method_colors,  # 使用固定的颜色映射
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
                    # 显示所有图中存在的算法在图例中
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
                    # 显示所有图中存在的算法在图例中
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
                    # 显示所有图中存在的算法在图例中
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
                    # 显示所有图中存在的算法在图例中
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
            
            i += 1  # 处理完一个场景的所有scalar_tags后递增索引

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"main_{mode}.pdf")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
                
