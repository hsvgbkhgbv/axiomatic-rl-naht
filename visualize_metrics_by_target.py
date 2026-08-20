import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Try to import scipy for t-distribution, use standard normal if unavailable
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# 配置：在这里设置要处理的目录路径
# ============================================================================
# 场景目录映射：键是场景名称，值是对应的目录路径
# 每个目录应该包含 all_mean_performance_by_n.json 文件
SCENARIO_DIRS = {
    "mpe-pp": "/scratch/project_2012831/naht-dev/naht_results/mpe-pp",
    "3sv5z": "/scratch/project_2012831/naht-dev/naht_results/3sv5z",
    "5v6": "/scratch/project_2012831/naht-dev/naht_results/5v6",
    "8v9": "/scratch/project_2012831/naht-dev/naht_results/8v9",
    "10v11": "/scratch/project_2012831/naht-dev/naht_results/10v11",
    # 可以添加更多场景...
}

# 指定要处理的子目录（subdir），会被拼接到SCENARIO_DIRS的每个路径后面
# 例如: TARGET_SUBDIR = "type_generalization_poam"
# 如果为None，则直接使用SCENARIO_DIRS中的路径
# TARGET_SUBDIR = "ood_generalization_main"  # None表示不使用子目录
TARGET_SUBDIR = "type_generalization_ippo"

# 指定要过滤的JSON数据中的子目录键（可选），如果为None则处理所有子目录
# 例如: FILTER_SUBDIRS = ["poam_shapley1_eps0.2-vs-ippo", "poam_banzhaf_eps0.2-vs-ippo"]
FILTER_SUBDIRS = None  # None表示处理所有子目录

# Algorithm name mapping (from ood_figure_type.py)
EXP_NAMEMAP = {
    "poam_shapley1_eps0.2": "SM-POAM",
    "poam_banzhaf_eps0.2": "BM-POAM",
    "poam_type": "POAM",
    "ippo_shapley2v_eps0.2": "SM-IPPO",
    "ippo_banzhaf_eps0.2": "BM-IPPO",
    "ippo_naht1v_type": "IPPO-NAHT",
}
# EXP_NAMEMAP = { # format is alg log name: display name
#         "poam_shapley1_eps0.2": "SM-POAM",
#         "poam_banzhaf_eps0.2": "BM-POAM",
#         "poam-pqvmq_open": "POAM",
#         "ippo_shapley2v_eps0.2": "SM-IPPO",
#         "ippo_banzhaf_eps0.2": "BM-IPPO",
#         "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
#         "poam_shapley1_eps0.2": "SM-POAM",
#         "ippo_shapley2v_eps0.2": "SM-IPPO",
#         "poam_banzhaf_eps0.2": "BM-POAM",
#         "ippo_banzhaf_eps0.2": "BM-IPPO",
#         "poam-pqvmq_open": "POAM",
#         "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
# }

# Output configuration
# Directory for saving the figure (if None, saves to first scenario's directory)
OUTPUT_DIR = "/scratch/project_2012831/naht-dev/plots-naht/iclr-rebuttal"  # 例如: "/scratch/project_2012831/naht-dev/plots-naht/5trials-95ci"

# Figure name prefix (the part before target algorithm name in the filename)
# Final filename will be: {FIGURE_NAME_PREFIX}_{target_algorithm}_by_n.pdf
# For example: "target_ippo_by_n.pdf" if FIGURE_NAME_PREFIX = "target"
FIGURE_NAME_PREFIX = "type_ippo"  # 例如: "target", "ood", "type", etc.

# Output format settings
OUTPUT_DPI = 600  # DPI for PDF output

# Statistical settings
# Number of samples used to calculate each mean value (for each random seed)
N_SAMPLES_PER_SEED = 128  # Set to None if values are not means of multiple samples
# ============================================================================

def load_results(json_file):
    """Load all_mean_performance_by_n.json file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Cannot load file {json_file}: {e}")
        return None

def extract_target_algorithm(subdir_name):
    """
    Extract target algorithm name from subdir name (the part after "-vs-")
    
    Args:
        subdir_name: Subdirectory name, e.g., "poam_banzhaf_eps0.2-vs-ippo"
        
    Returns:
        target_algorithm: Target algorithm name, e.g., "ippo"
    """
    if '-vs-' in subdir_name:
        return subdir_name.split('-vs-')[1]
    return None

def extract_eval_algorithm(subdir_name):
    """
    Extract evaluation algorithm name from subdir name (the part before "-vs-")
    
    Args:
        subdir_name: Subdirectory name, e.g., "poam_banzhaf_eps0.2-vs-ippo"
        
    Returns:
        eval_algorithm: Evaluation algorithm name, e.g., "poam_banzhaf_eps0.2"
    """
    if '-vs-' in subdir_name:
        return subdir_name.split('-vs-')[0]
    return subdir_name

def determine_metric_name(json_file_path):
    """
    Determine the metric name based on JSON file path
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        metric_name: Metric name ('test_return_mean' or 'test_battle_won_mean')
    """
    if 'mpe-pp' in json_file_path:
        return 'test_return_mean'
    else:
        return 'test_battle_won_mean'

def get_y_label(json_file_path):
    """
    Get y-axis label based on scenario
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        y_label: Y-axis label string
    """
    if 'mpe-pp' in json_file_path:
        return 'Mean Test Return'
    else:
        return 'Mean Test Winrate'

def extract_scenario_name(json_file_path):
    """
    Extract scenario name from JSON file path
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        scenario_name: Scenario name (e.g., 'mpe-pp', '10v11', '3sv5z')
    """
    scenarios = ['mpe-pp', '3sv5z', '5v6', '8v9', '10v11']
    
    for scenario in scenarios:
        if scenario in json_file_path:
            return scenario
    
    path_parts = json_file_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part == 'naht_results' and i + 1 < len(path_parts):
            return path_parts[i + 1]
    
    return 'Unknown'

def merge_results_by_eval_algorithm(results, metric_name, target_algorithm):
    """
    Merge results by evaluation algorithm for a specific target algorithm
    
    Args:
        results: Original results dictionary
        metric_name: Metric name
        target_algorithm: Target algorithm name (e.g., "ippo")
        
    Returns:
        merged_data: Merged data, format: {eval_algorithm: {n_value: {mean, ci_68_lower, ci_68_upper, values}}}
    """
    merged_data = {}
    
    for subdir, n_data in results.items():
        # Check if this subdir matches the target algorithm
        target = extract_target_algorithm(subdir)
        if target != target_algorithm:
            continue
        
        # Extract evaluation algorithm name
        eval_alg_name = extract_eval_algorithm(subdir)
        
        # Use mapping table to get display name
        display_name = EXP_NAMEMAP.get(eval_alg_name, eval_alg_name)
        
        if display_name not in merged_data:
            merged_data[display_name] = {}
        
        # Merge data for each n value
        for n_value, metrics_data in n_data.items():
            if metric_name not in metrics_data:
                continue
            
            if n_value not in merged_data[display_name]:
                merged_data[display_name][n_value] = {
                    'values': []
                }
            
            # Merge all values
            metric_stats = metrics_data[metric_name]
            merged_data[display_name][n_value]['values'].extend(metric_stats['values'])
    
    # Calculate merged statistics
    for alg_name in merged_data:
        for n_value in merged_data[alg_name]:
            values = merged_data[alg_name][n_value]['values']
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                count = len(values)
                
                # Calculate 68% confidence interval
                if HAS_SCIPY and count > 1:
                    t_critical = scipy_stats.t.ppf(0.84, df=count-1)  # 0.84 = (1 + 0.68) / 2
                else:
                    t_critical = 1.0  # Standard normal approximation for 68% CI
                
                standard_error = std / np.sqrt(count)
                margin = t_critical * standard_error
                
                merged_data[alg_name][n_value]['mean'] = mean
                merged_data[alg_name][n_value]['std'] = std
                merged_data[alg_name][n_value]['count'] = count
                merged_data[alg_name][n_value]['ci_68_lower'] = mean - margin
                merged_data[alg_name][n_value]['ci_68_upper'] = mean + margin
    
    return merged_data

def plot_by_target_algorithm(results, json_file_path, target_algorithm, output_file=None):
    """
    Plot metrics for a specific target algorithm, with each scenario as a subplot
    
    Args:
        results: Results data loaded from JSON file
        json_file_path: JSON file path
        target_algorithm: Target algorithm name (e.g., "ippo")
        output_file: Output image file path, None means display
    """
    metric_name = determine_metric_name(json_file_path)
    scenario_name = extract_scenario_name(json_file_path)
    y_label = get_y_label(json_file_path)
    
    # Filter subdirs if specified
    if FILTER_SUBDIRS is not None:
        filtered_results = {k: v for k, v in results.items() if k in FILTER_SUBDIRS}
    else:
        filtered_results = results
    
    # Merge results by evaluation algorithm for this target
    merged_data = merge_results_by_eval_algorithm(filtered_results, metric_name, target_algorithm)
    
    if not merged_data:
        print(f"  No data found for target algorithm: {target_algorithm}")
        return
    
    # This function will be called for each scenario, so we'll collect data
    # and plot all scenarios together in the main function
    return merged_data, scenario_name, y_label

def plot_all_targets_grouped(scenario_data_dict):
    """
    Plot all scenarios grouped by target algorithm
    
    Args:
        scenario_data_dict: Dictionary with scenario names as keys and (results, json_path) tuples as values
    """
    # Group data by target algorithm
    target_groups = {}
    
    for scenario_name, (results, json_path) in scenario_data_dict.items():
        metric_name = determine_metric_name(json_path)
        
        # Filter subdirs if specified
        if FILTER_SUBDIRS is not None:
            filtered_results = {k: v for k, v in results.items() if k in FILTER_SUBDIRS}
        else:
            filtered_results = results
        
        # Find all target algorithms in this scenario
        for subdir in filtered_results.keys():
            target_alg = extract_target_algorithm(subdir)
            if target_alg is None:
                continue
            
            if target_alg not in target_groups:
                target_groups[target_alg] = {}
            
            # Merge results for this target algorithm
            merged_data = merge_results_by_eval_algorithm(filtered_results, metric_name, target_alg)
            if merged_data:
                target_groups[target_alg][scenario_name] = {
                    'data': merged_data,
                    'json_path': json_path
                }
    
    # Plot each target algorithm in a separate figure
    for target_alg, scenario_data in target_groups.items():
        if not scenario_data:
            continue
        
        num_scenarios = len(scenario_data)
        
        # Create figure with subplots in a row
        fig, axes = plt.subplots(1, num_scenarios, figsize=(6 * num_scenarios, 5))
        
        if num_scenarios == 1:
            axes = [axes]
        
        # Get all evaluation algorithms across all scenarios for consistent colors
        all_algorithms = set()
        for scenario_name, data_dict in scenario_data.items():
            all_algorithms.update(data_dict['data'].keys())
        
        all_algorithms = sorted(all_algorithms)
        
        # Generate colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(all_algorithms), 10)))
        if len(all_algorithms) > 10:
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(all_algorithms) - 10))
            colors = np.concatenate([colors, colors2])
        
        algorithm_color_map = {alg: colors[i] for i, alg in enumerate(all_algorithms)}
        
        # Sort scenarios according to SCENARIO_DIRS order
        # Only include scenarios that exist in scenario_data, but maintain SCENARIO_DIRS order
        ordered_scenarios = []
        for scenario_name in SCENARIO_DIRS.keys():
            if scenario_name in scenario_data:
                ordered_scenarios.append((scenario_name, scenario_data[scenario_name]))
        
        # Process each scenario in the order of SCENARIO_DIRS
        for idx, (scenario_name, data_dict) in enumerate(ordered_scenarios):
            ax = axes[idx]
            merged_data = data_dict['data']
            json_path = data_dict['json_path']
            y_label = get_y_label(json_path)
            
            scenario_n_values = set()
            
            # Process each evaluation algorithm
            for alg_name in sorted(merged_data.keys()):
                alg_data = merged_data[alg_name]
                
                n_values = []
                means = []
                errors_lower = []
                errors_upper = []
                
                for n_value in sorted(alg_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                    if 'mean' not in alg_data[n_value]:
                        continue
                    
                    n_val = int(n_value)
                    n_values.append(n_val)
                    scenario_n_values.add(n_val)
                    
                    mean = alg_data[n_value]['mean']
                    ci_lower = alg_data[n_value]['ci_68_lower']
                    ci_upper = alg_data[n_value]['ci_68_upper']
                    
                    means.append(mean)
                    errors_lower.append(mean - ci_lower)
                    errors_upper.append(ci_upper - mean)
                
                if not n_values:
                    continue
                
                n_values = np.array(n_values)
                means = np.array(means)
                errors_lower = np.array(errors_lower)
                errors_upper = np.array(errors_upper)
                
                color = algorithm_color_map[alg_name]
                
                # Plot line
                ax.plot(n_values, means, 'o-', color=color, 
                       linewidth=2, markersize=6, label=alg_name)
                
                # Plot error bars
                ax.errorbar(n_values, means, 
                           yerr=[errors_lower, errors_upper],
                           fmt='none',
                           color=color,
                           capsize=5,
                           capthick=1.5,
                           elinewidth=1.5,
                           alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('N', fontsize=20)
            if idx == 0:
                ax.set_ylabel(y_label, fontsize=20)
            elif idx == 1:
                ax.set_ylabel('Mean Test Winrate', fontsize=20)
            ax.set_title(scenario_name, fontsize=22)  # Removed fontweight='bold'
            ax.legend(loc='best', fontsize=18)
            ax.grid(True, alpha=0.3)
            
            # Increase tick label font size
            ax.tick_params(labelsize=18)
            
            if scenario_n_values:
                sorted_n_values = sorted(scenario_n_values)
                ax.set_xticks(sorted_n_values)
        
        plt.suptitle(f'Target Algorithm: {target_alg}', fontsize=24, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        if OUTPUT_DIR is not None:
            output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
        else:
            first_scenario_name = list(SCENARIO_DIRS.keys())[0]
            first_scenario_dir = SCENARIO_DIRS[first_scenario_name]
            if TARGET_SUBDIR is not None:
                output_dir = os.path.join(first_scenario_dir, TARGET_SUBDIR)
            else:
                output_dir = first_scenario_dir
        
        output_filename = f"{FIGURE_NAME_PREFIX}_{target_alg}_by_n.pdf"
        output_file = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"  Chart saved to: {output_file}")
        plt.close()

def main():
    # Check if SCENARIO_DIRS is configured
    if not SCENARIO_DIRS:
        print("Error: SCENARIO_DIRS is empty. Please configure scenario directories in the script.")
        return
    
    print(f"Processing {len(SCENARIO_DIRS)} scenario(s)...")
    
    # Collect data from all scenarios
    scenario_data_dict = {}
    
    for scenario_name, scenario_dir in SCENARIO_DIRS.items():
        print(f"\nProcessing scenario: {scenario_name}")
        
        # Concatenate TARGET_SUBDIR to the scenario directory if specified
        if TARGET_SUBDIR is not None:
            full_scenario_dir = os.path.join(scenario_dir, TARGET_SUBDIR)
        else:
            full_scenario_dir = scenario_dir
        
        print(f"Directory: {full_scenario_dir}")
        
        if not os.path.exists(full_scenario_dir):
            print(f"Warning: Directory {full_scenario_dir} does not exist, skipping...")
            continue
        
        # Find JSON file in the directory
        json_file = os.path.join(full_scenario_dir, 'all_mean_performance_by_n.json')
        
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping...")
            continue
        
        # Load results
        results = load_results(json_file)
        if results is None:
            continue
        
        scenario_data_dict[scenario_name] = (results, json_file)
        print(f"  Loaded {len(results)} subdirectories")
    
    if not scenario_data_dict:
        print("Error: No valid scenario data found")
        return
    
    # Plot all scenarios grouped by target algorithm
    print(f"\nPlotting scenarios grouped by target algorithm...")
    plot_all_targets_grouped(scenario_data_dict)

if __name__ == "__main__":
    main()

