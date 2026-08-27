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
# Configuration: set the directories to process here
# ============================================================================
# Scenario directory map: keys are scenario names, values are directory paths.
# Each directory should contain all_mean_performance_by_n.json.
SCENARIO_DIRS = {
    "mpe-pp": "/scratch/project_2012831/naht-dev/naht_results/mpe-pp",
    "3sv5z": "/scratch/project_2012831/naht-dev/naht_results/3sv5z",
    "5v6": "/scratch/project_2012831/naht-dev/naht_results/5v6",
    "8v9": "/scratch/project_2012831/naht-dev/naht_results/8v9",
    "10v11": "/scratch/project_2012831/naht-dev/naht_results/10v11",
    # Add more scenarios here...
}

# Subdirectory appended to each path in SCENARIO_DIRS.
# Example: TARGET_SUBDIR = "type_generalization_poam"
# If None, use the paths in SCENARIO_DIRS as-is.
TARGET_SUBDIR = "ood_generalization_main"  # None means no extra subdirectory

# Optional filter on JSON subdirectory keys; None means process all subdirectories.
# Example: FILTER_SUBDIRS = ["poam_shapley1_eps0.2-vs-ippo", "poam_banzhaf_eps0.2-vs-ippo"]
FILTER_SUBDIRS = None  # None means process all subdirectories

# Algorithm name mapping (from ood_figure_type.py)
# EXP_NAMEMAP = {
#     # "poam_shapley1_eps0.2": "SM-POAM",
#     # "poam_banzhaf_eps0.2": "BM-POAM",
#     # "poam_type": "POAM",
#     "ippo_shapley2v_eps0.2": "SM-IPPO",
#     "ippo_banzhaf_eps0.2": "BM-IPPO",
#     "ippo_naht1v_type": "IPPO-NAHT",
# }
EXP_NAMEMAP = { # format is alg log name: display name
        "poam_shapley1_eps0.2": "SM-POAM",
        "poam_banzhaf_eps0.2": "BM-POAM",
        "poam-pqvmq_open": "POAM",
        "ippo_shapley2v_eps0.2": "SM-IPPO",
        "ippo_banzhaf_eps0.2": "BM-IPPO",
        "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
        "poam_shapley1_eps0.2": "SM-POAM",
        "ippo_shapley2v_eps0.2": "SM-IPPO",
        "poam_banzhaf_eps0.2": "BM-POAM",
        "ippo_banzhaf_eps0.2": "BM-IPPO",
        "poam-pqvmq_open": "POAM",
        "ippo_naht1v-pqvmq_open": "IPPO-NAHT",
}

# Output configuration
# Directory for saving the figure (if None, saves to first scenario's directory)
OUTPUT_DIR = "/scratch/project_2012831/naht-dev/plots-naht/5trials-95ci"  # e.g. "/scratch/project_2012831/naht-dev/plots-naht/5trials-95ci"

# Name of the output figure (if None, auto-generates based on scenario names)
FIGURE_NAME = "ood_poam_n"  # e.g. "metric_by_n" (saved as "metric_by_n.pdf")

# Output format settings
OUTPUT_DPI = 600  # DPI for PDF output

# Statistical settings
# Number of samples used to calculate each mean value (for each random seed)
# If each value in the data is a mean of N_SAMPLES_PER_SEED samples, set this value
# This affects the calculation of standard errors and confidence intervals
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

def list_available_metrics(results):
    """List all available metrics."""
    metrics = set()
    for subdir, n_data in results.items():
        for n_value, metrics_data in n_data.items():
            metrics.update(metrics_data.keys())
    return sorted(metrics)

def list_available_subdirs(results):
    """List all available subdirectories."""
    return sorted(results.keys())

def display_metric_by_n(results, metric_name, subdirs=None):
    """
    Print the mean and 95% CI of a metric at each N.

    Args:
        results: Results loaded from the JSON file
        metric_name: Metric to display
        subdirs: Subdirectories to display; None means all
    """
    if subdirs is None:
        subdirs = list_available_subdirs(results)
    
    print("\n" + "="*100)
    print(f"Metric: {metric_name}")
    print("="*100)
    
    for subdir in subdirs:
        if subdir not in results:
            print(f"\nWarning: subdirectory {subdir} does not exist")
            continue
        
        print(f"\nSubdirectory: {subdir}")
        print("-" * 100)
        print(f"{'N':<10} {'Mean':<15} {'95% CI lower':<15} {'95% CI upper':<15} {'Count':<10} {'Std':<15}")
        print("-" * 100)
        
        n_data = results[subdir]
        # Sort by N
        for n_value in sorted(n_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            if metric_name not in n_data[n_value]:
                continue
            
            metric_stats = n_data[n_value][metric_name]
            mean = metric_stats['mean']
            ci_lower = metric_stats['ci_95_lower']
            ci_upper = metric_stats['ci_95_upper']
            count = metric_stats['count']
            std = metric_stats['std']
            
            print(f"{n_value:<10} {mean:<15.6f} {ci_lower:<15.6f} {ci_upper:<15.6f} {count:<10} {std:<15.6f}")
        
        print()

def display_metric_summary(results, metric_name, subdirs=None):
    """
    Print a compact metric summary (mean and confidence interval only).

    Args:
        results: Results loaded from the JSON file
        metric_name: Metric to display
        subdirs: Subdirectories to display; None means all
    """
    if subdirs is None:
        subdirs = list_available_subdirs(results)
    
    print("\n" + "="*100)
    print(f"Metric summary: {metric_name}")
    print("="*100)
    
    for subdir in subdirs:
        if subdir not in results:
            continue
        
        print(f"\nSubdirectory: {subdir}")
        print("-" * 100)
        print(f"{'N':<10} {'Mean':<15} {'95% CI':<30}")
        print("-" * 100)
        
        n_data = results[subdir]
        for n_value in sorted(n_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            if metric_name not in n_data[n_value]:
                continue
            
            metric_stats = n_data[n_value][metric_name]
            mean = metric_stats['mean']
            ci_lower = metric_stats['ci_95_lower']
            ci_upper = metric_stats['ci_95_upper']
            
            print(f"{n_value:<10} {mean:<15.6f} [{ci_lower:.6f}, {ci_upper:.6f}]")
        
        print()

def extract_scenario_name(json_file_path):
    """
    Extract scenario name from JSON file path
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        scenario_name: Scenario name (e.g., 'mpe-pp', '10v11', '3sv5z')
    """
    # Common scenario names
    scenarios = ['mpe-pp', '3sv5z', '5v6', '8v9', '10v11']
    
    for scenario in scenarios:
        if scenario in json_file_path:
            return scenario
    
    # If no known scenario found, try to extract from path
    # Path format: .../naht_results/{scenario}/type_generalization_poam/...
    path_parts = json_file_path.split(os.sep)
    for i, part in enumerate(path_parts):
        if part == 'naht_results' and i + 1 < len(path_parts):
            return path_parts[i + 1]
    
    # Default fallback
    return 'Unknown'

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

def extract_algorithm_name(subdir_name):
    """
    Extract the algorithm name from a subdirectory name (the part before "-vs-").

    Args:
        subdir_name: Subdirectory name, e.g. "poam_banzhaf_eps0.2-vs-ippo"

    Returns:
        algorithm_name: Algorithm name, e.g. "poam_banzhaf_eps0.2"
    """
    if '-vs-' in subdir_name:
        return subdir_name.split('-vs-')[0]
    return subdir_name

def merge_results_by_algorithm(results, metric_name):
    """
    Merge results by algorithm (use the name before "-vs-", then pool matching algorithms).

    Args:
        results: Original results dictionary
        metric_name: Metric name

    Returns:
        merged_data: {algorithm_name: {n_value: {mean, ci_lower, ci_upper, values}}}
    """
    merged_data = {}
    
    for subdir, n_data in results.items():
        # Extract the algorithm name
        alg_name = extract_algorithm_name(subdir)
        
        # Map to a display name, or keep the original name if unmapped
        display_name = EXP_NAMEMAP.get(alg_name, alg_name)
        
        if display_name not in merged_data:
            merged_data[display_name] = {}
        
        # Merge data for each N
        for n_value, metrics_data in n_data.items():
            if metric_name not in metrics_data:
                continue
            
            if n_value not in merged_data[display_name]:
                merged_data[display_name][n_value] = {
                    'values': []
                }
            
            # Pool all values
            metric_stats = metrics_data[metric_name]
            merged_data[display_name][n_value]['values'].extend(metric_stats['values'])
    
    # Compute merged statistics
    for alg_name in merged_data:
        for n_value in merged_data[alg_name]:
            values = merged_data[alg_name][n_value]['values']
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                count = len(values)
                
                # 95% CI: t_{0.975, df} * SEM, or 1.96 * SEM without scipy
                if HAS_SCIPY and count > 1:
                    t_critical = scipy_stats.t.ppf(0.975, df=count-1)
                else:
                    t_critical = 1.96
                
                standard_error = std / np.sqrt(count)
                margin = t_critical * standard_error
                
                merged_data[alg_name][n_value]['mean'] = mean
                merged_data[alg_name][n_value]['std'] = std
                merged_data[alg_name][n_value]['count'] = count
                merged_data[alg_name][n_value]['ci_95_lower'] = mean - margin
                merged_data[alg_name][n_value]['ci_95_upper'] = mean + margin
    
    return merged_data

def perform_t_tests(merged_data, scenario_name):
    """
    Perform Welch's t-tests between all pairs of algorithms for each N value
    Welch's t-test does not assume equal variances, making it more robust when
    algorithms have different standard deviations.
    
    Args:
        merged_data: Merged data dictionary with algorithm names as keys
        scenario_name: Name of the scenario (for printing)
    """
    if not HAS_SCIPY:
        print(f"\nWarning: scipy not available, cannot perform t-tests for {scenario_name}")
        return
    
    algorithms = sorted(merged_data.keys())
    if len(algorithms) < 2:
        print(f"\nNot enough algorithms for t-tests in {scenario_name} (need at least 2)")
        return
    
    # Get all N values
    all_n_values = set()
    for alg_name, alg_data in merged_data.items():
        all_n_values.update(alg_data.keys())
    
    if not all_n_values:
        return
    
    print(f"\n{'='*80}")
    print(f"Welch's t-test Results for Scenario: {scenario_name}")
    print(f"{'='*80}")
    print("Note: Using Welch's t-test (does not assume equal variances)")
    if N_SAMPLES_PER_SEED is not None:
        print(f"Note: Each value represents a mean of {N_SAMPLES_PER_SEED} samples per random seed")
        print("      The t-test compares means across random seeds (hierarchical structure)")
    print("Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05")
    print(f"{'='*80}")
    
    # For each N value
    for n_value in sorted(all_n_values, key=lambda x: int(x) if x.isdigit() else float('inf')):
        print(f"\nN = {n_value}:")
        print("-" * 80)
        
        # Get data for all algorithms at this N value
        alg_data_dict = {}
        for alg_name in algorithms:
            if n_value in merged_data[alg_name] and 'values' in merged_data[alg_name][n_value]:
                values = merged_data[alg_name][n_value]['values']
                if len(values) > 0:
                    alg_data_dict[alg_name] = np.array(values)
        
        if len(alg_data_dict) < 2:
            print("  Not enough algorithms with data for t-test")
            continue
        
        # Perform t-test for all pairs
        alg_list = sorted(alg_data_dict.keys())
        for i in range(len(alg_list)):
            for j in range(i + 1, len(alg_list)):
                alg1 = alg_list[i]
                alg2 = alg_list[j]
                
                data1 = alg_data_dict[alg1]
                data2 = alg_data_dict[alg2]
                
                # Perform Welch's t-test (does not assume equal variances)
                # If each value is a mean of N_SAMPLES_PER_SEED samples, we need to account for this
                # The current t-test treats each value as an independent observation
                # This is appropriate if we're comparing means across seeds (hierarchical structure)
                # equal_var=False uses Welch's t-test which accounts for different stds
                t_stat, p_value = scipy_stats.ttest_ind(data1, data2, equal_var=False)
                
                # Note: If N_SAMPLES_PER_SEED is set, each value in data1/data2 represents
                # a mean of N_SAMPLES_PER_SEED samples. The t-test here compares means across
                # random seeds, which is the appropriate level for statistical inference.
                # The within-seed variance (from 128 samples) is already accounted for in
                # each mean value, so we're testing seed-level differences.
                
                # Calculate means for display
                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                
                # Get 95% confidence intervals for both algorithms
                ci1_lower = merged_data[alg1][n_value]['ci_95_lower']
                ci1_upper = merged_data[alg1][n_value]['ci_95_upper']
                ci2_lower = merged_data[alg2][n_value]['ci_95_lower']
                ci2_upper = merged_data[alg2][n_value]['ci_95_upper']
                
                # Check if 95% CIs overlap
                # Non-overlapping 95% CIs is a conservative significance check
                ci_overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
                
                # Determine significance based on p-value
                if p_value < 0.001:
                    sig = "***"
                    sig_text = "Highly Significant (p < 0.001)"
                elif p_value < 0.01:
                    sig = "**"
                    sig_text = "Very Significant (p < 0.01)"
                elif p_value < 0.05:
                    sig = "*"
                    sig_text = "Significant (p < 0.05)"
                else:
                    sig = ""
                    sig_text = "Not Significant (p >= 0.05)"
                
                # Determine significance based on 95% CI overlap
                if not ci_overlap:
                    ci_sig_text = "Significant (95% CIs do not overlap)"
                else:
                    ci_sig_text = "Not Significant (95% CIs overlap)"
                
                print(f"  {alg1} vs {alg2}:")
                print(f"    Mean1: {mean1:.6f} (n={len(data1)}), Mean2: {mean2:.6f} (n={len(data2)})")
                print(f"    95% CI1: [{ci1_lower:.6f}, {ci1_upper:.6f}]")
                print(f"    95% CI2: [{ci2_lower:.6f}, {ci2_upper:.6f}]")
                print(f"    t-statistic: {t_stat:.6f}, p-value: {p_value:.6f} {sig}")
                print(f"    Statistical Significance (t-test): {sig_text}")
                print(f"    Statistical Significance (95% CI): {ci_sig_text}")
        
        print()
    
    print(f"{'='*80}\n")

def plot_multiple_scenarios(scenario_data_dict, target_subdirs=None, output_file=None):
    """
    Plot metrics from multiple scenarios, each in a separate subplot arranged in a row
    
    Args:
        scenario_data_dict: Dictionary with scenario names as keys and (results, json_path) tuples as values
        target_subdirs: List of subdirs to process, None means all
        output_file: Output image file path, None means display
    """
    if not scenario_data_dict:
        print("Error: No scenario data provided")
        return
    
    num_scenarios = len(scenario_data_dict)
    
    # Create figure with subplots in a row
    # Width: 6 per subplot, height: 5
    fig, axes = plt.subplots(1, num_scenarios, figsize=(6 * num_scenarios, 5))
    
    # If only one scenario, axes is not a list, convert it
    if num_scenarios == 1:
        axes = [axes]
    
    # Get all algorithms across all scenarios to ensure consistent colors
    all_algorithms = set()
    for scenario_name, (results, json_path) in scenario_data_dict.items():
        metric_name = determine_metric_name(json_path)
        if target_subdirs is not None:
            filtered_results = {k: v for k, v in results.items() if k in target_subdirs}
        else:
            filtered_results = results
        merged_data = merge_results_by_algorithm(filtered_results, metric_name)
        all_algorithms.update(merged_data.keys())
    
    all_algorithms = sorted(all_algorithms)
    
    # Generate colors for algorithms (consistent across all subplots)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(all_algorithms), 10)))
    if len(all_algorithms) > 10:
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(all_algorithms) - 10))
        colors = np.concatenate([colors, colors2])
    
    algorithm_color_map = {alg: colors[i] for i, alg in enumerate(all_algorithms)}
    
    # Process each scenario in its own subplot
    for idx, (scenario_name, (results, json_path)) in enumerate(scenario_data_dict.items()):
        ax = axes[idx]
        metric_name = determine_metric_name(json_path)
        
        # Get y-label for this scenario
        y_label = get_y_label(json_path)
        
        # Filter subdirs if specified
        if target_subdirs is not None:
            filtered_results = {k: v for k, v in results.items() if k in target_subdirs}
        else:
            filtered_results = results
        
        # Check if metric exists
        available_metrics = list_available_metrics(filtered_results)
        if metric_name not in available_metrics:
            print(f"Warning: Metric '{metric_name}' not found in scenario '{scenario_name}', skipping...")
            ax.text(0.5, 0.5, f"No data for {scenario_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Merge results by algorithm
        merged_data = merge_results_by_algorithm(filtered_results, metric_name)
        
        if not merged_data:
            ax.text(0.5, 0.5, f"No data for {scenario_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Perform and print t-tests for this scenario
        perform_t_tests(merged_data, scenario_name)
        
        # Collect all n values for this scenario
        scenario_n_values = set()
        
        # Process each algorithm
        for alg_name in sorted(merged_data.keys()):
            alg_data = merged_data[alg_name]
            
            # Collect data
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
                ci_lower = alg_data[n_value]['ci_95_lower']
                ci_upper = alg_data[n_value]['ci_95_upper']
                
                means.append(mean)
                errors_lower.append(mean - ci_lower)
                errors_upper.append(ci_upper - mean)
            
            if not n_values:
                continue
            
            # Convert to numpy arrays
            n_values = np.array(n_values)
            means = np.array(means)
            errors_lower = np.array(errors_lower)
            errors_upper = np.array(errors_upper)
            
            # Get color for this algorithm
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
        
        # Set labels and title for this subplot
        ax.set_xlabel('N', fontsize=11)
        if idx == 0:  # First subplot shows its own y-label
            ax.set_ylabel(y_label, fontsize=11)
        elif idx == 1:  # Second subplot shows "Mean Test Winrate"
            ax.set_ylabel('Mean Test Winrate', fontsize=11)
        ax.set_title(scenario_name, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if scenario_n_values:
            sorted_n_values = sorted(scenario_n_values)
            ax.set_xticks(sorted_n_values)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"\nChart saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_metric_by_n(results, json_file_path, subdirs=None, output_file=None):
    """
    Plot metric vs N for a single scenario (kept for backward compatibility)
    
    Args:
        results: Results data loaded from JSON file
        json_file_path: JSON file path
        subdirs: List of subdirs to show, None means all
        output_file: Output image file path, None means display
    """
    # Filter subdirs if specified
    if subdirs is not None:
        filtered_results = {k: v for k, v in results.items() if k in subdirs}
    else:
        filtered_results = results
    
    # Use the multi-scenario function with single scenario
    scenario_name = extract_scenario_name(json_file_path)
    scenario_data_dict = {scenario_name: (filtered_results, json_file_path)}
    plot_multiple_scenarios(scenario_data_dict, target_subdirs=None, output_file=output_file)

def find_json_files(base_dir):
    """Find all_mean_performance_by_n.json files in the specified directory and its subdirectories"""
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        if 'all_mean_performance_by_n.json' in files:
            json_files.append(os.path.join(root, 'all_mean_performance_by_n.json'))
    return json_files

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
    
    # Generate output filename
    if FIGURE_NAME is not None:
        # Use specified figure name
        output_filename = f"{FIGURE_NAME}.pdf"
    else:
        # Auto-generate filename based on scenario names
        if len(scenario_data_dict) == 1:
            scenario_name = list(scenario_data_dict.keys())[0]
            output_filename = f"{scenario_name}_metric_by_n.pdf"
        else:
            scenario_names = "_".join(sorted(scenario_data_dict.keys()))
            output_filename = f"{scenario_names}_metric_by_n.pdf"
    
    # Determine output directory
    if OUTPUT_DIR is not None:
        # Use specified output directory
        output_dir = OUTPUT_DIR
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Use first scenario's directory as default
        first_scenario_name = list(SCENARIO_DIRS.keys())[0]
        first_scenario_dir = SCENARIO_DIRS[first_scenario_name]
        if TARGET_SUBDIR is not None:
            output_dir = os.path.join(first_scenario_dir, TARGET_SUBDIR)
        else:
            output_dir = first_scenario_dir
    
    output_file = os.path.join(output_dir, output_filename)
    
    # Plot all scenarios on the same figure
    print(f"\nPlotting all scenarios on the same figure...")
    plot_multiple_scenarios(scenario_data_dict, target_subdirs=FILTER_SUBDIRS, output_file=output_file)

if __name__ == "__main__":
    main()

