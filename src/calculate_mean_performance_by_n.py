import os
import json
import re
from collections import defaultdict
from pathlib import Path
import statistics
import math

# 尝试导入scipy用于t分布，如果不可用则使用标准正态分布
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def extract_n_value(dirname):
    """
    从目录名称中提取n值
    例如: eval_poam-seed=112358_vdn-seed=93718_n-1_seed=394820_09-18-03-43-29 -> 1
    """
    match = re.search(r'n-(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None

def load_metrics(metrics_path):
    """加载metrics.json文件"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载 {metrics_path}: {e}")
        return None

def calculate_95_ci(mean, std, n, confidence=0.95):
    """
    计算95%置信区间
    
    Args:
        mean: 样本均值
        std: 样本标准差
        n: 样本数量
        confidence: 置信水平（默认0.95）
    
    Returns:
        (lower_bound, upper_bound): 置信区间的下界和上界
    """
    if n <= 1:
        return (mean, mean)
    
    # 计算标准误差
    standard_error = std / math.sqrt(n)
    
    # 计算临界值
    if HAS_SCIPY:
        # 使用t分布（适用于小样本）
        alpha = 1 - confidence
        t_critical = scipy_stats.t.ppf(1 - alpha/2, df=n-1)
        margin = t_critical * standard_error
    else:
        # 使用标准正态分布近似（对于n>=30很准确，对于n=25也可以接受）
        # 95%置信区间的z值约为1.96
        z_critical = 1.96
        margin = z_critical * standard_error
    
    lower_bound = mean - margin
    upper_bound = mean + margin
    
    return (lower_bound, upper_bound)

def calculate_mean_performance_by_n(base_dir):
    """
    计算每个n值的平均性能
    
    Args:
        base_dir: 包含sacred子目录的基础目录路径
    """
    sacred_dir = os.path.join(base_dir, 'sacred')
    
    if not os.path.exists(sacred_dir):
        print(f"错误: 找不到目录 {sacred_dir}")
        return None
    
    # 按n值分组存储所有指标的值
    n_values_data = defaultdict(lambda: defaultdict(list))
    
    # 遍历所有子目录
    for subdir in os.listdir(sacred_dir):
        subdir_path = os.path.join(sacred_dir, subdir)
        
        # 跳过非目录项
        if not os.path.isdir(subdir_path):
            continue
        
        # 提取n值
        n_value = extract_n_value(subdir)
        if n_value is None:
            print(f"警告: 无法从目录名称中提取n值: {subdir}")
            continue
        
        # 读取metrics.json
        metrics_path = os.path.join(subdir_path, '1', 'metrics.json')
        if not os.path.exists(metrics_path):
            print(f"警告: 找不到文件 {metrics_path}")
            continue
        
        metrics = load_metrics(metrics_path)
        if metrics is None:
            continue
        
        # 提取所有指标的值
        for metric_name, metric_data in metrics.items():
            if 'values' in metric_data and len(metric_data['values']) > 0:
                # 取最后一个值（通常是最终性能）
                value = metric_data['values'][-1]
                n_values_data[n_value][metric_name].append(value)
    
    # 计算每个n值的平均值和95%置信区间
    results = {}
    for n_value in sorted(n_values_data.keys()):
        results[n_value] = {}
        for metric_name, values in n_values_data[n_value].items():
            if len(values) > 0:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                ci_lower, ci_upper = calculate_95_ci(mean, std, len(values))
                results[n_value][metric_name] = {
                    'mean': mean,
                    'std': std,
                    'count': len(values),
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper,
                    'values': values
                }
    
    return results

def print_results(results):
    """打印结果"""
    print("\n" + "="*80)
    print("每个n值的平均性能统计（包含95%置信区间）")
    print("="*80)
    
    for n_value in sorted(results.keys()):
        print(f"\nN = {n_value}:")
        print("-" * 80)
        for metric_name, stats in sorted(results[n_value].items()):
            print(f"  {metric_name}:")
            print(f"    平均值: {stats['mean']:.6f}")
            print(f"    标准差: {stats['std']:.6f}")
            print(f"    样本数: {stats['count']}")
            print(f"    95% CI: [{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}]")
            print()

def save_results(results, output_file):
    """保存结果到JSON文件"""
    # 转换为可序列化的格式
    serializable_results = {}
    for n_value, metrics in results.items():
        serializable_results[n_value] = {}
        for metric_name, stats in metrics.items():
            serializable_results[n_value][metric_name] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'count': int(stats['count']),
                'ci_95_lower': float(stats['ci_95_lower']),
                'ci_95_upper': float(stats['ci_95_upper']),
                'values': [float(v) for v in stats['values']]
            }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置基础目录路径
    base_dir = "/scratch/project_2012831/naht-dev/naht_results/mpe-pp/type_generalization_poam"
    
    print(f"正在分析基础目录: {base_dir}")
    print("正在提取n值并计算平均性能和95%置信区间...")
    if not HAS_SCIPY:
        print("注意: 未安装scipy，使用标准正态分布近似计算95%置信区间")
    
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 找不到基础目录 {base_dir}")
        exit(1)
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"警告: 基础目录 {base_dir} 下没有子目录")
        exit(1)
    
    print(f"\n找到 {len(subdirs)} 个子目录，开始处理...")
    print("="*80)
    
    # 迭代处理每个子目录
    all_results = {}
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(base_dir, subdir)
        sacred_dir = os.path.join(subdir_path, 'sacred')
        
        # 检查是否存在sacred子目录
        if not os.path.exists(sacred_dir):
            print(f"\n跳过 {subdir}: 未找到sacred子目录")
            continue
        
        print(f"\n正在处理子目录: {subdir}")
        print("-" * 80)
        
        # 计算平均性能
        results = calculate_mean_performance_by_n(subdir_path)
        
        if results:
            # 存储结果（使用子目录名作为键）
            all_results[subdir] = results
            
            # 打印结果
            print(f"\n子目录 {subdir} 的结果:")
            print_results(results)
        else:
            print(f"警告: 无法计算子目录 {subdir} 的平均性能")
    
    # 保存汇总结果（可选）
    if all_results:
        summary_output_file = os.path.join(base_dir, "all_mean_performance_by_n.json")
        with open(summary_output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n所有子目录的汇总结果已保存到: {summary_output_file}")
    else:
        print("\n错误: 没有成功处理任何子目录")

