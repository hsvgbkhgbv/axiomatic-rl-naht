import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
import os



# 参数
horizontal_ranges = [np.arange(1, 71)]
p_list = [0.15]

xticks = [1, 10, 20, 31, 40, 50, 60, 70]

fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharey=True)

for p, horizontal_range in zip(p_list, horizontal_ranges):
    probs = geom.pmf(horizontal_range, p)
    ax.bar(horizontal_range, probs)
    ax.set_xlabel('m', fontsize=18)
    ax.set_title(f'$\\lambda={1-p}$', fontsize=18)
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_xlim(0, 71)
ax.set_ylabel('Probability', fontsize=18)
plt.tight_layout()
save_path = os.path.join("plots-naht/", "main_gd_b.pdf")
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"✅ Saved: {save_path}")



# horizontal_ranges = [np.arange(1, 8), np.arange(1, 8)]
# p_list = [0.15, 0.5]

# xticks = [1, 2, 3, 4, 5, 6, 7]

# fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# for ax, p, horizontal_range in zip(axes, p_list, horizontal_ranges):
#     probs = geom.pmf(horizontal_range, p)
#     ax.bar(horizontal_range, probs)
#     ax.set_xlabel('m', fontsize=18)
#     ax.set_title(f'$\\lambda={1-p}$', fontsize=18)
#     ax.set_xticks(xticks)
#     ax.tick_params(labelsize=14)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
#     ax.set_xlim(0, 8)
# axes[0].set_ylabel('Probability', fontsize=18)
# plt.tight_layout()
# save_path = os.path.join("plots-naht/", "main_gd.pdf")
# plt.savefig(save_path, dpi=600, bbox_inches='tight')
# plt.close(fig)
# print(f"✅ Saved: {save_path}")