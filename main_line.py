import matplotlib.pyplot as plt
import os 
# 数据点，来自你的手绘图
x = [3, 5, 8, 10]
y = [7, 31, 120, 150]

plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', linestyle='-')

# 设置x和y的最小刻度为1
plt.xticks(x)
plt.yticks(y)

# 标注x轴和y轴
plt.xlabel("Number of agents")
plt.ylabel("Empirical m")

# 手动画风格，加网格
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

save_path = os.path.join("plots", "main_line.pdf")
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {save_path}")