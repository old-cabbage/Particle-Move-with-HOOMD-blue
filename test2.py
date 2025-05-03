import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

# 生成数据（可替换为你的数据）
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000000)

# 计算样本参数
mu = np.mean(data)
sigma = np.std(data)

# K-S检验
statistic, p_value = kstest(data, lambda x: norm.cdf(x, loc=mu, scale=sigma))
print(f"K-S检验结果: D={statistic:.4f}, p={p_value:.4f}")

# 可视化对比（直方图 vs 理论PDF）
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='k', label='直方图')

# 绘制理论正态曲线
x = np.linspace(data.min(), data.max(), 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False 
plt.plot(x, pdf, 'r-', linewidth=2, label=f'正态分布拟合 (μ={mu:.1f}, σ={sigma:.1f})')

plt.title(f'数据分布与正态拟合对比 (K-S p={p_value:.3f})')
plt.xlabel('数值')
plt.ylabel('概率密度')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
