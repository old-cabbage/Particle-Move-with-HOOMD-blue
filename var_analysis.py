import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,kstest
import json
import os
import math

data=[]
with open("result/sdf/convex/convex_sdf.json","r",encoding="utf-8") as file:
    oridata=json.load(file)

# 定义目标id
target_id = [1,2]
data=[]
times=0
target_pack="0.1"
#遍历寻找
item=None
for i in range(len(target_id)):
    for item in oridata[target_pack]:
        if item["id"] == target_id[i]:
            data += item["data"]
            times += item["times"]

data=np.array(data)

# 计算统计量
mean_cal = np.mean(data)                # 样本均值
median_cal=np.median(data)
var_cal = np.var(data, ddof=1)          # 样本方差（无偏估计）
std_cal = np.std(data)                  # 样本标准差
uncer_a=std_cal/np.sqrt(times)          # 不确定度

print(f"数据量：{times}")
print(f"计算均值: {mean_cal:.8f}")
print(f"不确定度:{uncer_a:.8f}")
print(f"计算方差: {var_cal:.8f},标准差：{std_cal:.8f}")
#print(f"真实均值: {mu_true}, 真实标准差: {sigma_true}")

np.random.shuffle(data)
# 执行K-S检验（检验数据是否符合均值为mu、标准差为sigma的正态分布）
statistic, p_value = kstest(data[:1000], lambda x: norm.cdf(x, loc=mean_cal, scale=std_cal))
print(f"统计量D={statistic:.6f}, p值={p_value:.12f}")

#data=data[:10000]
# 绘制散点图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(data[:10000])), data[:10000], s=5, alpha=0.1, color='red')
plt.title(rf'Density{target_pack}: ${-mean_cal:.5f}\pm{uncer_a:.5f}$; Variance:{var_cal:.5f}')
plt.xlabel('Index')
plt.ylabel('SDF Value')
plt.grid(alpha=0.3)


# 绘制密度分布图
plt.subplot(1, 2, 2)
bins_num=500
# 绘制直方图（归一化）
count, bins, _ = plt.hist(data, bins=bins_num, density=True, 
                         alpha=0.3, color='b', label='Histogragh')
#print(sum(count))
# 计算每个区间的中点（作为横坐标）
#bin_centers = (bins[:-1] + bins[1:]) / 2

# 绘制理论正态曲线
#plt.plot(x, norm.pdf(x, mu_true, sigma_true), 
#        'r--', lw=2, label='real distribution')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, bins_num)

# 绘制概率密度折线图
plt.plot(bins[1:bins_num+1], count, 'r-', linewidth=0.5, label='Fitted Distribution')

# 绘制计算得到的正态曲线
plt.plot(x, norm.pdf(x, mean_cal, std_cal), 
        'b-', lw=1, alpha=0.8, label='Expected Norm Distribution')

# 添加均值线和标准差线
plt.axvline(float(mean_cal), color='yellow', linestyle='-', linewidth=1, label=rf'Average $\mu={mean_cal:.5f}$')
plt.axvline(float(mean_cal + std_cal), color='orange', linestyle='--', linewidth=1, label=rf'$\mu+\sigma={mean_cal + std_cal:.5f}$')
plt.axvline(float(mean_cal - std_cal), color='orange', linestyle='--', linewidth=1, label=rf'$\mu-\sigma={mean_cal - std_cal:.5f}$')


plt.title(f'Probability Distribution;Data number:{times}')
plt.xlabel('SDF Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.suptitle(f'SDF Analysis for Convex Particles (packing fraction: {target_pack})')

# Create a directory to save figures if it doesn't exist
output_dir = '/home/ethan/图片/sdf_analysis'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"sdf_analysis_convex_{target_pack}.png")
plt.savefig(save_path)

plt.show()
