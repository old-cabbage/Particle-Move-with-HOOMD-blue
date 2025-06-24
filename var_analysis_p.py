import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,kstest
import json
import os
import math

data=[]
with open("result/convex/convex.json","r",encoding="utf-8") as file:
    oridata=json.load(file)

# 定义目标id
target_id = [1]
data = [[] for _ in range(25)]
times=0
target_pack="0.1"
target_value=[0]*25
num=list((x+1)*200 for x in range(25))
#遍历寻找
item=None
for i in range(len(target_id)):
    #times=oridata[target_pack]["200"]["data_num"]*oridata[target_pack]["200"]["sigle_point_times"]
    for j in range(25):
        for item in oridata[target_pack][str((j+1)*200)]:
            if item["id"] == target_id[i]:
                data[j] += item["data"]
                target_value[j]=np.log(sum(data[j])/len(data[j]))
                if j==0:
                    times=item["data_num"]*len(target_id)
                #times += item["times"]

data=np.array(data)

plt.figure(figsize=(12, 12))

for i in range(25):

    # 计算统计量
    mean_cal = np.mean(data[i])                # 样本均值
    median_cal=np.median(data[i])
    var_cal = np.var(data[i], ddof=1)          # 样本方差（无偏估计）
    std_cal = np.std(data[i])                  # 样本标准差
    uncer_a=std_cal/np.sqrt(times)          # 不确定度

    print(f"数据量：{times}")
    print(f"计算均值: {mean_cal:.8f}")
    print(f"不确定度{uncer_a:.8f}$")
    print(f"计算方差: {var_cal:.8f},标准差：{std_cal:.8f}")
    #print(f"真实均值: {mu_true}, 真实标准差: {sigma_true}")

    np.random.shuffle(data[i])
    # 执行K-S检验（检验数据是否符合均值为mu、标准差为sigma的正态分布）
    statistic, p_value = kstest(data[i][:1000], lambda x: norm.cdf(x, loc=mean_cal, scale=std_cal))
    print(f"统计量D={statistic:.6f}, p值={p_value:.12f}")

    plt.subplot(5, 6, i+1)
    
    # 绘制直方图（归一化）
    count, bins, _ = plt.hist(data[i], bins='auto', density=True, 
                            alpha=0.5, color='b')
    bins_num = len(count)
    #print(sum(count))
    # 计算每个区间的中点（作为横坐标）
    #bin_centers = (bins[:-1] + bins[1:]) / 2

    # 绘制理论正态曲线
    #plt.plot(x, norm.pdf(x, mu_true, sigma_true), 
    #        'r--', lw=2, label='real distribution')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins_num)

    # 绘制概率密度折线图
    plt.plot(bins[1:bins_num+1], count, 'r-', linewidth=0.5)

    # 绘制计算得到的正态曲线
    plt.plot(x, norm.pdf(x, mean_cal, std_cal), 
            'b-', lw=1, alpha=0.8)

    # 添加均值线和标准差线
    plt.axvline(float(mean_cal), color='yellow', linestyle='-', linewidth=1, label=rf'Average $\mu={mean_cal:.5f}$')
    plt.axvline(float(mean_cal + std_cal), color='orange', linestyle='--', linewidth=1, label=rf'$\mu+\sigma={mean_cal + std_cal:.5f}$')
    plt.axvline(float(mean_cal - std_cal), color='orange', linestyle='--', linewidth=1, label=rf'$\mu-\sigma={mean_cal - std_cal:.5f}$')
    plt.title(f'number:{num[i]}', fontsize=9)
    plt.legend(fontsize=6)
    plt.grid(alpha=0.3)

# 进行多项式拟合
degree = 6  # 选择多项式阶数
coefficients = np.polyfit(num, target_value, degree)

#coefficients[-1]=0

# 生成多项式函数
poly_func = np.poly1d(coefficients)

# 显示多项式
print("拟合的多项式为：")
print(poly_func)

sump=0
for i in range(5000):
    sump+=poly_func(i+1)
print(poly_func(5001)-sump/5000)

# 绘制原始数据和拟合曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = poly_func(x_fit)

plt.subplot(5,6,26)

plt.plot(num, target_value, 
         color='purple',             # 线的颜色
         linestyle='-',           # 线型为虚线
         linewidth=1.5,              # 线宽
         marker='s',               # 标记的形状
         markersize=5,            # 标记大小
         markerfacecolor='none',   # 标记内部为空心
         #markeredgecolor='purple',  # 标记边框颜色为黑色
         markeredgewidth=1,      # 标记边框宽度
         label=f'$\phi=${target_pack}')

# 添加标题和标签
plt.title('result')
plt.xlabel('i')
plt.ylabel('$lnP_i$')


plt.suptitle(f'Probability Distribution;Data number:{times}')
plt.legend(fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()

# Create a directory to save figures if it doesn't exist
output_dir = 'generated_figures'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"probability_distribution_convex_{target_pack}.png")
plt.savefig(save_path)

plt.show()
