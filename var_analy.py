import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import math

with open("result/sdf/convex/convex_sdf.json","r",encoding="utf-8") as file:
    oridata=json.load(file)

# 定义目标id
target_id = 4

# 遍历查找
result = None
for item in oridata["0.5"]:
    if item["id"] == target_id:
        result = item
        break

x=[]
data=item["data"]
mean = sum(data)/len(data)
var=np.var(data)
ua=math.sqrt(var/item["times"])
print(f"平均值：{mean}")
print(f"方差：{var},标准差：{math.sqrt(var)}")

#for i in range(len(oridata05)):
#    data+=oridata05[i]
data=data[:10000]
for i in range(len(data)):
    x.append((i+1)/len(data)*600)

plt.figure(figsize=(8,4))
plt.scatter(data, x, color='r',label='oringinal data',s=5,alpha=0.2)
plt.axvline(mean, color='r', linestyle='--', label='average value')
for x in data:
    plt.plot([x, mean], [0.1, 0.1], 'g-', alpha=0.3)  # 绘制偏差线
plt.title(f"var analysis (var={var:.5f})")
plt.title(f"{mean}")
sns.histplot(data, kde=True, bins=50, color='skyblue', linewidth=0)  # kde=True表示同时显示KDE图

plt.legend()
plt.show()
