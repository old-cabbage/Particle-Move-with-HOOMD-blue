import json

with open("result/convex/convex.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pack = ["0.1", "0.2", "0.3", "0.4", "0.5"]

for i in pack:
    data[i] = {}
    for j in range(50):
        data[i][str((j+1)*100)] = []

# 写回文件
with open("result/convex/convex.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)