import re

#for i in range(1,6):
result=[]
#with open("result/concave/concave_{:.2f}_result.txt".format(i*0.1)) as file:
with open("05.txt") as file:
    #content=file.read()
    lines=file.readlines()
for line in lines:
    #result.append(re.search(r'\d+$', line).group() if re.search(r'\d+$', line) else None)
    result.append(re.search(r'=(\S+)',line))
#results=[float(res) for res in result if res is not None]
#print(result)
results=[float(res.group(1)) for res in result if res is not None]
results.sort(reverse = True)
#print(f"堆叠密度为{i*0.1:.1f}")
print(results)