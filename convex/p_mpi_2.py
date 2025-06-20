import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system import System,add_data_with_auto_id_p
import time
import random
from multiprocessing import Pool, cpu_count

start_time=time.time()
#粒子总数
N0=5000
#packing_density = packing_density_0 * num_particles/5000
shape='A'
device=hoomd.device.CPU()
mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0.6,default_a=0.8)
mc.shape[shape] = dict(
            vertices = [
            (-2, 0),
            (2, 0),
            (11/8, 5*math.sqrt(63)/8),
            ]
)
particle_area=5*math.sqrt(63)/4

packing_density_0=0.5
strpacking_density_0="0.5"

# 运行循环模拟和插入测试
iterations = 200            # 循环次数
moves_per_cycle = 5            # 每个循环中移动的成功步数
insertions_per_cycle = 1000    # 每个循环中插入的粒子尝试次数
gsdcopy=20

def task(n):
    result=[]
    num_particles = (n+1)*200 #list(range(1000*(n%5)+200,1000*(n%5)+1200,200))

    packing_density = packing_density_0 * num_particles/N0

    total_success = 0
    total_attempts = 0
    simulation_start_time = time.time()

    print(f"粒子数{num_particles},开始进行循环模拟和插入测试")
    for k in range(gsdcopy):
    #创建实例
        system=System(
            num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            particle_area=particle_area,
            mc=mc,device=device,shape=shape
        )

        system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
        system.simulation.operations.integrator = mc
        system.simulation.create_state_from_gsd(filename='gsd/{}/P_convex_{:.2f}_{}_{}.gsd'.format(N0,packing_density_0,num_particles,k+1))

        for cycle in range(1, iterations + 1):
            # 进行移动
            system.simulation.run(moves_per_cycle)
            
            # 进行插入尝试  
            success = system.random_insert(
                insert_times=insertions_per_cycle
            )
            total_success += success
            total_attempts += insertions_per_cycle

            if success!=0:
                result.append(success/insertions_per_cycle)
            else:
                result.append(1)

    simulation_end_time = time.time()
    print(f"\n粒子N={N0},堆叠密度为{packing_density_0:.1f}，粒子数为{num_particles}的循环模拟和插入测试完成，耗时: {simulation_end_time - simulation_start_time:.2f} 秒")

    return result

def main():
    data = list(range(25)) 
    num_workers = 30

    with Pool(processes=num_workers) as pool:
        oriresults = np.array(pool.map(task, data))

    results=np.log(oriresults)

    target_value=[]
    for i in range(len(results)):
        target_value.append(sum(results[i])/len(results[i]))

    end_time=time.time()

    results=results.tolist()

    print(results)

    for i in range(len(results)):
        with open("result/convex/convex_p_{}_{:.2f}.txt".format(N0,packing_density_0),'a') as file:
            file.write(f"堆叠密度为{packing_density_0}, 数目为{(i+1)*200},计算结果为{target_value[i]},耗时 {end_time-start_time:.2f}秒\n\n")

        json_data={
            "info":f"堆叠密度为{packing_density_0}, 数目为{(i+1)*200},计算结果为{target_value[i]},耗时 {end_time-start_time:.2f}秒\n",
            "data_num":iterations*gsdcopy,
            "sigle_point_times":insertions_per_cycle,
            "data":results[i]
        }
        add_data_with_auto_id_p(new_data=json_data, target_class1=strpacking_density_0, target_class2=str((i+1)*200), filename="result/convex/convex.json")

if __name__ == '__main__':
    main()
