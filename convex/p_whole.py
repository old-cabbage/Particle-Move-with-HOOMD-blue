import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system import System
import time
import random

def main():
    #粒子总数
    num_particles = list(range(100,5100,100))
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    packing_density_0=[0.5,0.4,0.3,0.2,0.1]
    shape='A'
    device=hoomd.device.GPU()
    mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=3,default_a=3)
    mc.shape["A"] = dict(
                vertices = [
                (-2, 0),
                (2, 0),
                (11/8, 5*math.sqrt(63)/8),
                ]
    )
    particle_area=5*math.sqrt(63)/4

    results = [[0 for _ in range(50)] for _ in range(5)]

    for i in packing_density_0:

        for j in num_particles:

            total_success=0
            total_attempts=0

            simulation_start_time_j=time.time()

            for k in range(40):

                packing_density = i * j/5000
                
                #创建实例
                system=System(
                    num=j,packing_density=packing_density,
                    packing_density_0=i,
                    particle_area=particle_area,
                    mc=mc,device=device,shape=shape
                )

                system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
                system.simulation.operations.integrator = mc
                system.simulation.create_state_from_gsd(filename='./gsd/P_convex_{:.2f}_{}_{}.gsd'.format(i,j,k%20+1))

                # 运行循环模拟和插入测试
                iterations = 2000             # 循环次数
                moves_per_cycle = 5            # 每个循环中移动的成功步数
                insertions_per_cycle = 20000    # 每个循环中插入的粒子尝试次数

                print(f"\n次序{k+1}/40,堆叠密度为{i}，粒子数为{j}的循环开始进行循环模拟和插入测试...")
                simulation_start_time = time.time()
                for cycle in range(1, iterations + 1):
                    # 进行移动
                    system.simulation.run(moves_per_cycle)
                    
                    # 进行插入尝试  
                    success = system.random_insert(
                        insert_times=insertions_per_cycle
                    )
                    total_success += success
                    total_attempts += insertions_per_cycle


                    # 可选：打印每个循环的结果
                    if cycle % (iterations // 10 ) == 0:
                        simulation_interval_time = time.time()
                        print(f"循环 {cycle}/{iterations}: 成功插入 {success}/{insertions_per_cycle} 个粒子;耗时: {simulation_interval_time - simulation_start_time:.2f} 秒")
                
            simulation_end_time_j = time.time()

            final_probability = total_success / total_attempts

            if final_probability>0:
                lnpi=math.log(final_probability)
            else:
                lnpi=0

            results[round(i*10)-1][round(j/100)-1] += lnpi

            print(f"\n堆叠密度为{i}，粒子数为{j}的循环模拟和插入测试完成，耗时: {simulation_end_time_j - simulation_start_time_j:.2f} 秒")
            print(f"最终插入成功概率: {final_probability * 100:.5f}% ({total_success}/{total_attempts}) ;ln(Pi)={lnpi}")

            with open('./result/convex/convex_{:.2f}_result.txt'.format(i),'a') as file:
                file.write(f"\n堆叠密度为{i}，粒子数为{j}的循环模拟和插入测试完成，耗时: {simulation_end_time_j - simulation_start_time_j:.2f} 秒\n")
                file.write(f"最终插入成功概率: {final_probability * 100:.5f}% ({total_success}/{total_attempts}) ;ln(Pi)={lnpi}\n")
            
        with open('./result/convex/convex_{:.2f}_result.txt'.format(i),'a') as file:
                file.write(f'{results[round(i*10)-1]}\n')

    with open('./result/convex/convex_result.txt','a') as file:
        file.write(f'\n{results}\n')

if __name__ == '__main__':
    main()
