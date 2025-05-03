import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from system import System,add_data_with_auto_id
import time
from multiprocessing import Pool, cpu_count
import random
import json
import os

index=1
total_index=1
cyc_num=10

#粒子总数
N=5000
num_particles = 5000

packing_density_0=0.5
packing_density = packing_density_0 * num_particles/N

divice=hoomd.device.CPU()
shape='A'
mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=3,default_a=3)
mc.shape[shape] = dict(
                vertices = [
                (-2, 0),
                (2, 0),
                (11/8, 5*math.sqrt(63)/8),
                ]
    )
particle_area=5*math.sqrt(63)/4

sdf_mc=10
sdf_xmax=0.02
sdf_dx=1e-4
sdf_each_run=5

pho=packing_density/particle_area

sdf_compute=hoomd.hpmc.compute.SDF(xmax=sdf_xmax,dx=sdf_dx)

start_time=time.time()

def task(n):
    result=[]
    simulation = hoomd.Simulation(device=divice,seed=random.randint(1,10000))
    simulation.operations.integrator = mc
    simulation.operations.computes.append(sdf_compute)

    simulation.create_state_from_gsd(filename='gsd_sdf/convex/P_convex_{:.2f}_{}_{}.gsd'.format(packing_density_0,num_particles,n+1))

    target_value=0

    for i in range(sdf_mc):
        simulation.run(sdf_each_run)
        result.append(sdf_compute.betaP/pho-1)

    return result

def main():
    
    if __name__ == '__main__':
        data = list(range(cyc_num*(index-1),cyc_num*index,1)) 
        num_workers = 25

        with Pool(processes=num_workers) as pool:
            oriresults = pool.map(task, data)

    results=[]
    for i in range(len(oriresults)):
        results+=oriresults[i]

    target_value=0
    for i in range(len(results)):
        target_value+=results[i]
    target_value/=len(results)

    end_time=time.time()
    with open("result/sdf/convex/convex_sdf_bp_{:.2f}_{}.txt".format(packing_density_0,num_particles),'a') as file:
        file.write(f"序列{index}/{total_index}\n堆叠密度为{packing_density_0}, {cyc_num}个{sdf_mc}次sdf计算结束,计算结果为{target_value},耗时 {end_time-start_time:.2f}秒\n")
        file.write(f"原始数据\n{results}\n\n")

    json_data={
        "info":f"序列{index}/{total_index}\n堆叠密度为{packing_density_0}, {cyc_num}个{sdf_mc}次sdf计算结束,计算结果为{target_value},耗时 {end_time-start_time:.2f}秒\n",
        "times":sdf_mc*cyc_num,
        "data":results
    }
    add_data_with_auto_id(new_data=json_data, target_class="0.5", filename="result/sdf/convex/convex_sdf.json")

if __name__ == '__main__':
    main()