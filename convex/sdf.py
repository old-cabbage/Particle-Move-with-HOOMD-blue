import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from system import System
import time
from multiprocessing import Pool, cpu_count
import random

index=1
total_index=5
cyc_num=20

#粒子总数
N=5000
num_particles = 5000

packing_density_0=0.3
packing_density = packing_density_0 * num_particles/N

divice=hoomd.device.CPU()
shape='A'
mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=3,default_a=2)
mc.shape[shape] = dict(
                vertices = [
                (-2, 0),
                (2, 0),
                (11/8, 5*math.sqrt(63)/8),
                ]
    )
particle_area=5*math.sqrt(63)/4

sdf_mc=5000
sdf_xmax=0.02
sdf_dx=1e-4
sdf_each_run=5

def task(n):

    system=System(
            num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            particle_area=particle_area,
            mc=mc,device=divice,shape=shape
        )

    system.simulation = hoomd.Simulation(device=divice,seed=random.randint(1,10000))
    system.simulation.operations.integrator = mc
    system.simulation.create_state_from_gsd(filename='./gsd_sdf/convex/P_convex_{:.2f}_{}_{}.gsd'.format(packing_density_0,num_particles,n+1))

    total_sdf_xcompression,total_sdf_xexpansion,total_sdf_sdfcompression,total_sdf_sdfexpansion=system.calculate_sdf(sdf_mc,sdf_xmax,sdf_dx,sdf_each_run)

    return total_sdf_xcompression,total_sdf_xexpansion,total_sdf_sdfcompression,total_sdf_sdfexpansion

def main():
    
    if __name__ == '__main__':
        data = list(range(cyc_num*(index-1),cyc_num*index,1))
        num_workers = cpu_count()-1

        with Pool(processes=num_workers) as pool:
            results = pool.map(task, data)
    
    total_sdf_xcompression=np.zeros(int(sdf_xmax/sdf_dx))
    total_sdf_xexpansion=np.zeros(int(sdf_xmax/sdf_dx))
    total_sdf_sdfcompression=np.zeros(int(sdf_xmax/sdf_dx))
    total_sdf_sdfexpansion=np.zeros(int(sdf_xmax/sdf_dx))
    
    for i in range(len(results)):
        interval_sdf_xcompression,interval_sdf_xexpansion,interval_sdf_sdfcompression,interval_sdf_sdfexpansion=results[i]
        total_sdf_xcompression += interval_sdf_xcompression
        total_sdf_xexpansion += interval_sdf_xexpansion
        total_sdf_sdfcompression += interval_sdf_sdfcompression
        total_sdf_sdfexpansion += interval_sdf_sdfexpansion
    
    total_sdf_xcompression/=len(results)
    total_sdf_xexpansion/=len(results)
    total_sdf_sdfcompression/=len(results)
    total_sdf_sdfexpansion/=len(results)

    print(total_sdf_xcompression.tolist()) 
    print(total_sdf_xexpansion.tolist()) 
    print(total_sdf_sdfcompression.tolist())
    print(total_sdf_sdfexpansion.tolist()) 

    with open("result/sdf/convex_sdf_{:.2f}_{}.txt".format(packing_density_0,num_particles),'a') as file:
        file.write(f"序列{index}/{total_index}\n堆叠密度为{packing_density_0},{cyc_num}个{sdf_mc}次sdf_mc循环,调用了{num_workers}个核心\n压缩sdf\n{total_sdf_xcompression.tolist()}\n{total_sdf_sdfcompression.tolist()}\n膨胀sdf\n{total_sdf_xexpansion.tolist()}\n{total_sdf_sdfexpansion.tolist()}\n\n")

if __name__ == '__main__':
    main()