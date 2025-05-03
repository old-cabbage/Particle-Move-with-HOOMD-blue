import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from system_mixture import System
import time
from multiprocessing import Pool, cpu_count
import random

#index=1
packing_density_0=0.4
#mpi
cyc_num=10
#total_upload=10
#混合比例
concave_mixture_ratio=0.9

num_particles = 5000

sdf_mc=100
sdf_xmax=0.01
sdf_dx=1e-4
sdf_each_run=5

#粒子总数
N=5000
#粒子的总面积/盒子面积为packing_density
packing_density = packing_density_0 * num_particles/N
device=hoomd.device.CPU()

d=[20.0,7.25,2.5,1.24,0.66]
a=[20.0,20.0,10.0,1.66,0.67]

#mc = hoomd.hpmc.integrate.SimplePolygon(default_d=d[round(packing_density_0)-1],default_a=a[round(packing_density_0)-1])
mc = hoomd.hpmc.integrate.SimplePolygon(default_d=3,default_a=3)
mc.shape['A'] = dict(
            vertices = [
            (-1, 0),
            (1, 0),
            (11/16, 5*math.sqrt(63)/16)
            ]
)
mc.shape['B'] = dict(
            vertices = [
            (-1, 0),
            (1, 0),
            (1,2),
            (0,1),
            (-2,2)
            ]
)
particle_area_equivalent=5*math.sqrt(63)/16*(1-concave_mixture_ratio)+7/2*concave_mixture_ratio

def task(n):


    system=System(
            num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            particle_area=particle_area_equivalent,
            mc=mc,device=device,concave_mixture_ratio=concave_mixture_ratio
        )

    system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
    system.simulation.operations.integrator = mc
    system.simulation.create_state_from_gsd(filename='gsd_sdf/mixture/P_{:.2f}_{:.2f}_{}_{}.gsd'.format(system.concave_mixture_ratio,system.packing_density_0,system.num,n+1))

    print(f"堆叠密度为{packing_density_0},第{n+1}个{sdf_mc}次sdf_mc循环")

    total_sdf_xcompression,total_sdf_xexpansion,total_sdf_sdfcompression,total_sdf_sdfexpansion=system.calculate_sdf(sdf_mc,sdf_xmax,sdf_dx,sdf_each_run)

    return total_sdf_xcompression,total_sdf_xexpansion,total_sdf_sdfcompression,total_sdf_sdfexpansion

def main():
  
    if __name__ == '__main__':
        data = list(range(cyc_num))
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

    with open("result/sdf/mixture_sdf_{:.2f}_{:.2f}_{}.txt".format(concave_mixture_ratio,packing_density_0,num_particles),'a') as file:
        file.write(f"堆叠密度为{packing_density_0},{cyc_num}个{sdf_mc}次sdf_mc循环,调用了{num_workers}个核心\n压缩sdf\n{total_sdf_xcompression.tolist()}\n{total_sdf_sdfcompression.tolist()}\n膨胀sdf\n{total_sdf_xexpansion.tolist()}\n{total_sdf_sdfexpansion.tolist()}\n\n")

if __name__ == '__main__':
    main()