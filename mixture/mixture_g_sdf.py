import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system_mixture import System
import time
import random
from multiprocessing import Pool, cpu_count

#粒子总数
num_particles = 5000
#每次开始插入之前，先对基础的系统预处理pre_random步数
pre_random = 100
#混合比例
concave_mixture_ratio=0.7
#packing_density = packing_density_0 * num_particles/5000
device=hoomd.device.CPU()
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
    
    packing_density_0= (1+n//100)*0.1

    #for i in num_particles:

    #for j in packing_density_0:

    packing_density = packing_density_0 * num_particles /5000

    #创建实例
    system=System(
        num=num_particles,packing_density=packing_density,
        packing_density_0=packing_density_0,
        particle_area=particle_area_equivalent,
        mc=mc,pre_random=pre_random,concave_mixture_ratio=concave_mixture_ratio,
        device=device,sdf_file_num=n%100+1,mixture=True
    )
    system.generate_system()

    print(f"\n正在预热系统，进行 {pre_random} 次移动...")
    start_time=time.time()
    logger = hoomd.logging.Logger()
    logger.add(system.mc, quantities=['type_shapes'])
    gsd_writer = hoomd.write.GSD(filename='gsd_sdf/mixture/P_{:.2f}_{:.2f}_{}_{}.gsd'.format(system.concave_mixture_ratio,system.packing_density_0,system.num,n%100+1),
                                trigger=hoomd.trigger.Periodic(100),
                                mode='wb',filter=hoomd.filter.All(),
                                logger=logger)
    system.simulation.operations.writers.append(gsd_writer)
    system.simulation.run(pre_random)
    end_time = time.time()
    print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")

def main():
    if __name__ == '__main__':
        data = list(range(500)) 
        num_workers = cpu_count()

        with Pool(processes=num_workers) as pool:
            pool.map(task, data)

if __name__ == '__main__':
    main()