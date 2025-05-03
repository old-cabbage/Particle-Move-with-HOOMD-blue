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
from multiprocessing import Pool, cpu_count

#粒子总数
num_particles = 5000
pre_random=100
device=hoomd.device.CPU()
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

def task(n):

    packing_density_0= (1+n//100)*0.1
    #for i in num_particles:

    #for j in packing_density_0:

    packing_density = packing_density_0 * num_particles /5000

    #创建实例
    system=System(
        num=num_particles,packing_density=packing_density,
        packing_density_0=packing_density_0,
        particle_area=particle_area,
        mc=mc,device=device,shape=shape,sdf_file_num=n%100+1
    )
    system.generate_system()

    print(f"\n密度为{packing_density_0},第{system.sdf_file_num},正在预热系统，进行 {pre_random} 次移动...")
    start_time=time.time()
    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=['type_shapes'])
    gsd_writer = hoomd.write.GSD(filename='gsd_sdf/convex/P_convex_{:.2f}_{}_{}.gsd'.format(system.packing_density_0,system.num,system.sdf_file_num),
                                    trigger=hoomd.trigger.Periodic(100),
                                    mode='wb',filter=hoomd.filter.All(),
                                    logger=logger)
    system.simulation.operations.writers.append(gsd_writer)
    system.simulation.run(pre_random)
    end_time = time.time()
    print(f"密度为{packing_density_0},第{system.sdf_file_num},预热完成，耗时: {end_time - start_time:.2f} 秒")

def main():
    if __name__ == '__main__':
        data = list(range(500)) 
        num_workers = cpu_count()-1

        with Pool(processes=num_workers) as pool:
            pool.map(task, data)

if __name__ == '__main__':
    main()