import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system import System
import time

def main():
    #粒子总数
    num_particles = list(range(100,5100,100))
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 100
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=[0.1,0.2,0.3,0.4,0.5]
    device=hoomd.device.GPU()
    shape='A'
    mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=2,default_a=3)
    mc.shape[shape] = dict(
                vertices = [
                (-2, 0),
                (2, 0),
                (11/8, 5*math.sqrt(63)/8),
                ]
    )
    particle_area=5*math.sqrt(63)/4

    start_time=time.time()

    for j in packing_density_0:

        for i in num_particles:

            for k in range(20):

                packing_density = j * i /5000
                
                #创建实例
                system=System(
                    num=i,packing_density=packing_density,
                    packing_density_0=j,
                    particle_area=particle_area,
                    mc=mc,pre_random=pre_random,device=device)
                
                print(f"正在初始化系统， 密度{j}，粒子数{i}，次序{k+1}进行 {pre_random} 次移动")
                system.generate_system()

                logger = hoomd.logging.Logger()
                logger.add(system.mc, quantities=['type_shapes'])
                gsd_writer = hoomd.write.GSD(filename='./gsd/P_convex_{:.2f}_{}_{}.gsd'.format(system.packing_density_0,system.num,k+1),
                                            trigger=hoomd.trigger.Periodic(100),
                                            mode='wb',filter=hoomd.filter.All(),
                                            logger=logger)
                system.simulation.operations.writers.append(gsd_writer)
                system.simulation.run(pre_random)
                end_time = time.time()
                print(f"密度{j}，粒子数{i}，次序{k+1}，初始化系统完成,耗时{end_time-start_time:.2f}秒\n")
        j_end_time=time.time()
    print(f"密度{j}初始化完成，耗时{j_end_time-start_time:2f}秒")
    
if __name__ == '__main__':
    main()