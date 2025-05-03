import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system_mixture import System
import time

def main():
    #粒子总数
    num_particles = list(range(200,5200,200))
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 1000
    #混合比例
    concave_mixture_ratio=0.5
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=0.5
    device=hoomd.device.GPU()
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=0.5,default_a=0.2)
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

    good_d=[0]*int(5000/200)
    good_a=[0]*int(5000/200)

    for index in range(10):

        for i in num_particles:
            packing_density = packing_density_0 * i /5000
            #创建实例
            system=System(
                num=i,packing_density=packing_density,
                packing_density_0=packing_density_0,
                particle_area=particle_area_equivalent,
                mc=mc,device=device,
                concave_mixture_ratio=concave_mixture_ratio,mixture=True
            )
            system.generate_system()

            print(f"\n正在预热系统，进行 {pre_random} 次移动...")
            start_time=time.time()
            logger = hoomd.logging.Logger()
            logger.add(system.mc, quantities=['type_shapes'])
            gsd_writer = hoomd.write.GSD(filename='./gsd_mixture/P_{:.2f}_{:.2f}_{}_{}.gsd'.format(system.concave_mixture_ratio,system.packing_density_0,system.num,index+1),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)
            system.simulation.operations.writers.append(gsd_writer)
            system.simulation.run(pre_random)
            end_time = time.time()
            print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")
            good_d[int(i/200)-1]+=system.mc.d['A']
            good_a[int(i/200)-1]+=system.mc.a['A']

            #for _ in range(5):
            #    system.simulation.run(100)
    good_d=[d/10 for d in good_d]
    good_a=[a/10 for a in good_a]
    print(good_d)
    print(good_a)

if __name__ == '__main__':
    main()