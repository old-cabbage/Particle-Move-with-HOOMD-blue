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

def main():
    #粒子总数
    num_particles = list(range(200,5200,200))
    #混合比例
    concave_mixture_ratio=0.5
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 2000
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=0.5

    device=hoomd.device.GPU()
    d=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.746718169951576, 4.3082473952995315, 3.877652377233043, 3.5066465278503713, 3.1963617307912258, 2.856466975681768, 2.5499559595837953, 2.2734688500455986, 2.02649131322099, 1.8030549620277818, 1.6098857687883146, 1.4350287445670382, 1.2784644172418649, 1.142809827106684, 1.023651206519947, 0.9164468721198238, 0.8200582955812928, 0.7347982280728644, 0.6627190205913235]
    a=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.772556076277117, 2.879167247884582, 2.310629346218518, 1.8889315328460161, 1.570407140816734, 1.3027796612473226, 1.1025217244193217, 0.9444887437013632, 0.8214364540103443]
    particle_area_equivalent=5*math.sqrt(63)/16*(1-concave_mixture_ratio)+7/2*concave_mixture_ratio
    for index in range(10):
        for i in num_particles:
            packing_density = packing_density_0 * i/5000

            mc = hoomd.hpmc.integrate.SimplePolygon(default_d=d[int(i/200)-1],default_a=a[int(i/200)-1])
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
            
            #创建实例
            system=System(
                num=i,packing_density=packing_density,
                packing_density_0=packing_density_0,
                particle_area=particle_area_equivalent,
                mc=mc,
                pre_random=pre_random,
                device=device,mixture=True
            )

            system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
            system.simulation.operations.integrator = mc
            system.simulation.create_state_from_gsd(filename='./gsd_mixture/P_{:.2f}_{:.2f}_{}_{}.gsd'.format(system.concave_mixture_ratio,system.packing_density_0,system.num,index+1))

            system.save_to_gsd()
            #print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")
            system.simulation.run(2000)
            print(f"循环{index+1}/10,密度为{packing_density_0},粒子数为{i}，的重打乱完成")

if __name__ == '__main__':
    main()