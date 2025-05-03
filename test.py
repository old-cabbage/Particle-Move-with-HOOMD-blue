import math
import hoomd
import hoomd.simulation
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import random
from convex.system import System

#粒子总数
N=5000
num_particles = 5000

packing_density_0=0.2
packing_density = packing_density_0 * num_particles/N

divice=hoomd.device.GPU()


mc = hoomd.hpmc.integrate.ConvexSpheropolygon(
    default_d=0.3, default_a=0.4
)
mc.shape["A"] = dict(
    vertices=[
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
    ],
    sweep_radius=0.3,
)

area=2.2+0.09*math.pi


system=System(
    num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            mc=mc,device=divice,shape='A',particle_area=area
)

system.generate_system()

print(mc.kT)
print(mc.type_shapes)

logger = hoomd.logging.Logger()
logger.add(system.mc, quantities=['type_shapes'])

gsd_writer = hoomd.write.GSD(filename='gsd_test/P_{:.2f}_{}.gsd'.format(system.packing_density_0,system.num),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)

system.simulation.operations.writers.append(gsd_writer)
#print("vertices = ", mc.shape["A"]["vertices"])

#simulation = hoomd.Simulation(device=divice,seed=random.randint(1,10000))
#simulation.operations.integrator = mc

#simulation.create_state_from_gsd(filename='gsd/P_convex_0.50_5000_1.gsd')

#print(simulation.state.box.Lx,simulation.state.box.Ly,simulation.state.box.Lz)

#print(simulation.state.box.volume)
#simulation.run(100)

#print(simulation.state.N_particles)

system.simulation.run(100)

print(mc.kT)
print(mc.translate_moves)


total_sdf_xcompression,total_sdf_xexpansion=system.calculate_sdf(sdf_mc=10,sdf_xmax=0.02,sdf_dx=1e-4,sdf_each_run=10)

print(system.sdf_compute.betaP)
print(system.sdf_compute.P)