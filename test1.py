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

packing_density_0=0.5
packing_density = packing_density_0 * num_particles/N

device=hoomd.device.GPU()

mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=2,default_a=3)
mc.shape['A'] = dict(
            vertices = [
            (-2, 0),
            (2, 0),
            (11/8, 5*math.sqrt(63)/8),
            ]
)
particle_area=5*math.sqrt(63)/4


system=System(
    num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            mc=mc,device=device,shape='A',particle_area=particle_area
)

system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
system.simulation.operations.integrator = mc
system.simulation.create_state_from_gsd(filename="gsd/P_0.50_4900.gsd")

#print(mc.kT)
#print(mc.type_shapes)
#print(mc.translate_moves)

#print("vertices = ", mc.shape["A"]["vertices"])

#simulation = hoomd.Simulation(device=divice,seed=random.randint(1,10000))
#simulation.operations.integrator = mc

#simulation.create_state_from_gsd(filename='gsd/P_convex_0.50_5000_1.gsd')

#print(simulation.state.box.Lx,simulation.state.box.Ly,simulation.state.box.Lz)

#print(simulation.state.box.volume)
#simulation.run(100)

#print(simulation.state.N_particles)

system.simulation.run(100)

logger = hoomd.logging.Logger()
logger.add(system.mc, quantities=['type_shapes'])

hoomd.write.GSD.write(state=system.simulation.state, mode='wb', filter=hoomd.filter.All(),filename="P_0.50_4900.gsd",logger=logger)

#print(mc.kT)
print(mc.translate_moves)


total_sdf_xcompression,total_sdf_xexpansion,total_sdf_sdfcompression,total_sdf_sdfexpansion=system.calculate_sdf(sdf_mc=10,sdf_xmax=0.02,sdf_dx=1e-4,sdf_each_run=10)

print(total_sdf_xcompression)
print(total_sdf_sdfcompression)

pho=packing_density/particle_area
print(system.sdf_compute.betaP/pho-1)
#print(system.sdf_compute.P)