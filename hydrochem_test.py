import numpy
from matplotlib import pyplot

from amuse.units import units, constants, nbody_system
from amuse.ext.molecular_cloud import molecular_cloud

from amuse.community.fi.interface import Fi
from amuse.community.krome.interface import Krome

N_part = 1000
M_cloud = 10 | units.MSun
R_cloud = 1 | units.pc
dt = 1 | units.kyr
t_end = 1 | units.Myr

conv = nbody_system.nbody_to_si(M_cloud, R_cloud)
sph = Fi(conv, mode='openmp', redirection='none')

particles = molecular_cloud(targetN=N_part,convert_nbody=conv, ethep_ratio=0.05, ekep_ratio=0.5).result
print(particles.position)


sph.parameters.self_gravity_flag = True
sph.parameters.use_hydro_flag = True
sph.parameters.isothermal_flag = True
sph.parameters.integrate_entropy_flag = False
sph.parameters.gamma = 1
sph.parameters.verbosity = 0
sph.parameters.timestep = dt

sph.gas_particles.add_particles(particles)
t = 0 | units.kyr
while t <= t_end:
    t += dt
    sph.evolve_model(t)
    print('Time is {}'.format(t))
    print('Virial radius is {}'.format(sph.gas_particles.Virial()))

pyplot.plot(particles.x.value_in(units.pc),particles.y.value_in(units.pc))
pyplot.show()