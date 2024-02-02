import numpy
from matplotlib import pyplot

from amuse.units import units, constants, nbody_system
from amuse.ext.molecular_cloud import molecular_cloud

from amuse.community.fi.interface import Fi
from amuse.community.krome.interface import Krome

def get_n_T_xi(density,u):
    gamma = 1.4
    meanmwt = 1.35 | units.amu
    ionization_rate = 2.e-17 | units.s**-1
    number_density = density/meanmwt
    temperature = ((gamma-1)*meanmwt*u/constants.kB)
    ionrate=ionization_rate*numpy.ones_like(density)
    return (number_density, temperature, ionrate)
    
N_part = 100
M_cloud = 10 | units.MSun
R_cloud = 1 | units.pc
dt = 1 | units.kyr
t_end = 1 | units.Myr

conv = nbody_system.nbody_to_si(M_cloud, R_cloud)
chem = Krome(redirection='none')

sph = Fi(conv, mode='openmp', redirection='none')


particles = molecular_cloud(targetN=N_part,convert_nbody=conv, ethep_ratio=0.05, ekep_ratio=0.5).result
rho_cloud = M_cloud / (4/3 * numpy.pi * R_cloud**3)
particles.density = rho_cloud
particles.number_density, particles.temperature, particles.ionrate = get_n_T_xi(particles.density, particles.u)
channel = sph.gas_particles.new_channel_to(particles)

print(particles)
sph.parameters.self_gravity_flag = True
sph.parameters.use_hydro_flag = True
sph.parameters.isothermal_flag = True
sph.parameters.integrate_entropy_flag = False
sph.parameters.gamma = 1
sph.parameters.verbosity = 0
sph.parameters.timestep = dt

sph.gas_particles.add_particles(particles)
chem.particles.add_particles(particles)


f=pyplot.figure()
pyplot.ion()
pyplot.show()

t = 0 | units.kyr
density = [rho_cloud.value_in(units.g*units.cm**-3)]
time = [t.value_in(units.kyr)]
while t <= t_end:
    t += dt
    
    sph.evolve_model(t)
    chem_channel = sph.particles.new_channel_to(chem.particles)
    chem_channel.transform(['number_density','temperature','ionrate'], get_n_T_xi, ['density','u'])
    chem.evolve_model(t)
    channel.copy()
    if t.value_in(units.kyr)%5 == 0:
        print('Time is {}'.format(t))
        time.append(t.value_in(units.kyr))
        n = (sph.particles.density/(1.35|units.amu)).value_in(units.cm**-3)
        rho_cloud = M_cloud / (4/3 * numpy.pi * R_cloud**3)
        density.append(rho_cloud.value_in(units.g*units.cm**-3))
        h2 = (chem.particles.abundances[:,chem.species['H2']])
        co = chem.particles.abundances[:,chem.species['CO']]
    
        pyplot.clf()
        pyplot.loglog(n,h2,'r.',label='H2')
        pyplot.loglog(n,co,'g.',label='CO')
        pyplot.xlabel('density(cm^-3)')
        pyplot.ylabel('Abundance')
        pyplot.legend()
        f.canvas.flush_events()

    


pyplot.plot(time,density)
pyplot.show()

pyplot.plot(particles.x.value_in(units.pc),particles.y.value_in(units.pc),'.')
pyplot.show()
