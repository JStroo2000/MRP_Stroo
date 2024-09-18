from amuse.community.fi.interface import Fi
from amuse.units import units, constants, nbody_system
from amuse.ic.molecular_cloud import new_molecular_cloud
from tests.plotting import HydroPlotter
import matplotlib.pyplot as plt
n_particles = 5000
M_cloud = 1e4 | units.Msun
r_cloud = 19 | units.pc
T_gas = 300 | units.K
v_collision = 7 | units.kms
t_end = 5.0e6 | units.yr
dt = 44e3 | units.yr

conv =  nbody_system.nbody_to_si(M_cloud, r_cloud)
cloud1 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud1.x += r_cloud
cloud1.vx -= v_collision

cloud2 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud2.x -= r_cloud
cloud2.vx += v_collision
cloud2.add_particles(cloud1)
particles = cloud2.copy()
#PLACEHOLDER: replace Fi with Phantom when ready
sph = Fi(conv, redirection='none')
sph.parameters.self_gravity_flag = True
sph.parameters.use_hydro_flag = True
sph.parameters.isothermal_flag = True
sph.parameters.integrate_entropy_flag = False
sph.parameters.gamma = 1
sph.parameters.verbosity = 0
sph.parameters.timestep = dt

sph.gas_particles.add_particles(particles)

sph_channel = sph.particles.new_channel_to(particles)

t = 0|units.yr
sph.evolve_model(100|units.yr)
sph_channel.copy()
plotter = HydroPlotter(500, use_torch=False)
plotter.add_gas_particles(particles)
plotter.plot_projection(200,params_to_plot=['density'],use_gaussian_kernel=True)
while t <= t_end:
    t += dt
    sph.evolve_model(t)
    sph_channel.copy()
    #print(t)
plotter.plot_projection(200,params_to_plot=['density'],use_gaussian_kernel=False)
plotter.clear_all_particles()