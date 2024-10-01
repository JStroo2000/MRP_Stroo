from amuse.community.fi.interface import Fi
from amuse.community.uclchem.interface import UCLchem
from amuse.units import units, constants, nbody_system
from amuse.ic.molecular_cloud import new_molecular_cloud
from amuse.ext.sink import new_sink_particles
from amuse.datamodel.particles import Particles
from amuse.io import write_set_to_file

from tests.plotting import HydroPlotter
import matplotlib.pyplot as plt
import numpy
import time 

def get_n_T_xi(density, u):
    gamma = 1.4
    meanmwt = 1.35 | units.amu
    number_density=density/meanmwt
    temperature=((gamma - 1) * meanmwt * u / constants.kB)
    ionrate=ionization_rate*numpy.ones_like(density)
    radfield = 1.7|units.habing
    return (number_density, temperature, ionrate, radfield)

def update_chem(sph_parts, chem_parts):
    channel = sph_parts.new_channel_to(chem_parts)
    channel.transform(["number_density", "temperature", "ionrate","radfield"], get_n_T_xi, ["density", "u"])

def check_new_sinks(parts, sink_density):
    over_thresh = parts.select_array(lambda x : x >= sink_density, ['density'])
    new_sinks = over_thresh.copy()
    parts.remove_particles(over_thresh)
    return new_sinks

# def out_of_bounds(parts, bounds):
#     far_away = parts.select_array(lambda x, y, z: (x>box_size/2)or(x<-1*box_size/2)or(y>box_size/2)or(y<-1*box_size/2)or(z>box_size/2)or(z<-1*box_size/2),['x','y','z'])
#     parts.remove_particles(far_away)
start = time.perf_counter()
n_cores = 2
n_particles = 100
M_cloud = 1e4 | units.Msun
r_cloud = 19 | units.pc
T_gas = 300 | units.K
v_collision = 7 | units.kms
UV_field = 1.7 | units.habing
ionization_rate = 1e-16 | units.s**-1
t_end = 5.0e6 | units.yr
dt = 44e3 | units.yr
sink_density_threshold = 2e-16 |units.g*units.cm**-3
sink_radius = 9e-4 | units.pc
box_size = 96 | units.pc

conv =  nbody_system.nbody_to_si(M_cloud, r_cloud)
cloud1 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud1.x += r_cloud
cloud1.vx -= v_collision

cloud2 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud2.x -= r_cloud
cloud2.vx += v_collision
cloud2.add_particles(cloud1)
particles = cloud2.copy()

chem = UCLchem(number_of_workers=n_cores)
#PLACEHOLDER: replace Fi with Phantom when ready
sph = Fi(conv, redirection='none')
sph.parameters.self_gravity_flag = True
sph.parameters.use_hydro_flag = True
sph.parameters.isothermal_flag = True
sph.parameters.integrate_entropy_flag = False
sph.parameters.gamma = 1
sph.parameters.verbosity = 0
sph.parameters.timestep = dt
#sph.parameters.periodic_boundaries_flag = True
sph.parameters.periodic_box_size = box_size
sph.gas_particles.add_particles(particles)

sph_channel = sph.particles.new_channel_to(particles)
particles_to_sph_channel = particles.new_channel_to(sph.particles)

sph.evolve_model(100|units.yr)
sph_channel.copy()
update_chem(particles,particles)
print(particles)

chem.particles.add_particles(particles)
chem.out_species = ["OH", "OCS", "CO", "CS", "CH3OH"]

chem_channel = chem.particles.new_channel_to(particles)
particles_to_chem_channel = particles.new_channel_to(chem.particles)
t = 0|units.yr

sink_particles = new_sink_particles(sph.dm_particles, sink_radius)

plotter = HydroPlotter(500, use_torch=False)
plotter.add_gas_particles(particles)
#plotter.plot_projection(200,params_to_plot=['density'],use_gaussian_kernel=True)
counter = 0
while t <= t_end:
    print(t)
    write_set_to_file(particles, 'data/cloud_collision_{}.txt'.format(counter),format='amuse', overwrite_file=True)
    t += dt
    sph.evolve_model(t)
    update_chem(sph.particles, chem.particles)
    chem.evolve_model(t)
    sph_channel.copy()
    chem_channel.copy()
    new_sinks = check_new_sinks(particles, sink_density_threshold)
    sph.dm_particles.add_particles(new_sinks)
    sink_particles.add_sinks(new_sinks)
    accreted = sink_particles.accrete(particles)
    particles.remove_particles(accreted)
    # out_of_bounds(particles, box_size)
    particles_to_sph_channel.copy()
    particles_to_chem_channel.copy()
    counter+=1

plotter.plot_projection(200,params_to_plot=['density'],use_gaussian_kernel=True)
plotter.clear_all_particles()
end = timer.perf_counter()
print("Time elapsed with {} particles and {} cores: {} seconds".format(n_particles,n_cores, start-end))