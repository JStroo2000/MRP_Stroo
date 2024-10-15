from amuse.community.fi.interface import Fi
from amuse.community.phantom.interface import Phantom
from amuse.community.uclchem.interface import UCLchem
from amuse.units import units, constants, nbody_system
from amuse.ic.molecular_cloud import new_molecular_cloud
from amuse.ext.sink import new_sink_particles
from amuse.datamodel.particles import Particles
from amuse.io import write_set_to_file, read_set_from_file

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

def update_chem(hydro_parts, chem_parts):
    channel = hydro_parts.new_channel_to(chem_parts)
    channel.transform(["number_density", "temperature", "ionrate","radfield"], get_n_T_xi, ["density", "u"])

def check_new_sinks(parts, sink_density):
    over_thresh = parts.select_array(lambda x : x >= sink_density, ['density'])
    new_sinks = over_thresh.copy()
    parts.remove_particles(over_thresh)
    return new_sinks

def accrete(parts, sinks):
    removed_particles = Particles()
    for s in sinks:
        xs,ys,zs=s.x,s.y,s.z
        radius_squared = s.radius**2
        sinks_collide = sinks[~s].select_array(lambda x,y,z: (x-xs)**2+(y-ys)**2+(z-zs)**2 <radius_squared,["x","y","z"])
        if len(sinks_collide) != 0:
            cm=s.position*s.mass
            p=s.velocity*s.mass
            s.mass+=sinks_collide.total_mass()
            s.position=(cm+sinks_collide.center_of_mass()*sinks_collide.total_mass())/s.mass
            s.velocity=(p+sinks_collide.total_momentum())/s.mass
            sinks.remove_particles(sinks_collide)
        insink=parts.select_array(lambda x,y,z: (x-xs)**2+(y-ys)**2+(z-zs)**2 <radius_squared,["x","y","z"])
        if len(insink)==0:
            return insink
        cm=s.position*s.mass
        p=s.velocity*s.mass
        s.mass+=insink.total_mass()
        s.position=(cm+insink.center_of_mass()*insink.total_mass())/s.mass
        s.velocity=(p+insink.total_momentum())/s.mass
        removed_particles.add_particles(insink)
    return removed_particles

def sample_chem_particles(particles, n):
    rng = numpy.random.default_rng()
    sample = rng.integers(0,high=len(particles.key),size=n,endpoint=True)
    uniques, ind = numpy.unique(sample, return_index=True)
    print(sample, uniques)
    if numpy.sort(sample) != uniques:
        sample[~ind] = rng.integers(low=len(particles.key), size=len(~ind))
        uniques, ind = numpy.unique(sample, return_index=True)
    return particles[sample]

def copy_chem_to_hydro(chem, hydro):
    CO_index = chem.get_index_of_species('CO')
    e_index = chem.get_index_of_species('E-')
    H2_index = chem.get_index_of_species('H2')
    Hi_index = chem.get_index_of_species('H+')

    hydro.gas_particles.co_abundance = chem.abundances[CO_index]


# def out_of_bounds(parts, bounds):
#     far_away = parts.select_array(lambda x, y, z: (x>box_size/2)or(x<-1*box_size/2)or(y>box_size/2)or(y<-1*box_size/2)or(z>box_size/2)or(z<-1*box_size/2),['x','y','z'])
#     parts.remove_particles(far_away)
start = time.perf_counter()
n_cores = 10
n_particles = 100000
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
print(sample_chem_particles(particles, 1000))
#particles = read_set_from_file('data/cloud_collision_hydro_104.txt', format='amuse')
chem = UCLchem(number_of_workers=n_cores)
#PLACEHOLDER: replace Fi with Phantom when ready
#sph = Fi(conv, redirection='none')
hydro = Phantom(conv, number_of_workers=n_cores)
# hydro.parameters.self_gravity_flag = True
# hydro.parameters.use_hydro_flag = True
# hydro.parameters.isothermal_flag = True
# hydro.parameters.integrate_entropy_flag = False
hydro.gamma = 1
# hydro.parameters.verbosity = 0
hydro.time_step = dt
hydro.rho_crit = sink_density_threshold
hydro.h_acc = sink_radius
#hydro.parameters.periodic_boundaries_flag = True
# hydro.parameters.periodic_box_size = box_size
hydro.gas_particles.add_particles(particles)

hydro_channel = hydro.gas_particles.new_channel_to(particles)
particles_to_hydro_channel = particles.new_channel_to(hydro.gas_particles)
print(hydro.gas_particles)
hydro.evolve_model(100|units.yr)
hydro_channel.copy()

chem.particles.abundances[chem.index_of_species('C')] = 1.40e-4
chem.particles.abundances[chem.index_of_species('N')] = 7.60e-5
chem.particles.abundances[chem.index_of_species('O')] = 3.20e-4
chem.particles.abundances[chem.index_of_species('S')] = 1.20e-5
chem.particles.abundances[chem.index_of_species('Mg')] = 1.40e-7
chem.particles.abundances[chem.index_of_species('si')] = 1.50e-7


update_chem(particles,particles)

#chem.particles.add_particles(particles)
#chem.out_species = ["OH", "OCS", "CO", "CS", "CH3OH"]

#chem_channel = chem.particles.new_channel_to(particles)
#particles_to_chem_channel = particles.new_channel_to(chem.particles)
t = 0|units.yr

sink_particles = Particles()


#plotter.plot_projection(200,params_to_plot=['density'],use_gaussian_kernel=True)
counter = 0
while t <= t_end:
    print(t)
    write_set_to_file(particles, 'data/cloud_collision_hydro_{}.txt'.format(counter),format='amuse', overwrite_file=True)
    write_set_to_file(sink_particles, 'data/cloud_collision_hydro_sinks_{}.txt'.format(counter), format='amuse', overwrite_file=True)
    t += dt
    hydro.evolve_model(t)
    update_chem(hydro.particles, chem.particles)
    #chem.evolve_model(t)
    hydro_channel.copy()
    #chem_channel.copy()
    new_sinks = check_new_sinks(particles, sink_density_threshold)
    if len(new_sinks)!=0:
        new_sinks.radius = sink_radius
        sink_particles.add_particles(new_sinks)
    accreted = accrete(particles, hydro.sink_particles)
    print(sink_particles)
    particles.remove_particles(accreted)
    # out_of_bounds(particles, box_size)
    particles_to_hydro_channel.copy()
    #particles_to_chem_channel.copy()
    counter+=1

end = time.perf_counter()
print("Time elapsed with {} particles and {} cores: {} seconds".format(n_particles,n_cores, start-end))
hydro.stop()
#chem.stop()
