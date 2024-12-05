from amuse.community.fi.interface import Fi
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
from random import sample
try:
    import queue
except:
    import Queue as queue
import threading
import multiprocessing

code_queue = queue.Queue()

def remote_worker_code():
    code = code_queue.get()
    evolve_single_chem(code)
    code_queue.task_done()

def evolve_with_multithreading(codes):
    for ci in codes:
        code_queue.put(ci)
    n_cpu = multiprocessing.cpu_count()
    for i in range(n_cpu):
        th = threading.Thread(target=remote_worker_code)
        th.daemon = True
        th.start()
    code_queue.join()

def evolve_single_chem(code):
    time = code.uclchem_time
    code.evolve_model(time+(44e2|units.yr))

    

def get_n_T_xi(density, u):
    gamma = 7/5
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
    #Function for sink particles; unused
    over_thresh = parts.select_array(lambda x : x >= sink_density, ['density'])
    new_sinks = over_thresh.copy()
    parts.remove_particles(over_thresh)
    return new_sinks

def accrete(parts, sinks):
    #Function for sink particles; unused
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
    chem_sample = sample(range(len(particles.key)+1),n)
    return particles[chem_sample]

#Parameters used in the simulation
start = time.perf_counter()
n_cores = 194
n_particles = 100000
M_cloud = 1e4 | units.Msun
r_cloud = 19 | units.pc
T_gas = 300 | units.K
v_collision = 7 | units.kms
UV_field = 1.7 | units.habing
ionization_rate = 1e-16 | units.s**-1
t_end = 5.0e6 | units.yr
dt = 44e2 | units.yr
sink_density_threshold = 2e-16 |units.g*units.cm**-3
sink_radius = 9e-4 | units.pc
box_size = 96 | units.pc

conv =  nbody_system.nbody_to_si(M_cloud, r_cloud)
#Initial cloud setup
cloud1 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud1.x += r_cloud
cloud1.vx -= v_collision

cloud2 = new_molecular_cloud(targetN=n_particles, convert_nbody=conv)
cloud2.x -= r_cloud
cloud2.vx += v_collision
cloud2.add_particles(cloud1)
particles = cloud2.copy()
#Sample chemical particles randomly from hydro particles
chem_sample = sample_chem_particles(particles, 2000)

hydro = Fi(conv,mode='openmp', redirection='none')
#Parameters for hydro
hydro.parameters.self_gravity_flag = True
hydro.parameters.use_hydro_flag = True
hydro.parameters.radiation_flag = True
hydro.star_formation_flag = False
hydro.parameters.isothermal_flag = False
hydro.parameters.integrate_entropy_flag = False
hydro.parameters.gamma = 7/5
hydro.parameters.gas_epsilon = 0.1|units.pc
hydro.parameters.sph_h_const = 0.1|units.pc
hydro.parameters.eps_is_h_flag = False
hydro.parameters.timestep = dt
hydro.parameters.periodic_box_size = box_size
hydro.commit_parameters()
hydro.gas_particles.add_particles(particles)

hydro_channel = hydro.particles.new_channel_to(particles)
particles_to_hydro_channel = particles.new_channel_to(hydro.gas_particles)
#Relax hydro
hydro.evolve_model(100|units.yr)
hydro_channel.copy()

particles.u = (constants.kB * T_gas)/ ((7/5-1)*1.35|units.amu)
particles_to_hydro_channel.copy()
update_chem(particles,particles)
update_chem(particles,chem_sample)
print(hydro.particles.get_attribute_names_defined_in_store())
workers = []
chem_samples = numpy.array_split(numpy.arange(len(chem_sample)), n_cores)
#Create chem instances for parallel computing
for i in range(n_cores):
    chem = UCLchem()
    
    chem.particles.add_particles(chem_sample[chem_samples[i][0]:chem_samples[i][-1]+1])
    chem.particles.abundances[:,chem.get_index_of_species('C')] = 1.40e-4
    chem.particles.abundances[:,chem.get_index_of_species('N')] = 7.60e-5
    chem.particles.abundances[:,chem.get_index_of_species('O')] = 3.20e-4
    chem.particles.abundances[:,chem.get_index_of_species('S')] = 1.20e-5
    chem.particles.abundances[:,chem.get_index_of_species('Mg')] = 1.40e-7
    chem.particles.abundances[:,chem.get_index_of_species('Si')] = 1.50e-7
    chem.out_species = ["OH", "OCS", "CO", "CS", "CH3OH"]
    workers.append(chem)
    

t = 0|units.yr
counter = 0
while t <= t_end:
    #print(t)
    write_set_to_file(hydro.particles, 'data/cloud_collision_{}.txt'.format(counter),format='amuse', overwrite_file=True)
    #Reunite the particles in all the chem instances to save them in 1 file
    chem_to_save = Particles()
    
    for i in workers:
        chem_to_save.add_particles(i.particles)
    write_set_to_file(chem_to_save, 'data/chem_sample_{}.txt'.format(counter), format='amuse', overwrite_file=True)
    t += dt
    #Evolve hydro
    hydro.evolve_model(t)
    #Update values in chem particles
    for i in workers:
        update_chem(hydro.particles, i.particles)
    #Evolve chem
    evolve_with_multithreading(workers)
    #Copy hydro back to the main particle set
    hydro_channel.copy()
    counter+=1
   
end = time.perf_counter()
print("Time elapsed with {} particles and {} cores: {} seconds".format(n_particles,n_cores, end-start))
hydro.stop()
for i in workers:
    i.stop()
