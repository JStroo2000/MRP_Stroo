import matplotlib.pyplot as plt
import numpy as np
from amuse.io import read_set_from_file
from tests.plotting import HydroPlotter
from amuse. units import units, constants
from amuse.community.uclchem.interface import Uclchem
from amuse.datamodel.particles import Particles
import pandas as pd
#import uclchem
#from uclchem.utils import get_species


# plt.rcParams['text.usetex']=True

def u_to_temperature(internal_energy):
    """
    Function that transforms internal energy to temperature
    (For now) assumes all gas is molecular hydrogen and gamma=5/3
    """
    temperature = (5/3 - 1) * (2.333|units.amu) * internal_energy / constants.kB

    return temperature

def find_keys(hydro, chem):
   
    temp_chem = np.load('chem_temperature.npy')
    dens_chem = np.load('chem_density.npy')[-1,:]
    keys= np.array([0,0])
    hydro_to_search = hydro.copy()
    hydro_to_search.number_density = (hydro_to_search.density/(1.35|units.amu))
    counter=0
    for i in chem.number_density.value_in(units.cm**-3):
        index = np.argmin(np.abs(i-hydro_to_search.number_density.value_in(units.cm**-3)))
        closest = hydro_to_search.number_density[index]
        #print(index)
        keys = np.vstack((keys,[hydro_to_search.key[index],chem.key[counter]]))
        hydro_to_search.remove_particle(hydro_to_search[index])

        counter+=1
    keys = np.delete(keys,0,0)
    print(np.unique(keys[:,0]).shape)

    return keys

def parse_data():
    plt.rc('font', size=60/1.5)            # controls default text sizes
    plt.rc('axes', titlesize=96/1.5)     # fontsize of the axes title
    plt.rc('axes', labelsize=72/1.5)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=60/1.5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=60/1.5)    # fontsize of the tick labels
    plt.rc('legend', fontsize=72/1.5)    # legend fontsize
    plt.rc('figure', titlesize=96/1.5) 
    dt = 44e2
    plotter = HydroPlotter(500, use_torch=False)
    particles = read_set_from_file('chem_run_data/data/cloud_collision_hydro_0.txt', format='amuse')
    chem_particles = read_set_from_file('chem_run_data/data/chem_sample_0.txt', format='amuse')
    print(np.intersect1d(particles.key, chem_particles.key))
    print(chem_particles.get_attribute_names_defined_in_store())
    t = [0]
    density = np.array([particles.density.value_in(units.amu/units.cm**3)])
    temperature = np.array([u_to_temperature(particles.u).value_in(units.K)])
    density_chem = np.array([chem_particles.number_density.value_in(units.cm**-3)])
    temperature_chem = np.array([chem_particles.temperature.value_in(units.K)])
    abundances = chem_particles.abundances
    print(abundances.shape)
    #plotter.add_gas_particles(particles)
    #plotter.add_star_particles(sinks)
    #plotter.plot_projection(96, save_fig="plots/hydro_0.pdf",params_to_plot=['density','temperature'], use_gaussian_kernel=False, cmap='magma', title='0 yr')
    #plotter.clear_all_particles()
    for i in range(1,1137):
        particles = read_set_from_file('chem_run_data/data/cloud_collision_hydro_{}.txt'.format(i), format='amuse')
        chem_particles = read_set_from_file('chem_run_data/data/chem_sample_{}.txt'.format(i), format='amuse')
        #sinks = read_set_from_file('data/cloud_collision_hydro_sinks_{}.txt'.format(i), format='amuse')
        t.append(i*dt)
        
        if i%20 == 0:
            density=np.vstack((density, particles.density.value_in(units.amu/units.cm**3)))
            temperature=np.vstack((temperature, u_to_temperature(particles.u).value_in(units.K)))
            density_chem=np.vstack((density_chem, chem_particles.number_density.value_in(units.cm**-3)))
            temperature_chem=np.vstack((temperature_chem, chem_particles.temperature.value_in(units.K)))
            print(abundances.shape, chem_particles.abundances.shape)
            abundances = np.dstack((abundances,chem_particles.abundances))
            print(abundances.shape)
            #plotter.add_gas_particles(particles)
            #plotter.add_star_particles(sinks)
            #plotter.plot_projection(96, save_fig="plots/hydro_{}.pdf".format(i),params_to_plot=['density'], use_gaussian_kernel=False, cmap='magma', title='{} yr'.format(i*dt))
           # plotter.clear_all_particles()
            print('plotted')
    plotter.add_gas_particles(particles)
    plotter.plot_projection(96, save_fig='plots/hydro_frontpage.pdf',params_to_plot=['density'],use_gaussian_kernel=False, cmap='magma')
    np.save('hydro_run_density.npy', density)
    np.save('hydro_run_temperature.npy', temperature)
    np.save('chem_density.npy', density_chem)
    np.save('chem_temperature.npy', temperature_chem)
    np.save('chem_abundances.npy', abundances)
    print('saved')

def plot_density_temperature():

    radius = 19|units.pc
    speed = 7|units.km/units.s
    col_time = (radius/speed).value_in(units.yr)
    print('col_time',col_time)
    density = np.load('hydro_run_density.npy')
    #density_avg = density.mean(axis=1)
    temperature = np.load('hydro_run_temperature.npy')
    #temperature_avg = temperature.mean(axis=1)
    density_chem = np.load('chem_density.npy')
    temperature_chem = np.load('chem_temperature.npy')

    y_dens = np.log10(density.flatten())
    y_temp = np.log10(temperature.flatten())
    y_dens_chem = np.log10(density_chem.flatten())
    y_temp_chem = np.log10(temperature_chem.flatten())


    time = np.arange(0, 1137*44e2, 20*44e2)
    time_hydro = np.repeat(time, 200000)
    time_chem = np.repeat(time, 2000)

    print(len(time_hydro),len(y_dens))

    hist_dens, xedge_dens, yedge_dens = np.histogram2d(time_hydro, y_dens, bins=56)
    hist_temp, xedge_temp, yedge_temp = np.histogram2d(time_hydro, y_temp, bins=56)

    hist_dens_chem, xedge_dens_chem, yedge_dens_chem = np.histogram2d(time_chem, y_dens_chem, bins=56)
    hist_temp_chem, xedge_temp_chem, yedge_temp_chem = np.histogram2d(time_chem, y_temp_chem, bins=56)

    hist_dens[hist_dens==0] = "Nan"
    hist_temp[hist_temp==0] = "Nan"
    hist_dens_chem[hist_dens_chem==0] = "Nan"
    hist_temp_chem[hist_temp_chem==0] = "Nan"

    # plt.plot(time, density_avg)
    # plt.title('average density')
    # plt.xlabel('time (yr)')
    # plt.ylabel('density (amu/cm^3)')
    # plt.savefig('avg_dens_hydrotest.pdf')
    # plt.close()

    # plt.plot(time, temperature_avg)
    # plt.title('average temperature')
    # plt.xlabel('time (yr)')
    # plt.ylabel('temperature (K)')
    # plt.savefig('avg_temp_hydrotest.pdf')
    # plt.close()

    # plt.plot(density[:,-1], temperature[:,-1], '.', alpha=0.1)
    # plt.title('Temperature as a function of density')
    # plt.xlabel('Density (amu/cm^3)')
    # plt.ylabel('Temperature (K)')
    # plt.savefig('density_temperature.pdf')
    # plt.close()

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    im =ax[0].pcolormesh(xedge_dens,yedge_dens, hist_dens.T, cmap='magma')
    ax[1].pcolormesh(xedge_temp,yedge_temp, hist_temp.T, cmap='magma')
    ax[0].set_xlabel('Time (yr)')
    ax[1].set_xlabel('Time (yr)')
    ax[0].set_ylabel('log density (amu/$cm^{-3}$)')
    ax[1].set_ylabel('log temperature (K)')
    # cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    # cbar.set_ticks([])
    # cbar.set_ticklabels([])
    fig.savefig('hydro_dens_temp.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1,2,figsize=(11,4))
    im = ax[0].pcolormesh(xedge_dens_chem,yedge_dens_chem, hist_dens_chem.T, cmap='magma')
    ax[1].pcolormesh(xedge_temp_chem,yedge_temp_chem, hist_temp_chem.T, cmap='magma')
    ax[0].set_xlabel('Time (yr)')
    ax[1].set_xlabel('Time (yr)')
    ax[0].vlines(col_time, -1,4, linestyle='dashed', color='r')
    ax[1].vlines(col_time,1,2.8, linestyle='dashed', color='r')
    ax[0].set_ylabel('log number density ($cm^{-3}$)')
    ax[1].set_ylim(10)
    ax[1].set_ylabel('log temperature (K)')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label("Number of particles")
    fig.savefig('chem_dens_temp.pdf', bbox_inches='tight')
    plt.close()

    hist_last, xedge_last, yedge_last = np.histogram2d(np.log10(density_chem[-1,:]),np.log10(temperature_chem[-1,:]), bins=56)
    hist_last[hist_last==0] = 'NaN'

    plt.pcolormesh(xedge_last, yedge_last, hist_last.T, cmap='magma')
    plt.xlabel('log Number density ($cm^{-3}$)')
    plt.ylabel('log temperature (K)')
    plt.colorbar(ticks=[],label='Number of particles')
    plt.savefig('temp_dens_last.pdf', bbox_inches='tight')
    plt.close()

def plot_abundances(species):
    abundances = np.load('chem_abundances.npy')
    temperatures = np.load('chem_temperature.npy')
    density = np.load('chem_density.npy')
    chem = Uclchem()

    radius = 19|units.pc
    speed = 7|units.km/units.s
    col_time = (radius/speed).value_in(units.yr)
    polynomials = np.array([[0.0073,-0.12,0.55,-0.55,-0.92,-8.2],
                            [-0.0041,0.12,-1.2,4.8,-6,-10],
                            [-0.018,0.52,-5.7,28,-62,34],
                            [-0.022,0.62,-6.6,33,-74,51],
                            [-0.022,0.5,-4.2,15,-22,-2.2],
                            [-0.0011,0.077,-1.1,5.9,-12,-3.1],
                            [-0.062,1.5,-13,54,-100,55],
                            [0.0064,-0.17,1.9,-11,29,-38],
                            [0.036,-0.89,8.6,-40,90,-86]])
    
    square = int(np.ceil(np.sqrt(len(species))))
    fig, ax = plt.subplots(square, square)#, figsize=(10,10))
    ax = ax.flatten()
    for i in range(len(species)):
        y_polynomials = np.zeros_like(density[-1,:])
        x_polynomials = np.linspace(1e2,5e6,2000)
        for j in range(6):
            y_polynomials += polynomials[i,j]*(np.log10(x_polynomials))**j
            print(polynomials[i,j]*(np.log10(x_polynomials))**j)
        index = chem.get_index_of_species(species[i])
        ax[i].scatter(np.log10(density[-1,:]), np.log10(abundances[:,index,-1]), s=1, marker='.')
        # ax[i].plot(np.log10(x_polynomials), (y_polynomials), linestyle='dashed', color='k')
        # ax[i].set_xscale('log')
        # ax[i].set_yscale('log')
        ax[i].set_title(species[i])
        ax[i].hlines(-12,-2,3,linestyle='dashed', color='r')
    fig.supxlabel('Number Density (cm^-3)')
    fig.supylabel('Abundances')
    fig.savefig('abundances_density_CO.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(square, square)#, figsize=(10,10))
    ax=ax.flatten()
    for i in range(len(species)):
        index = chem.get_index_of_species(species[i])
        ax[i].set_xlim(10,300)
        ax[i].scatter(temperatures[-1,:], abundances[:,index,-1], s=1, marker='.')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_title(species[i])
        ax[i].hlines(10**-12, 10,300,linestyle='dashed', color='r')

    fig.supxlabel('Temperatures (K)')
    fig.supylabel('Abundances')
    fig.savefig('abundances_temperature_CO.pdf', bbox_inches='tight')
    plt.close()

    time = np.arange(0, 1137*44e2, 20*44e2)
    times = np.repeat(time,2000)

    fig, ax = plt.subplots(square, square)#, figsize=(10,10))
    ax = ax.flatten()
    for i in range(len(species)):
        index = chem.get_index_of_species(species[i])
        species_abundance = np.log10(abundances[:,index,:].T.flatten())
        histogram, xedge, yedge = np.histogram2d(times, species_abundance, bins=56)
        histogram[histogram==0] = 'NaN'
        histogram = histogram.T
        ax[i].set_ylim(-16,-6)
        ax[i].vlines(col_time, -40,0, linestyle='dashed', color='r')
        ax[i].hlines(-12, 0,5e6,linestyle='dashed', color='r')
        ax[i].pcolormesh(xedge,yedge,histogram,cmap='magma')
        ax[i].set_title(species[i])
    fig.supxlabel('Time (yr)')
    fig.supylabel('Abundances')
    fig.savefig('abundances_time_CO.pdf', bbox_inches='tight')
    plt.close()

    particle_rich = abundances[1788,:,:]
    particle_poor = abundances[1730,:,:]
    fig, ax = plt.subplots(3,2, figsize=(12,9), sharex=True, gridspec_kw={'height_ratios':[3,1,1]})
    ax = ax.flatten()
    for i in range(len(species)):
        ax[0].plot(time,particle_rich[chem.get_index_of_species(species[i]),:], label=species[i])
        ax[1].plot(time,particle_poor[chem.get_index_of_species(species[i]),:])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    ax[0].set_ylim(1e-22)
    ax[1].set_ylim(1e-22)
    ax[2].plot(time,density[:,1788])
    ax[3].plot(time,density[:,1730])
    ax[4].plot(time,temperatures[:,1788])
    ax[5].plot(time,temperatures[:,1730])
    fig.supxlabel('Time (yr)')
    ax[0].set_ylabel('Abundances')
    ax[2].set_ylabel('Number Density (cm^-3)')
    ax[4].set_ylabel('Temperature (K)')
    ax[0].set_title('Depleted Chemistry')
    ax[1].set_title('Rich Chemistry')
    ax[0].legend(loc='upper left')
    fig.savefig('rich_poor_chemistry.pdf',bbox_inches='tight')
    plt.close()

def slices():
    plt.rc('font', size=60/1.5)            # controls default text sizes
    plt.rc('axes', titlesize=96/1.5)     # fontsize of the axes title
    plt.rc('axes', labelsize=72/1.5)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=60/1.5)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=60/1.5)    # fontsize of the tick labels
    plt.rc('legend', fontsize=72/1.5)    # legend fontsize
    plt.rc('figure', titlesize=96/1.5) 
    uclchem = Uclchem()
    hydro = read_set_from_file('chem_run_data/data/cloud_collision_hydro_1120.txt', format='amuse')
    chem = read_set_from_file('chem_run_data/data/chem_sample_1120.txt', format='amuse')
    keys = find_keys(hydro, chem)
    particles = Particles()
    for i in range(keys.shape[0]):
        hydro_particle = (hydro.select(lambda x: x == keys[i,0],['key']))
        chem_particle = chem.select(lambda x: x==keys[i,1],['key'])
        chem_particle.x = hydro_particle.x
        chem_particle.y = hydro_particle.y
        chem_particle.z = hydro_particle.z
        particles.add_particle(chem_particle)

    species = ['NH3','HCN','N2H+','HCO+','CS','HNC','CH3OH','CN','C2H']
    indices = []
    print(np.unique(particles.position.value_in(units.pc)).shape)
    for i in species:
        indices.append(uclchem.get_index_of_species(i))
    plotter = HydroPlotter(500, use_torch=False)
    plotter.add_gas_particles(particles)
    params_to_plot = np.repeat(np.array(['abundance']),9)
    plotter.plot_projection(40,params_to_plot=params_to_plot, save_fig='abundance_slices.pdf', cmap='cividis',max_cols=3, species_index=indices, species_names=species,use_gaussian_kernel=True)
    plotter.clear_all_particles()

#slices()
#parse_data()
plot_abundances(['NH3','HCN','N2H+','HCO+','CS','HNC','CH3OH','CN','C2H'])
slices()
# plot_density_temperature()
