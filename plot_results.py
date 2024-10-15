import matplotlib.pyplot as plt
import numpy as np
from amuse.io import read_set_from_file
from tests.plotting import HydroPlotter

plt.rc('font', size=60/1.5)            # controls default text sizes
plt.rc('axes', titlesize=96/1.5)     # fontsize of the axes title
plt.rc('axes', labelsize=72/1.5)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=60/1.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=60/1.5)    # fontsize of the tick labels
plt.rc('legend', fontsize=72/1.5)    # legend fontsize
plt.rc('figure', titlesize=96/1.5) 

plotter = HydroPlotter(500, use_torch=False)
for i in range(106):
    particles = read_set_from_file('data/cloud_collision_hydro_{}.txt'.format(i), format='amuse')
    sinks = read_set_from_file('data/cloud_collision_hydro_sinks_{}.txt'.format(i), format='amuse')
    plotter.add_gas_particles(particles)
    plotter.add_star_particles(sinks)
    plotter.plot_projection(96, save_fig="plots/hydro_{}.pdf".format(i),params_to_plot=['density','temperature'], use_gaussian_kernel=False, cmap='magma', plot_stars=True)
    plotter.clear_all_particles()
