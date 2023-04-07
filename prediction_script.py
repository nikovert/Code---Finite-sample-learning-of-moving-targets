from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Polygon
from aeb import AEB
from mip import Model, OptimizationStatus
from dill import dump_session, load_session
from matplotlib.transforms import Affine2D

if False:
    filepath = 'session.pkl'
    load_session(filepath) # Load the session
    model = Model()
    model.read('model.lp')
    print('model has {} vars, {} constraints and {} nzs'.format(model.num_cols, model.num_rows, model.num_nz))

delta = 10**-4
d = 4
k = 1
eps = 0.19
# For speed = 0.1
a_high = 0.009
a_low  = 0.001
# For speed = 0.01
a_high = 0.002096
a_low = 0.000505

# Estimated to need 52342 samples
# Discarded 51960 samples.
# Model reduced to using 382 samples.
np.random.seed(19680801)
simulator = AEB(save_change = False)

full_run = False
if full_run:
    alpha = np.outer(np.linspace(0.001, 1-0.001, 100), np.ones(100))
    delta_ratio = alpha.copy().T
    alpha, delta_ratio = np.meshgrid(np.linspace(0.001, 1-0.001, 100), np.linspace(0.001, 1-0.001, 100))

    sample_count_range = np.maximum(5*(2*(1+alpha)*a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(2*(1+alpha)*a_high + eps)/eps**2), -3*k/(alpha**2 * a_low) * np.log(((1-delta_ratio)*delta)))
    sample_count = int(np.min(sample_count_range))
    ind = np.unravel_index(np.argmin(sample_count_range, axis=None), sample_count_range.shape)
else:
    sample_count = 1000

print("Estimated to need %d samples"%sample_count)
model = simulator.generateMsampleModel(sample_count)
#model.emphasis = 1 # FEASIBILITY

# fig1 = plt.figure(figsize=(5,5))
# simulator.im = plt.imshow(simulator.map, cmap=simulator.CM, interpolation='none')

# fig2 = plt.figure(figsize=(5,5))
# sample_array = np.zeros(simulator.map.shape)
# sample_array_min = np.zeros(simulator.map.shape)
# for i in range(sample_count):
#     sample_array[round(model.x[i][0]), round(model.x[i][1])] = max(sample_array[round(model.x[i][0]), round(model.x[i][1])], 1 + int(model.f[i]))
#     if model.discarded[i] < 1:
#         sample_array_min[round(model.x[i][0]), round(model.x[i][1])] = max(sample_array_min[round(model.x[i][0]), round(model.x[i][1])], 1 + int(model.f[i]))

# # Colours for potting
# two_colour = simulator.car_colour  
# one_colour = simulator.ground_colour 
# zero_colour = "gold" 
# CM = mpl.colors.ListedColormap([zero_colour,one_colour,two_colour])
# plt.imshow(sample_array, cmap=CM, interpolation='none')
# plt.show(block=False)

# fig3 = plt.figure(figsize=(5,5))
# plt.imshow(sample_array_min, cmap=CM, interpolation='none')
# plt.show(block=False)

#model.max_mip_gap = k
status = model.optimize(max_nodes = 10)
model.write('model.lp')
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))
    plt.show(block=True)

plot_results = True
if plot_results:
    fig1 = plt.figure(figsize=(5,5))
    t_samples = model.x[np.argwhere(model.f)[:, 0]]
    f_samples = model.x[np.argwhere(model.f<1)[:, 0]]
    plt.scatter(t_samples[:,0], t_samples[:,1], marker='^', alpha=0.3) # Plot samples with f=1
    plt.scatter(f_samples[:,0], f_samples[:,1], marker='o', alpha=0.3) # Plot samples with f=0
    
    d_sampes = model.x[np.argwhere(model.discarded)[:, 0]]
    plt.scatter(d_sampes[:,0], d_sampes[:,1], marker='o', alpha=0.3, c='k') # Plot discarded samples
    
    plt.xlabel('distance (m)')
    plt.ylabel('speed (m/s)^2')

    simulator.next_step() # t = m+1
    ## Find lower left corner
    #       p4 ----- p3
    #       |        |
    #       |        |
    #       p1 ----- p2
    #
    theta  = model.theta
    p1 = [model.b[2].x * cos(theta)-model.b[3].x*sin(theta), model.b[2].x * sin(theta)+model.b[3].x*cos(theta)]
    p2 = [-model.b[0].x * cos(theta)-model.b[3].x*sin(theta), -model.b[0].x * sin(theta)+model.b[3].x*cos(theta)]
    p3 = [-model.b[0].x * cos(theta)+model.b[1].x*sin(theta), -model.b[0].x * sin(theta)-model.b[1].x*cos(theta)]
    p4 = [model.b[2].x * cos(theta)+model.b[1].x*sin(theta), model.b[2].x * sin(theta)-model.b[1].x*cos(theta)]

    polygon = Polygon([p1, p2, p3, p4], alpha = 0.4)
    plt.gca().add_patch(polygon)



m = len(model.f)
max_val = -inf
Nf = 2*len(model.x[0])
boundary_points = []
violation_points = []
for i in range(m):
    if model.f[i]: # constraints for i in I1
        if model.v[i].x:
            violation_points.append(model.x[i])
            plt.scatter(model.x[i,0], model.x[i,1], c='r', marker='^')
            continue
        for j in range(Nf):
            val = np.matmul(model.a[j], model.x[i]) + model.b[j].x
            if val > -0.1:
                max_val = max(max_val, val)
                boundary_points.append(model.x[i])
    elif model.v[i].x:
        plt.scatter(model.x[i,0], model.x[i,1], c='r', marker='o')

# for deg in range(0, 360, 45):
#     rec = Rectangle((x_min,y_min), model.width[0].x,model.width[1].x,
#                         edgecolor='blue',
#                         facecolor='none',
#                         lw=1,
#                         angle = deg)
#     plt.gca().add_patch(rec)

# fig5 = plt.figure(figsize=(5,5))
# simulator.reset_map()
# simulator.stay_marked = False # only show current position of target

# simulator.next_step()
# simulator.im = plt.imshow(simulator.map, cmap=simulator.CM, interpolation='none')
# y_min = min(abs(model.b[0].x), abs(model.b[2].x))
# x_min = min(abs(model.b[1].x), abs(model.b[3].x))
# plt.gca().add_patch(Rectangle((x_min,y_min),model.width[1],model.width[0],
#                     edgecolor='blue',
#                     facecolor='none',
#                     lw=1))

# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()