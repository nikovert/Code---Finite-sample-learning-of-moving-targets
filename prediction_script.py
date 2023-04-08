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

a_high = 0.07671622306528939

# Estimated to need 52342 samples
# Discarded 51960 samples.
# Model reduced to using 382 samples.
np.random.seed(19680801)
simulator = AEB(save_change = False)

full_run = False
if full_run:
    delta_ratio = 0.5;
    t = np.linspace(0.001, 1-0.001, 100)

    m_min = 5*(2*(a_high+t) + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* np.log(40*(2*(a_high+t) + eps)/eps**2))
    m_max = -1/(2 * t**2) * np.log(((1-delta_ratio)*delta))

    condition = abs(m_min - m_max+1)
    ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
    sample_count_range = [m_min[ind], m_max[ind]]
    sample_count = ceil(np.min(sample_count_range))
else:
    sample_count = 40329

print("Estimated to need %d samples"%sample_count)
model = simulator.generateMsampleModel(sample_count)
#model.emphasis = 1 # FEASIBILITY

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
    
    t_d_samples = model.x[np.argwhere(model.discarded_t)[:, 0]]
    f_d_samples = model.x[np.argwhere(model.discarded_f)[:, 0]]
    plt.scatter(t_d_samples[:,0], t_d_samples[:,1], marker='^', alpha=0.3, c='k') # Plot discarded samples
    plt.scatter(f_d_samples[:,0], f_d_samples[:,1], marker='o', alpha=0.3, c='k')

    plt.xlabel('distance (m)')
    plt.ylabel('speed (m/s)^2')

    print("Discarded %f %% of true samples." % (len(t_d_samples)/len(t_samples)))
    print("Discarded %f %% of false samples." % (len(f_d_samples)/len(f_samples)))
        
    print("Discarded %d %% of %d samples." % (sum(model.discarded_t) + sum(model.discarded_f), sample_count))

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

# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()