from math import cos, sin, ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mip import OptimizationStatus
# from dill import dump_session, load_session
from hypothesis import Hypothesis


##########################################################
#                        MAIN SCRIPT
##########################################################

# load_model = False
# if load_model:
#    filepath = 'session.pkl'
#    load_session(filepath)  # Load the session
#    model = Model()
#    model.read('model.lp')
#    print(
#        f"model has {model.num_cols} vars, {model.num_rows} constraints and {model.num_nz} nzs")


delta = 10**-4
d = 1
k = 1
eps = 0.01
a_high = 0.035

# Estimated to need 18 728 samples
# Discarded _  samples.
# Model reduced to using _ samples.
np.random.seed(19682802)

full_run = False
if full_run:
    delta_ratio = 1-10**-6
    t = np.linspace(10**-6, 1-10**-6, 10000)

    m_min = 5*(2*(a_high+t) + eps)/eps**2 * (-np.log((delta_ratio *
                                                      delta)/4) + d * np.log(40*(2*(a_high+t) + eps)/eps**2))
    m_max = -1/(2 * t**2) * np.log(((1-delta_ratio)*delta))

    condition = abs(m_min - m_max+1)
    ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
    sample_count_range = [m_min[ind], m_max[ind]]
    sample_count = ceil((m_min[ind] + m_max[ind])/2)
else:
    sample_count = 500

print(f"Estimated to need {sample_count} samples")
model = Hypothesis()  # Generate empty Hypothesis with a simulator attached
model.generateMsampleModel(sample_count, reduce=True)

solveMIP = True
if solveMIP:
    status = model.optimize(max_nodes=2000)
    model.write('model_archive/model.lp')
    if status == OptimizationStatus.OPTIMAL:
        print(f"optimal solution cost {model.objective_value} found")
    elif status == OptimizationStatus.FEASIBLE:
        print(
            f"sol.cost {model.objective_value} found, best possible: {model.objective_bound}")
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print(
            f"no feasible solution found, lower bound is: {model.objective_bound}")
        plt.show(block=True)

if solveMIP:
    violation_points = []
    for i in range(len(model.f)):
        if not model.v[i].x:
            violation_points.append(model.x[i])

prune_dist = 0.014
prnd_model = model.prune(prune_dist)

m = len(prnd_model.f)
if solveMIP:
    prnd_violation_points = []
    for p in violation_points:
        if p in prnd_model.x:
            prnd_violation_points.append(p)

    fig0 = plt.figure(figsize=(5, 5))
    t_samples = prnd_model.x[np.argwhere(prnd_model.f)[:, 0]]
    f_samples = prnd_model.x[np.argwhere(prnd_model.f < 1)[:, 0]]
    plt.scatter(t_samples[:, 0], t_samples[:, 1],
                marker='^', alpha=0.3)  # Plot samples with f=1
    plt.scatter(f_samples[:, 0], f_samples[:, 1],
                marker='o', alpha=0.3)  # Plot samples with f=0

    plt.xlabel('distance (m)')
    plt.ylabel('speed (m/s)^2')

    model.simulator.next_step()  # t = m+1

    # Find lower left corner
    #       p4 ----- p3
    #       |        |
    #       |        |
    #       p1 ----- p2
    #
    theta = model.theta
    if model.simulator.singleFacet:
        b0 = -100
        b1 = -600
        b2 = model.b[0].x
        b3 = 100
    else:
        b0 = model.b[0].x
        b1 = model.b[1].x
        b2 = model.b[2].x
        b3 = model.b[3].x

    p1 = [b2 * cos(theta)-b3*sin(theta),  b2 * sin(theta)+b3*cos(theta)]
    p2 = [-b0 * cos(theta)-b3*sin(theta), -b0 * sin(theta)+b3*cos(theta)]
    p3 = [-b0 * cos(theta)+b1*sin(theta), -b0 * sin(theta)-b1*cos(theta)]
    p4 = [b2 * cos(theta)+b1*sin(theta),  b2 * sin(theta)-b1*cos(theta)]

    polygon = Polygon([p1, p2, p3, p4], alpha=0.4)
    ax = plt.gca()
    ax.add_patch(polygon)
    ax.set_xlim([model.simulator.l_min, model.simulator.l_max])
    ax.set_ylim([floor(np.min(prnd_model.x[:, 1])),
                ceil(np.max(prnd_model.x[:, 1]))])

    # Draw extended Facet p1--p4
    ax.axline(p1, p4, lw=2, color='r')
    for i in range(m):
        if not any(np.all(prnd_model.x[i] == elem) for elem in prnd_violation_points):
            if prnd_model.f[i]:
                plt.scatter(
                    prnd_model.x[i, 0], prnd_model.x[i, 1], c='b', marker='^', edgecolors='red')
            else:
                plt.scatter(
                    prnd_model.x[i, 0], prnd_model.x[i, 1], c='g', marker='o', edgecolors='red')

#  Figure 1
# The evolution of the samples over 6 time steps
gradient = 0.5*model.simulator.M/prnd_model.F_list

fig1 = plt.figure(figsize=(5, 5))
for index in range(1, 7):
    #  Get uppder index for sample range
    upper = floor(index * m/6)

    #  Create subplot
    ax = plt.subplot(2, 3, index)
    ax.set_title(f"(t={index}/6 T)")
    plt.xlabel('distance (m)')
    plt.ylabel('speed (m/s)^2')

    #  Plot previous true oracle line
    p0 = 0
    for sub_index in range(1, index):
        sub_upper = floor(sub_index * m/6)
        p1 = 1/gradient[sub_upper]
        ax.axline(xy1=(0, p0), slope=p1, color='r',
                  lw=2, alpha=0.5*sub_index/index)

        prev_upper = floor((sub_index-1) * m/6)
        t_samples = prnd_model.x[prev_upper +
                                 np.argwhere(prnd_model.f[prev_upper:sub_upper])[:, 0], :]
        f_samples = prnd_model.x[prev_upper +
                                 np.argwhere(prnd_model.f[prev_upper:sub_upper] < 1)[:, 0], :]
        plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
                    c='b', alpha=0.1*sub_index/index)  # Plot samples with f=1
        plt.scatter(f_samples[:, 0], f_samples[:, 1],
                    marker='o', c='g', alpha=0.1*sub_index/index)

    #  Plot true oracle line at current step
    p1 = 1/gradient[upper-1]
    ax.axline(xy1=(0, p0), slope=p1, color='r', lw=2)

    #  Extract 1-0 labeled samples
    prev_upper = floor((index-1) * m/6)
    t_samples = prnd_model.x[prev_upper +
                             np.argwhere(prnd_model.f[prev_upper:upper] > 0)[:, 0], :]
    f_samples = prnd_model.x[prev_upper +
                             np.argwhere(prnd_model.f[prev_upper:upper] < 1)[:, 0], :]
    plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
                alpha=0.5, c='b')  # Plot samples with f=1
    plt.scatter(f_samples[:, 0], f_samples[:, 1], marker='o',
                alpha=0.5, c='g')  # Plot samples with f=0

    #  Set Axis limits
    ax.set_xlim([model.simulator.l_min, model.simulator.l_max])
    ax.set_ylim([floor(np.min(prnd_model.x[:, 1])),
                ceil(np.max(prnd_model.x[:, 1]))])

plt.subplots_adjust(hspace=0.3)

#  Figure 2
# # Prune lists
# figure, axes = plt.subplots()
# plt.scatter(prnd_model.x[:,0], prnd_model.x[:,1], c='b', alpha = 0.8, edgecolors = None)
# plt.scatter(prnd_model.x[:,0], prnd_model.x[:,1], c='b', alpha = 0.8, edgecolors = 'red')

# for index in range(0,prnd_model.x.shape[0]):
#     cirlces = plt.Circle(prnd_model.x[index,:], prune_dist, color='g', fill = False, clip_on=True)
#     axes.add_artist(cirlces)
# axes.set_aspect(0.22)

# Extract discarded samples
fig2 = plt.figure(figsize=(5, 5))

# Plot all samples
t_samples = prnd_model.x[np.argwhere(prnd_model.f > 0)[:, 0], :]
f_samples = prnd_model.x[np.argwhere(prnd_model.f < 1)[:, 0], :]
plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
            alpha=0.1, c='b')  # Plot samples with f=1
plt.scatter(f_samples[:, 0], f_samples[:, 1], marker='o',
            alpha=0.1, c='g')  # Plot samples with f=0

#  Highlight discarded samples
for i in prnd_model.discard_indices:
    if prnd_model.f[i] > 0:
        plt.scatter(prnd_model.x[i, 0], prnd_model.x[i, 1], marker='^',
                    alpha=0.4, c='b', edgecolors='red')  # Plot discarded samples
    else:
        plt.scatter(prnd_model.x[i, 0], prnd_model.x[i, 1], marker='o',
                    alpha=0.4, c='g', edgecolors='red')  # Plot discarded samples
plt.xlabel('distance (m)')
plt.ylabel('speed (m/s)^2')

# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()
