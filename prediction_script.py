from math import cos, sin, ceil, floor
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numpy import linalg as LA
from mip import Model, OptimizationStatus
#from dill import dump_session, load_session
from aeb import AEB


def prune(model, distance):
    """
    Prunes the model by removing samples within a certain distance from each other.

    Args:
        model (Model): The model to be pruned.
        distance (float): The distance threshold for pruning.

    Returns:
        Model: The pruned model.
    """
    index = 0
    check = model.x.shape[0]
    discard_list = np.zeros(model.x.shape)
    discard_list[model.discard_indices] = 1
    while check > 0:
        if check == model.x.shape[0]:
            index = random.randint(0, model.x.shape[0]-1)
        else:
            index = (index+1) % model.x.shape[0]

        elements = np.argwhere(
            LA.norm(abs(model.x[index, :]-model.x), axis=1) < distance)[1:]

        if elements.shape[0] > 0:
            model.x = np.delete(model.x, elements, 0)
            model.f = np.delete(model.f, elements, 0)
            discard_list = np.delete(discard_list, elements, 0)
            model.F_list = np.delete(model.F_list, elements, 0)
            check = min(check, model.x.shape[0])
        else:
            check -= 1

    model.discard_indices = np.argwhere(discard_list[:, 0])[:, 0]
    return model


load_model = False
if load_model:
    filepath = 'session.pkl'
    load_session(filepath)  # Load the session
    model = Model()
    model.read('model.lp')
    print(
        f"model has {model.num_cols} vars, {model.num_rows} constraints and {model.num_nz} nzs")


delta = 10**-4
d = 4
k = 1
eps = 0.3
a_high = 0.001843

# Estimated to need 18 728 samples
# Discarded _  samples.
# Model reduced to using _ samples.
np.random.seed(19681800)
simulator = AEB()

full_run = False
if full_run:
    delta_ratio = 1-10**-5
    t = np.linspace(10**-5, 1-10**-5, 10000)

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
model = simulator.generateMsampleModel(sample_count)

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

violation_points = []
for i in range(len(model.f)):
    if not model.v[i].x:
        violation_points.append(model.x[i])

prune_dist = 3
prnd_model = prune(model, prune_dist)

prnd_violation_points = []
for p in violation_points:
    if p in prnd_model.x:
        prnd_violation_points.append(p)

m = len(prnd_model.f)
plot_results = True
if plot_results:
    fig0 = plt.figure(figsize=(5, 5))
    t_samples = prnd_model.x[np.argwhere(prnd_model.f)[:, 0]]
    f_samples = prnd_model.x[np.argwhere(prnd_model.f < 1)[:, 0]]
    plt.scatter(t_samples[:, 0], t_samples[:, 1],
                marker='^', alpha=0.3)  # Plot samples with f=1
    plt.scatter(f_samples[:, 0], f_samples[:, 1],
                marker='o', alpha=0.3)  # Plot samples with f=0

    plt.xlabel('distance (m)')
    plt.ylabel('speed (m/s)^2')

    simulator.next_step()  # t = m+1

    # Find lower left corner
    #       p4 ----- p3
    #       |        |
    #       |        |
    #       p1 ----- p2
    #
    theta = model.theta
    if simulator.singleFacet:
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
    ax.set_xlim([simulator.l_min, simulator.l_max])
    ax.set_ylim([floor(np.min(prnd_model.x[:, 1])),
                ceil(np.max(prnd_model.x[:, 1]))])

    # Draw extended Facet p1--p4
    ax.axline(p1, p4, lw=2, color='r')
    for i in range(m):
        if any([np.all(prnd_model.x[i] == elem) for elem in prnd_violation_points]):
            if prnd_model.f[i]:
                plt.scatter(
                    prnd_model.x[i, 0], prnd_model.x[i, 1], c='b', marker='^', edgecolors='red')
            else:
                plt.scatter(
                    prnd_model.x[i, 0], prnd_model.x[i, 1], c='g', marker='o', edgecolors='red')

#  Figure 1
# The evolution of the samples over 6 time steps
gradient = 0.5*simulator.M/prnd_model.F_list
assert all((prnd_model.x[:, 1] * gradient <
           prnd_model.x[:, 0]) == (prnd_model.f > 0))

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
    ax.set_xlim([simulator.l_min, simulator.l_max])
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
discarded_samples = prnd_model.x[model.discard_indices]
plt.scatter(discarded_samples[:, 0], discarded_samples[:, 1], marker='o',
            alpha=0.8, c='b', edgecolors='red')  # Plot discarded samples
plt.xlabel('distance (m)')
plt.ylabel('speed (m/s)^2')

# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()
