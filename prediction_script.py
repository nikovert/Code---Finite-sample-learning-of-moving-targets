from math import cos, sin, tan, ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from mip import OptimizationStatus
# from dill import dump_session, load_session
from hypothesis import Hypothesis, compute_required_samples, prune_samples


##########################################################
#                        MAIN SCRIPT
##########################################################

np.random.seed(19681800)

plot_hypothesis = True
plot_drift = True
plot_discards = True
plot_montecarlo = True

all_samples = True
load_model = False
solveMIP = True
if all_samples:
    sample_count = compute_required_samples()
else:
    sample_count = 100

print(f"Estimated to need {sample_count} samples")

# Generate empty Hypothesis with a simulator attached
model = Hypothesis()
model.generateMsampleModel(sample_count, reduce=True, addMILP=solveMIP)
if load_model:
    model.read(
        '/Users/nikovertovec/Documents/PYTHON/ForestFire/model_archive/model.lp')
    print(
        f"model has {model.num_cols} vars, {model.num_rows} constraints and {model.num_nz} nzs")


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

prune_dist = 0.03

y_min, y_max = floor(np.min(model.x[:, 1])), ceil(np.max(model.x[:, 1]))
x_min, x_max = model.simulator.l_min, model.simulator.l_max

if solveMIP and plot_hypothesis:
    prnd_violation_points = []
    for p in violation_points:
        if p in model.x:
            prnd_violation_points.append(p)

    fig0 = plt.figure(figsize=(7, 5))
    t_samples = model.x[np.argwhere(model.f)[:, 0]]
    f_samples = model.x[np.argwhere(model.f < 1)[:, 0]]
    t_samples = prune_samples(t_samples, prune_dist)
    f_samples = prune_samples(f_samples, prune_dist)
    plt.scatter(t_samples[:, 0], t_samples[:, 1],
                c='b', marker='^', alpha=0.3)  # Plot samples with f=1
    plt.scatter(f_samples[:, 0], f_samples[:, 1],
                c='g', marker='o', alpha=0.3)  # Plot samples with f=0

    plt.xlabel(r'$l [m]$', fontsize=15)
    plt.ylabel(r'$v^2 [(m/s)^2]$', fontsize=15)

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
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min,
                 y_max])

    # Draw extended Facet p1--p4
    ax.axline(p1, p4, lw=2, color='r')
    h_t = model.hypothesis_label(t_samples)
    t_violations = t_samples[np.argwhere(h_t < 1)[:, 0]]
    plt.scatter(t_violations[:, 0], t_violations[:, 1],
                c='b', marker='^', edgecolors='red')

    h_f = model.hypothesis_label(f_samples)
    f_violations = f_samples[np.argwhere(h_f)[:, 0]]
    plt.scatter(f_violations[:, 0], f_violations[:, 1],
                c='g', marker='o', edgecolors='red')

#  Figure 1
# The evolution of the samples over 6 time steps
gradient = 0.5*model.simulator.M/model.F_list
m = len(model.f)

if plot_drift:
    fig1 = plt.figure(figsize=(13, 7))
    for index in range(1, 7):
        #  Get uppder index for sample range
        upper = floor(index * m/6)

        #  Create subplot
        ax = plt.subplot(2, 3, index)
        ax.set_title(f'i={upper}')
        plt.xlabel(r'$l [m]$', fontsize=10)
        plt.ylabel(r'$v^2 [(m/s)^2]$', fontsize=10)

        #  Plot previous true oracle line
        p0 = 0
        for sub_index in range(1, index):
            sub_upper = floor(sub_index * m/6)
            p1 = 1/gradient[sub_upper]
            ax.axline(xy1=(0, p0), slope=p1, color='r',
                      lw=2, alpha=0.5*sub_index/index)

            prev_upper = floor((sub_index-1) * m/6)

            t_samples = model.x[prev_upper +
                                np.argwhere(model.f[prev_upper:sub_upper])[:, 0], :]
            f_samples = model.x[prev_upper +
                                np.argwhere(model.f[prev_upper:sub_upper] < 1)[:, 0], :]

            t_samples = prune_samples(t_samples, prune_dist)
            f_samples = prune_samples(f_samples, prune_dist)

            plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
                        c='b', alpha=0.1*sub_index/index)  # Plot samples with f=1
            plt.scatter(f_samples[:, 0], f_samples[:, 1],
                        marker='o', c='g', alpha=0.1*sub_index/index)

        #  Plot true oracle line at current step
        p1 = 1/gradient[upper-1]
        ax.axline(xy1=(0, p0), slope=p1, color='r', lw=2)

        #  Extract 1-0 labeled samples
        prev_upper = floor((index-1) * m/6)
        t_samples = model.x[prev_upper +
                            np.argwhere(model.f[prev_upper:upper] > 0)[:, 0], :]
        f_samples = model.x[prev_upper +
                            np.argwhere(model.f[prev_upper:upper] < 1)[:, 0], :]

        t_samples = prune_samples(t_samples, prune_dist*2)
        f_samples = prune_samples(f_samples, prune_dist*2)

        plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
                    alpha=0.5, c='b')  # Plot samples with f=1
        plt.scatter(f_samples[:, 0], f_samples[:, 1], marker='o',
                    alpha=0.5, c='g')  # Plot samples with f=0

        #  Set Axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    plt.subplots_adjust(hspace=0.3)

if plot_discards:
    # Extract discarded samples
    fig2 = plt.figure(figsize=(7, 5))

    # Plot all samples
    t_samples = model.x[np.argwhere(model.f > 0)[:, 0], :]
    f_samples = model.x[np.argwhere(model.f < 1)[:, 0], :]
    t_samples = prune_samples(t_samples, prune_dist)
    f_samples = prune_samples(f_samples, prune_dist)
    plt.scatter(t_samples[:, 0], t_samples[:, 1], marker='^',
                alpha=1, c='red')  # Plot samples with f=1
    plt.scatter(f_samples[:, 0], f_samples[:, 1], marker='o',
                alpha=1, c='red')  # Plot samples with f=0

    #  Highlight discarded samples
    for i in model.discard_indices:
        if model.f[i] > 0:
            if model.x[i, :] not in t_samples:
                continue
            plt.scatter(model.x[i, 0], model.x[i, 1], marker='^',
                        alpha=1, c='k')  # Plot discarded samples
        else:
            if model.x[i, :] not in f_samples:
                continue
            plt.scatter(model.x[i, 0], model.x[i, 1], marker='o',
                        alpha=1, c='k')  # Plot discarded samples
    ax = plt.gca()
    theta = model.theta
    R = np.array(((np.cos(theta), np.sin(theta)),
                  (-np.sin(theta), np.cos(theta))))
    # Rotate the t_samples and samples using the rotation matrix
    rotated_t_samples = np.matmul(R, np.transpose(t_samples))
    rotated_f_samples = np.matmul(R, np.transpose(f_samples))
    x = np.arange(x_min, x_max+20)

    # Draw lower Halfplane bound
    t_sam = t_samples[np.argmin(rotated_t_samples[0, :])]
    b_t = t_sam[0] + t_sam[1]*tan(theta)
    ax.axline([b_t, 0], t_sam, lw=2, color='c', linestyle=":")
    ax.fill_between(x, (b_t-x)/tan(theta), y_max, facecolor='c', alpha=0.2)

    # Draw upper Halfplane bound
    f_sam = f_samples[np.argmax(rotated_f_samples[0, :])]
    b_f = f_sam[0] + f_sam[1]*tan(theta)
    ax.axline([b_f, 0], f_sam, lw=2, color='m', linestyle=":")
    ax.fill_between(x, (b_f-x)/tan(theta), y_min, facecolor='m', alpha=0.2)

    plt.xlabel(r'$l [m]$', fontsize=15)
    plt.ylabel(r'$v^2 [(m/s)^2]$', fontsize=15)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

if plot_montecarlo:
    # Monte Carlo simulation of the disagreement of fm+1 and h
    p_m = model.simulator.get_p()
    m_m = model.simulator.get_m()
    K = 500
    dissagreement = np.zeros(K)
    for k in range(K):
        # compute f_m+1
        model.simulator.set_p(p_m)
        model.simulator.set_m(m_m)
        model.simulator.next_step()

        samples = int(K*10)
        x, f = model.simulator.genSamples(m=samples, noStep=True)
        h = model.hypothesis_label(x)
        dissagreement[k] = sum(f != h)/samples

    # Create a histogram
    sns.displot(dissagreement*100, kde=False, height=5, aspect=7/5)
    # Add labels and a legend
    plt.xlabel(r'$\widehat{\mathrm{er}}_m(f_{m+1},h_m)$ in %', fontsize=15)
    plt.ylabel(r'Nr. of Monte Carlo runs', fontsize=15)
    plt.tight_layout()
    print(f"Monte Carlo disagreement between h and f: {sum(dissagreement)/K}")

# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()
