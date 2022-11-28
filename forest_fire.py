from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from forest import Forest
from mip import Model, OptimizationStatus

my_forest = Forest(100, 0.6)

# initilaise a fire
my_forest.forest[50:52, 60:62] = 2
# Let the fire propagate a little
# for i in range(200):
#     my_forest.basic_fire_prop()
# plt.show(block=False)
# plt.pause(0.001)

# Plot m to eps
a_low  = my_forest.p_burnout * np.count_nonzero(my_forest.forest == 2)/my_forest.forest.size + my_forest.p.min() *(1-np.count_nonzero(my_forest.forest == 2)/my_forest.forest.size)
a_high = my_forest.p_burnout * np.count_nonzero(my_forest.forest == 2)/my_forest.forest.size + my_forest.p.max() *(1-np.count_nonzero(my_forest.forest == 2)/my_forest.forest.size)
delta = 10**-5
d = 2
k = 1
eps = np.linspace(0.05,0.3,1000)
m = np.maximum(5*(4*a_high + eps)/eps**2 * (log(8/delta) + d* 40*(4*a_high + eps)/eps**2), 3*k/a_low * log(2/delta))
#plt.plot(eps,m, 'r')

eps = 0.1
sample_count = np.maximum(5*(4*a_high + eps)/eps**2 * (log(8/delta) + d* 40*(4*a_high + eps)/eps**2), 3*k/a_low * log(2/delta))
sample_count = int(sample_count)

generate_hypothesis = False

if generate_hypothesis:
    model = my_forest.generateMsampleModel(sample_count)
    my_forest.update_plt()
    status = model.optimize()
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(model.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))

    fig2 = plt.figure(figsize=(5,5))
    sample_array = np.zeros(my_forest.forest.shape)
    for i in range(sample_count):
        sample_array[model.x[i][0], model.x[i][1]] = 1 + int(model.f[i])

    # Colours for potting
    two_colour = [0.156, 1,0,1]  
    one_colour = [1,0.01,0,0.6] 
    zero_colour = [1 ,1 , 1 ,1]  
    CM = mpl.colors.ListedColormap([zero_colour,one_colour,two_colour])
    plt.imshow(sample_array, cmap=CM, interpolation='none')
    plt.show(block=False)

    Nf_hf = 2
    y_min = min(abs(model.b[0].x), abs(model.b[2].x))
    x_min = min(abs(model.b[1].x), abs(model.b[3].x))
    plt.gca().add_patch(Rectangle((x_min,y_min),model.width[1],model.width[0],
                        edgecolor='blue',
                        facecolor='none',
                        lw=1))
else:

    fig1 = plt.figure(figsize=(5,5))
    my_forest.im = plt.imshow(my_forest.forest, cmap=my_forest.CM, interpolation='none')

    frame_count = 100
    anim = animation.FuncAnimation(fig1, my_forest.updatefig, frames=range(frame_count))
    #anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
    plt.axis('off')
    plt.show()
    



