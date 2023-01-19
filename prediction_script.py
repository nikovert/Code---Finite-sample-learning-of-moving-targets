from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from car import Car
from mip import Model, OptimizationStatus
from dill import dump_session, load_session

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

map_size = 250
my_car = Car(map_size, save_change = False)
my_car.stay_marked = False

full_run = True
if full_run:
    alpha = np.outer(np.linspace(0.001, 1-0.001, 100), np.ones(100))
    delta_ratio = alpha.copy().T
    alpha, delta_ratio = np.meshgrid(np.linspace(0.001, 1-0.001, 100), np.linspace(0.001, 1-0.001, 100))

    sample_count_range = np.maximum(5*(2*(1+alpha)*a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(2*(1+alpha)*a_high + eps)/eps**2), -3*k/(alpha**2 * a_low) * np.log(((1-delta_ratio)*delta)))
    sample_count = int(np.min(sample_count_range))
    ind = np.unravel_index(np.argmin(sample_count_range, axis=None), sample_count_range.shape)
else:
    sample_count = 20000

print("Estimated to need %d samples"%sample_count)
model = my_car.generateMsampleModel(sample_count)
#model.emphasis = 1 # FEASIBILITY

fig1 = plt.figure(figsize=(5,5))
my_car.im = plt.imshow(my_car.map, cmap=my_car.CM, interpolation='none')

fig2 = plt.figure(figsize=(5,5))
sample_array = np.zeros(my_car.map.shape)
sample_array_min = np.zeros(my_car.map.shape)
for i in range(sample_count):
    sample_array[round(model.x[i][0]), round(model.x[i][1])] = max(sample_array[round(model.x[i][0]), round(model.x[i][1])], 1 + int(model.f[i]))
    if model.discarded[i] < 1:
        sample_array_min[round(model.x[i][0]), round(model.x[i][1])] = max(sample_array_min[round(model.x[i][0]), round(model.x[i][1])], 1 + int(model.f[i]))

# Colours for potting
two_colour = my_car.car_colour  
one_colour = my_car.ground_colour 
zero_colour = "gold" 
CM = mpl.colors.ListedColormap([zero_colour,one_colour,two_colour])
plt.imshow(sample_array, cmap=CM, interpolation='none')
plt.show(block=False)

fig3 = plt.figure(figsize=(5,5))
plt.imshow(sample_array_min, cmap=CM, interpolation='none')
plt.show(block=False)

model.max_mip_gap = k
status = model.optimize(max_nodes = 10)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))
    plt.show(block=True)


car_postion_m = my_car.car_position
travel_dir_m = my_car.travel_dir

fig4 = plt.figure(figsize=(5,5))
my_car.reset_map()
my_car.stay_marked = False # only show current position of target

my_car.next_step() # t = m+1
my_car.im = plt.imshow(my_car.map, cmap=my_car.CM, interpolation='none')
y_min = min(abs(model.b[0].x), abs(model.b[2].x))
x_min = min(abs(model.b[1].x), abs(model.b[3].x))
plt.gca().add_patch(Rectangle((x_min,y_min),model.width[1],model.width[0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))

fig5 = plt.figure(figsize=(5,5))
my_car.reset_map()
my_car.stay_marked = False # only show current position of target

my_car.next_step()
my_car.im = plt.imshow(my_car.map, cmap=my_car.CM, interpolation='none')
y_min = min(abs(model.b[0].x), abs(model.b[2].x))
x_min = min(abs(model.b[1].x), abs(model.b[3].x))
plt.gca().add_patch(Rectangle((x_min,y_min),model.width[1],model.width[0],
                    edgecolor='blue',
                    facecolor='none',
                    lw=1))

frame_count = 200
anim = animation.FuncAnimation(fig5, my_car.updatefig, frames=frame_count, interval = 1)
anim.save('animation.mp4', fps=30, writer="ffmpeg", codec="libx264")
plt.axis('off')

model.write('model.lp')
# del(model)
# filepath = 'session.pkl'
# dump_session(filepath) # Save the session
plt.show()