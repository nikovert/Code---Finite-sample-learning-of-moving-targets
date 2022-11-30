from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from car import Car
from mip import Model, OptimizationStatus

delta = 10**-6
d = 4
k = 1
eps = 0.1
a_high = 0.0009
a_low  = 0.0007
map_size = 100

sample_count = int(np.maximum(5*(4*a_high + eps)/eps**2 * (log(8/delta) + d* 40*(4*a_high + eps)/eps**2), 3*k/a_low * log(2/delta)))
my_car = Car(map_size, save_change = True)

model = my_car.generateMsampleModel(sample_count)
my_car.update_plt()
status = model.optimize()
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))

fig2 = plt.figure(figsize=(5,5))
sample_array = np.zeros(my_car.map.shape)
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