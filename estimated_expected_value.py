from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from forest import Forest
from car import Car
from mip import Model, OptimizationStatus

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

delta = 10**-6
d = 4
k = 1
eps = 0.1
a_high = 0.0009
a_low  = 0.0007
map_size = 100

sample_count = np.maximum(5*(4*a_high + eps)/eps**2 * (log(8/delta) + d* 40*(4*a_high + eps)/eps**2), 3*k/a_low * log(2/delta))
m = 100
while sample_count > int(m+1):
    # Generate initial Map
    my_car = Car(map_size, save_change = True)
    mu = 0
    m = int(sample_count)
    for i in range(m):
        my_car.next_step()
    change = np.ones(my_car.map.shape, dtype=bool)
    for i in range(m):
        change ^= my_car.map_change[m-1-i]
        p_i = np.count_nonzero(~change)/my_car.map.size
        mu += p_i/m
    a_high = 1.01*mu
    a_low = 0.99*mu
    # Calculate the number of samples needed
    sample_count = np.maximum(5*(4*a_high + eps)/eps**2 * (log(8/delta) + d* 40*(4*a_high + eps)/eps**2), 3*k/a_low * log(2/delta))
    print(sample_count)

my_car = Car(map_size, save_change = True)
fig1 = plt.figure(figsize=(5,5))
my_car.im = plt.imshow(my_car.map, cmap=my_car.CM, interpolation='none')

frame_count = 20
anim = animation.FuncAnimation(fig1, my_car.updatefig, frames=range(frame_count), interval = 100)
#anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
plt.axis('off')
plt.show()
    



