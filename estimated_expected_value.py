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
import time

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

delta = 10**-4
d = 4
k = 1
eps = 0.25
a_high = 0.017
a_low  = 0.016
map_size = 100
stay_marked = False

delta_ratio = np.array([x / 100.0 for x in range(1, 100)])
sample_count_range = np.maximum(5*(4*a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(4*a_high + eps)/eps**2), -3*k/a_low * np.log(((1-delta_ratio)*delta)))
sample_count = int(min(sample_count_range))
print("Starting with %d samples" % sample_count)

rounds = 1000
global_a_high = 0
global_a_low = 1
global_mu = np.zeros(rounds)
for iter in range(rounds):
    print("Starting round with %d" % iter)
    m = sample_count-2
    while sample_count > int(m+1):
        # Generate initial Map
        my_car = Car(map_size, save_change = True)
        my_car.stay_marked = stay_marked
        mu = 0
        m = int(sample_count)
        for i in range(m):
            my_car.next_step()
        for i in range(m):
            change = my_car.map_archive[m-1-i]^my_car.map
            p_i = np.count_nonzero(change)/my_car.map.size
            mu += p_i/m
        print("Calculated mu = %f" % mu)
        a_high = 1.01*mu
        a_low = 0.99*mu
        global_a_high = max(global_a_high, a_high)
        global_a_low  = min(global_a_low, a_low)
        global_mu[iter] = max(global_mu[iter],mu)
        # Calculate the number of samples needed
        delta_ratio = np.array([x / 100.0 for x in range(1, 100)])
        sample_count_range = np.maximum(5*(4*global_a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(4*global_a_high + eps)/eps**2), -3*k/global_a_low * np.log(((1-delta_ratio)*delta)))
        sample_count = int(min(sample_count_range))
    print("Estimated to need %d samples" % sample_count)
    
    
print("After %d rounds the following bounds have been found"%rounds)
print("a_high: %f" % global_a_high)
print("a_low: %f" % global_a_low)


plt.plot(np.ones_like(global_mu) * global_a_high)
plt.plot(np.ones_like(global_mu) * global_a_low)
plt.plot(global_mu)
plt.xlabel('Iteration')
plt.ylabel('Expected Value')
plt.show()

# my_car = Car(map_size, save_change = True)
# my_car.stay_marked = stay_marked
# fig1 = plt.figure(figsize=(5,5))
# my_car.im = plt.imshow(my_car.map, cmap=my_car.CM, interpolation='none')

# frame_count = 2000
# anim = animation.FuncAnimation(fig1, my_car.updatefig, frames=range(frame_count), interval = 1)
# #anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
# plt.axis('off')
# plt.show()
    



