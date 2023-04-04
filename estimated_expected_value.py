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
from scipy.optimize import fmin
import time

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

delta = 10**-4
d = 4
k = 1
eps = 0.15
a_high =  0.001336
a_low  = 0.000529
map_size = 250
stay_marked = False

# alpha = np.outer(np.linspace(0.001, 1-0.001, 100), np.ones(100))
# delta_ratio = alpha.copy().T
# alpha, delta_ratio = np.meshgrid(np.linspace(0.001, 1-0.001, 100), np.linspace(0.001, 1-0.001, 100))

delta_ratio = 0.5;
alpha = np.linspace(0.001, 1-0.001, 100)

m_min = 5*(2*(1+alpha)*a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(2*(1+alpha)*a_high + eps)/eps**2)
m_max = -3*k/(alpha**2 * a_low) * np.log(((1-delta_ratio)*delta))

condition = abs(m_min - m_max+1)
ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
sample_count_range = [m_min[ind], m_max[ind]]
sample_count = ceil(np.min(sample_count_range))
# ind = np.unravel_index(np.argmin(sample_count_range, axis=None), sample_count_range.shape)
print("Starting with %d samples" % sample_count)
# print("minimum alpha: %f" % alpha[ind])
# print("minimum delta_ratio: %f" % delta_ratio[ind])


# Time horizon
T = 2000; #in seconds
dt = T/sample_count

rounds = 5
global_a_high = np.zeros(rounds)
global_a_low = np.zeros(rounds)
global_mu = np.zeros(rounds)
global_dt = np.zeros(rounds)
global_T = np.zeros(rounds)
global_m = np.zeros(rounds)
for iter in range(rounds):
    print("Starting round %d" % iter)
    repeat = True
    while repeat:
        # Generate initial Map
        my_car = Car(map_size, save_change = True)
        my_car.stay_marked = stay_marked
        mu = 0
        m = int(sample_count)
        for i in range(m):
            my_car.next_step(dt)
        for i in range(m):
            change = my_car.map_archive[m-1-i]^my_car.map
            p_i = np.count_nonzero(change)/my_car.map.size
            mu += p_i
        print("Calculated mu = %f" % mu)

        # mu = a*m
        a_high = 1.01*mu/m
        a_low = 0.99*mu/m
        global_a_high[iter] = a_high
        global_a_low[iter]  = a_low
        global_mu[iter] = mu
        # Calculate the number of samples needed
        m_min = 5*(2*(1+alpha)*a_high + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* 40*(2*(1+alpha)*a_high + eps)/eps**2)
        m_max = -3*k/(alpha**2 * a_low) * np.log(((1-delta_ratio)*delta))

        condition = abs(m_min - m_max+1)
        ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
        sample_count_range = [m_min[ind], m_max[ind]]
        sample_count = ceil(np.max(sample_count_range))-1

        sample_count *= (iter+1)

        dt = T/ sample_count
        global_dt[iter] = dt
        global_T[iter] = T
        global_m[iter] = m
        print("Estimated to need %d samples" % sample_count)
        print("Calculated dt = %f" % dt)
        if (sample_count >= m_min[ind]) & (sample_count <= m_max[ind]):
            repeat = False
        repeat = False
    
    
print("After %d rounds the following bounds have been found"%rounds)
print("a_high: %f" % max(global_a_high))
print("a_low: %f" % min(global_a_low))


plt.plot(np.ones_like(global_mu) * max(global_a_high))
plt.plot(np.ones_like(global_mu) * min(global_a_low))
plt.plot(global_mu/global_m)
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
    



