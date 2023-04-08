from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from aeb import AEB
from mip import Model, OptimizationStatus
from scipy.optimize import fmin
import time

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

delta = 10**-4
d = 4
k = 1
eps = 0.03
a_high =  0.077
a_low = 0

# alpha = np.outer(np.linspace(0.001, 1-0.001, 100), np.ones(100))
# delta_ratio = alpha.copy().T
# alpha, delta_ratio = np.meshgrid(np.linspace(0.001, 1-0.001, 100), np.linspace(0.001, 1-0.001, 100))

delta_ratio = 1-10**-5;
t = np.linspace(10**-5, 1-10**-5, 10000)

m_min = 5*(2*(a_high+t) + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* np.log(40*(2*(a_high+t) + eps)/eps**2))
m_max = -1/(2 * t**2) * np.log(((1-delta_ratio)*delta))

condition = abs(m_min - m_max+1)
ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
sample_count_range = [m_min[ind], m_max[ind]]
sample_count = ceil((m_min[ind] + m_max[ind])/2)

# ind = np.unravel_index(np.argmin(sample_count_range, axis=None), sample_count_range.shape)
print("Starting with %d samples" % sample_count)
# print("minimum alpha: %f" % alpha[ind])
# print("minimum delta_ratio: %f" % delta_ratio[ind])

rounds = 5
global_a_high = np.zeros(rounds)
global_a_low = np.zeros(rounds)
global_mu = np.zeros(rounds)
global_m = np.zeros(rounds)
for iter in range(rounds):
    print("Starting round %d" % iter)
    repeat = True
    while repeat:
        # Generate initial Map
        simulator = AEB()
        mu = 0
        m = int(sample_count)
        K = 2000
        f = np.zeros((K,m))
        x = np.zeros((K,2,m))
        for i in range(m):
            (x[:,:,i],f[:,i]) = simulator.genSamples_nostep(K)
            simulator.next_step()

        simulator.next_step() # proceed to m+1
        for i in range(m):
            f_last = np.zeros(K)
            for j in range(K):
                f_last[j] = simulator.safety_label(x[j,0,i], x[j,1,i])
            mu += np.count_nonzero(f_last != f[:,i])/K

        print("Calculated mu = %f" % mu)

        # mu = a*m
        a_high = 1.01*mu/m
        #a_low = 0.99*mu/m
        global_a_high[iter] = a_high
        #global_a_low[iter]  = a_low
        global_mu[iter] = mu
        # Calculate the number of samples needed
        m_min = 5*(2*(a_high+t) + eps)/eps**2 * (-np.log((delta_ratio*delta)/4) + d* np.log(40*(2*(a_high+t) + eps)/eps**2))
        m_max = -1/(2 * t**2) * np.log(((1-delta_ratio)*delta))

        condition = abs(m_min - m_max+1)
        ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
        sample_count_range = [m_min[ind], m_max[ind]]
        sample_count = ceil((m_min[ind] + m_max[ind])/2)
        
        global_m[iter] = m
        print("Estimated to need %d samples" % sample_count)
        if (m >= m_min[ind]) & (m <= m_max[ind]):
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
    



