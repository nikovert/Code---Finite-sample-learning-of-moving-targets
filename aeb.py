from math import*
import numpy as np
import random
import matplotlib as mpl
from mip import Model, minimize, xsum, BINARY, CONTINUOUS
import copy as cp
class AEB:
    # Parameters
    braking_force = 2000 # braking force in N for p=100%
    m = 900 # mass of the vehicle in kg
    _l_min = 30
    _l_max = 100
    v_min = 5 # in km/h
    v_max = 160 # in km/h
    _vsqr_min = (0.27777778*v_min)**2
    _vsqr_max = (0.27777778*v_max)**2

    def __init__(self, save_change=False):
        self._p = 1 # Brake performance in range [0,1]
        self._mu_vsqr = (0.27777778*70)**2 # assume 70km/h average speed
        self._sigma_vsqr = (0.27777778*20)**2
        
    @property
    def F(self):
        # return braking force taking perfromance into account
        return self._p * self.braking_force

    def reset_brake(self):
        # reset the brake performance to 100%
        self._p = 1 

    def next_step(self, dt=1):
        mult_loss = min(1,random.gauss(mu=1.0, sigma=10**-5))
        self._p = self._p*mult_loss*dt 

    def genSamples(self, dt=1, m=1):
        # Assuming map.shape[0] = map.shape[1]
        l = np.random.uniform(low=self._l_min, high=self._l_max, size=m)
        vsqr = np.minimum(np.maximum(np.random.normal(self._mu_vsqr, self._sigma_vsqr, size=m), self._vsqr_min), self._vsqr_max)
        f = np.zeros(m)
        for i in range(m):
            self.next_step(dt)
            # Generate label
            f[i] = self.safety_label(l[i], vsqr[i])
        x = np.stack((l, vsqr), axis=1)
        return (x, f)

    def safety_label(self, l, vsqr):
        # return safety label for a distance l in m and speed v in km/h
        # 1 Kilometer/hour = 0.27777778 Meter/second
        return 0.5*vsqr * self.m/self.F < l

    def generateMsampleModel(self, nr_samples, dt=1, reduce=True):
        # Get samples
        (x,f) = self.genSamples(dt, m=nr_samples)

        if reduce:
           target_samples =  x[np.argwhere(f)[:, 0]]
           max_l, max_vsqr = np.max(target_samples,0)
           min_l, min_vsqr = np.min(target_samples,0)
        else:
            max_l, max_vsqr = self._l_max, self._vsqr_max
            min_l, min_vsqr = self._l_min, self._vsqr_min

        m = len(f)

        # Number of facets
        Nf_hf = len(x[0])
        Nf = 2*Nf_hf
        
        # a = np.identity(Nf_hf) # aj = a[j]
        # a = np.block([[a], [-a]])

        # Assuming Mj = -mj = self.map_width (upperbound/lowerbound for x[i,j])
        M = max(max_l, max_vsqr)
        eps = 0.0001

        model = Model()
        v = {i: model.add_var(var_type=BINARY, name="v_%d" % i) for i in range(m)}
        b = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="b_%d" % j) for j in range(Nf)}
        z = {(i,j): model.add_var(var_type=BINARY, name="z_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}
        s = {(i,j): model.add_var(var_type=CONTINUOUS, lb = 0, ub = M, name="s_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}

        theta = -13/180 * pi
        a = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        a = np.block([[a], [-a]])

        discarded = np.zeros_like(f)
        for i in range(m):
            if f[i]: # constraints for i in I1
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - s[i,j] <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:# constraints for i in I0
                if any((x[i] < [min_l, min_vsqr]) ^ (x[i] > [max_l, max_vsqr])): 
                    discarded[i] = 1
                    continue
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - M*(1 - z[i,j]) <= 0)
                    model.add_constr(eps + (-M - eps)*z[i,j] - s[i,j] - np.matmul(a[j], x[i])- b[j]<= 0)
                    model.add_constr(xsum(z[i, j] for j in range(Nf)) + 1 - Nf <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
        width = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="width_%d" % j) for j in range(Nf_hf)}
        for j in range(Nf_hf):
            model.add_constr(-b[j] - b[j + Nf_hf] == width[j])
            model.add_constr(width[j] >= 0)
        
        volume_weight = 10**-4
        model.objective = minimize(xsum(v[i] for i in range(m)) + volume_weight * xsum(width[j] for j in range(Nf_hf)))
        #model.objective = minimize(xsum(v[i] for i in range(m)))
                                   
        print("Discarded %d samples." % sum(discarded))
        model.x = x
        model.f = f
        model.discarded = discarded

        model.v = v
        model.a = a
        model.b = b
        model.z = z
        model.s = s
        model.width = width
        model.theta = theta

        print("Model reduced to using %d samples." % (nr_samples-sum(discarded)))

        return model