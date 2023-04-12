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
    
    def genSamples_nostep(self, m=1):
        # Assuming map.shape[0] = map.shape[1]
        l = np.random.uniform(low=self._l_min, high=self._l_max, size=m)
        vsqr = np.minimum(np.maximum(np.random.normal(self._mu_vsqr, self._sigma_vsqr, size=m), self._vsqr_min), self._vsqr_max)
        f = np.zeros(m)
        for i in range(m):
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

        theta_degrees = 45 * 0.5*self.m/self.F
        theta = -13/180 * pi # Fixed rotation angle
        R = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))

        if reduce:
           margin = 0.0
           t_samples =  x[np.argwhere(f)[:, 0]]
           t_max_l, t_max_vsqr =  margin + np.max(np.matmul(R, np.transpose(t_samples)),1)
           t_min_l, t_min_vsqr = -margin + np.min(np.matmul(R, np.transpose(t_samples)),1)
           
           f_samples = x[np.argwhere(f<1)[:, 0]]
           f_max_l, f_max_vsqr =  margin + np.max(np.matmul(R, np.transpose(f_samples)),1)
           f_min_l, f_min_vsqr = -margin + np.min(np.matmul(R, np.transpose(f_samples)),1)

        m = len(f)

        # Number of facets
        Nf_hf = len(x[0])
        Nf = 2*Nf_hf
        
        # a = np.identity(Nf_hf) # aj = a[j]
        # a = np.block([[a], [-a]])

        # Assuming Mj = -mj = self.map_width (upperbound/lowerbound for x[i,j])
        M = max(self._l_max, self._vsqr_max)
        eps = 0.0001

        model = Model()
        v = {i: model.add_var(var_type=BINARY, name="v_%d" % i) for i in range(m)}
        b = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="b_%d" % j) for j in range(Nf)}
        z = {(i,j): model.add_var(var_type=BINARY, name="z_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}
        s = {(i,j): model.add_var(var_type=CONTINUOUS, lb = 0, ub = M, name="s_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}

        a = np.block([[R], [-R]])

        discarded_f = np.zeros_like(f)
        discarded_t = np.zeros_like(f)
        min_target = [0,0]
        max_target = [self._l_max, self._vsqr_max]

        for i in range(m):
            if f[i]: # constraints for i in I1
                # if reduce and any((np.matmul(R, np.transpose(x[i])) < [f_min_l, f_min_vsqr]) ^ (np.matmul(R, np.transpose(x[i])) > [f_max_l, f_max_vsqr])): 
                #     if all(x[i] < min_target):
                #         if all(x[i] > max_target):
                #             discarded_t[i] = 1
                #             continue
                #         else:
                #             max_target = x[i]
                #     else:
                #         min_target = x[i]
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - s[i,j] <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:# constraints for i in I0
                if reduce and any((np.matmul(R, np.transpose(x[i])) < [t_min_l, t_min_vsqr]) ^ (np.matmul(R, np.transpose(x[i])) > [t_max_l, t_max_vsqr])): 
                    discarded_f[i] = 1
                    continue
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - M*(1 - z[i,j]) <= 0)
                    model.add_constr(eps + (-M - eps)*z[i,j] - s[i,j] - np.matmul(a[j], x[i])- b[j]<= 0)
                    model.add_constr(xsum(z[i, j] for j in range(Nf)) + 1 - Nf <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
        width = {j: model.add_var(var_type=CONTINUOUS, lb = 0, ub = 2*M, name="width_%d" % j) for j in range(Nf_hf)}
        for j in range(Nf_hf):
            model.add_constr(-b[j] - b[j + Nf_hf] == width[j])
        
        volume_weight = 10**-ceil(log(m)-8)
        model.objective = minimize(xsum(v[i] for i in range(m)) + volume_weight * xsum(width[j] for j in range(Nf_hf)))
        #model.objective = minimize(xsum(v[i] for i in range(m)))
                                   
        print("Discarded %d samples." % (sum(discarded_f)+sum(discarded_t)))
        print("Discarded %f %% of true samples." % (sum(discarded_t)/len(t_samples)))
        print("Discarded %f %% of false samples." % (sum(discarded_f)/len(f_samples)))
        model.x = x
        model.f = f
        model.discarded_f = discarded_f
        model.discarded_t = discarded_t

        model.v = v
        model.a = a
        model.b = b
        model.z = z
        model.s = s
        model.width = width
        model.theta = theta

        print("Model reduced to using %d samples." % (nr_samples-sum(discarded_f)-sum(discarded_t)))

        return model