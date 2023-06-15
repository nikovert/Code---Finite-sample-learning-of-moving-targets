from math import*
import numpy as np
import random
import matplotlib as mpl
from mip import Model, minimize, xsum, BINARY, CONTINUOUS
import copy as cp
class AEB:
    # Parameters
    braking_force = 2000 # braking force in N for p=100%
    M = 900 # mass of the vehicle in kg
    _l_min = 60
    _l_max = 120
    v_min = 5 # in km/h
    v_max = 160 # in km/h
    _vsqr_min = (0.27777778*v_min)**2
    _vsqr_max = (0.27777778*v_max)**2

    def __init__(self, save_change=False):
        self._p = 1 # Brake performance in range [0,1]
        self._mu_vsqr = (0.27777778*70)**2 # assume 70km/h average speed
        self._sigma_vsqr = (0.27777778*20)**2
        self._singleFacet = True
        
    @property
    def F(self):
        # return braking force taking perfromance into account
        return self._p * self.braking_force

    def measure_F(self):
        # reset the brake performance to 100%
        return self.F + random.gauss(mu=0.0, sigma=1/10)

    def reset_brake(self):
        # reset the brake performance to 100%
        self._p = 1 

    def next_step(self, dt=1):
        mult_loss = min(1.00+10**-4,random.gauss(mu=1.0, sigma=10**-3))
        self._p = self._p*mult_loss*dt 

    def safety_label(self, l, vsqr):
        # return safety label for a distance l in m and speed v in km/h
        # 1 Kilometer/hour = 0.27777778 Meter/second
        F = self.measure_F()
        return 0.5*vsqr * self.M/F < l

    def genSamples(self, dt=1, m=1):
        # Assuming map.shape[0] = map.shape[1]
        l = np.random.uniform(low=self._l_min, high=self._l_max, size=m)
        vsqr = np.minimum(np.maximum(np.random.normal(self._mu_vsqr, self._sigma_vsqr, size=m), self._vsqr_min), self._vsqr_max)
        f = np.zeros(m)
        self.F_list = np.zeros(m) # Log true braking force (for reference)
        for i in range(m):
            self.next_step(dt)
            self.F_list[i] = self.F
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

    def discard_samples(self, samples, label, theta):
        # Create a rotation matrix using the given theta angle
        R = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        
        # Rotate the t_samples and samples using the rotation matrix
        rotated_t_samples = np.matmul(R, np.transpose(samples[np.argwhere(label>0)[:, 0]]))
        rotated_samples = np.matmul(R, np.transpose(samples))

        # Calculate the minimum and maximum coordinates for each dimension in the rotated_t_samples
        min_coords = np.min(rotated_t_samples, axis=1)
        max_coords = np.max(rotated_t_samples, axis=1)

        # Initialize a list to store the indices of samples to be discarded
        discard_indices = []

        # Iterate over each dimension
        for i in range(len(min_coords)):
            if self._singleFacet:
                indices = np.argwhere(np.logical_and(rotated_samples[0] < min_coords[0], label<1)).flatten()
                discard_indices.extend(indices)
            else:
                # Find the indices where the rotated_f_samples are lower/higher than the minimum coordinate
                lower_indices = np.argwhere(np.logical_and(rotated_samples[i] < min_coords[i], label<1)).flatten()
                upper_indices = np.argwhere(np.logical_and(rotated_samples[i] > max_coords[i], label<1)).flatten()
                
                # Add the lower_indices and upper_indices to the discard_indices list
                discard_indices.extend(lower_indices)
                discard_indices.extend(upper_indices)

        # Return the list of discard indices
        return discard_indices

    def generateMsampleModel(self, m, dt=1, reduce=True):
        # Get samples
        (x,f) = self.genSamples(dt, m)

        # Add fixed rotation of the hyperrectangle based on a measurment of the braking force
        theta = atan(-0.5*self.M/self.measure_F())
        R = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        
        # Compute samples that are irrelevant for the MIP
        discard_indices = self.discard_samples(x, f, theta)

        # Number of facets
        if self._singleFacet:
            Nf = 1
        else:
            Nf = len(x[0])*2

        # Assuming Mj = -mj = self.map_width (upperbound/lowerbound for x[i,j])
        M = max(self._l_max, self._vsqr_max)
        eps = 0.0001

        model = Model()
        v = {i: model.add_var(var_type=BINARY, name="v_%d" % i) for i in range(m)}
        b = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="b_%d" % j) for j in range(Nf)}
        z = {(i,j): model.add_var(var_type=BINARY, name="z_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}
        s = {(i,j): model.add_var(var_type=CONTINUOUS, lb = 0, ub = M, name="s_%d,%d" % (i, j)) for i in range(m) for j in range(Nf)}

        a = np.block([[R], [-R]])

        for i in range(m):
            if i in discard_indices:
                continue
            if f[i]:
                if self._singleFacet:
                    model.add_constr(np.matmul(a[2], x[i]) + b[0] - s[i,0] <= 0)
                    model.add_constr(s[i, 0] - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        model.add_constr(np.matmul(a[j], x[i]) + b[j] - s[i,j] <= 0)
                        model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:# constraints for i in I0
                if self._singleFacet:
                    model.add_constr(np.matmul(a[2], x[i]) + b[0] - M*(1 - z[i,0]) <= 0)
                    model.add_constr(eps + (-M - eps)*z[i,0] - s[i,0] - np.matmul(a[2], x[i])- b[0]<= 0)
                    model.add_constr(z[i, 0] <= 0)
                    model.add_constr(s[i, 0]  - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        model.add_constr(np.matmul(a[j], x[i]) + b[j] - M*(1 - z[i,j]) <= 0)
                        model.add_constr(eps + (-M - eps)*z[i,j] - s[i,j] - np.matmul(a[j], x[i])- b[j]<= 0)
                        model.add_constr(xsum(z[i, j] for j in range(Nf)) + 1 - Nf <= 0)
                        model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
        if self._singleFacet:
            model.objective = minimize(xsum(v[i] for i in range(m)))
        else:
            Nf_hf = floor(Nf/2)
            width = {j: model.add_var(var_type=CONTINUOUS, lb = 0, ub = 2*M, name="width_%d" % j) for j in range(Nf_hf)}
            for j in range(Nf_hf):
                model.add_constr(-b[j] - b[j + Nf_hf] == width[j])
            volume_weight = 10**-ceil(log(m)-8) # for m = 20000
            volume_weight = 0.01 
            model.objective = minimize(xsum(v[i] for i in range(m)) + volume_weight * xsum(width[j] for j in range(Nf_hf)))
            model.width = width
                                   
        print("Discarded %d samples." % (len(discard_indices)))
        print("Discarded %f %% of samples." % (len(discard_indices)/len(x)))
        model.x = x
        model.f = f
        model.discard_indices = discard_indices
        model.F_list = self.F_list

        model.v = v
        model.a = a
        model.b = b
        model.z = z
        model.s = s
        model.theta = theta

        return model