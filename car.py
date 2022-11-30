from math import*
import numpy as np
import random
import matplotlib as mpl
from mip import Model, minimize, xsum, BINARY, CONTINUOUS
class Car:
    # Parameters for propagation
    travel_vel = 0.1
    travel_dir = 0

    # Colours for potting
    car_colour = [0.156,0.59,0,1]     # map == 1
    ground_colour = [0.29,0.01,0,0.6]   # map == 0
    CM = mpl.colors.ListedColormap([ground_colour,car_colour])
    save_change = False # save copies of the map change

    def __init__(self, w, save_change=False):
        # Create a simple map
        #    w - width for a square map
        self._map = np.zeros((w,w), dtype=bool)
        self.map_width = w
        self.t = 0 # current time stamp

        # initialize a car
        start_x = random.randint(5, w-5)
        start_y = random.randint(5, w-5)
        self.car_position = (start_x, start_y)
        self._map[start_x-1:start_x+1, start_y-1:start_y+1] = 1
        if save_change:
            self.save_change = save_change
            self.map_change = []

    @property
    def map(self):
        return self._map

    def next_step(self):
        map_change = np.ones((self.map_width, self.map_width), dtype=bool)
        self.travel_dir += random.random() * pi/100
        x, y = self.car_position
        self._map[round(x)-1:round(x)+1, round(y)-1:round(y)+1] = 0
        map_change[round(x)-1:round(x)+1, round(y)-1:round(y)+1] = 0

        # If we are heading towards the top edge
        if (y > self.map_width-10) & (self.travel_dir%(2*pi) > 0) & (self.travel_dir%(2*pi) < pi):
                self.travel_dir += pi

        # If we are heading towards the bottom edge
        if (y < 10) & (self.travel_dir%(2*pi) > pi) & (self.travel_dir%(2*pi) < 2*pi):
                self.travel_dir += pi

        # If we are heading towards the top edge
        if (x > self.map_width-10) & (self.travel_dir%(2*pi) >= 0) & (self.travel_dir%(2*pi) < pi/2):
                self.travel_dir += pi/2
        if (x > self.map_width-10) & (self.travel_dir%(2*pi) > 3*pi/2) & (self.travel_dir%(2*pi) < 2*pi):
                self.travel_dir -= pi/2
                
        # If we are heading towards the bottom edge
        if (x < 10) & (self.travel_dir%(2*pi) > pi/2) & (self.travel_dir%(2*pi) < 3*pi/2):
                self.travel_dir += pi/2

        x_new, y_new = (x + self.travel_vel * cos(self.travel_dir), y + self.travel_vel * sin(self.travel_dir))
        self._map[round(x_new)-1:round(x_new)+1, round(y_new)-1:round(y_new)+1] = 1
        self.t += 1
        self.car_position = (x_new, y_new)

        if self.save_change:
            self.map_change.append(~(map_change^self._map))

    def genSamples(self, m=1):
        # Assuming map.shape[0] = map.shape[1]
        x = np.random.randint(1, high=self.map_width, size=(m,2))
        f = np.zeros((m,1))
        for i in range(m):
            self.next_step()
            # Burnt or on fire land
            f[i] = self._map[x[i, 0], x[i,1]] == 1
        return (x, f)


    def generateMsampleModel(self, nr_samples):
        # Get samples
        (x,f) = self.genSamples(m=nr_samples)
        m = len(f)

        # Number of facets
        Nf_hf = len(x[0])
        Nf = 2*Nf_hf
        
        a = np.identity(Nf_hf) # aj = a[j]
        a = np.block([[a], [-a]])
        # Assuming Mj = -mj = self.map_width (upperbound/lowerbound for x[i,j])
        M = self.map_width
        eps = 0.0001

        model = Model()
        v = {i: model.add_var(var_type=BINARY, name="v[%d ]" % i) for i in range(m)}
        b = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="b[%d]" % j) for j in range(Nf)}
        z = {(i,j): model.add_var(var_type=BINARY, name="z[%d, %d]" % (i, j)) for i in range(m) for j in range(Nf)}
        s = {(i,j): model.add_var(var_type=CONTINUOUS, lb = 0, ub = M, name="s[%d, %d]" % (i, j)) for i in range(m) for j in range(Nf)}
        
        model.objective = minimize(xsum(v[i] for i in range(m)))

        for i in range(m):
            if f[i]: # constraints for i in I1
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - s[i,j] <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:# constraints for i in I0
                for j in range(Nf):
                    model.add_constr(np.matmul(a[j], x[i]) + b[j] - M*(1 - z[i,j]) <= 0)
                    model.add_constr(eps + (-M - eps)*z[i,j] - s[i,j] - np.matmul(a[j], x[i])- b[j]<= 0)
                    model.add_constr(xsum(z[i, j] for j in range(Nf)) + 1 - Nf <= 0)
                    model.add_constr(xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
        width = {j: model.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="width[%d]" % j) for j in range(Nf_hf)}
        for j in range(Nf_hf):
            model.add_constr(-b[j] - b[j + Nf_hf] == width[j])
            model.add_constr(width[j] >= 0)
        
        model.x = x
        model.f = f

        model.v = v
        model.a = a
        model.b = b
        model.z = z
        model.s = s
        model.width = width

        return model

    def updatefig(self, frame):
        self.next_step()
        self.im.set_array(self._map)
        return self.im # Return value only used if blit=True