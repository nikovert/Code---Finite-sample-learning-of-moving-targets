from math import*
import numpy as np
import random
import matplotlib as mpl
from mip import Model, minimize, xsum, BINARY, CONTINUOUS
class Forest:
    # Parameters for propagation
    wind_direction = 2*pi * 0
    wind_strength = 0.001
    base_prop = 0.001
    p_burnout = 0.000001
    p_rotation = np.array([[0.7*sin(wind_direction) - 0.7*cos(wind_direction), sin(wind_direction), 0.7*sin(wind_direction) + 0.7*cos(wind_direction)], [-cos(wind_direction), 1, cos(wind_direction)], [-0.7*sin(wind_direction) - 0.7*cos(wind_direction), -sin(wind_direction), -0.7*sin(wind_direction) + 0.7*cos(wind_direction)]])
    p = base_prop + np.clip(wind_strength * p_rotation, 0, 1-base_prop) # Propagation prob
    
    # Colours for potting
    fire_colour = [0.87,0.3,0.2,0.9]    # forest == 2
    tree_colour = [0.156,0.59,0,1]      # forest == 3
    ground_colour = [0.29,0.01,0,0.6]   # forest == 0
    burnt_colour = [0.1,0.1,0.1,1]      # forest == 1
    CM = mpl.colors.ListedColormap([ground_colour,burnt_colour,fire_colour, tree_colour])
    save_forest = False # save copies of the forest when propogating the fire.

    def __init__(self,w,d, save_forest=False):
        # Create a simple forest
        #    w - width for a square forest
        #    d - forest denisty
        self._forest = np.zeros((w,w))
        self.forest_width = w
        for y in range(w):
            for x in range(w):
                self._forest[y,x] = 3*(random.random() <= d)
        self.t = 0 # current time stamp

        # initialize a fire
        start_x = random.randint(5, w-5)
        start_y = random.randint(5, w-5)
        self._forest[start_x-1:start_x+1, start_y-1:start_y+1] = 2
        self._burn_time = np.zeros((w,w))
        if save_forest:
            self.save_forest = save_forest
            self.forest_change = []

    @property
    def forest(self):
        return self._forest

    def basic_fire_prop(self):
        if 2 not in self._forest:
            return
        forest_size = self._forest.shape
        forest_change = np.zeros(forest_size, dtype=bool)

        fire_row, fire_col = np.where(self._forest == 2)
        burnt_row, burnt_col = np.where(self._forest == 1)

        fire_count = np.count_nonzero(self._forest == 2)

        for i in range(fire_count):
            if fire_row[i] in [0, 1, forest_size[0]-1, forest_size[0]] or fire_col[i] in [0, 1, forest_size[1]-1, forest_size[1]]:
                change = (random.random() <= self.base_prop) # ignore wind
            else:
                change = (np.random.rand(self.p.shape[0], self.p.shape[1]) <= self.p)
            # Catch fire with probability less than self.p
            forest_change[max(int(fire_row[i])-1,0):min(int(fire_row[i])+2,forest_size[0]), max(int(fire_col[i])-1,0):min(int(fire_col[i])+2,forest_size[1])] += change
    
        forest_change[burnt_row, burnt_col] = False # Always stay burnt
        forest_change[fire_row, fire_col] = np.random.rand(len(fire_row),)**((1+self._burn_time[fire_row, fire_col])**(-100)) <= self.p_burnout # Fire goes out with prob less than base_prop
        
        self.t += 1
        self._forest -= forest_change
        if self.save_forest:
            self.forest_change.append(forest_change)

        self._burn_time[fire_row, fire_col] += 1
            
        fire_count = np.count_nonzero(self._forest == 2)
        burnt_count = np.count_nonzero(self._forest == 1)
        if self.t % 1000 == 0:
            print('fire count: ' + str(fire_count))
            print('burnt count: ' + str(burnt_count))

        # for i in range(len(burnt_row)):
        #     self._forest[burnt_row[i]][burnt_col[i]] = 1
        # for i in range(len(fire_row)):
        #     if (random.random() >= self.base_prop): # Keep Burning
        #         self._forest[fire_row[i]][fire_col[i]] = 2    
        #self._forest = np.clip(self._forest,0,3)

    def genSamples(self, m=1):
        # Assuming forest.shape[0] = forest.shape[1]
        x = np.random.randint(1, high=self.forest_width, size=(m,2))
        f = np.zeros((m,1))
        for i in range(m):
            self.basic_fire_prop()
            # Burnt or on fire land
            f[i] = self._forest[x[i, 0], x[i,1]] == 2
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
        # Assuming Mj = -mj = self.forest_width (upperbound/lowerbound for x[i,j])
        M = self.forest_width
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

    # ###### Test Model ########
    #     x_true = np.empty((int(sum(f)), 2))
    #     iter_true = 0
    #     model_true = Model()
    #     b = {j: model_true.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="b[%d]" % j) for j in range(Nf)}
    #     for i in range(m):
    #         if f[i]: # constraints for i in I1
    #             x_true[iter_true] = x[i]
    #             iter_true += 1
    #             for j in range(Nf):
    #                 model_true.add_constr(np.matmul(a[j], x[i]) + b[j] <= 0)

    #     width = {j: model_true.add_var(var_type=CONTINUOUS, lb = -M, ub = M, name="width[%d]" % j) for j in range(Nf_hf)}
    #     for j in range(Nf_hf):
    #         model_true.add_constr(-b[j] - b[j + Nf_hf] == width[j])
    #         model_true.add_constr(width[j] >= 0)
    #     model_true.objective = minimize(xsum(width[j] for j in range(Nf_hf)))


    #     b_check = [-max(x_true[:,0]), -max(x_true[:,1]), min(x_true[:,0]), min(x_true[:,1])]
    #     for example in x_true:
    #         for j in range(Nf):
    #             if np.matmul(a[j], example) + b_check[j] > 0:
    #                 print('found error')
    #     model = model_true
    # #######
        
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
        self.basic_fire_prop()
        self.im.set_array(self._forest)
        return self.im # Return value only used if blit=True