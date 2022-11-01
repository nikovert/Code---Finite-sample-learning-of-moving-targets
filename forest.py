from math import*
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
class Forest:
    # Parameters for propagation
    wind_direction = pi * 5/8
    wind_strength = 0.5
    base_prop = 0.1
    p = np.clip(base_prop + wind_strength * np.cos(wind_direction - pi/8 * np.array([[3.,  2, 1], [4.,  2.,  0], [5,  6,  7]])), 0, 1) # Propagation prob
    
    # Colours for potting
    fire_colour = [0.87,0.3,0.2,0.9]
    tree_colour = [0.156,0.59,0,1]
    ground_colour = [0.29,0.01,0,0.6]
    burnt_colour = [0.1,0.1,0.1,1]
    CM = mpl.colors.ListedColormap([ground_colour,burnt_colour,fire_colour, tree_colour])
    

    def __init__(self, l,h,d):
        # Create a simple forest
        #    l - length
        #    h - height
        #    d - forest denisty
        self.forest = np.zeros((l,h))
        for y in range(h):
            for x in range(l):
                self.forest[y,x] = 3*(random.random() <= d)
        self.im = plt.imshow(self.forest, cmap=self.CM, interpolation='none')
        plt.show()
    def basic_fire_prop(self):
        if 2 not in self.forest:
            return
        forest_size = self.forest.shape

        fire_row, fire_col = np.where(self.forest == 2)
        burnt_row, burnt_col = np.where(self.forest == 1)

        fire_count = np.count_nonzero(self.forest == 2)

        for i in range(fire_count):
            if fire_row[i] in [0, 1, forest_size[0]-1, forest_size[0]] or fire_col[i] in [0, 1, forest_size[1]-1, forest_size[1]]:
                p_tmp = self.base_prop # ignore wind
            else:
                p_tmp = self.p  
            self.forest[max(int(fire_row[i])-1,0):min(int(fire_row[i])+2,forest_size[0]), max(int(fire_col[i])-1,0):min(int(fire_col[i])+2,forest_size[1])] -= (random.random() <= p_tmp) 
    
        fire_count = np.count_nonzero(self.forest == 2)
        burnt_count = np.count_nonzero(self.forest == 1)
        print('fire count: ' + str(fire_count))
        print('burnt count: ' + str(burnt_count))

        for i in range(len(burnt_row)):
            self.forest[burnt_row[i]][burnt_col[i]] = 1
        self.forest = np.clip(self.forest,0,3)

    def updatefig(self, *args):
        self.basic_fire_prop()
        self.im.set_array(self.forest)
        return self.im,