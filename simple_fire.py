from math import*
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl


def create_simple_forest(l,h,d):
    # Create a simple forest
    #    l - length
    #    h - height
    #    d - forest denisty
    forest=np.zeros((l,h))
    for y in range(h):
        for x in range(l):
            forest[y,x] = 3*(random.random() <= d)
    return forest


forest = create_simple_forest(400, 400, 0.6)
fig = plt.figure(figsize=(15,15))

# initilaise a fire
forest[200, 150] = 2

def basic_fire_prop():
    if 2 not in forest:
        return forest
    forest_size = forest.shape

    fire_row, fire_col = np.where(forest == 2)
    burnt_row, burnt_col = np.where(forest == 1)

    fire_count = np.count_nonzero(forest == 2)

    wind_direction = pi * 5/8
    wind_strength = 0.5
    base_prop = 0.1
    p = np.clip(base_prop + wind_strength * np.cos(wind_direction - pi/8 * np.array([[3.,  2, 1], [4.,  2.,  0], [5,  6,  7]])), 0, 1) # Propagation prob
    for i in range(fire_count):
        if fire_row[i] in [0, 1, forest_size[0]-1, forest_size[0]] or fire_col[i] in [0, 1, forest_size[1]-1, forest_size[1]]:
            p_tmp = base_prop # ignore wind
        else:
            p_tmp = p  
        forest[max(int(fire_row[i])-1,0):min(int(fire_row[i])+2,forest_size[0]), max(int(fire_col[i])-1,0):min(int(fire_col[i])+2,forest_size[1])] -= (random.random() <= p_tmp) 
    
    fire_count = np.count_nonzero(forest == 2)
    burnt_count = np.count_nonzero(forest == 1)
    print('fire count: ' + str(fire_count))
    print('burnt count: ' + str(burnt_count))

    for i in range(len(burnt_row)):
        forest[burnt_row[i]][burnt_col[i]] = 1
    return np.clip(forest,0,3)

fire_colour = [0.87,0.3,0.2,0.9]
tree_colour = [0.156,0.59,0,1]
ground_colour = [0.29,0.01,0,0.6]
burnt_colour = [0.1,0.1,0.1,1]

CM = mpl.colors.ListedColormap([ground_colour,tree_colour,burnt_colour, fire_colour])
CM = mpl.colors.ListedColormap([ground_colour,burnt_colour,fire_colour, tree_colour])
im = plt.imshow(basic_fire_prop(), cmap=CM, interpolation='none')

def updatefig(*args):
    #forest_new = basic_fire_prop(forest_old)
    im.set_array(basic_fire_prop())
    return im,
 
anim = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
plt.axis('off')
plt.show()



