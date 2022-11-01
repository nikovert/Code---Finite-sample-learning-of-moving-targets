from math import*
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from forest import Forest


fig = plt.figure(figsize=(5,5))
my_forest = Forest(400, 400, 0.6)

# initilaise a fire
my_forest.forest[200, 150] = 2
plt.pause(0.001)

anim = animation.FuncAnimation(fig, my_forest.updatefig, interval=10, blit=True)
anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
plt.axis('off')
plt.show()



