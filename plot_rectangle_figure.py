from math import cos, sin, tan, ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mip import OptimizationStatus
from hypothesis import Hypothesis


##########################################################
#                        MAIN SCRIPT
##########################################################

# Create an empty Hypothesis with a simulator attached
model = Hypothesis(singleFacet=False)

# Generate a model with multiple samples and perform optimization
np.random.seed(19681800)
sample_count = 100
model.generateMsampleModel(sample_count, reduce=True, addMILP=True)
status = model.optimize(max_nodes=2000)

# Print optimization results
if status == OptimizationStatus.OPTIMAL:
    print(f"optimal solution cost {model.objective_value} found")
elif status == OptimizationStatus.FEASIBLE:
    print(
        f"sol.cost {model.objective_value} found, best possible: {model.objective_bound}")
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print(
        f"no feasible solution found, lower bound is: {model.objective_bound}")

# Store violation points that contribute to infeasibility
violation_points = []
for i in range(len(model.f)):
    if not model.v[i].x:
        violation_points.append(model.x[i])

# Visualize the results using matplotlib
fig0 = plt.figure(figsize=(5, 5))

# Set labels for the plot
plt.xlabel('distance (m)')
plt.ylabel('speed (m/s)^2')

# Calculate polygon corners based on model parameters
#       p4 ----- p3
#       |        |
#       |        |
#       p1 ----- p2
#
theta = model.theta
b0 = model.b[0].x
b1 = model.b[1].x
b2 = model.b[2].x
b3 = model.b[3].x

# Define polygon corners
p1 = [b2 * cos(theta)-b3*sin(theta),  b2 * sin(theta)+b3*cos(theta)]
p2 = [-b0 * cos(theta)-b3*sin(theta), -b0 * sin(theta)+b3*cos(theta)]
p3 = [-b0 * cos(theta)+b1*sin(theta), -b0 * sin(theta)-b1*cos(theta)]
p4 = [b2 * cos(theta)+b1*sin(theta),  b2 * sin(theta)-b1*cos(theta)]

# Create a polygon patch
polygon = Polygon([p1, p2, p3, p4], alpha=0.4)
ax = plt.gca()
ax.add_patch(polygon)

# Set plot limits
ax.set_xlim([floor(min(polygon.xy[:,0])), ceil(max(polygon.xy[:,0]))])
ax.set_ylim([floor(min(polygon.xy[:,1]))-10, ceil(max(polygon.xy[:,1]))+10])

# Annotate polygon facets
for j in range(4):
    p = (polygon.xy[j]+polygon.xy[j+1])/2+1
    text = str(j)
    if j==0:
        text = str(4)
        p[1] -=9
    if j==2:
        p[1] -=2
    if j==3:
        p[0] -=3
    ax.annotate(text, xy=p, xycoords='data', xytext=(2, 2), textcoords='offset points')

# Draw extended Facet p1--p4
ax.axline(p1, p4, lw=2, color='r')

# Plot a circle
angle = np.linspace(np.pi/2+theta, np.pi/2, 100)  # 0 <= θ <= 2π
r =  30 #circle radius
x1 = min(polygon.xy[:,0])  + r * np.cos(angle)
x2 = min(polygon.xy[:,0])*tan(np.pi/2+theta) + r * np.sin(angle)
plt.plot(x1, x2, color='green')

# Annotate the angle
ax.annotate(r'$\theta$', xy=[68, 323], xycoords='data', usetex=True)

# Display the plot
plt.show()