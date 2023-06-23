from matplotlib import pyplot as plt
import numpy as np
from aeb import AEB
from hypothesis import compute_required_samples

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

sample_count = compute_required_samples()

print(f"Starting with {sample_count} samples")

K = 100
a_list = []
repeat = True
while repeat:
    # Generate initial Map
    simulator = AEB()
    mu = 0
    m = int(sample_count)
    f = np.zeros((K, m))
    x = np.zeros((K, 2, m))
    # Draw K samples for each timestep 0->m
    for i in range(m):
        (x[:, :, i], f[:, i]) = simulator.genSamples(m=K, noStep=True)
        simulator.next_step()

    # Compute the disagreement between the safety labels at the original timestep and m+1
    disagreements = np.sum(simulator.safety_label(
        x[:, 0, :m], x[:, 1, :m]) != f[:, :m], axis=0)
    mu = np.sum(disagreements)/K
    a = mu/m
    print(f"Calculated mu = {mu}")
    print(f"Calculated a = {a}")

    a_list.append(a)
    a_high = 1.01*a

    # Calculate the number of samples needed
    sample_count = compute_required_samples(a_high=a_high)

    print(f"Estimated to need {sample_count} samples")
    if (sample_count <= m) and (0.9*m <= sample_count):
        repeat = False
        print("Done")

# Plot evolution of 'a'
plt.plot(a_list)
plt.show()

##### Outcome ######
# Starting with 105630 samples
# Calculated mu = 3683.75
# Calculated a = 0.034874088800530154
# Estimated to need 105877 samples
# Calculated mu = 3647.28
# Calculated a = 0.034448274885008075
# Estimated to need 105400 samples
# Done
