from matplotlib import pyplot as plt
import numpy as np
from aeb import AEB
from hypothesis import compute_required_samples

# This script runs a basic monte carlo simulation to estimate the expected value of probability of the disagreement between f_i and f_m+1

T = 100000 # Time horizon
sample_count = T

print(f"Starting with {sample_count} samples")

K = 100
mu_list = []
repeat = True
while repeat:
    # Generate initial Map
    simulator = AEB()
    mu = 0
    m = int(sample_count)
    delta = T - m
    gamma = simulator.expected_loss**(-delta/m)
    f = np.zeros((K, m))
    x = np.zeros((K, 2, m))
    # Draw K samples for each timestep 0->m
    for i in range(m):
        (x[:, :, i], f[:, i]) = simulator.genSamples(m=K, noStep=True)
        simulator.next_step(gamma=gamma)

    # Compute the disagreement between the safety labels at the original timestep and m+1
    disagreements = np.sum(simulator.safety_label(
        x[:, 0, :m], x[:, 1, :m]) != f[:, :m], axis=0)
    mu = np.sum(disagreements)/K
    mu_bar = mu/m
    print(f"Calculated mu = {mu}")

    mu_list.append(mu_bar)
    mu_high = np.ceil(mu)/m
    mu_low  = np.floor(mu)/m
    print(f"Calculated mu_high = {mu_high}")
    print(f"Calculated mu_low = {mu_low}")

    # Calculate the number of samples needed
    sample_count = compute_required_samples(mu_high=mu_high, mu_low=mu_low)

    print(f"Estimated to need {sample_count} samples")
    if (sample_count <= m) and (0.9*m <= sample_count):
        repeat = False
        print("Done")

# Plot evolution of 'mu_bar'
plt.plot(mu_list)
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
