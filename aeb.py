import random
import numpy as np


class AEB:
    """Class for AEB (Automatic Emergency Braking) system."""
    braking_force = 2000  # braking force in N for p=100%
    M = 900  # mass of the vehicle in kg
    l_min = 6
    l_max = 120
    vsqr_min = (0.27777778 * 5) ** 2  # (5 km/h)^2
    vsqr_max = (0.27777778 * 160) ** 2  # (160 km/h)^2
    mu_vsqr = (0.27777778 * 70) ** 2  # assume 70km/h average speed
    std_vsqr = (0.27777778 * 20) ** 2

    def __init__(self, singleFacet=True):
        """Initialize the AEB system."""
        self.F_list = []
        self._p = 1  # Brake performance in range [0,1]
        self.singleFacet = singleFacet

    @property
    def F(self):
        """Calculate braking force taking performance into account."""
        return self._p * self.braking_force

    def measure_F(self):
        """Measure the braking force."""
        return self.F + random.gauss(mu=0.0, sigma=1/10)

    def reset_brake(self):
        """Reset the brake performance to 100%."""
        self._p = 1

    def next_step(self, dt=1):
        """Calculate the brake performance for the next time step."""
        mult_loss = min(1.00, random.gauss(mu=1.0-10**-7, sigma=10**-6))
        self._p = self._p*mult_loss*dt

    def safety_label(self, l, vsqr):
        """Calculate the safety label for a distance l in meters and speed v in km/h."""
        F = self.measure_F()
        return 0.5*vsqr * self.M/F < l

    def genSamples(self, dt=1, m=1, noStep=False):
        """Generate samples for the AEB system.

        Args:
            dt (int, optional): Time step for sample generation. Defaults to 1.
            m (int, optional): Number of samples to generate. Defaults to 1.
            noStep (bool, optional): Flag indicating whether to skip step calculations.

        Returns:
            tuple: A tuple containing the input samples and labels.

        """
        l = np.random.uniform(low=self.l_min, high=self.l_max, size=m)
        vsqr = np.minimum(np.maximum(np.random.normal(
            self.mu_vsqr, self.std_vsqr, size=m), self.vsqr_min), self.vsqr_max)
        f = np.zeros(m)
        self.F_list = np.zeros(m)  # Log true braking force (for reference)
        for i in range(m):
            if not noStep:
                self.next_step(dt)
            self.F_list[i] = self.F
            #  Generate label
            f[i] = self.safety_label(l[i], vsqr[i])
        x = np.stack((l, vsqr), axis=1)
        return (x, f)
    