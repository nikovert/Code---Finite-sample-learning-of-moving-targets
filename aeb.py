from math import atan, floor
import random
import numpy as np
from mip import Model, minimize, xsum, BINARY, CONTINUOUS


class AEB:
    """Class for AEB (Automatic Emergency Braking) system."""
    braking_force = 2000  # braking force in N for p=100%
    M = 900  # mass of the vehicle in kg
    l_min = 6
    l_max = 120
    vsqr_min = (0.27777778 * 5) ** 2  # (5 km/h)^2
    vsqr_max = (0.27777778 * 160) ** 2  # (160 km/h)^2
    _mu_vsqr = (0.27777778 * 70) ** 2  # assume 70km/h average speed
    _sigma_vsqr = (0.27777778 * 20) ** 2

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
            self._mu_vsqr, self._sigma_vsqr, size=m), self.vsqr_min), self.vsqr_max)
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

    def discard_samples(self, samples, label, theta):
        """Discard samples that are irrelevant for the MIP optimization.

        Args:
            samples (ndarray): Array of input samples.
            label (ndarray): Array of labels indicating whether each sample is in the safe set.
            theta (float): Angle for rotation.

        Returns:
            list: Indices of samples to be discarded.

        """

        # Create a rotation matrix using the given theta angle
        R = np.array(((np.cos(theta), np.sin(theta)),
                     (-np.sin(theta), np.cos(theta))))

        # Rotate the t_samples and samples using the rotation matrix
        rotated_t_samples = np.matmul(R, np.transpose(
            samples[np.argwhere(label > 0)[:, 0]]))
        rotated_samples = np.matmul(R, np.transpose(samples))

        # Calculate the minimum and maximum coordinates for each dimension in the rotated_t_samples
        min_coords = np.min(rotated_t_samples, axis=1)

        if self.singleFacet:
            max_coords = np.max(np.matmul(R, np.transpose(
                samples[np.argwhere(label < 1)[:, 0]])), axis=1)
        else:
            max_coords = np.max(rotated_t_samples, axis=1)

        # Initialize a list to store the indices of samples to be discarded
        discard_indices = []

        if self.singleFacet:
            indices = np.argwhere(np.logical_and(
                rotated_samples[0] < min_coords[0], label < 1)).flatten()
            discard_indices.extend(indices)
            indices = np.argwhere(np.logical_and(
                rotated_samples[0] > max_coords[0], label > 0)).flatten()
            discard_indices.extend(indices)
        else:
            # Iterate over each dimension
            for i, min_coord in enumerate(min_coords):
                # Find the indices where the rotated_f_samples are
                # lower/higher than the minimum coordinate
                lower_indices = np.argwhere(np.logical_and(
                    rotated_samples[i] < min_coord, label < 1)).flatten()
                upper_indices = np.argwhere(np.logical_and(
                    rotated_samples[i] > max_coords[i], label < 1)).flatten()

                # Add the lower_indices and upper_indices to the discard_indices list
                discard_indices.extend(lower_indices)
                discard_indices.extend(upper_indices)

        # Return the list of discard indices
        return discard_indices

    def generateMsampleModel(self, m, dt=1, reduce=True):
        """Generate an M-sample model for the AEB system using the MIP optimization.

        Args:
            m (int): Number of samples to generate.
            dt (int, optional): Time step for sample generation. Defaults to 1.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            Model: The generated M-sample model.

        """
        (x, f) = self.genSamples(dt=dt, m=m)
        model = self.build_model(x, f, reduce=reduce)
        return model

    def preprocess_samples(self, x, f, reduce=True):
        """Preprocess the samples for the AEB system.

        Args:
            x (ndarray): Array of input samples.
            f (ndarray): Array of labels indicating whether each sample is in the safe set.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            tuple: A tuple containing the preprocessing results.

        """
        theta = atan(-0.5 * self.M / self.measure_F())
        R = np.array(((np.cos(theta), np.sin(theta)),
                     (-np.sin(theta), np.cos(theta))))
        a = np.block([[R], [-R]])
        discard_indices = self.discard_samples(x, f, theta) if reduce else []
        eps = 10**-4
        Nf = 1 if self.singleFacet else len(x[0]) * 2
        M = max(self.l_max, self.vsqr_max)
        m = f.shape[0]
        return theta, a, discard_indices, eps, Nf, m, M

    def build_model(self, x, f, reduce=True):
        """Build the MIP optimization model for the AEB system.

        Args:
            x (ndarray): Array of input samples.
            f (ndarray): Array of labels indicating whether each sample is in the safe set.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            Model: The constructed optimization model.

        """

        theta, a, discard_indices, eps, Nf, m, M = self.preprocess_samples(
            x, f, reduce)
        model = Model()
        v = {i: model.add_var(var_type=BINARY, name=f"v_{i}")
             for i in range(m)}
        b = {j: model.add_var(var_type=CONTINUOUS, lb=-M,
                              ub=M, name=f"b_{j}") for j in range(Nf)}
        z = {(i, j): model.add_var(var_type=BINARY, name=f"z_{i},{j}")
             for i in range(m) for j in range(Nf)}
        s = {(i, j): model.add_var(var_type=CONTINUOUS, lb=0, ub=M,
                                   name=f"s_{i},{j}") for i in range(m) for j in range(Nf)}

        for i in range(m):
            if i in discard_indices:
                continue
            if f[i]:
                if self.singleFacet:
                    model.add_constr(
                        np.matmul(a[2], x[i]) + b[0] - s[i, 0] <= 0)
                    model.add_constr(s[i, 0] - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        model.add_constr(
                            np.matmul(a[j], x[i]) + b[j] - s[i, j] <= 0)
                        model.add_constr(
                            xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:
                if self.singleFacet:
                    model.add_constr(
                        np.matmul(a[2], x[i]) + b[0] - M * (1 - z[i, 0]) <= 0)
                    model.add_constr(
                        eps + (-M - eps) * z[i, 0] - s[i, 0] - np.matmul(a[2], x[i]) - b[0] <= 0)
                    model.add_constr(z[i, 0] <= 0)
                    model.add_constr(s[i, 0] - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        model.add_constr(
                            np.matmul(a[j], x[i]) + b[j] - M * (1 - z[i, j]) <= 0)
                        model.add_constr(
                            eps + (-M - eps) * z[i, j] - s[i, j]
                            - np.matmul(a[j], x[i]) - b[j] <= 0)
                        model.add_constr(xsum(z[i, j]
                                         for j in range(Nf)) + 1 - Nf <= 0)
                        model.add_constr(
                            xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)

        if self.singleFacet:
            model.objective = minimize(xsum(v[i] for i in range(m)))
        else:
            width = {j: model.add_var(
                var_type=CONTINUOUS, lb=0, ub=2 * M, name=f"width_{j}") for j in range(floor(Nf/2))}
            for j in range(floor(Nf/2)):
                model.add_constr(-b[j] - b[j + floor(Nf/2)] == width[j])
            volume_weight = 0.01
            model.objective = minimize(xsum(v[i] for i in range(
                m)) + volume_weight * xsum(width[j] for j in range(floor(Nf/2))))
            model.width = width

        print(f"Discarded {len(discard_indices)} samples.")
        print(f"Discarded {(len(discard_indices)/len(x))} %% of samples.")

        # Save values to model
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
