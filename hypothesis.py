from math import atan, floor, ceil
import numpy as np
from mip import Model, minimize, xsum, BINARY, CONTINUOUS
from numpy import linalg as LA
from scipy.stats import norm
from aeb import AEB

"""
    Main results from paper: Hypothesis class for hypothesis generation and compute_required_samples function to find m
"""


def compute_required_samples(eps=0.01, delta=10**-4, a_high=0.035, vc_dim=1):
    """
    Compute the required number of samples for a hypothesis test.

    Args:
        eps (float): The desired precision. The default value is 0.01.
        delta (float): The desired maximum error probability. The default value is 10**-4.
        a_high (float): The upper bound on the empirical error of the hypothesis. The default value is 0.035.
        vc_dim (int): The VC dimension of the hypothesis class. The default value is 1.

    Returns:
        int: The required number of samples for the hypothesis test.
    """
    delta_ratio = 1-10**-6
    t = np.linspace(10**-6, 1-10**-6, 10000)

    m_min = 5*(2*(a_high+t) + eps)/eps**2 * (-np.log((delta_ratio *
                                                      delta)/4) + vc_dim * np.log(40*(2*(a_high+t) + eps)/eps**2))
    m_max = -1/(2 * t**2) * np.log(((1-delta_ratio)*delta))

    condition = abs(m_min - m_max+1)
    ind = np.unravel_index(np.argmin(condition, axis=None), condition.shape)
    sample_count = ceil((m_min[ind] + m_max[ind])/2)
    return sample_count


class Hypothesis(Model):
    """
    A class representing a hypothesis model for the AEB system.

    Args:
        singleFacet (bool, optional): Flag indicating whether to use a single facet or multiple facets. 
                                        Defaults to True.
    """

    def __init__(self, singleFacet=True):
        super().__init__()
        self.singleFacet = singleFacet
        self.simulator = AEB()

        # Save values to self
        self.x = []
        self.f = []
        self.discard_indices = []
        self.F_list = self.simulator.F_list

        self.v = []
        self.a = []
        self.b = []
        self.z = []
        self.s = []
        self.theta = []

        self.width = []

    def copy(self, solver_name=""):
        """
        Create a copy of the Hypothesis model.

        Returns:
            Hypothesis: A copy of the Hypothesis model.
        """
        new_instance = super().copy(solver_name)
        new_instance.__class__ = Hypothesis

        # Save values to self
        new_instance.singleFacet = self.singleFacet
        new_instance.simulator = self.simulator
        new_instance.x = self.x
        new_instance.f = self.f
        new_instance.discard_indices = self.discard_indices
        new_instance.F_list = self.F_list

        new_instance.v = self.v
        new_instance.a = self.a
        new_instance.b = self.b
        new_instance.z = self.z
        new_instance.s = self.s
        new_instance.theta = self.theta

        new_instance.width = self.width

        # Optionally copy any additional attributes or state
        return new_instance

    def prune(self, distance):
        """
        Prunes the model by removing samples within a certain distance from each other.

        Args:
            distance (float): The distance threshold for pruning.
            copy_model (bool, optional): Flag indicating whether to create a copy of the model before pruning.
                                         Defaults to True.

        Returns:
            Hypothesis: The pruned model.
        """
        prnd_model = self.copy()

        # Normalising constants
        cx = self.simulator.l_max - self.simulator.l_min
        cy = ceil(np.max(prnd_model.x[:, 1])) - \
            floor(np.min(prnd_model.x[:, 1]))
        c = np.array([cx, cy])

        check = self.x.shape[0]-1
        discard_list = np.zeros(self.x.shape)
        discard_list[self.discard_indices] = 1
        index_list = np.random.permutation(self.x.shape[0])

        while check > 0:
            #  Get all elments close to the indexed sample
            index = index_list[check]
            if check > prnd_model.x.shape[0]-1:
                reference_point, f = prnd_model.simulator.genSamples()
            else:
                reference_point = prnd_model.x[index, :]
            elements = np.argwhere(
                LA.norm((reference_point-prnd_model.x)/c, axis=1) < distance)[1:]

            #  Prune elements
            if elements.shape[0] > 0:
                prnd_model.x = np.delete(prnd_model.x, elements, 0)
                prnd_model.f = np.delete(prnd_model.f, elements, 0)
                discard_list = np.delete(discard_list, elements, 0)
                prnd_model.F_list = np.delete(prnd_model.F_list, elements, 0)
                index_list = np.random.permutation(prnd_model.x.shape[0])
                check = min(check, prnd_model.x.shape[0]-1)
            else:
                check -= 1

        prnd_model.discard_indices = np.argwhere(discard_list[:, 0])[:, 0]

        # Fit a Gaussian distribution to the samples
        mu, std = norm.fit(self.x[:, 1])
        if abs(self.simulator.mu_vsqr - mu)/mu > 0.01 or abs(self.simulator.std_vsqr - std)/std > 0.1:
            print("Warning, distribution doesn't match before pruning")

        # Fit a Gaussian distribution to the samples
        mu, std = norm.fit(prnd_model.x[:, 1])
        if abs(prnd_model.simulator.mu_vsqr - mu)/mu > 0.01 or abs(prnd_model.simulator.std_vsqr - std)/std > 0.1:
            print("Warning, distribution doesn't match after pruning")

        return prnd_model

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

    def preprocess_samples(self, x, f, reduce=True):
        """Preprocess the samples for the AEB system.

        Args:
            x (ndarray): Array of input samples.
            f (ndarray): Array of labels indicating whether each sample is in the safe set.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            tuple: A tuple containing the preprocessing results.

        """
        theta = atan(-0.5 * self.simulator.M / self.simulator.measure_F())
        R = np.array(((np.cos(theta), np.sin(theta)),
                     (-np.sin(theta), np.cos(theta))))
        a = np.block([[R], [-R]])
        discard_indices = self.discard_samples(x, f, theta) if reduce else []
        eps = 10**-4
        Nf = 1 if self.singleFacet else len(x[0]) * 2
        M = max(self.simulator.l_max, self.simulator.vsqr_max)
        m = f.shape[0]
        return theta, a, discard_indices, eps, Nf, m, M

    def build_model(self, x, f, reduce=True):
        """Build the MIP optimization self for the AEB system.

        Args:
            x (ndarray): Array of input samples.
            f (ndarray): Array of labels indicating whether each sample is in the safe set.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            Model: The constructed optimization self.

        """

        theta, a, discard_indices, eps, Nf, m, M = self.preprocess_samples(
            x, f, reduce)
        v = {i: self.add_var(var_type=BINARY, name=f"v_{i}")
             for i in range(m)}
        b = {j: self.add_var(var_type=CONTINUOUS, lb=-M,
                             ub=M, name=f"b_{j}") for j in range(Nf)}
        z = {(i, j): self.add_var(var_type=BINARY, name=f"z_{i},{j}")
             for i in range(m) for j in range(Nf)}
        s = {(i, j): self.add_var(var_type=CONTINUOUS, lb=0, ub=M,
                                  name=f"s_{i},{j}") for i in range(m) for j in range(Nf)}

        for i in range(m):
            if i in discard_indices:
                continue
            if f[i]:
                if self.singleFacet:
                    self.add_constr(
                        np.matmul(a[2], x[i]) + b[0] - s[i, 0] <= 0)
                    self.add_constr(s[i, 0] - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        self.add_constr(
                            np.matmul(a[j], x[i]) + b[j] - s[i, j] <= 0)
                        self.add_constr(
                            xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)
            else:
                if self.singleFacet:
                    self.add_constr(
                        np.matmul(a[2], x[i]) + b[0] - M * (1 - z[i, 0]) <= 0)
                    self.add_constr(
                        eps + (-M - eps) * z[i, 0] - s[i, 0] - np.matmul(a[2], x[i]) - b[0] <= 0)
                    self.add_constr(z[i, 0] <= 0)
                    self.add_constr(s[i, 0] - v[i] * M <= 0)
                else:
                    for j in range(Nf):
                        self.add_constr(
                            np.matmul(a[j], x[i]) + b[j] - M * (1 - z[i, j]) <= 0)
                        self.add_constr(
                            eps + (-M - eps) * z[i, j] - s[i, j]
                            - np.matmul(a[j], x[i]) - b[j] <= 0)
                        self.add_constr(xsum(z[i, j]
                                             for j in range(Nf)) + 1 - Nf <= 0)
                        self.add_constr(
                            xsum(s[i, j] for j in range(Nf)) - v[i] * Nf * M <= 0)

        if self.singleFacet:
            self.objective = minimize(xsum(v[i] for i in range(m)))
        else:
            width = {j: self.add_var(
                var_type=CONTINUOUS, lb=0, ub=2 * M, name=f"width_{j}") for j in range(floor(Nf/2))}
            for j in range(floor(Nf/2)):
                self.add_constr(-b[j] - b[j + floor(Nf/2)] == width[j])
            volume_weight = 0.01
            self.objective = minimize(xsum(v[i] for i in range(
                m)) + volume_weight * xsum(width[j] for j in range(floor(Nf/2))))
            self.width = width

        print(f"Discarded {len(discard_indices)} samples.")
        print(f"Discarded {(len(discard_indices)/len(x))} %% of samples.")

        # Save values to self
        self.x = x
        self.f = f
        self.discard_indices = discard_indices
        self.F_list = self.simulator.F_list

        self.v = v
        self.a = a
        self.b = b
        self.z = z
        self.s = s
        self.theta = theta

    def generateMsampleModel(self, m, dt=1, reduce=True):
        """Generate an M-sample self for the AEB system using the MIP optimization.

        Args:
            m (int): Number of samples to generate.
            dt (int, optional): Time step for sample generation. Defaults to 1.
            reduce (bool, optional): Flag indicating whether to reduce samples. Defaults to True.

        Returns:
            Model: The generated M-sample self.

        """
        (x, f) = self.simulator.genSamples(dt=dt, m=m)
        self.build_model(x, f, reduce=reduce)
