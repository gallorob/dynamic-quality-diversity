from typing import Tuple

import numpy as np
import numpy.typing as npt

from configs import configs


class DynamicSphere:
    def __init__(self):
        self.max_bound = 100 / 2 * 5.12

        self.ranges = np.asarray([(-self.max_bound, self.max_bound), (-self.max_bound, self.max_bound)])

        self.t = 0
        self.just_shifted = False
        self.rng = np.random.default_rng(configs.rng_seed)

        self.shift_multiplier = [-1, 1] if not configs.directed else [1]

        self.objective_shift = np.zeros((1,))
        self.measures_shift = np.zeros((1, 2))

        self.objective_shift_strength = np.asarray(configs.objective_shift_strength)
        self.measures_shift_strength = np.asarray(configs.measures_shift_strength)

        # center and range for local shifts (only solutions with measure within the center +- range undergo shift)
        if configs.influence_ranges is None:
            # NOTE: `*2` ensures all solutions within the domain are placed within the influence range
            self.influence_ranges = np.asarray(
                [(-self.max_bound, self.max_bound), (-self.max_bound, self.max_bound)]) * 2
        else:
            self.influence_ranges = np.asarray(configs.influence_ranges)
        self.influence_center = self.ranges[:, 0] + self.rng.random(size=(2,)) * (self.ranges[:, 1] - self.ranges[:, 0])

    @property
    def tracked_shifts(self):
        return self.objective_shift[0], self.measures_shift[0, 0], self.measures_shift[0, 1]

    def next_timestep(self):
        self.t += 1
        self.just_shifted = False
        if self.t % configs.time_shift_val == 0 and self.rng.random(1) <= configs.objective_shift_prob:
            self.objective_shift += self.rng.choice(self.shift_multiplier) * self.rng.uniform(low=0,
                                                                                               high=self.objective_shift_strength,
                                                                                               size=self.objective_shift.shape)
            self.just_shifted = True
        if self.t % configs.time_shift_val == 0 and self.rng.random(1) <= configs.measures_shift_prob:
            self.measures_shift += self.rng.choice(self.shift_multiplier) * self.rng.uniform(low=0,
                                                                                              high=self.measures_shift_strength,
                                                                                              size=self.measures_shift.shape)
            self.just_shifted = True
        if self.just_shifted:
            self.influence_center = self.ranges[:, 0] + self.rng.random(size=(2,)) * (self.ranges[:, 1] - self.ranges[:, 0])

    def reset(self):
        self.t = 0
        self.just_shifted = False
        self.rng = np.random.default_rng(configs.rng_seed)
        self.objective_shift = np.zeros((1,))
        self.measures_shift = np.zeros((1, 2))

    def __call__(self, solution_batch: npt.NDArray[np.float32], client) -> Tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Sphere function evaluation and measures for a batch of solutions.

        Args:
            solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
        Returns:
            objective_batch (np.ndarray): (batch_size,) batch of objectives.
            measures_batch (np.ndarray): (batch_size, 2) batch of measures.
        """
        dim = solution_batch.shape[1]

        # Shift the Sphere function so that the optimal value is at x_i = 2.048.
        # sphere_shift = 5.12 * 0.4
        sphere_shift = (5.12 * 0.4) + self.objective_shift
        sphere_shift = np.clip(sphere_shift, 0, 5.12)

        # Compute raw objectives
        best_obj = 0.0
        worst_obj = (-5.12 - sphere_shift) ** 2 * dim
        raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)

        # Calculate measures.
        clipped = solution_batch.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]
        measures_batch = np.concatenate(
            (
                np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
                np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
            ),
            axis=1,
        )

        in_influence_range = np.where(
            np.logical_and(self.influence_center + self.influence_ranges[:, 0] <= measures_batch,
                           measures_batch <= self.influence_center + self.influence_ranges[:, 1]).all(axis=1))

        # raw_obj[in_influence_range] += self.objective_shift
        measures_batch[in_influence_range] += self.measures_shift

        # Normalize the objective to the range [0, 100] where 100 is optimal.
        objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100
        # objective_batch = raw_obj

        # clipping measures within min and max ranges to avoid NaN values when indexing in the archive
        measures_batch[:, 0] = np.clip(measures_batch[:, 0], self.ranges[0, 0], self.ranges[0, 1])
        measures_batch[:, 1] = np.clip(measures_batch[:, 1], self.ranges[1, 0], self.ranges[1, 1])

        return objective_batch, measures_batch
