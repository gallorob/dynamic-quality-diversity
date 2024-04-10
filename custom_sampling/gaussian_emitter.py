import numpy as np
from ribs._utils import readonly
from ribs.emitters import GaussianEmitter

from configs import configs
from utils import vectorized_beta


class CustomGaussianEmitter(GaussianEmitter):
    def __init__(self,
                 archive,
                 *,
                 sigma,
                 x0=None,
                 initial_solutions=None,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        GaussianEmitter.__init__(self,
                                 archive=archive,
                                 sigma=sigma,
                                 x0=x0,
                                 initial_solutions=initial_solutions,
                                 bounds=bounds,
                                 batch_size=batch_size,
                                 seed=seed)
        self.parents_indices = None

        self.beta_dists = np.ones((archive.cells, 2), dtype=np.float64)
        self.eps = 1e-5

    def update_emitter(self,
                       reevaluated_indices,
                       parents_indices,
                       has_reset):
        # NOTE: Beta distribution -> beta < alpha = higher sample probability
        if not has_reset:
            valid_indices = self.archive._occupied_indices[:self.archive._num_occupied]
            self.beta_dists[valid_indices, :] += 1

            # convert reevaluated_indices to int
            reevaluated_indices = np.where(reevaluated_indices)[0]

            if len(reevaluated_indices) > 0:
                assert np.isin(reevaluated_indices, valid_indices).all(), 'Re-evaluated indices not in valid indices; should be reset'
            if len(parents_indices) > 0:
                assert np.isin(parents_indices, valid_indices).all(), 'Parent indices are not among valid indices; should be reset'

            if configs.sampling_strategy == 'replacees_only':
                if len(reevaluated_indices) > 0: self.beta_dists[reevaluated_indices, 1] -= 1
            elif configs.sampling_strategy == 'parents_only':
                if len(parents_indices) > 0: self.beta_dists[parents_indices, 1] -= 1
            elif configs.sampling_strategy == 'replacees_and_parents':
                combined_indices = np.hstack([reevaluated_indices, parents_indices])
                if len(combined_indices) > 0: self.beta_dists[combined_indices, 1] -= 1
            elif configs.sampling_strategy == 'not_replacees_only':
                filtered_indices = np.setdiff1d(valid_indices, reevaluated_indices, assume_unique=True)
                if len(filtered_indices) > 0: self.beta_dists[filtered_indices, 1] -= 1
            elif configs.sampling_strategy == 'not_parents_only':
                filtered_indices = np.setdiff1d(valid_indices, parents_indices, assume_unique=True)
                if len(filtered_indices) > 0: self.beta_dists[filtered_indices, 1] -= 1
            elif configs.sampling_strategy == 'not_replacees_and_parents':
                filtered_indices = np.setdiff1d(valid_indices, reevaluated_indices, assume_unique=True)
                filtered_indices = np.setdiff1d(filtered_indices, parents_indices, assume_unique=True)
                if len(filtered_indices) > 0: self.beta_dists[filtered_indices, 1] -= 1
            else:
                raise NotImplementedError(f'Unknown sampling strategy {configs.sampling_strategy}!')
        else:
            self.beta_dists[:, :] = 1

    def ask(self):
        if self.archive.empty:
            if self._initial_solutions is not None:
                self.parents_indices = None
                return np.clip(self._initial_solutions, self.lower_bounds,
                               self.upper_bounds)
            self.parents_indices = None
            parents = np.expand_dims(self.x0, axis=0)
        else:
            valid_indices = self.archive._occupied_indices[:self.archive._num_occupied]
            # probs = np.stack([vectorized_beta(self.beta_dists[valid_indices, 0],
            #                                   self.beta_dists[valid_indices, 1],
            #                                   self._rng) for _ in range(self.batch_size)])
            probs = np.stack([self._rng.exponential(scale=self.beta_dists[valid_indices, 1]) for _ in range(self.batch_size)])
            self.parents_indices = valid_indices[np.argmax(probs, axis=1)]
            parents = readonly(self.archive._solution_arr[self.parents_indices])

        noise = self._rng.normal(
            scale=self._sigma,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        return np.clip(parents + noise, self.lower_bounds, self.upper_bounds)
