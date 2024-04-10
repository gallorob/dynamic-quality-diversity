from typing import Optional

import numpy as np
import numpy.typing as npt
from ribs.emitters import EvolutionStrategyEmitter
from ribs.emitters.opt import CMAEvolutionStrategy


class CustomEvolutionStrategyEmitter(EvolutionStrategyEmitter):
    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 ranker="2imp",
                 es="cma_es",
                 es_kwargs=None,
                 selection_rule="filter",
                 restart_rule="no_improvement",
                 bounds=None,
                 batch_size=None,
                 seed=None,
                 ):
        EvolutionStrategyEmitter.__init__(self,
                                          archive=archive,
                                          x0=x0,
                                          sigma0=sigma0,
                                          ranker=ranker,
                                          es=es,
                                          es_kwargs=es_kwargs,
                                          selection_rule=selection_rule,
                                          restart_rule=restart_rule,
                                          bounds=bounds,
                                          batch_size=batch_size,
                                          seed=seed)
        self.n_outdated_sols = batch_size #// 2
        self.scaling = 0.5

    def update_emitter(self,
                       reevaluated: npt.NDArray[np.float32],
                       parents: Optional[npt.NDArray[np.float32]] = None,
                       has_reset: bool = False) -> None:
        assert isinstance(self._opt, CMAEvolutionStrategy), f'Currently only CMA-ES is supported!'

        valid_indices = self.archive._occupied_indices[:self.archive._num_occupied]
        non_updated_indices = valid_indices[np.isin(valid_indices, reevaluated, assume_unique=True, invert=True)]

        if len(reevaluated) > 0 and len(non_updated_indices) > 0 and self._opt.current_eval > 0:
            non_updated_sampled_indices = self._rng.integers(0, len(non_updated_indices), self.n_outdated_sols)
            non_updated_sampled_solutions = self.archive._solution_arr[non_updated_indices[non_updated_sampled_indices]]

            self._opt.current_eval += self.n_outdated_sols

            weights, mueff, cc, cs, c1, cmu = self._opt._calc_strat_params(self.n_outdated_sols)

            # scaling
            weights *= self.scaling
            mueff *= self.scaling

            damps = (1 + 2 * max(
                0,
                np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
            ) + cs)

            # Recombination of the new mean.
            old_mean = self._opt.mean
            self._opt.mean = np.sum(non_updated_sampled_solutions * np.expand_dims(weights, axis=1), axis=0)

            # Update the evolution path.
            y = self._opt.mean - old_mean
            z = np.matmul(self._opt.cov.invsqrt, y)
            self._opt.ps = ((1 - cs) * self._opt.ps +
                            (np.sqrt(cs * (2 - cs) * mueff) / self._opt.sigma) * z)
            left = (np.sum(np.square(self._opt.ps)) / self._opt.solution_dim /
                    (1 - (1 - cs) ** (2 * self._opt.current_eval / self._opt.batch_size)))
            right = 2 + 4. / (self._opt.solution_dim + 1)
            hsig = 1 if left < right else 0

            self._opt.pc = ((1 - cc) * self._opt.pc + hsig * np.sqrt(cc *
                                                                     (2 - cc) * mueff) * y)

            # Adapt the covariance matrix.
            ys = non_updated_sampled_solutions - np.expand_dims(old_mean, axis=0)
            weighted_ys = ys * np.expand_dims(weights, axis=1)
            # Equivalent to calculating the outer product of each ys[i] with itself
            # and taking a weighted sum of the outer products.
            rank_mu_update = np.einsum("ki,kj", weighted_ys, ys)
            c1a = c1 * (1 - (1 - hsig ** 2) * cc * (2 - cc))
            self._opt.cov.cov = self._opt._calc_cov_update(self._opt.cov.cov, c1a, cmu, c1,
                                                           self._opt.pc, self._opt.sigma,
                                                           rank_mu_update, weights)

            # Update sigma.
            cn, sum_square_ps = cs / damps, np.sum(np.square(self._opt.ps))
            self._opt.sigma *= np.exp(
                min(1,
                    cn * (sum_square_ps / self.solution_dim - 1) / 2))
