import argparse
import datetime
import json
import os
import sys
import warnings
from typing import Optional, List, Dict, Union, Any

import numpy as np
import numpy.typing as npt
from EntropyHub import CoSiEn
from dask.distributed import Client
from ribs.archives import GridArchive
from scipy.stats import pearsonr, spearmanr
from tqdm import trange

from configs import configs, reload_configs, update_configs
from dyn_envs.dyn_lunar_lander import DynamicLunarLanderWrapper
from dyn_envs.dyn_sphere import DynamicSphere

# suppress all warnings (raised by entropy and correlation computations)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(prog='Dynamic Environment Explorer',
                                 description='A tool to explore properties of a dynamic environment.',
                                 epilog='Programmed by Roberto Gallotta (UoM)')
parser.add_argument("--configs", help="The configuration to use.", action='store', default='./configs/configs.yml',
                    type=str)
args = parser.parse_args()


def simulate_and_collect(env, initial_model: npt.NDArray[np.float32],
                         target_variables: List[str],
                         control_variables: Dict[str, List[int]],
                         n_individuals: int = 1000, n_iters: int = 1000,
                         client: Optional[Client] = None) -> List[Dict[str, Union[Dict, npt.NDArray[Any]]]]:
    # get the cartesian product of all possible control variables values
    combinations = [[control_variables[x][y] for x in control_variables.keys()]
                    for y in range(len(control_variables[list(control_variables.keys())[0]]))]
    results = []
    rng = np.random.default_rng(seed=configs.rng_seed)
    # generate n_individuals random individuals and evaluate them
    population = rng.random(size=(n_individuals, np.prod(initial_model.shape)))
    for combination in combinations:
        env.reset()
        objective_population, measures_population = env(population, client)
        # update the environment parameters to the current setting
        setting = {k: v for k, v in zip(control_variables.keys(), combination)}
        for k, v in setting.items():
            env.__dict__[k] = v
        with trange(0, n_iters, file=sys.stdout, desc=f'Collecting data ({setting=})...') as iters:
            # define the archive
            threshold_min = configs.threshold_min if configs.threshold_min is not None else -np.inf
            archive = GridArchive(solution_dim=initial_model.size,
                                  dims=configs.archive_dims,
                                  ranges=env.ranges,
                                  threshold_min=threshold_min,
                                  qd_score_offset=configs.qd_score_offset,
                                  seed=configs.rng_seed)
            # add the population to the archive
            archive.add(solution_batch=population,
                        objective_batch=objective_population,
                        measures_batch=measures_population)
            # get the existing solutions in the archive
            indices = archive._occupied_indices[:archive._num_occupied]
            solutions = archive._solution_arr[indices]
            # let the user know how many solutions are in the archive
            iters.set_description(f'Collecting data ({setting=}) ({len(solutions)} out of {n_individuals})...')
            # create the data tracking arrays
            objectives = np.zeros(shape=(n_iters, len(solutions)))
            measures = np.zeros(shape=(n_iters, len(solutions), 2))
            target_variables_values = [np.zeros(shape=(n_iters, *env.__dict__[x].shape)) for x in target_variables]
            solutions_indices = np.zeros(shape=(n_iters, len(solutions)), dtype=np.int32)

            for itr in iters:
                # simulate
                objective_batch, measures_batch = env(solutions, client)
                # collect data
                objectives[itr] = objective_batch
                measures[itr] = measures_batch
                for i, target_variable in enumerate(target_variables):
                    target_variables_values[i][itr] = env.__dict__[target_variable]
                # collect the (int) indices of the surviving solutions
                archive.clear()
                archive.add(solution_batch=solutions,
                            objective_batch=objective_batch,
                            measures_batch=measures_batch)
                for i in range(solutions.shape[0]):
                    equals = np.all(solutions[i] == archive._solution_arr, axis=1)
                    match_idx = np.where(equals)[0]
                    # missing solutions will have index -1
                    solutions_indices[itr, i] = match_idx[0] if match_idx.size > 0 else -1
                # increase the time in the environment
                env.next_timestep()
            # update results with current setting data
            results.append({
                'control_variables_values': setting,
                'objectives': objectives,
                'measures': measures,
                'solutions_indices': solutions_indices,
                'target_variables_values': target_variables_values
            })

    return results


def compute_correlations(arr1, target_variable_values):
    return {
        'Pearson': pearsonr(arr1, target_variable_values, alternative='two-sided'),
        'Spearman': spearmanr(arr1, target_variable_values, alternative='two-sided')
    }


def compute_entropy(objectives_diffs, client):
    aps = []
    # https://www.mdpi.com/1099-4300/19/12/652
    futures = client.map(lambda matrix: CoSiEn(matrix.flatten()), objectives_diffs)
    results = client.gather(futures)
    for ap, _ in results:
        aps.append(ap)
    return np.mean(aps)


if __name__ == '__main__':
    reload_configs(configs, args.configs)
    update_configs(configs, argparse.Namespace(**{
        'time_shift_val': 1,
        'total_iters': 100
    })
                   )
    fname = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}_environment.analysis'

    with (open(os.path.join(configs.analysis_dir, fname), 'w', encoding='utf-8') as f):

        print(f'Analysis results will be saved in {fname}')

        configs_str = json.dumps(configs.__dict__)
        f.write(f'{configs_str}\n\n')
        f.flush()

        client = Client(n_workers=configs.n_workers, threads_per_worker=configs.threads_per_worker)

        if configs.env == 'sphere':
            env = DynamicSphere()
            initial_model = np.zeros(configs.solution_dim)
            target_variables = ['objective_shift', 'measures_shift']
            control_variables = {
                'objective_shift_strength': [0, configs.objective_shift_strength],
                'measures_shift_strength': [configs.measures_shift_strength, 0]
            }
        elif configs.env == 'lunar-lander':
            env = DynamicLunarLanderWrapper()
            initial_model = np.zeros((env.action_space.n, env.observation_space.shape[0]))
            target_variables = ['wind_power', 'turbulence_power']
            control_variables = {
                'wind_shift_strength': [0, configs.wind_shift_strength],
                'turbulence_shift_strength': [configs.turbulence_shift_strength, 0]
            }
        else:
            raise NotImplementedError(f'Unrecognized environment: {configs.env}.')

        print('Running simulations for RQ1...')

        results = simulate_and_collect(env=env, initial_model=initial_model, control_variables=control_variables,
                                       target_variables=target_variables, client=client, n_iters=configs.total_iters)

        for i in trange(len(results), desc='Analysing results (RQ1)...'):
            result = results[i]
            f.write(f'Setting: {result["control_variables_values"]}\n')

            objectives = result['objectives']
            measures = result['measures']
            solutions_indices = result['solutions_indices']
            target_variables_values = result['target_variables_values']

            for i, target_variable in enumerate(target_variables):
                target_variable_values = np.atleast_2d(target_variables_values[i].squeeze())
                if target_variable_values.shape[0] < target_variable_values.shape[1]:
                    target_variable_values = target_variable_values.reshape(target_variable_values.shape[1], -1)

                for j in range(target_variable_values.shape[1]):
                    # Research Question #1.A
                    f.write('What is the correlation between changes in the environment and objective?\n')
                    correlations = compute_correlations(np.mean(objectives, axis=1), target_variable_values[:, j])
                    for k in correlations.keys():
                        f.write(f'\t=== {k} correlation\n')
                        f.write(
                            f'\t\tObjective wrt {target_variable} (dim={j}): {correlations[k].statistic:4.2f} (p={correlations[k].pvalue:4.2f})\n')

                    # Research Question #1.B
                    f.write('What is the correlation between changes in the environment and BC1?\n')
                    correlations = compute_correlations(np.mean(measures, axis=1)[:, 0], target_variable_values[:, j])
                    for k in correlations.keys():
                        f.write(f'\t=== {k} correlation\n')
                        f.write(
                            f'\t\tBC1 wrt {target_variable} (dim={j}): {correlations[k].statistic:4.2f} (p={correlations[k].pvalue:4.2f})\n')

                    # Research Question #1.C
                    f.write('What is the correlation between changes in the environment and BC2?\n')
                    correlations = compute_correlations(np.mean(measures, axis=1)[:, 1], target_variable_values[:, j])
                    for k in correlations.keys():
                        f.write(f'\t=== {k} correlation\n')
                        f.write(
                            f'\t\tBC2 wrt {target_variable} (dim={j}): {correlations[k].statistic:4.2f} (p={correlations[k].pvalue:4.2f})\n')

            f.flush()

        print('Computing entropies and standard deviations for RQ2...')

        objectives = results[-1]['objectives']
        measures = results[-1]['measures']
        solutions_indices = results[-1]['solutions_indices']
        target_variables_values = results[-1]['target_variables_values']

        # Research Question #2
        avg_objectives_diffs = np.zeros(shape=(configs.total_iters - 1, *configs.archive_dims))
        avg_measures_diffs = np.zeros(shape=(configs.total_iters - 1, *configs.archive_dims, 2))
        for i in range(configs.total_iters - 1):
            at = np.where(solutions_indices[i + 1] != -1)[0]
            this_iter_int_indices = solutions_indices[i + 1][at]
            this_iter_grid_indices = np.asarray(np.unravel_index(this_iter_int_indices, configs.archive_dims)).astype(
                np.int32)
            avg_objectives_diffs[i, this_iter_grid_indices[0, :], this_iter_grid_indices[1, :]] = objectives[
                                                                                                      i + 1, at] - \
                                                                                                  objectives[i, at]
            avg_measures_diffs[i, this_iter_grid_indices[0, :], this_iter_grid_indices[1, :], :] = measures[i + 1, at,
                                                                                                   :] - measures[i - 1,
                                                                                                        at,
                                                                                                        :]
        entropy = compute_entropy(avg_objectives_diffs, client)
        std = np.mean(np.std(avg_objectives_diffs.reshape(avg_objectives_diffs.shape[0], -1), axis=1))
        f.write('Do changes in the environment lead to local or global changes in the objective?\n')
        f.write(f'\tAverage Cosine Similarity Entropy: {entropy:4.2f}\n\t\tAverage Standard Deviation: {std:4.2f}\n')
        entropy = compute_entropy(avg_measures_diffs[:, :, :, 0], client)
        std = np.mean(
            np.std(avg_measures_diffs[:, :, :, 0].reshape(avg_measures_diffs[:, :, :, 0].shape[0], -1), axis=1))
        f.write('Do changes in the environment lead to local or global changes in the BC1?\n')
        f.write(f'\tAverage Cosine Similarity Entropy: {entropy:4.2f}\n\t\tAverage Standard Deviation: {std:4.2f}\n')
        entropy = compute_entropy(avg_measures_diffs[:, :, :, 1], client)
        std = np.mean(
            np.std(avg_measures_diffs[:, :, :, 1].reshape(avg_measures_diffs[:, :, :, 1].shape[0], -1), axis=1))
        f.write('Do changes in the environment lead to local or global changes in the BC2?\n')
        f.write(f'\tAverage Cosine Similarity Entropy: {entropy:4.2f}\n\t\tAverage Standard Deviation: {std:4.2f}\n')

        f.flush()

        print('Running simulations for RQ3...')

        # Research Question #3
        if configs.env == 'sphere':
            control_variables = {
                'objective_shift_strength': [0, 0, 0, 1, 1, 1, 3, 3, 3],
                'measures_shift_strength': [0, 1, 3, 0, 1, 3, 0, 1, 3]
            }
        elif configs.env == 'lunar-lander':
            control_variables = {
                'wind_shift_strength': [0, 0, 0, 0.05, 0.1, 0.15, 0.2],
                'turbulence_shift_strength': [0, 0.1, 0.2, 0, 0, 0, 0]
            }
        else:
            raise NotImplementedError(f'Unrecognized environment: {configs.env}.')

        results = simulate_and_collect(env=env, initial_model=initial_model, control_variables=control_variables,
                                       target_variables=target_variables, client=client, n_iters=configs.total_iters)

        avg_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        avg_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        for i in trange(len(results), desc='Analysing results (RQ3)...'):
            result = results[i]
            objectives = result['objectives']
            measures = result['measures']
            solutions_indices = result['solutions_indices']
            target_variables_values = result['target_variables_values']

            for j in range(configs.total_iters - 1):
                avg_objectives_diffs[i, j] = np.mean(objectives[j + 1] - objectives[j])
                avg_abs_objectives_diffs[i, j] = np.mean(np.abs(objectives[j + 1] - objectives[j]))
                avg_measures_diffs[i, j] = np.mean(measures[j + 1] - measures[j], axis=0)
                avg_abs_measures_diffs[i, j] = np.mean(np.abs(measures[j + 1] - measures[j]), axis=0)
                std_objectives_diffs[i, j] = np.std(objectives[j + 1] - objectives[j])
                std_abs_objectives_diffs[i, j] = np.std(np.abs(objectives[j + 1] - objectives[j]))
                std_measures_diffs[i, j] = np.std(measures[j + 1] - measures[j], axis=0)
                std_abs_measures_diffs[i, j] = np.std(np.abs(measures[j + 1] - measures[j]), axis=0)
        f.write(f'Do small changes in the environment result in small changes in fitness?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_objectives_diffs[i], axis=0)
            avg_std_diff = np.mean(std_objectives_diffs[i], axis=0)
            abs_avg_diff = np.mean(avg_abs_objectives_diffs[i], axis=0)
            avg_std_abs_diff = np.mean(std_abs_objectives_diffs[i], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do small changes in the environment result in small changes in BC1?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 0], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 0], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 0], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 0], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do small changes in the environment result in small changes in BC2?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 1], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 1], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 1], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 1], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')

        f.flush()

        print('Running simulations for RQ4.A...')

        # Research Question #4
        if configs.env == 'sphere':
            control_variables = {
                'objective_shift_strength': [0, 0, 0, 1, 1, 1, 3, 3, 3],
                'measures_shift_strength': [0, 1, 3, 0, 1, 3, 0, 1, 3]
            }
        elif configs.env == 'lunar-lander':
            control_variables = {
                'wind_shift_strength': [0, 0, 0, 0.05, 0.1, 0.15, 0.2],
                'turbulence_shift_strength': [0, 0.1, 0.2, 0, 0, 0, 0]
            }
        else:
            raise NotImplementedError(f'Unrecognized environment: {configs.env}.')

        env.shift_multiplier = [1]

        results = simulate_and_collect(env=env, initial_model=initial_model, control_variables=control_variables,
                                       target_variables=target_variables, client=client, n_iters=configs.total_iters)

        avg_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        avg_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        for i in trange(len(results), desc='Analysing results (RQ4)...'):
            result = results[i]
            objectives = result['objectives']
            measures = result['measures']
            solutions_indices = result['solutions_indices']
            target_variables_values = result['target_variables_values']

            for j in range(configs.total_iters - 1):
                avg_objectives_diffs[i, j] = np.mean(objectives[j + 1] - objectives[j])
                avg_abs_objectives_diffs[i, j] = np.mean(np.abs(objectives[j + 1] - objectives[j]))
                avg_measures_diffs[i, j] = np.mean(measures[j + 1] - measures[j], axis=0)
                avg_abs_measures_diffs[i, j] = np.mean(np.abs(measures[j + 1] - measures[j]), axis=0)
                std_objectives_diffs[i, j] = np.std(objectives[j + 1] - objectives[j])
                std_abs_objectives_diffs[i, j] = np.std(np.abs(objectives[j + 1] - objectives[j]))
                std_measures_diffs[i, j] = np.std(measures[j + 1] - measures[j], axis=0)
                std_abs_measures_diffs[i, j] = np.std(np.abs(measures[j + 1] - measures[j]), axis=0)
        f.write(
            f'Do constantly positively directed small changes in the environment result in small changes in fitness?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_objectives_diffs[i], axis=0)
            avg_std_diff = np.mean(std_objectives_diffs[i], axis=0)
            abs_avg_diff = np.mean(avg_abs_objectives_diffs[i], axis=0)
            avg_std_abs_diff = np.mean(std_abs_objectives_diffs[i], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do constantly positively directed small changes in the environment result in small changes in BC1?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 0], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 0], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 0], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 0], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do constantly positively directed small changes in the environment result in small changes in BC2?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 1], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 1], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 1], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 1], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')

        f.flush()

        print('Running simulations for RQ4.B...')

        # Research Question #4
        if configs.env == 'sphere':
            control_variables = {
                'objective_shift_strength': [0, 0, 0, 1, 1, 1, 3, 3, 3],
                'measures_shift_strength': [0, 1, 3, 0, 1, 3, 0, 1, 3]
            }
        elif configs.env == 'lunar-lander':
            control_variables = {
                'wind_shift_strength': [0, 0, 0, 0.05, 0.1, 0.15, 0.2],
                'turbulence_shift_strength': [0, 0.1, 0.2, 0, 0, 0, 0]
            }
        else:
            raise NotImplementedError(f'Unrecognized environment: {configs.env}.')

        env.shift_multiplier = [-1]

        results = simulate_and_collect(env=env, initial_model=initial_model, control_variables=control_variables,
                                       target_variables=target_variables, client=client, n_iters=configs.total_iters)

        avg_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        avg_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        avg_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        std_abs_objectives_diffs = np.zeros(shape=(len(results), configs.total_iters))
        std_abs_measures_diffs = np.zeros(shape=(len(results), configs.total_iters, 2))
        for i in trange(len(results), desc='Analysing results (RQ4)...'):
            result = results[i]
            objectives = result['objectives']
            measures = result['measures']
            solutions_indices = result['solutions_indices']
            target_variables_values = result['target_variables_values']

            for j in range(configs.total_iters - 1):
                avg_objectives_diffs[i, j] = np.mean(objectives[j + 1] - objectives[j])
                avg_abs_objectives_diffs[i, j] = np.mean(np.abs(objectives[j + 1] - objectives[j]))
                avg_measures_diffs[i, j] = np.mean(measures[j + 1] - measures[j], axis=0)
                avg_abs_measures_diffs[i, j] = np.mean(np.abs(measures[j + 1] - measures[j]), axis=0)
                std_objectives_diffs[i, j] = np.std(objectives[j + 1] - objectives[j])
                std_abs_objectives_diffs[i, j] = np.std(np.abs(objectives[j + 1] - objectives[j]))
                std_measures_diffs[i, j] = np.std(measures[j + 1] - measures[j], axis=0)
                std_abs_measures_diffs[i, j] = np.std(np.abs(measures[j + 1] - measures[j]), axis=0)
        f.write(
            f'Do constantly negatively directed small changes in the environment result in small changes in fitness?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_objectives_diffs[i], axis=0)
            avg_std_diff = np.mean(std_objectives_diffs[i], axis=0)
            abs_avg_diff = np.mean(avg_abs_objectives_diffs[i], axis=0)
            avg_std_abs_diff = np.mean(std_abs_objectives_diffs[i], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do constantly negatively directed small changes in the environment result in small changes in BC1?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 0], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 0], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 0], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 0], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')
        f.write(f'Do constantly negatively directed small changes in the environment result in small changes in BC2?\n')
        for i, result in enumerate(results):
            avg_diff = np.mean(avg_measures_diffs[i, :, 1], axis=0)
            avg_std_diff = np.mean(std_measures_diffs[i, :, 1], axis=0)
            abs_avg_diff = np.mean(avg_abs_measures_diffs[i, :, 1], axis=0)
            avg_std_abs_diff = np.mean(std_abs_measures_diffs[i, :, 1], axis=0)
            f.write(f'\t{result["control_variables_values"]}: {avg_diff:4.2f}±{avg_std_diff:4.2f}\n')
            f.write(f'\t{result["control_variables_values"]} ABS: {abs_avg_diff:4.2f}±{avg_std_abs_diff:4.2f}\n')

        f.flush()

    client.close()
