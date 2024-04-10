import argparse
import datetime
import importlib
import json
import os
import random
import sys

import decorator
import numpy as np
from dask.distributed import Client
from ribs.archives import GridArchive, ArchiveDataFrame
from ribs.emitters import GaussianEmitter, EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from scipy import ndimage
from sklearn.metrics import auc
from tqdm.auto import trange

from configs import configs, reload_configs, update_configs
from custom_sampling.es_emitter import CustomEvolutionStrategyEmitter
from custom_sampling.gaussian_emitter import CustomGaussianEmitter
from dyn_envs.dyn_lunar_lander import DynamicLunarLanderWrapper
from dyn_envs.dyn_sphere import DynamicSphere
from utils import plot_metric, make_folders, vectorized_beta

# An uninstalled version of decorator is occasionally loaded. This loads the
# newly installed version of decorator so that moviepy works properly -- see
# https://github.com/Zulko/moviepy/issues/1625
importlib.reload(decorator)

all_metrics = {}

parser = argparse.ArgumentParser(prog='Dynamic Quality-Diversity Search',
                                 description='A dynamic Evolutionary Algorithm for Quality-Diversity to handle changes in the fitness landscape',
                                 epilog='Programmed by Roberto Gallotta (UoM)')
parser.add_argument("--configs", help="The configuration to use.", action='store', default='./configs/configs.yml',
                    type=str)
parser.add_argument("--config_args", help="Additional config arguments (valid JSON format)", action='store',
                    default='{}', type=str)

args = parser.parse_args()


def run_experiment(policy: str, client: Client) -> ArchiveDataFrame:
    with trange(0, configs.total_iters, file=sys.stdout, desc=f'Running {policy}...') as iters:
        n_evals = 0

        # setting rng seed for reproducibility
        random.seed(configs.rng_seed)
        np.random.seed(configs.rng_seed)

        rng = np.random.default_rng(configs.rng_seed)

        # evaluation budget is computed based on the no_updates ideal number of evaluations
        evaluations_budget = configs.batch_size * configs.n_emitters * configs.total_iters if configs.early_stopping else 0

        metrics = {
            "Offline Error (Objective)": [],
            "Offline Error (BC 1)": [],
            "Offline Error (BC 2)": [],
            "Offline Error (QD)": [],
            "Survival (%)": [],
            "Evaluations": [],
            'Cumulative Evaluations': [],

            "Mean ± Standard Deviation (MSE Objective)": (0.0, 0.0),
            "Mean ± Standard Deviation (MSE BC 1)": (0.0, 0.0),
            "Mean ± Standard Deviation (MSE BC 2)": (0.0, 0.0),
            "Mean ± Standard Deviation (MSE QD)": (0.0, 0.0),
            "Mean ± Standard Deviation (Coverage Error)": (0.0, 0.0),
            "AUC (MSE Objective)": 0.0,
            "AUC (MSE BC 1)": 0.0,
            "AUC (MSE BC 2)": 0.0,
            "AUC (MSE QD)": 0.0,
            "Iterations": 0,

            "Detected Shift": [],
            "Actual Shift": [],
            "Environment Shift Values": [],
        }

        if configs.env == 'sphere':
            env = DynamicSphere()
            initial_model = np.zeros(configs.solution_dim)
        elif configs.env == 'lunar-lander':
            env = DynamicLunarLanderWrapper()
            initial_model = np.zeros((env.action_space.n, env.observation_space.shape[0]))
        else:
            raise NotImplementedError(f'Unrecognized environment: {configs.env}.')

        threshold_min = configs.threshold_min if configs.threshold_min is not None else -np.inf
        archive = GridArchive(solution_dim=initial_model.size,
                              dims=configs.archive_dims,
                              ranges=env.ranges,
                              threshold_min=threshold_min,
                              qd_score_offset=configs.qd_score_offset,
                              seed=configs.rng_seed)

        solutions_age = np.ones(shape=np.prod(configs.archive_dims), dtype=np.uint32)

        if configs.alg == 'D-MAP-Elites':
            emitter_class = CustomGaussianEmitter if configs.custom_sampling else GaussianEmitter
            emitters = [
                emitter_class(
                    archive=archive,
                    sigma=0.0,
                    x0=initial_model.flatten(),
                    seed=configs.rng_seed,
                    batch_size=configs.batch_size,
                ) for _ in range(configs.n_emitters)
            ]
        elif configs.alg == 'D-CMA-ME':
            emitter_class = CustomEvolutionStrategyEmitter if configs.custom_sampling else EvolutionStrategyEmitter
            emitters = [
                emitter_class(
                    archive=archive,
                    x0=initial_model.flatten(),
                    sigma0=1.0,  # Initial step size.
                    ranker="2imp",
                    batch_size=configs.batch_size,
                ) for _ in range(configs.n_emitters)
            ]
        else:
            raise NotImplementedError(f'Unrecognized algorithm: {configs.alg}.')

        scheduler = Scheduler(archive, emitters)

        for _ in iters:
            # Update the age of solutions
            solutions_age[archive._occupied_arr] += 1

            current_indices_bool = archive._occupied_arr
            current_indices = np.where(archive._occupied_arr)[0]
            current_solutions = archive._solution_arr[archive._occupied_arr]

            # OFFSPRING GENERATION AND EVALUATION STEP
            # get new solutions
            solution_batch = scheduler.ask()

            parent_int_indices = np.zeros((0,))
            if configs.alg == 'D-MAP-Elites':
                if configs.custom_sampling and archive.stats.num_elites >= len(scheduler.emitters):
                    parent_int_indices = np.stack([emitter.parents_indices for emitter in scheduler.emitters]).flatten()
                # add noise to solution_batch to get new solutions
                noise = emitters[0]._rng.normal(
                    scale=configs.sigma,
                    size=(configs.n_emitters * configs.batch_size, archive.solution_dim)
                ).astype(archive.dtype)
                solution_batch = np.clip(solution_batch + noise, emitters[0].lower_bounds, emitters[0].upper_bounds)
                # update the scheduler `_solution_batch` so it saves the correct solutions to the archive
                scheduler._solution_batch = solution_batch

            # evaluate new solutions
            objective_batch, measure_batch = env(solution_batch, client)
            n_evals += len(solution_batch)

            # DETECTION STEP
            needs_update = False
            if policy != 'no_updates':
                if configs.detection_on == 'replacees':
                    # always recompute the replacees solutions since we don't know when a shift may occur
                    # get the indices of occupied cells that would be compared with the new solutions
                    filtered_int_indices = np.intersect1d(archive.index_of(measure_batch),
                                                          archive._occupied_indices[:archive._num_occupied])
                elif configs.detection_on == 'oldest_elites':
                    # keep track of which solutions actually exist in the archive
                    existing_mask = np.zeros_like(solutions_age)
                    existing_mask[archive._occupied_arr] = 1
                    # compute the probabilities for all existing solutions based on their age using a beta distribution
                    # probs = np.stack([vectorized_beta(solutions_age,
                    #                                   np.ones(shape=solutions_age.shape, dtype=np.float32),
                    #                                   rng) * existing_mask for _ in range(configs.batch_size)])
                    probs = np.stack([rng.exponential(scale=solutions_age) * existing_mask for _ in range(configs.batch_size * configs.n_emitters)])
                    # get the indices of occupied cells that are also probably old (ie: not reevaluated recently)
                    # NOTE: probs = 0 when existing_mask is all 0s (ie: on empty archive)
                    filtered_int_indices = np.argmax(probs, axis=1) if np.any(probs > 0) else np.empty((0,))
                elif configs.detection_on == 'random_elites':
                    if archive._num_occupied > 0:
                        filtered_int_indices = archive._occupied_indices[rng.integers(low=0, high=archive._num_occupied,
                                                                                      size=configs.batch_size * configs.n_emitters,
                                                                                      endpoint=True)]
                    else:
                        filtered_int_indices = np.empty((0,))
                else:
                    raise NotImplementedError(f'Unrecognized detection configuration: {configs.detection_on}!')

                if filtered_int_indices.size:
                    detecting_solutions = archive._solution_arr[filtered_int_indices]
                    target_objectives, target_measures = env(detecting_solutions, client)
                    # update number of evaluations for this iteration
                    n_evals += len(detecting_solutions)
                    existing_objectives = archive._objective_arr[filtered_int_indices]
                    existing_measures = archive._measures_arr[filtered_int_indices]
                    # check for changes in objective
                    needs_update = not np.isclose(existing_objectives, target_objectives, configs.atol,
                                                  configs.rtol).all()
                    # check for changes in measures
                    needs_update |= not np.isclose(existing_measures, target_measures, configs.atol, configs.rtol).all()

            if needs_update:
                if policy == 'no_updates':
                    raise AssertionError('Environment shift detected but policy is no_updates.')
                elif policy == 'update_all':
                    reeval_indices = archive._occupied_arr
                    all_replacees_solutions = archive._solution_arr[reeval_indices]
                    all_replacees_objectives, all_replacees_measures = env(all_replacees_solutions, client)
                    n_evals += len(all_replacees_solutions)
                elif policy.startswith('update_local'):
                    # if re-evaluate strategy is different from detection strategy, expand the detecting_solutions
                    if configs.reevaluate_on == 'replacees' and configs.detection_on != configs.reevaluate_on:
                        replacees_int_indices = np.intersect1d(archive.index_of(measure_batch),
                                                               archive._occupied_indices[:archive._num_occupied])
                        replacees_int_indices = np.setdiff1d(replacees_int_indices, filtered_int_indices, assume_unique=True)
                        if replacees_int_indices.size:
                            replacees_solutions = archive._solution_arr[replacees_int_indices]
                            replacees_objectives, replacees_measures = env(replacees_solutions, client)
                            # update number of evaluations for this iteration
                            n_evals += len(replacees_solutions)
                            # add replacees_* to the current detection batch
                            detecting_solutions = np.vstack([detecting_solutions, replacees_solutions])
                            filtered_int_indices = np.hstack([filtered_int_indices, replacees_int_indices])
                            target_objectives = np.hstack([target_objectives, replacees_objectives])
                            target_measures = np.vstack([target_measures, replacees_measures])
                            existing_objectives = np.hstack([existing_objectives, archive._objective_arr[replacees_int_indices]])
                            existing_measures = np.vstack([existing_measures, archive._measures_arr[replacees_int_indices]])
                    elif configs.reevaluate_on == 'oldest_elites' and configs.detection_on != configs.reevaluate_on:
                        existing_mask = np.zeros_like(solutions_age)
                        existing_mask[archive._occupied_arr] = 1
                        # probs = np.stack([vectorized_beta(solutions_age,
                        #                                   np.ones(shape=solutions_age.shape, dtype=np.float32),
                        #                                   rng) * existing_mask for _ in range(configs.batch_size)])
                        probs = np.stack([rng.exponential(scale=solutions_age) * existing_mask for _ in range(configs.batch_size * configs.n_emitters)])
                        oldest_int_indices = np.argmax(probs, axis=1) if np.any(probs > 0) else np.empty((0,))
                        oldest_int_indices = np.setdiff1d(oldest_int_indices, filtered_int_indices, assume_unique=True)
                        oldest_solutions = archive._solution_arr[filtered_int_indices]
                        oldest_objectives, oldest_measures = env(oldest_solutions, client)
                        # update number of evaluations for this iteration
                        n_evals += len(oldest_solutions)
                        # add oldest_* to the current detection batch
                        detecting_solutions = np.vstack([detecting_solutions, oldest_solutions])
                        filtered_int_indices = np.hstack([filtered_int_indices, oldest_int_indices])
                        target_objectives = np.hstack([target_objectives, oldest_objectives])
                        target_measures = np.vstack([target_measures, oldest_measures])
                        existing_objectives = np.hstack([existing_objectives, archive._objective_arr[oldest_int_indices]])
                        existing_measures = np.vstack([existing_measures, archive._measures_arr[oldest_int_indices]])
                    elif configs.reevaluate_on != 'oldest_elites' and configs.reevaluate_on != 'replacees':
                        raise NotImplementedError(f'Unrecognized/Invalid reevaluation configuration: {configs.reevaluate_on}!')
                    # get indices of solutions that have changed
                    indices = np.logical_or(
                        ~np.isclose(existing_objectives, target_objectives, configs.atol, configs.rtol),
                        ~np.isclose(existing_measures, target_measures, configs.atol, configs.rtol).any(axis=1)
                        # measures are 2D arrays
                    ) if (
                            existing_objectives.size and target_objectives.size and existing_measures.size and target_measures.size) else np.zeros(
                        (0,), dtype=bool)

                    all_replacees_solutions = detecting_solutions[indices]
                    all_replacees_objectives = target_objectives[indices]
                    all_replacees_measures = target_measures[indices]
                    all_replacees_indices = filtered_int_indices[indices]

                    # get the grid indices of existing solutions
                    grid_indices_existing = archive.int_to_grid_index(archive._occupied_indices[:archive._num_occupied])

                    cascading_iter = configs.cascade_max_depth
                    cascading_measures = target_measures[indices]
                    while cascading_iter > 0 and len(cascading_measures) > 0:
                        # get the cell indices of changed solutions
                        grid_indices_changed = archive.int_to_grid_index(archive.index_of(cascading_measures))
                        # make a copy of the archive array
                        archive_arr = np.zeros(archive.dims, dtype=np.uint8)
                        # mark occupied cells of changed solutions
                        archive_arr[grid_indices_changed[:, 0], grid_indices_changed[:, 1]] = 1
                        # mark cells in a neighborhood of the given radius
                        radius = int(policy.split('update_local_')[1])
                        for _ in range(radius):
                            archive_arr = ndimage.binary_dilation(archive_arr).astype(archive_arr.dtype)
                        # make another copy for occupied cells
                        archive_arr_2 = np.ones(archive.dims, dtype=np.uint8)
                        # set the occupied cells to 0
                        archive_arr_2[grid_indices_existing[:, 0], grid_indices_existing[:, 1]] = 0
                        # occupied cells that have changed + neighborhood -> 1
                        # occupied cells that have not changed -> 0
                        # unoccupied cells -> -1
                        # (can't have an unoccupied cell that has changed)
                        archive_arr = archive_arr - archive_arr_2
                        reeval_indices = np.where(archive_arr.flatten() == 1)[0]
                        # remove already reevaluated indices from reeval_indices
                        reeval_indices = np.setdiff1d(reeval_indices, all_replacees_indices)
                        if len(reeval_indices) > 0:
                            # get solutions at reeval_indices
                            reeval_solutions = archive._solution_arr[reeval_indices]
                            reeval_objectives, reeval_measures = env(reeval_solutions, client)
                            n_evals += len(reeval_solutions)
                            # update all_replacees_*
                            all_replacees_solutions = np.vstack([all_replacees_solutions, reeval_solutions])
                            all_replacees_objectives = np.hstack([all_replacees_objectives, reeval_objectives])
                            all_replacees_measures = np.vstack([all_replacees_measures, reeval_measures])
                            all_replacees_indices = np.hstack([all_replacees_indices, reeval_indices])
                            # update cascading variables
                            cascading_iter -= 1
                            new_reeval_indices = np.setdiff1d(archive.index_of(reeval_measures), all_replacees_indices)
                            cascading_measures = archive._measures_arr[
                                np.intersect1d(new_reeval_indices, archive._occupied_indices[:archive._num_occupied])]
                        else:
                            cascading_measures = np.zeros((0,))

                    reeval_indices = np.zeros((archive.cells,), dtype=bool)
                    reeval_indices[all_replacees_indices] = True
                else:
                    raise NotImplementedError(f'Unrecognized policy: {configs.policy}.')

                # update the archive if we have at least one new solution
                if np.any(reeval_indices):
                    # get the indices of the solutions we're not updating
                    not_updating_indices = np.logical_and(archive._occupied_arr, ~reeval_indices)
                    # get the solutions, objectives, and measures that we're not updating
                    not_updating_solutions = archive._solution_arr[not_updating_indices]
                    not_updating_objectives = archive._objective_arr[not_updating_indices]
                    not_updating_measures = archive._measures_arr[not_updating_indices]

                    # reset the archive
                    archive.clear()
                    # add unchanged solutions
                    archive.add(not_updating_solutions, not_updating_objectives, not_updating_measures)
                    # add re-evaluated solutions
                    archive.add(all_replacees_solutions, all_replacees_objectives, all_replacees_measures)
                    # Reset age of re-evaluated solutions
                    # NOTE: We don't care here if the re-evaluated solutions have moved
                    solutions_age[reeval_indices] = 1
            else:
                reeval_indices = np.zeros((0,), dtype=archive.dtype)

            # attempt to add the new solutions to the archive
            scheduler.tell(objective_batch, measure_batch)

            # Reset age of replaced/moved solutions
            new_indices_bool = archive._occupied_arr
            new_indices = archive._occupied_indices[:archive._num_occupied]
            new_solutions = archive._solution_arr[archive._occupied_arr]
            if current_indices.size:
                # new_solutions_at_bool is true in cells where the solution is different from previously and currently
                new_solutions_at_bool = np.zeros_like(archive._occupied_arr)
                shared_indices, current_shared_indices_idx, new_shared_indices_idx = np.intersect1d(current_indices,
                                                                                                    new_indices,
                                                                                                    assume_unique=True,
                                                                                                    return_indices=True)
                new_solutions_at_bool[shared_indices] = [~np.array_equal(current_solutions[i1], new_solutions[i2])
                                                         for i1, i2 in
                                                         zip(current_shared_indices_idx, new_shared_indices_idx)]
            else:
                new_solutions_at_bool = archive._occupied_arr
            # indices of cell solutions that were previously empty/occupied and are now occupied/empty
            new_indices_bool_masked = np.logical_xor(new_indices_bool, current_indices_bool)
            resetting_indices = np.logical_or(new_solutions_at_bool, new_indices_bool_masked)
            solutions_age[resetting_indices] = 1

            # if possible, update the emitters
            if configs.custom_sampling:
                for emitter in emitters:
                    emitter.update_emitter(reeval_indices, parent_int_indices, needs_update)

            # COMPUTE METRICS STEP
            # get the current archive of solutions
            current_solutions = archive._solution_arr[archive._occupied_arr]

            # create the ideal (up-to-date) archive for metrics computation
            ideal_archive = GridArchive(solution_dim=initial_model.size,
                                        dims=configs.archive_dims,
                                        ranges=env.ranges,
                                        threshold_min=threshold_min,
                                        qd_score_offset=configs.qd_score_offset,
                                        seed=configs.rng_seed)
            # compute the correct values for both objectives and measures
            # NOTE: we don't consider this re-evaluation in our metric since it's just for the offline error
            updated_objectives, updated_measures = env(current_solutions, client)
            ideal_archive.add(solution_batch=current_solutions,
                              objective_batch=updated_objectives,
                              measures_batch=updated_measures)

            # get the solutions shared by both archives
            # NOTE: The ideal archive contains all shared solutions already
            shared_solutions = ideal_archive._solution_arr[ideal_archive._occupied_arr]
            ideal_objs = ideal_archive._objective_arr[ideal_archive._occupied_arr]
            ideal_measures = ideal_archive._measures_arr[ideal_archive._occupied_arr]
            # get the shared solutions objectives and measures in the current archive
            surviving_solutions_bool = np.all(np.equal(current_solutions, shared_solutions[:, None, :]), axis=2).any(axis=0)
            current_objs = archive._objective_arr[archive._occupied_arr][surviving_solutions_bool]
            current_measures = archive._measures_arr[archive._occupied_arr][surviving_solutions_bool]
            # compute MSE on objective and measures
            obj_mse = np.mean(np.sqrt(np.square(current_objs - ideal_objs)))
            meas_mse = np.mean(np.sqrt(np.square(current_measures - ideal_measures)), axis=0)
            current_qd_score = np.sum(current_objs + configs.qd_score_offset)
            ideaL_qd_score = np.sum(ideal_objs + configs.qd_score_offset)
            # update the metrics
            metrics["Offline Error (Objective)"].append(obj_mse)
            metrics["Offline Error (BC 1)"].append(meas_mse[0])
            metrics["Offline Error (BC 2)"].append(meas_mse[1])
            metrics["Offline Error (QD)"].append(np.abs(current_qd_score - ideaL_qd_score))
            # metrics["Offline Error (Coverage)"].append(coverage_err)
            metrics["Evaluations"].append(n_evals)
            metrics["Survival (%)"].append((np.sum(surviving_solutions_bool) / archive.stats.num_elites) * 100)
            metrics["Iterations"] += 1

            metrics["Detected Shift"].append(1 if needs_update else 0)
            metrics["Actual Shift"].append(1 if env.just_shifted else 0)
            metrics["Environment Shift Values"].append(env.tracked_shifts)

            # update the tqdm bar postfix
            iters.set_postfix(
                ordered_dict={"MSE Obj": f"{metrics['Offline Error (Objective)'][-1]:6.3f}",
                              "MSE BC1": f"{metrics['Offline Error (BC 1)'][-1]:6.3f}",
                              "MSE BC2": f"{metrics['Offline Error (BC 2)'][-1]:6.3f}",
                              # "Err Coverage": f"{metrics['Offline Error (Coverage)'][-1]:6.3f}",
                              "Survival (%)": f"{metrics['Survival (%)'][-1]:6.3f}",
                              "Evals": f"{np.sum(metrics['Evaluations'], dtype=int)})"
                              })
            # increase the time in the environment
            env.next_timestep()

            n_evals = 0

            if configs.early_stopping and np.sum(metrics['Evaluations']) >= evaluations_budget:
                break

    metrics["Mean ± Standard Deviation (MSE Objective)"] = (np.mean(metrics["Offline Error (Objective)"]),
                                                            np.std(metrics["Offline Error (Objective)"]))
    metrics["Mean ± Standard Deviation (MSE BC 1)"] = (np.mean(metrics["Offline Error (BC 1)"]),
                                                       np.std(metrics["Offline Error (BC 1)"]))
    metrics["Mean ± Standard Deviation (MSE BC 2)"] = (np.mean(metrics["Offline Error (BC 2)"]),
                                                       np.std(metrics["Offline Error (BC 2)"]))
    metrics["Mean ± Standard Deviation (MSE QD)"] = (np.mean(metrics["Offline Error (QD)"]),
                                                     np.std(metrics["Offline Error (QD)"]))
    metrics["AUC (MSE Objective)"] = auc(np.arange(len(metrics["Offline Error (Objective)"])),
                                         metrics["Offline Error (Objective)"])
    metrics["AUC (MSE BC 1)"] = auc(np.arange(len(metrics["Offline Error (BC 1)"])),
                                    metrics["Offline Error (BC 1)"])
    metrics["AUC (MSE BC 2)"] = auc(np.arange(len(metrics["Offline Error (BC 2)"])),
                                    metrics["Offline Error (BC 2)"])
    metrics["AUC (MSE QD)"] = auc(np.arange(len(metrics["Offline Error (QD)"])),
                                  metrics["Offline Error (QD)"])
    metrics["Cumulative Evaluations"] = list(np.cumsum(metrics["Evaluations"], dtype=float))  # Can't encode int in JSON
    # add the experiment metrics to all metrics
    all_metrics[policy] = metrics
    # save the archives history
    return archive.as_pandas()


if __name__ == '__main__':
    reload_configs(configs, args.configs)
    update_configs(configs, json.loads(args.config_args))

    experiment_seed = configs.rng_seed

    experiment_name = f'{configs.alg} {configs.sampling_strategy if hasattr(configs, "sampling_strategy") else "custom" if configs.custom_sampling else "default"} {configs.detection_on} {configs.reevaluate_on}'

    client = Client(n_workers=configs.n_workers, threads_per_worker=configs.threads_per_worker, memory_limit='4GB')
    client.amm.start()

    for n_run in range(configs.n_runs):
        zf_run = f'{str(n_run).zfill(int(np.log10(configs.n_runs)))}'
        print(f'=== Run {n_run + 1}/{configs.n_runs} for {experiment_name}')
        update_configs(configs, {'rng_seed': experiment_seed + n_run})
        experiment_fname = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + f'_{zf_run}_' + experiment_name.replace(' ', '_')
        make_folders(experiment_fname)
        experiment_path = os.path.join(configs.plots_dir, experiment_fname)

        with open(os.path.join(experiment_path, f'settings.metadata'), 'w') as f:
            json.dump(configs.__dict__, f)

        for policy in configs.policies:
            archive = run_experiment(policy=policy, client=client)
            with open(os.path.join(experiment_path, f'{policy}_archive.csv'), 'w') as f:
                archive.to_csv(f, index=False)

        with open(os.path.join(experiment_path, f'results.json'), 'w') as f:
            json.dump(all_metrics, f)

        if configs.make_plots:
            plot_metric(metric='Offline Error (Objective)',
                        title=f'Offline Error (Objective) for {configs.alg} ({configs.env})',
                        ylabel='MSE',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'mse_obj_plot'),
                        metrics=all_metrics)
            plot_metric(metric='Offline Error (BC 1)',
                        title=f'Offline Error (BC 1) for {configs.alg} ({configs.env})',
                        ylabel='MSE',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'mse_bc1_plot'),
                        metrics=all_metrics)
            plot_metric(metric='Offline Error (BC 2)',
                        title=f'Offline Error (BC 2) for {configs.alg} ({configs.env})',
                        ylabel='MSE',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'mse_bc2_plot'),
                        metrics=all_metrics)
            plot_metric(metric='Offline Error (QD)',
                        title=f'Offline Error (QD) for {configs.alg} ({configs.env})',
                        ylabel='QD',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'qd_err_plot'),
                        metrics=all_metrics)
            plot_metric(metric='Survival (%)',
                        title=f'Survival (%) for {configs.alg} ({configs.env})',
                        ylabel='% of elites survived',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'survival_plot'),
                        metrics=all_metrics)
            plot_metric(metric='Cumulative Evaluations',
                        title=f'(Cumulative) Number of Evaluations for {configs.alg} ({configs.env})',
                        ylabel='#evaluations',
                        policies=configs.policies,
                        fname=os.path.join(experiment_path, 'evaluations_plot'),
                        metrics=all_metrics)

    client.close()
