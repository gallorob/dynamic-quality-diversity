import json
import os

import numpy as np
import pandas as pd
from tqdm.auto import trange

from configs import configs

column_to_metric = {
    'Environment': 'env', 'Algorithm': 'alg', 'Policy': 'policy', 'Sampling': 'custom_sampling',
    'Detection': 'detection_on', 'Re-Evaluation': 'reevaluate_on',

    "Mean (MSE Objective)": "Mean ± Standard Deviation (MSE Objective)",
    "Standard Deviation (MSE Objective)": "Mean ± Standard Deviation (MSE Objective)",
    "Mean (MSE BC 1)": "Mean ± Standard Deviation (MSE BC 1)",
    "Standard Deviation (MSE BC 1)": "Mean ± Standard Deviation (MSE BC 1)",
    "Mean (MSE BC 2)": "Mean ± Standard Deviation (MSE BC 2)",
    "Standard Deviation (MSE BC 2)": "Mean ± Standard Deviation (MSE BC 2)",
    "Mean (MSE QD)": "Mean ± Standard Deviation (MSE QD)",
    "Standard Deviation (MSE QD)": "Mean ± Standard Deviation (MSE QD)",
    "Mean (Coverage Error)": "Mean ± Standard Deviation (Coverage Error)",

    "AUC (MSE Objective)": "AUC (MSE Objective)",
    "AUC (MSE BC 1)": "AUC (MSE BC 1)",
    "AUC (MSE BC 2)": "AUC (MSE BC 2)",
    "AUC (MSE QD)": "AUC (MSE QD)",
    "AUC (Coverage Error)": "AUC (Coverage Error)",
    "Mean (Survival %)": "Survival (%)",
    "Evaluations": "Evaluations",

    "Rate of Change (MSE Objective)": "Offline Error (Objective)",
    "Rate of Change (MSE QD)": "Offline Error (QD)",
}

get_mean = lambda x, settings: x[0]
get_std = lambda x, settings: x[1]
compute_sum = lambda x, settings: np.cumsum(x)[-1]
compute_mean = lambda x, settings: np.mean(x)
compute_std = lambda x, settings: np.std(x)
get_sample_method = lambda x, settings: settings[
    'sampling_strategy'] if 'sampling_strategy' in settings.keys() else 'custom' if x else 'default'


def get_rate_of_change(values, settings):
    t = settings['time_shift_val']
    v = np.asarray(values)
    a = v[0:len(values):t]
    b = v[t-1:len(values):t]
    return np.mean(b - a)


column_to_op = {
    'Sampling': get_sample_method,

    "Mean (MSE Objective)": get_mean, "Standard Deviation (MSE Objective)": get_std,
    "Mean (MSE BC 1)": get_mean, "Standard Deviation (MSE BC 1)": get_std,
    "Mean (MSE BC 2)": get_mean, "Standard Deviation (MSE BC 2)": get_std,
    "Mean (MSE QD)": get_mean, "Standard Deviation (MSE QD)": get_std,
    "Mean (Coverage Error)": get_mean, "Standard Deviation (Coverage Error)": get_std,
    "Mean (Survival %)": compute_mean, "Standard Deviation (Survival %)": compute_std,
    "Rate of Change (MSE Objective)": get_rate_of_change,
    "Rate of Change (MSE QD)": get_rate_of_change,

    "Evaluations": compute_sum
}

data = {name: [] for name in column_to_metric.keys()}
df = pd.DataFrame()

ref_dir = os.path.join(configs.plots_dir, '')

items = os.listdir(ref_dir)
folders = [item for item in items if os.path.isdir(os.path.join(ref_dir, item)) and item != 'tmp' and item != 'plots']

with trange(0, len(folders), desc='Processing experiments...') as ii:
    for i in ii:
        try:
            experiment_fname = folders[i]
            ii.set_postfix_str(f'{experiment_fname}', refresh=True)
            experiment_path = os.path.join(ref_dir, experiment_fname)
            with open(os.path.join(experiment_path, f'settings.metadata'), 'r') as f:
                settings = json.load(f)
            with open(os.path.join(experiment_path, 'results.json'), 'r') as f:
                results = json.load(f)
            env, alg, policies = settings['env'], settings['alg'], settings['policies']
            for policy in policies:
                new_row = {'run': experiment_fname.split('_')[1]}
                for c in column_to_metric.keys():
                    if c == 'Policy':
                        m = policy
                    elif column_to_metric[c] in settings.keys():
                        m = settings[column_to_metric[c]]
                    else:
                        m = results[policy][column_to_metric[c]]
                    if c in column_to_op.keys():
                        m = column_to_op[c](m, settings)
                    new_row.update({c: m})
                df2 = pd.DataFrame([new_row])
                df = pd.concat([df, df2], ignore_index=True)
        except Exception:
            print(f'Skipped {experiment_fname}...')

output_fname = os.path.join(ref_dir, 'data.csv')
df.to_csv(output_fname, index=False)
