import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as st

from configs import configs

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['FreeSerif']
plt.rcParams['font.size'] = 16
ref_colors = [
    (0.12, 0.12, 0.12),
    (0.64, 0.64, 0.64),
    (1.0, 0.0, 0.1607843137254902),
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    (0.4, 0.6509803921568628, 0.11764705882352941),
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    (0.0, 0.8235294117647058, 0.8352941176470589),
    (1.0, 0.4980392156862745, 0.0),
    (0.6862745098039216, 0.5529411764705883, 0.0),
    (0.4980392156862745, 0.5019607843137255, 0.803921568627451),
    (0.7019607843137254, 0.9137254901960784, 0.0),
    (0.7686274509803922, 0.1803921568627451, 0.3764705882352941),
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
    (0.5529411764705883, 0.8274509803921568, 0.7803921568627451),
    (0.7450980392156863, 0.7294117647058823, 0.8549019607843137),
    (0.984313725490196, 0.5019607843137255, 0.4470588235294118),
    (0.5019607843137255, 0.6941176470588235, 0.8274509803921568)
]


ref_dir = configs.plots_dir
rawdata_fname = 'data'

df = pd.read_csv(os.path.join(ref_dir, f'{rawdata_fname}.csv'))

envs = set(df['Environment'])
algs = set(df['Algorithm'])
policies = list(sorted(set(df['Policy'])))
samplings = list(sorted(set(df['Sampling'])))
detections = list(sorted(set(df['Detection'])))
reevaluations = list(sorted(set(df['Re-Evaluation'])))
metrics = df.columns.drop(['run', 'Environment', 'Algorithm', 'Policy', 'Sampling', 'Detection', 'Re-Evaluation'])

use_valid = False
valid_combinations = {
    'no_updates': {'default': True,
                   'replacees_only': False,
                   'parents_only': False,
                   'replacees_and_parents': False,
                   'not_replacees_only': False,
                   'not_parents_only': False,
                   'not_replacees_and_parents': False,
                   'custom': False},
    'update_all': {'default': True,
                   'replacees_only': False,
                   'parents_only': False,
                   'replacees_and_parents': False,
                   'not_replacees_only': False,
                   'not_parents_only': False,
                   'not_replacees_and_parents': False,
                   'custom': True},
    'update_local_0': {'default': False,
                   'replacees_only': True,
                   'parents_only': True,
                   'replacees_and_parents': True,
                   'not_replacees_only': True,
                   'not_parents_only': True,
                   'not_replacees_and_parents': True,
                   'custom': True},
    'update_local_1': {'default': False,
                   'replacees_only': True,
                   'parents_only': True,
                   'replacees_and_parents': True,
                   'not_replacees_only': True,
                   'not_parents_only': True,
                   'not_replacees_and_parents': True,
                   'custom': True},
    'update_local_2': {'default': False,
                   'replacees_only': True,
                   'parents_only': True,
                   'replacees_and_parents': True,
                   'not_replacees_only': True,
                   'not_parents_only': True,
                   'not_replacees_and_parents': True,
                   'custom': True}
}

processed_df = pd.DataFrame()

for env in envs:
    for alg in algs:
        for metric in metrics:
            labels = []
            x_pos = [0]
            mean_vals = []
            ci_vals = []
            colors = []
            for policy in policies:
                for sampling in samplings:
                    for detection in detections:
                        for reevaluation in reevaluations:
                            values = df.loc[(df['Environment'] == env) & (df['Algorithm'] == alg) \
                                            & (df['Policy'] == policy) & (df['Sampling'] == sampling) \
                                            & (df['Detection'] == detection) & (df['Re-Evaluation'] == reevaluation)]
                            ref_values = df.loc[(df['Environment'] == env) & (df['Algorithm'] == alg) \
                                                & (df['Policy'] == 'no_updates') & (df['Sampling'] == 'default') \
                                                & (df['Detection'] == 'replacees') & (
                                                            df['Re-Evaluation'] == 'replacees')]
                            if len(values) == 0: continue
                            mean_value = values[metric].mean()
                            ref_mean_value = ref_values[metric].mean()
                            std_value = values[metric].std()
                            ci_value = st.t.interval(0.95, len(values[metric]) - 1, loc=mean_value, scale=st.sem(values[metric]))
                            new_row = pd.DataFrame([{'Environment': env, 'Algorithm': alg, 'Policy': policy,
                                                               'Sampling': sampling, 'Detection': detection, 'Re-Evaluation': reevaluation,
                                                               'Metric': metric,
                                                               'Mean': mean_value, 'Std': std_value, '95% CI': ci_value,
                                                               'Difference (%)': ((mean_value - ref_mean_value) / ref_mean_value) * 100}])
                            if not use_valid or valid_combinations[policy][sampling]:
                                labels.append(f'{policy}_{sampling}_{detection}_{reevaluation}')
                                x_pos.append(x_pos[-1] + 1)
                                colors.append(ref_colors[len(colors) % len(ref_colors)])
                                mean_vals.append(mean_value)
                                ci_vals.append([mean_value - ci_value[0], ci_value[1] - mean_value])
                                processed_df = pd.concat([processed_df, new_row], ignore_index=True)
            # fig, ax = plt.subplots()
            plt.bar(x_pos[:-1], mean_vals,  color=colors, yerr=np.asarray(ci_vals).T, align='center', alpha=1.0, ecolor='black', capsize=10)
            plt.ylabel('Value')
            plt.xticks(x_pos[:-1], labels, rotation=45, ha='right', rotation_mode='anchor')
            plt.title(f'{metric}')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(ref_dir, f'{"alt_plots" if use_valid else "plots"}/{env}_{alg}_{metric}.png'), transparent=True)
            # plt.show()

            plt.clf()
            plt.close()

output_fname = os.path.join(ref_dir, f'processed_{rawdata_fname}{"_alt" if use_valid else ""}.csv')
processed_df.to_csv(output_fname, index=False)
