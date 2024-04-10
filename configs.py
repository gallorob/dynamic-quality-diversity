from argparse import Namespace

import yaml

with open('./configs/configs.yml', 'r') as file:
    configs: Namespace = Namespace(**yaml.safe_load(file))


def reload_configs(configs, config_filename):
    with open(config_filename, 'r') as file:
        new_configs: Namespace = Namespace(**yaml.safe_load(file))
        for attr in new_configs.__dict__.keys():
            # if hasattr(configs, attr):
            configs.__dict__[attr] = new_configs.__dict__[attr]


def update_configs(configs, kwargs):
    for attr in kwargs.keys():
        configs.__dict__[attr] = kwargs[attr]
