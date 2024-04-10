clear

echo 'Analyzing global shifts in sphere environment...'
python -m sklearnex dyn_env_explorer.py --configs ./configs/sphere_configs.yml

echo 'Analyzing local shifts in spherev2 environment...'
python -m sklearnex dyn_env_explorer.py --configs ./configs/spherev2_configs.yml

echo 'Analyzing lunar lander environment...'
python -m sklearnex dyn_env_explorer.py --configs ./configs/lunar_lander_configs.yml