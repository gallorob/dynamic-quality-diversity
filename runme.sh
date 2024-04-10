clear

echo 'Running Sphere Environment with D-MAP-Elites...'
echo 'Baselines...'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-MAP-Elites", "policies": ["no_updates", "update_all"]}'
echo 'Normal sphere, other configurations...'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-MAP-Elites", "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-MAP-Elites", "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-MAP-Elites", "custom_sampling": true, "sampling_strategy": "parents_only", "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-MAP-Elites", "custom_sampling": true, "sampling_strategy": "parents_only", "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'

echo 'Running Sphere Environment with D-CMA-ME...'
echo 'Baselines...'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "policies": ["no_updates", "update_all"]}'
echo 'Normal sphere, other configurations...'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "custom_sampling": true, "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/sphere_configs.yml --config_args '{"objective_shift_prob": 1.0,  "measures_shift_prob": 1.0, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "custom_sampling": true, "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'


echo 'Running Lunar Lander Environment with D-MAP-Elites...'
echo 'Baselines...'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-MAP-Elites", "policies": ["no_updates", "update_all"]}'
echo 'Lunar Lander, other configurations'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-MAP-Elites", "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-MAP-Elites", "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-MAP-Elites", "custom_sampling": true, "sampling_strategy": "parents_only",  "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-MAP-Elites", "custom_sampling": true, "sampling_strategy": "parents_only",  "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'

echo 'Running Lunar Lander Environment with D-CMA-ME...'
echo 'Baselines...'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-CMA-ME", "batch_size": 8, "n_emitters": 1, "policies": ["no_updates", "update_all"]}'
echo 'Lunar Lander, other configurations'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-CMA-ME", "batch_size": 1, "n_emitters": 1, "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-CMA-ME", "batch_size": 1, "n_emitters": 1, "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-CMA-ME", "batch_size": 1, "n_emitters": 1, "custom_sampling": true, "policies": ["update_local_0"], "detection_on": "oldest_elites", "reevaluate_on": "replacees"}'
python -m sklearnex main.py --configs ./configs/lunar_lander_configs.yml --config_args '{"wind_shift_prob": 1.0, "turbulence_shift_prob": 1.0, "n_runs": 5, "alg": "D-CMA-ME", "batch_size": 1, "n_emitters": 1, "custom_sampling": true, "policies": ["update_local_0"], "detection_on": "replacees", "reevaluate_on": "replacees"}'

python runs_comparator.py
python results_analyzer.py