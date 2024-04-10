__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import numpy.typing as npt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d import LunarLander
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle, colorize
from dask.distributed import Client
import gc

from configs import configs


class DynamicLunarLanderWrapper:
    def __init__(self):
        self.t = 0
        self.just_shifted = False
        self.rng = np.random.default_rng(configs.rng_seed)
        self.ranges = np.asarray(configs.ranges)
        # re-set wind_idx and torque_idx as `np.random.randint(-9999, 9999)`, but with rng
        self.wind_idx = self.rng.random() * (9999 * 2) - 9999
        self.torque_idx = self.rng.random() * (9999 * 2) - 9999
        self.wind_power = np.asarray(10.)
        self.wind_shift_strength = np.asarray(configs.wind_shift_strength)
        self.turbulence_power = np.asarray(1.0)
        self.turbulence_shift_strength = np.asarray(configs.turbulence_shift_strength)
        self.shift_multiplier = [-1, 1] if not configs.directed else [1]

        self.truncation_timestep = configs.truncation_timestep

    @property
    def action_space(self):
        return gym.make('LunarLander-v2').action_space

    @property
    def observation_space(self):
        return gym.make('LunarLander-v2').observation_space

    @property
    def tracked_shifts(self):
        return self.wind_power.item(), self.turbulence_power.item()

    def next_timestep(self):
        self.t += 1
        self.just_shifted = False
        if self.t % configs.time_shift_val == 0 and self.rng.random(1) <= configs.wind_shift_prob:
            wind_shift = self.rng.choice(self.shift_multiplier) * self.rng.uniform(low=0, high=1, size=None) * self.wind_shift_strength
            # self.wind_power = np.clip(self.wind_power + wind_shift, 0, 20)
            self.wind_power = self.wind_power + wind_shift
            self.just_shifted = True
        if self.t % configs.time_shift_val == 0 and self.rng.random(1) <= configs.turbulence_shift_prob:
            turbulence_shift = self.rng.choice(self.shift_multiplier) * self.rng.uniform(low=0, high=1, size=None) * self.turbulence_shift_strength
            # self.turbulence_power = np.clip(self.turbulence_power + turbulence_shift, 0, 2)
            self.turbulence_power = self.turbulence_power + turbulence_shift
            self.just_shifted = True

    def reset(self):
        self.t = 0
        self.just_shifted = False
        self.rng = np.random.default_rng(configs.rng_seed)
        self.wind_idx = self.rng.random() * (9999 * 2) - 9999
        self.torque_idx = self.rng.random() * (9999 * 2) - 9999
        self.wind_power = np.asarray(10.)
        self.turbulence_power = np.asarray(1.0)

    def _simulate(self, model: npt.NDArray[np.float32]) -> Tuple[float, Tuple[float, float]]:
        env = gym.make('LunarLander-v2')  # NOTE: Passing `enable_wind = True` here leads to non-determinstic simulations
        total_reward = 0.0
        impact_x_pos = None
        impact_y_vel = None
        all_y_vels = []
        # For the love of all that is holy, do not EVER change this few lines
        # Seriously though, changing the order either removes the wind/turbulence effect or introduces non-determinism
        obs, _ = env.reset(seed=configs.rng_seed)
        # `gym.make()` returns a bunch of wrappers, but we need to change the parameters of the LunarLander environment
        lunar_lander_env = env.env.env.env
        lunar_lander_env.enable_wind = True
        lunar_lander_env.wind_idx, lunar_lander_env.torque_idx = self.wind_idx, self.torque_idx
        lunar_lander_env.wind_power, lunar_lander_env.turbulence_power = self.wind_power, self.turbulence_power
        done, _t = False, 0
        model = model.reshape((env.action_space.n, env.observation_space.shape[0]))
        while not done:
            action = np.argmax(model @ obs)  # Linear policy.
            obs, reward, terminated, _, _ = env.step(action)
            done = terminated or _t == self.truncation_timestep
            total_reward += reward

            # Refer to the definition of state here:
            # https://gymnasium.farama.org/environments/box2d/lunar_lander/
            x_pos = obs[0]
            y_vel = obs[3]
            leg0_touch = bool(obs[6])
            leg1_touch = bool(obs[7])
            all_y_vels.append(y_vel)

            # Check if the lunar lander is impacting for the first time.
            if impact_x_pos is None and (leg0_touch or leg1_touch):
                impact_x_pos = x_pos
                impact_y_vel = y_vel

            _t += 1

        # If the lunar lander did not land, set the x-pos to the one from the final
        # timestep, and set the y-vel to the max y-vel (we use min since the lander
        # goes down).
        if impact_x_pos is None:
            impact_x_pos = x_pos
            impact_y_vel = min(all_y_vels)

        env.close()

        return total_reward, (impact_x_pos, impact_y_vel)

    def __call__(self, solution_batch: npt.NDArray[np.float32], client: Client) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        objective_batch, measure_batch = [], []
        # futures = client.map(lambda model: self._simulate(model), solution_batch)
        # futures = client.map(self._simulate, [solution_batch[i] for i in range(len(solution_batch))])
        # results = client.gather(futures)

        # for obj, meas in results:
        # for model in solution_batch:
        for i in range(solution_batch.shape[0]):
            model = solution_batch[i]
            obj, meas = self._simulate(model)
            objective_batch.append(obj)
            measure_batch.append(list(meas))

        # client.cancel(solution_batch)
        # del futures
        # del results
        # client.run(gc.collect)
        # client.rebalance()

        return np.asarray(objective_batch), np.asarray(measure_batch)
