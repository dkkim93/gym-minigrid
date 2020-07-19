#!/usr/bin/env python3
import gym
from gym_minigrid.wrappers import *


def load_env(env_name):
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    return VectorObsWrapper(env)


if __name__ == "__main__":
    env = load_env("MiniGrid-Empty-5x5-v0")  # Either "MiniGrid-Empty-5x5-v0" or "MiniGrid-Unlock-v0"
    obs = env.reset()
    env.unwrapped.reset_task(task=(4, 4))
    while True:
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
