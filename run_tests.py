#!/usr/bin/env python3
import gym
from gym_minigrid.wrappers import *


def load_env(env_name):
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
    return RGBImgObsWrapper(env)


if __name__ == "__main__":
    # env = load_env("MiniGrid-Empty-5x5-v0")
    env = load_env("MiniGrid-Unlock-v0")
    obs = env.reset(task=(3, 3))
    while True:
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
