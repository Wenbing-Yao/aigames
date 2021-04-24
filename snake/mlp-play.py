# -*- coding: utf-8 -*-

import time
from snake_env import SnakeEnv

from simple_mlp import Policy


def play(env, policy):
    obs = env.reset()
    while True:
        action = policy.predict(obs)
        reward, obs, done, _ = env(action)
        env.render()
        if done:
            obs = env.reset()
            time.sleep(1)
        # time.sleep(0.05)


if __name__ == '__main__':
    policy = Policy(pre_trained='pretrained/mlp-v0.joblib')
    env = SnakeEnv(alg='MLP')
    env.reset()
    env.render()
    input()
    play(env, policy)
