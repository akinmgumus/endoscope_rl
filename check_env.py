"""
check_env.py — Checks before training.
1. SB3 check_env: does the environment follow the Gymnasium API?
2. Baselines: how well do simple policies do? PPO must beat these numbers.
"""

import numpy as np
from stable_baselines3.common.env_checker import check_env

from endoscope_env import EndoscopeEnv


def run_policy(env, policy, n_episodes: int, seed: int):
    """Run a policy for many episodes and collect results.

    Input : env, policy (function: obs (12,) -> action (1,)), n_episodes, seed
    Output: returns (n,), lengths (n,), progress (n,), outcomes (n,) strings
    """
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    progress = np.zeros(n_episodes)            # fraction of the tube passed
    outcomes = []

    env.action_space.seed(seed)                # random actions repeat too
    for ep in range(n_episodes):               # loop over episodes
        obs, info = env.reset(seed=seed + ep)  # tube ep is the same for every policy
        total = 0.0
        while True:                            # loop over time steps
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break

        returns[ep] = total
        lengths[ep] = env.steps
        progress[ep] = info["progress"] / env.tube.total_length
        if info["crashed"]:
            outcomes.append("crash")
        elif info["reached"]:
            outcomes.append("goal")
        else:
            outcomes.append("timeout")

    return returns, lengths, progress, np.array(outcomes)


def report(name, returns, lengths, progress, outcomes):
    print(f"\n--- {name} ---")
    print(f"return   : {returns.mean():8.2f} +- {returns.std():.2f}")
    print(f"length   : {lengths.mean():8.1f} steps")
    print(f"progress : {100 * progress.mean():8.1f} % of the tube")
    for o in ["goal", "crash", "timeout"]:
        print(f"{o:9}: {100 * np.mean(outcomes == o):8.1f} %")


if __name__ == "__main__":
    # 1. API check: raises an error if something is wrong, may print warnings
    env = EndoscopeEnv()
    check_env(env, warn=True)
    print("check_env: OK")

    # 2. Three simple policies
    def random_policy(obs):
        return env.action_space.sample()                 # (1,)

    def straight_policy(obs):
        return np.zeros(1, dtype=np.float32)             # never turn

    def pursuit_policy(obs):
        # Hand-made controller, not RL: turn toward the 2nd point ahead.
        # obs[4:6] = that point in the tip's frame (x forward, y left).
        lx, ly = obs[4], obs[5]
        angle = np.arctan2(ly, lx)                       # angle to the point
        return np.array([np.clip(angle / env.max_turn, -1, 1)], dtype=np.float32)

    n = 300
    for name, pol in [("random", random_policy),
                      ("straight", straight_policy),
                      ("pursuit (hand-made)", pursuit_policy)]:
        report(name, *run_policy(env, pol, n, seed=0))