"""
check_env.py — Checks before training.
1. SB3 check_env: does the environment follow the Gymnasium API?
2. Baselines: how well do simple policies do? PPO must beat these numbers.

Run: python check_env.py            (2D)
     python check_env.py --dim 3    (3D)
"""

import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from endoscope_env import EndoscopeEnv
from endoscope_env3d import EndoscopeEnv3D


def get_env_class(dim: int):
    """Input: dim (2 or 3). Output: the environment class (not an object)."""
    if dim == 2:
        return EndoscopeEnv
    if dim == 3:
        return EndoscopeEnv3D
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def make_pursuit(env):
    """Hand-made controller, not RL: turn toward the 2nd point ahead.

    Input : env (2D or 3D). The action size tells which one.
    Output: a policy function, obs -> action
    """
    if env.action_space.shape[0] == 1:            # 2D
        def pursuit_policy(obs):
            # obs[4:6] = 2nd point ahead in tip frame (x forward, y left)
            lx, ly = obs[4], obs[5]
            angle = np.arctan2(ly, lx)
            return np.array([np.clip(angle / env.max_turn, -1, 1)], dtype=np.float32)
    else:                                         # 3D
        def pursuit_policy(obs):
            # obs[7:10] = 2nd point ahead in tip frame (forward, left, up)
            lx, ly, lz = obs[7], obs[8], obs[9]
            yaw = np.arctan2(ly, lx)
            pitch = np.arctan2(lz, np.hypot(lx, ly))
            a = np.array([yaw, pitch]) / env.max_turn
            return np.clip(a, -1, 1).astype(np.float32)
    return pursuit_policy


def run_policy(env, policy, n_episodes: int, seed: int):
    """Run a policy for many episodes and collect results.

    Input : env, policy (function: obs -> action), n_episodes, seed
    Output: returns (n,), lengths (n,), progress (n,), outcomes (n,) strings
    """
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    progress = np.zeros(n_episodes)              # fraction of the tube passed
    outcomes = []

    env.action_space.seed(seed)                  # random actions repeat too
    for ep in range(n_episodes):                 # loop over episodes
        obs, info = env.reset(seed=seed + ep)    # tube ep is the same for every policy
        total = 0.0
        while True:                              # loop over time steps
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    # 1. API check: raises an error if something is wrong, may print warnings
    env = get_env_class(args.dim)()
    check_env(env, warn=True)
    print(f"check_env ({args.dim}D): OK")

    # 2. Three simple policies
    def random_policy(obs):
        return env.action_space.sample()

    def straight_policy(obs):
        return np.zeros(env.action_space.shape, dtype=np.float32)   # never turn

    for name, pol in [("random", random_policy),
                      ("straight", straight_policy),
                      ("pursuit (hand-made)", make_pursuit(env))]:
        report(name, *run_policy(env, pol, args.episodes, seed=0))