"""
evaluate.py — Test a trained model on new tubes, draw paths, make a GIF.
Run: python evaluate.py --model models/ppo_endoscope/best_model.zip
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # no window, only files
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3 import PPO

from endoscope_env import EndoscopeEnv
from check_env import run_policy, report

TEST_SEED = 100_000                       # tubes never seen in training


def make_pursuit(env):
    def pursuit_policy(obs):
        lx, ly = obs[4], obs[5]           # 2nd point ahead, tip frame
        angle = np.arctan2(ly, lx)
        return np.array([np.clip(angle / env.max_turn, -1, 1)], dtype=np.float32)
    return pursuit_policy


def rollout(env, policy, seed):
    """One episode. Output: path (steps+1, 2), last info."""
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    return np.array(env.path), info


def plot_paths(env, policies, seeds, file):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, seed in zip(axes.ravel(), seeds):
        for (name, pol), color in zip(policies, ["g", "orange"]):
            path, info = rollout(env, pol, seed)
            result = "goal" if info["reached"] else ("crash" if info["crashed"] else "timeout")
            ax.plot(*path.T, color=color, lw=1.5, label=f"{name}: {result}")
        ax.plot(*env.tube.left_wall.T, "b-", lw=2)
        ax.plot(*env.tube.right_wall.T, "b-", lw=2)
        ax.plot(*env.tube.centerline.T, "k--", lw=0.8)
        ax.set_title(f"test tube seed {seed}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(file, dpi=120)
    plt.close(fig)


def make_gif(policy, seed, file, every=2):
    env = EndoscopeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    while True:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        if env.steps % every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    frames.append(env.render())
    env.close()
    images = [Image.fromarray(f) for f in frames]
    images[0].save(file, save_all=True, append_images=images[1:], duration=50, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_endoscope/best_model.zip")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    env = EndoscopeEnv()

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)   # (1,)
        return action

    pursuit_policy = make_pursuit(env)

    for name, pol in [("PPO", ppo_policy), ("pursuit (hand-made)", pursuit_policy)]:
        report(name, *run_policy(env, pol, args.episodes, seed=TEST_SEED))

    seeds = [TEST_SEED + k for k in range(4)]
    plot_paths(env, [("PPO", ppo_policy), ("pursuit", pursuit_policy)],
               seeds, out / "paths.png")
    make_gif(ppo_policy, seeds[0], out / "ppo.gif")
    print(f"saved: {out}/paths.png and {out}/ppo.gif")


if __name__ == "__main__":
    main()