"""
evaluate.py — Test a trained model on new tubes, draw paths, make a GIF.
Run: python evaluate.py --model models/ppo_300k/best_model.zip --out results_300k
     python evaluate.py --dim 3 --model models/ppo3d_300k/best_model.zip --out results3d_300k
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # no window, only files
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3 import PPO

from check_env import get_env_class, make_pursuit, run_policy, report

TEST_SEED = 100_000                       # tubes never seen in training
COLORS = ["g", "orange"]                  # PPO, pursuit


def rollout(env, policy, seed):
    """One episode. Output: path (steps+1, dim), last info."""
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    return np.array(env.path), info


def outcome(info):
    return "goal" if info["reached"] else ("crash" if info["crashed"] else "timeout")


def plot_paths_2d(env, policies, seeds, file):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, seed in zip(axes.ravel(), seeds):
        for (name, pol), color in zip(policies, COLORS):
            path, info = rollout(env, pol, seed)
            ax.plot(*path.T, color=color, lw=1.5, label=f"{name}: {outcome(info)}")
        ax.plot(*env.tube.left_wall.T, "b-", lw=2)
        ax.plot(*env.tube.right_wall.T, "b-", lw=2)
        ax.plot(*env.tube.centerline.T, "k--", lw=0.8)
        ax.set_title(f"test tube seed {seed}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(file, dpi=120)
    plt.close(fig)


def plot_paths_3d(env, policies, seeds, file):
    fig = plt.figure(figsize=(12, 10))
    for k, seed in enumerate(seeds):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        for (name, pol), color in zip(policies, COLORS):
            path, info = rollout(env, pol, seed)          # path: (steps+1, 3)
            ax.plot(*path.T, color=color, lw=1.5, label=f"{name}: {outcome(info)}")
        X, Y, Z = env.tube.wall_surface(n_around=16)      # each (N, 16)
        ax.plot_wireframe(X[::3], Y[::3], Z[::3], color="lightblue", lw=0.4)

        # equal axis scale
        c = env.tube.centerline
        mid = c.mean(axis=0)
        half = np.ptp(c, axis=0).max() / 2 + env.tube.radius
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_title(f"test tube seed {seed}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(file, dpi=110)
    plt.close(fig)


def make_gif(env_class, policy, seed, file, every=2, max_width=700):
    env = env_class(render_mode="rgb_array")
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

    images = []
    for f in frames:                                      # f: (height, width, 3)
        img = Image.fromarray(f)
        if img.width > max_width:                         # keep the GIF file small
            h = int(img.height * max_width / img.width)
            img = img.resize((max_width, h))
        images.append(img)
    images[0].save(file, save_all=True, append_images=images[1:], duration=50, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.model is None:
        name = "ppo_endoscope" if args.dim == 2 else "ppo3d_endoscope"
        args.model = f"models/{name}/best_model.zip"
    if args.out is None:
        args.out = "results" if args.dim == 2 else "results3d"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_class = get_env_class(args.dim)
    model = PPO.load(args.model, device="cpu")
    env = env_class()

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    pursuit_policy = make_pursuit(env)

    for name, pol in [("PPO", ppo_policy), ("pursuit (hand-made)", pursuit_policy)]:
        report(name, *run_policy(env, pol, args.episodes, seed=TEST_SEED))

    seeds = [TEST_SEED + k for k in range(4)]
    policies = [("PPO", ppo_policy), ("pursuit", pursuit_policy)]
    if args.dim == 2:
        plot_paths_2d(env, policies, seeds, out / "paths.png")
    else:
        plot_paths_3d(env, policies, seeds, out / "paths.png")
    make_gif(env_class, ppo_policy, seeds[0], out / "ppo.gif")
    print(f"saved: {out}/paths.png and {out}/ppo.gif")


if __name__ == "__main__":
    main()