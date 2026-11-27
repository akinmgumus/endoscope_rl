"""
evaluate_mujoco.py — Compare PPO with hand-made policies in the MuJoCo environment.

Metrics per episode (averaged over test tubes):
    goal          : tip reached the end
    tip contact   : % of steps in which the bending section touches the wall
    tip mean F    : mean tip contact force per step
    tip max F     : largest tip contact force in the episode
    body mean F   : mean contact force of the rest of the endoscope
    return        : sum of rewards

Run: python evaluate_mujoco.py --model models/ppo_mujoco_300k/best_model.zip --out results_mujoco
Headless machine: MUJOCO_GL=egl python evaluate_mujoco.py ...
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3 import PPO

from endoscope_mujoco_env import EndoscopeMujocoEnv, make_straight, make_pursuit_mujoco

TEST_SEED = 100_000
COLORS = {"straight": "gray", "pursuit-2": "orange", "pursuit-5": "purple", "PPO": "g"}


def run_episode(env, policy, seed):
    """One episode. Output: dict of metrics, tip path (steps+1, 3)."""
    obs, _ = env.reset(seed=seed)
    f_tip, f_body, total = [], [], 0.0
    while True:                                          # loop over time steps
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        f_tip.append(info["f_tip"])
        f_body.append(info["f_body"])
        if terminated or truncated:
            break
    f_tip, f_body = np.array(f_tip), np.array(f_body)
    metrics = dict(goal=float(info["reached"]), escape=float(info["crashed"]),
                   tip_contact=100 * np.mean(f_tip > 0.01), tip_mean=f_tip.mean(),
                   tip_max=f_tip.max(), body_mean=f_body.mean(), ret=total)
    return metrics, np.array(env.path)


def plot_paths(env, policies, seeds, file):
    fig = plt.figure(figsize=(12, 10))
    for k, seed in enumerate(seeds):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        for name, pol in policies:
            m, path = run_episode(env, pol, seed)          # path: (steps+1, 3)
            ax.plot(*path.T, color=COLORS[name], lw=1.5,
                    label=f"{name}: tip contact {m['tip_contact']:.0f} %")
        X, Y, Z = env.tube.wall_surface(n_around=16)
        ax.plot_wireframe(X[::2], Y[::2], Z[::2], color="lightblue", lw=0.4)
        c = env.tube.centerline
        mid, half = c.mean(axis=0), np.ptp(c, axis=0).max() / 2 + env.tube.radius
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_title(f"test tube seed {seed}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(file, dpi=110)
    plt.close(fig)


def make_gif(policy, seed, file, every=2):
    env = EndoscopeMujocoEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    while True:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        if env.steps % every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    images = [Image.fromarray(f) for f in frames]           # each (480, 640, 3)
    images[0].save(file, save_all=True, append_images=images[1:], duration=80, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_mujoco/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="results_mujoco")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = EndoscopeMujocoEnv()
    model = PPO.load(args.model, device="cpu")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    policies = [("straight", make_straight(env)),
                ("pursuit-2", make_pursuit_mujoco(env, k=2)),
                ("pursuit-5", make_pursuit_mujoco(env, k=5)),
                ("PPO", ppo_policy)]

    lines = ["| Policy | Goal | Tip contact (% of steps) | Tip mean F | Tip max F | Body mean F | Return |",
             "|--------|------|--------------------------|------------|-----------|-------------|--------|"]
    for name, pol in policies:                              # loop over policies
        rows = [run_episode(env, pol, TEST_SEED + ep)[0] for ep in range(args.episodes)]
        avg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{name:9} | goal {100 * avg['goal']:5.1f} % | tip contact {avg['tip_contact']:5.1f} % | "
              f"tip mean F {avg['tip_mean']:.3f} | tip max F {avg['tip_max']:.2f} | "
              f"body mean F {avg['body_mean']:.3f} | return {avg['ret']:7.2f}")
        lines.append(f"| {name} | {100 * avg['goal']:.1f} % | {avg['tip_contact']:.1f} % | "
                     f"{avg['tip_mean']:.3f} | {avg['tip_max']:.2f} | {avg['body_mean']:.3f} | "
                     f"{avg['ret']:.2f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    seeds = [TEST_SEED + k for k in range(4)]
    plot_paths(env, [p for p in policies if p[0] in ("straight", "pursuit-2", "PPO")],
               seeds, out / "paths.png")
    try:
        make_gif(ppo_policy, seeds[0], out / "ppo.gif")
        print(f"saved: {out}/table.md, {out}/paths.png and {out}/ppo.gif")
    except Exception as e:                                  # no OpenGL on this machine
        print(f"GIF skipped ({e}). saved: {out}/table.md and {out}/paths.png")


if __name__ == "__main__":
    main()
