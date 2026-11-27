"""
hard_test.py — Test trained PPO models in conditions that differ from training.
The models are NOT retrained. This tests how far the learned policy generalizes.

Training: curviness c in [0, 0.08], width w = 0.5, tip turn up to 10 deg/step,
          exact observations (no noise).

Test conditions:
    train       : same as training (reference)
    sharp tube  : c in [0.08, 0.16], w = 0.30. A narrow tube also allows sharper
                  bends, up to about 13 deg/step, more than the tip can turn.
    weak tip    : c in [0.08, 0.16], tip turn only up to 6 deg/step
    noise 0.1   : Gaussian noise with std 0.1 added to every observation number
    noise 0.2   : same, std 0.2
    noise 0.3   : same, std 0.3

Policies:
    PPO         : trained model, deterministic
    pursuit-2   : hand-made, aims at the 2nd point ahead (0.10 ahead)
    pursuit-5   : hand-made, aims at the 5th point ahead (0.25 ahead)

Run: python hard_test.py
     python hard_test.py --dims 3 --episodes 100      (faster, 3D only)
Output: table in the terminal, results_hard/table_<dim>d.md, path plots
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from stable_baselines3 import PPO

from check_env import get_env_class, make_pursuit, run_policy
import evaluate
from evaluate import plot_paths_2d, plot_paths_3d, TEST_SEED

# Each condition: env settings + observation noise std
CONDITIONS = {
    "train":      dict(env=dict(), noise=0.0),
    "sharp tube": dict(env=dict(tube_kwargs=dict(width=0.30, curviness_range=(0.08, 0.16))),
                       noise=0.0),
    "weak tip":   dict(env=dict(max_turn_deg=6.0,
                                tube_kwargs=dict(curviness_range=(0.08, 0.16))),
                       noise=0.0),
    "noise 0.1":  dict(env=dict(), noise=0.1),
    "noise 0.2":  dict(env=dict(), noise=0.2),
    "noise 0.3":  dict(env=dict(), noise=0.3),
}

DEFAULT_MODELS = {
    2: "models/ppo_300k/best_model.zip",
    3: "models/ppo3d_300k/best_model.zip",
}


def with_noise(policy, std: float, seed: int = 0):
    """Wrap a policy so it sees noisy observations.

    Input : policy (obs -> action), noise std, seed
    Output: new policy. obs_noisy = obs + noise,  noise ~ Normal(0, std^2), same shape
    """
    if std == 0.0:
        return policy
    rng = np.random.default_rng(seed)

    def noisy_policy(obs):
        noise = rng.normal(0.0, std, size=obs.shape).astype(np.float32)
        return policy(obs + noise)
    return noisy_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--model2d", type=str, default=DEFAULT_MODELS[2])
    parser.add_argument("--model3d", type=str, default=DEFAULT_MODELS[3])
    parser.add_argument("--out", type=str, default="results_hard")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_paths = {2: args.model2d, 3: args.model3d}
    evaluate.COLORS = ["g", "orange", "purple"]      # PPO, pursuit-2, pursuit-5

    for dim in args.dims:                            # loop over 2D / 3D
        model = PPO.load(model_paths[dim], device="cpu")
        env_class = get_env_class(dim)

        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        lines = ["| Condition | Policy | Goal | Crash | Progress | Mean return |",
                 "|-----------|--------|------|-------|----------|-------------|"]

        for cname, cond in CONDITIONS.items():       # loop over test conditions
            env = env_class(**cond["env"])
            base = [("PPO", ppo_policy),
                    ("pursuit-2", make_pursuit(env, k=2)),
                    ("pursuit-5", make_pursuit(env, k=5))]

            for pname, pol in base:                  # loop over policies
                noisy = with_noise(pol, cond["noise"], seed=0)
                returns, lengths, progress, outcomes = run_policy(
                    env, noisy, args.episodes, seed=TEST_SEED
                )
                goal = 100 * np.mean(outcomes == "goal")
                crash = 100 * np.mean(outcomes == "crash")
                prog = 100 * progress.mean()
                print(f"{dim}D {cname:10} {pname:9} | goal {goal:5.1f} % | "
                      f"crash {crash:5.1f} % | progress {prog:5.1f} % | "
                      f"return {returns.mean():7.2f}")
                lines.append(f"| {cname} | {pname} | {goal:.1f} % | {crash:.1f} % | "
                             f"{prog:.1f} % | {returns.mean():.2f} |")

            # Path plots on 4 test tubes of this condition
            plot_pols = [(n, with_noise(p, cond["noise"], seed=1)) for n, p in base]
            seeds = [TEST_SEED + k for k in range(4)]
            file = out / f"paths_{dim}d_{cname.replace(' ', '_').replace('.', '')}.png"
            if dim == 2:
                plot_paths_2d(env, plot_pols, seeds, file)
            else:
                plot_paths_3d(env, plot_pols, seeds, file)

        (out / f"table_{dim}d.md").write_text("\n".join(lines) + "\n")
        print(f"saved: {out}/table_{dim}d.md\n")


if __name__ == "__main__":
    main()
