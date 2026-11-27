"""
train_mujoco.py — Train PPO on the MuJoCo endoscope environment (CPU).

The MuJoCo environment is slower than the 2D/3D ones, so the environments run
in parallel processes (SubprocVecEnv), one per CPU core.

Run : python train_mujoco.py --steps 300000 --name ppo_mujoco_300k
Logs: tensorboard --logdir runs
"""

import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from endoscope_mujoco_env import EndoscopeMujocoEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo_mujoco")
    args = parser.parse_args()

    torch.set_num_threads(1)

    models_dir = Path("models") / args.name
    models_dir.mkdir(parents=True, exist_ok=True)

    # One process per environment
    train_env = make_vec_env(EndoscopeMujocoEnv, n_envs=args.n_envs, seed=args.seed,
                             vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(EndoscopeMujocoEnv, n_envs=1, seed=args.seed + 10_000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(models_dir),
        eval_freq=max(10_000 // args.n_envs, 1),   # every 10k total steps
        n_eval_episodes=10,
        deterministic=True,
    )

    # Rollout of about 8192 steps in total, as in the 2D/3D runs
    n_steps = max(8192 // args.n_envs, 64)

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        # log_std_init=-1 -> start with action noise std exp(-1) = 0.37 instead of 1.
        # Large random bending makes the tip hit the wall too often early in training.
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0),
        tensorboard_log="runs",
        device="cpu",
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.steps, callback=eval_cb, tb_log_name=args.name)
    model.save(models_dir / "final_model")
    print(f"saved: {models_dir}/final_model.zip and {models_dir}/best_model.zip")


if __name__ == "__main__":
    main()
