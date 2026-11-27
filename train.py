"""
train.py — Train PPO on EndoscopeEnv (CPU).
Run : python train.py --steps 500000
Logs: tensorboard --logdir runs
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from endoscope_env import EndoscopeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo_endoscope")
    args = parser.parse_args()

    torch.set_num_threads(1)                      # small MLP: 1 thread is fastest

    models_dir = Path("models") / args.name
    models_dir.mkdir(parents=True, exist_ok=True)

    # 8 environments run side by side -> more data per update
    train_env = make_vec_env(EndoscopeEnv, n_envs=args.n_envs, seed=args.seed)
    # separate environment for evaluation, different tubes
    eval_env = make_vec_env(EndoscopeEnv, n_envs=1, seed=args.seed + 10_000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),     # saves best_model.zip
        log_path=str(models_dir),
        eval_freq=max(10_000 // args.n_envs, 1),  # every 10k total steps
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,          # steps per env per rollout -> 8192 total
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
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