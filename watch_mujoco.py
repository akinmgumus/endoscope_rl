"""
watch_mujoco.py — Watch a policy live in the MuJoCo viewer.

Each episode has a new random tube, so it has a new MuJoCo model.
The viewer belongs to one model, so a new viewer window opens for every episode.

Run: python watch_mujoco.py --model models/ppo_mujoco_300k/best_model.zip
     python watch_mujoco.py --policy pursuit             (no model needed)
     python watch_mujoco.py --policy straight --episodes 1
Keys in the viewer: mouse to rotate/zoom, Space to pause, close the window to go on.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from endoscope_mujoco_env import EndoscopeMujocoEnv, make_straight, make_pursuit_mujoco

TEST_SEED = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "pursuit", "straight"], default="ppo")
    parser.add_argument("--model", type=str, default="models/ppo_mujoco_300k/best_model.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    args = parser.parse_args()

    env = EndoscopeMujocoEnv()

    if args.policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    elif args.policy == "pursuit":
        policy = make_pursuit_mujoco(env, k=2)
    else:
        policy = make_straight(env)

    for ep in range(args.episodes):                       # loop over episodes
        obs, _ = env.reset(seed=TEST_SEED + ep)
        step_time = env.frame_skip * env.model.opt.timestep   # 25 * 0.002 = 0.05 s

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            # Point the camera at the middle of the tube
            c = env.tube.centerline
            with viewer.lock():
                viewer.cam.lookat[:] = c.mean(axis=0)
                viewer.cam.distance = 1.6 * np.ptp(c, axis=0).max() + 1.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -35

            f_tip = []
            while viewer.is_running():                    # loop over env steps
                t0 = time.time()
                obs, _, terminated, truncated, info = env.step(policy(obs))
                f_tip.append(info["f_tip"])
                viewer.sync()
                if terminated or truncated:
                    break
                time.sleep(max(0.0, step_time / args.speed - (time.time() - t0)))

            f_tip = np.array(f_tip)
            print(f"episode {ep}: steps={env.steps} reached={info['reached']} "
                  f"tip contact {100 * np.mean(f_tip > 0.01):.1f} % of steps")
            time.sleep(1.0)                               # short pause at the end


if __name__ == "__main__":
    main()
