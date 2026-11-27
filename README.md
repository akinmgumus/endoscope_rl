# Endoscope tip navigation in random curved tubes (RL)

A PPO agent steers an endoscope tip through random curved 2D tubes
(a simplified colon model) without touching the wall.
Every episode uses a new random tube, so the agent cannot memorize one path.

![PPO agent, 300k steps](assets/ppo_300k.gif)

## Task

- **Tube:** a random centerline with constant width. Curviness is sampled per episode,
  from straight to strongly curved. Turn rate is limited so the walls never fold.
- **Action:** one number in [-1, 1], scaled to a turn of at most 10° per step.
  The tip moves forward at constant speed.
- **Observation (12 numbers):**
  - offset from the centerline, normalized so the walls are at ±1,
  - heading error between the tip and the tube,
  - the next 5 centerline points, in the tip's own frame (x forward, y left).
- **Reward:**
  - progress along the centerline (×10),
  - a small penalty for distance from the center,
  - −10 for hitting the wall,
  - +10 for reaching the end.
- **Episode end:** reaching the end or hitting the wall (terminated),
  or 400 steps (truncated).

## Training

PPO from Stable-Baselines3, on CPU.

| Setting                | Value                     |
|------------------------|---------------------------|
| Policy / value network | MLP, 2 × 64 units each    |
| Parallel environments  | 8                         |
| Rollout                | 1024 steps per env (8192) |
| Batch size / epochs    | 256 / 10                  |
| γ / GAE λ              | 0.99 / 0.95               |
| Learning rate / clip   | 3e-4 / 0.2                |

Two runs with the same settings:

- `ppo_endoscope`: 50k steps (stops after the 7th rollout, at 57,344 steps).
- `ppo_300k`: 300k steps, about 1.2 minutes on a laptop CPU.

### Learning curve

![Learning curve, 50k and 300k runs](assets/learning_curve.png)

`rollout/ep_rew_mean` (mean return of the last 100 training episodes):

- 0–20k steps: the tip hits the wall almost at once.
- 20k–100k steps: most of the learning happens here.
- After 100k steps: the curve is flat at about 108.9.
  The maximum possible return is about 109.5.
- The 50k run stops in the middle of the rise. Its training curve ends at 95,
  but the deterministic policy already reaches the goal on every test tube (see below).
  Training actions are sampled from a Gaussian policy, so exploration noise
  causes some crashes during training.

## Results

Evaluation on 300 test tubes (seeds 100000+), never seen in training.
PPO uses the deterministic policy (mean action).
For each run, the PPO model is the best checkpoint: the one with the highest
evaluation score during training (`best_model.zip`).

| Policy                   | Goal   | Crash  | Mean return     | Mean length |
|--------------------------|--------|--------|-----------------|-------------|
| PPO, 50k steps           | 100 %  | 0 %    | 109.10 ± 0.17   | 199.7       |
| PPO, 300k steps          | 100 %  | 0 %    | 109.22 ± 0.14   | 198.5       |
| Pursuit (hand-made)      | 100 %  | 0 %    | 109.09 ± 0.07   | 198.0       |
| Straight (never turns)*  | 1.7 %  | 98.3 % | 8.13 ± 17.68    | 38.7        |
| Random*                  | 0 %    | 100 %  | 0.99 ± 6.54     | 25.1        |

\* Baselines from `check_env.py`, on 300 tubes with seeds 0–299.

**Pursuit** is a hand-made controller, not RL. It turns the tip toward
the 2nd centerline point ahead. It shows that the task is solvable
and gives a strong reference for PPO.

### Paths on test tubes (PPO in green, pursuit in orange)

| PPO, 50k steps | PPO, 300k steps |
|----------------|-----------------|
| ![Paths, 50k](assets/paths_50k.png) | ![Paths, 300k](assets/paths_300k.png) |

| PPO, 50k steps | PPO, 300k steps |
|----------------|-----------------|
| ![PPO 50k](assets/ppo_50k.gif) | ![PPO 300k](assets/ppo_300k.gif) |

## Discussion

- PPO matches the hand-made pursuit controller on unseen tubes.
  The 2D task with a perfect centerline observation is easy enough
  for a simple geometric controller.
- This stage checks the full pipeline: tube generator, environment,
  baselines, training and evaluation.
- RL becomes useful where a hand-made controller is hard to write:
  a camera image as the observation, variable speed, or physical
  contact between a bendable endoscope and the wall.

## Files

- `tube.py`: random curved tube and geometry queries (offset, progress, points ahead)
- `endoscope_env.py`: Gymnasium environment
- `check_env.py`: Gymnasium API check and baseline policies
- `train.py`: PPO training (Stable-Baselines3, CPU), TensorBoard logs
- `evaluate.py`: test on unseen tubes, path plots, GIF

## Setup

Python 3.11, [uv](https://github.com/astral-sh/uv):

```
uv sync
```

## Usage

```
uv run python tube.py
uv run python check_env.py
uv run python train.py --steps 300000 --name ppo_300k
uv run tensorboard --logdir runs
uv run python evaluate.py --model models/ppo_300k/best_model.zip --out results_300k
```

## Next steps

- 3D tubes
- Harder test: tubes more curved than in training
- MuJoCo: real wall geometry, bendable endoscope, physical contact
- Camera image as the observation (CNN)
