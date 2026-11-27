"""
endoscope_env.py — Gymnasium environment.
An endoscope tip moves at constant speed through a random curved 2D tube.
The agent controls only the turn angle.

Notation (Latin letters only):
    q      : tip position, shape (2,)
    h      : tip heading (radians), scalar
    p_k    : k-th centerline point, shape (2,)
    m      : number of points ahead in the observation
    w      : tube width
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from tube import Tube


def wrap_angle(x: float) -> float:
    """Wrap an angle to [-pi, pi]. Example: 350 deg -> -10 deg."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class EndoscopeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n_ahead: int = 5,             # m
        speed: float = 0.05,          # distance per step
        max_turn_deg: float = 10.0,   # largest turn per step
        max_steps: int = 400,         # time limit (truncation)
        w_progress: float = 10.0,     # reward weight for progress
        w_center: float = 0.1,        # penalty weight for distance from center
        crash_penalty: float = 10.0,  # penalty for hitting the wall
        goal_bonus: float = 10.0,     # bonus for reaching the end
        render_mode=None,
    ):
        self.n_ahead = n_ahead
        self.speed = speed
        self.max_turn = np.deg2rad(max_turn_deg)
        self.max_steps = max_steps
        self.w_progress = w_progress
        self.w_center = w_center
        self.crash_penalty = crash_penalty
        self.goal_bonus = goal_bonus
        self.render_mode = render_mode

        # Action: one number in [-1, 1]. step() scales it to [-max_turn, +max_turn].
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # Observation: offset (1) + heading error (1) + m points ahead (2m)
        obs_size = 2 + 2 * n_ahead                     # 12 when m = 5
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )

        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
    # reset: start a new episode with a new random tube
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                       # creates self.np_random
        self.tube = Tube(self.np_random)               # new tube every episode

        self.pos = self.tube.centerline[0].copy()      # q, (2,)
        self.heading = 0.0                             # h, start in +x direction
        self.steps = 0
        _, self.progress, _ = self.tube.project(self.pos)
        self.path = [self.pos.copy()]                  # for drawing

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}                     # (observation, info)

    # ------------------------------------------------------------------
    # step: apply one action, return the result
    # ------------------------------------------------------------------
    def step(self, action):
        # 1. Turn: action in [-1, 1] -> angle in [-max_turn, +max_turn]
        a = float(np.clip(action[0], -1.0, 1.0))
        self.heading = wrap_angle(self.heading + a * self.max_turn)

        # 2. Move forward at constant speed: q <- q + speed [cos h, sin h]
        self.pos = self.pos + self.speed * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        self.steps += 1
        self.path.append(self.pos.copy())

        # 3. Where is the tip now?
        offset, progress, _ = self.tube.project(self.pos)
        half_w = self.tube.width / 2.0

        crashed = abs(offset) >= half_w
        reached = progress >= self.tube.total_length - self.speed

        # 4. Reward
        reward = self.w_progress * (progress - self.progress)   # progress this step
        reward -= self.w_center * (offset / half_w) ** 2        # stay near center
        if crashed:
            reward -= self.crash_penalty
        if reached:
            reward += self.goal_bonus
        self.progress = progress

        # 5. End of episode?
        terminated = crashed or reached                          # real end of task
        truncated = (self.steps >= self.max_steps) and not terminated  # time limit

        info = {"crashed": crashed, "reached": reached, "progress": progress}

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        offset, _, i = self.tube.project(self.pos)

        # Heading error: tip heading minus tube heading
        d = self.tube.seg_vec[i]                       # (2,)
        tube_heading = np.arctan2(d[1], d[0])
        err = wrap_angle(self.heading - tube_heading)

        # Points ahead in the tip's own frame: l_k = R(-h) (p_k - q)
        ahead = self.tube.points_ahead(i, self.n_ahead)    # (m, 2)
        rel = ahead - self.pos                             # (m, 2)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array([[c, s], [-s, c]])                    # R(-h), (2, 2)
        local = rel @ R.T                                  # (m, 2)

        obs = np.concatenate(
            [[offset / (self.tube.width / 2.0), err], local.ravel()]
        )                                                  # (2 + 2m,)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(7, 5))

        ax = self._ax
        ax.clear()
        ax.plot(*self.tube.left_wall.T, "b-", lw=2)
        ax.plot(*self.tube.right_wall.T, "b-", lw=2)
        ax.plot(*self.tube.centerline.T, "k--", lw=0.8)
        path = np.array(self.path)                     # (steps+1, 2)
        ax.plot(*path.T, "g-", lw=1.5)
        ax.plot(*self.pos, "ro", ms=6)
        ax.set_aspect("equal")
        ax.set_title(f"step {self.steps}")

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        # rgb_array: return the picture as an array, (height, width, 3)
        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ----------------------------------------------------------------------
# Quick test: random agent, one episode.  Run: python endoscope_env.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = EndoscopeEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print("first observation:", obs.shape, obs)

    total = 0.0
    while True:
        action = env.action_space.sample()             # random action, (1,)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break

    print(f"steps={env.steps}  total reward={total:.2f}  info={info}")
    plt.show()
    env.close()