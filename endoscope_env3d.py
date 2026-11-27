"""
endoscope_env3d.py — Gymnasium environment, 3D version.
An endoscope tip moves at constant speed through a random curved 3D tube.
The agent controls two turn angles: yaw (left/right) and pitch (up/down).

Notation (Latin letters only):
    q      : tip position, shape (3,)
    R      : tip orientation, rotation matrix, shape (3, 3)
             columns of R = tip axes in world coordinates:
             R[:, 0] = f (forward), R[:, 1] = l (left), R[:, 2] = u (up)
    p_k    : k-th centerline point, shape (3,)
    m      : number of points ahead in the observation
    r      : tube radius
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from tube3d import Tube3D


def rot_z(a: float) -> np.ndarray:
    """Rotation by angle a around the z axis (yaw). Positive a turns left."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    """Rotation by angle a around the y axis. rot_y(-a) tilts forward up."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Remove small numerical errors, so R stays a rotation matrix."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


class EndoscopeEnv3D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n_ahead: int = 5,             # m
        speed: float = 0.05,          # distance per step
        max_turn_deg: float = 10.0,   # largest turn per step, per axis
        max_steps: int = 400,         # time limit (truncation)
        w_progress: float = 10.0,
        w_center: float = 0.1,
        crash_penalty: float = 10.0,
        goal_bonus: float = 10.0,
        tube_kwargs: dict = None,     # extra Tube3D settings, e.g. width, curviness_range
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
        self.tube_kwargs = tube_kwargs or {}           # {} -> training defaults
        self.render_mode = render_mode

        # Action: [yaw, pitch], each in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Observation: offset (2) + tube direction (2) + m points ahead (3m)
        obs_size = 4 + 3 * n_ahead                     # 19 when m = 5
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )

        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tube = Tube3D(self.np_random, **self.tube_kwargs)

        self.pos = self.tube.centerline[0].copy()      # q, (3,)
        self.R = np.eye(3)                             # facing +x, left = +y, up = +z
        self.steps = 0
        _, _, self.progress, _ = self.tube.project(self.pos)
        self.path = [self.pos.copy()]

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        # 1. Turn in the tip's own frame: R <- R Rz(yaw) Ry(-pitch)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        yaw, pitch = a[0] * self.max_turn, a[1] * self.max_turn
        self.R = orthonormalize(self.R @ rot_z(yaw) @ rot_y(-pitch))

        # 2. Move forward: q <- q + speed f,  f = R[:, 0]
        self.pos = self.pos + self.speed * self.R[:, 0]
        self.steps += 1
        self.path.append(self.pos.copy())

        # 3. Where is the tip now?
        _, dist, progress, _ = self.tube.project(self.pos)
        r = self.tube.radius

        crashed = dist >= r
        reached = progress >= self.tube.total_length - self.speed

        # 4. Reward (same as 2D, with dist instead of |offset|)
        reward = self.w_progress * (progress - self.progress)
        reward -= self.w_center * (dist / r) ** 2
        if crashed:
            reward -= self.crash_penalty
        if reached:
            reward += self.goal_bonus
        self.progress = progress

        # 5. End of episode?
        terminated = crashed or reached
        truncated = (self.steps >= self.max_steps) and not terminated

        info = {"crashed": crashed, "reached": reached, "progress": progress}

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation: everything in the tip's own frame
    # ------------------------------------------------------------------
    def _get_obs(self):
        _, _, _, i = self.tube.project(self.pos)
        q, R = self.pos, self.R

        # Closest centerline point on segment i: C = A + u D
        A, D = self.tube.centerline[i], self.tube.seg_vec[i]
        u = np.clip(((q - A) @ D) / (D @ D), 0.0, 1.0)
        C = A + u * D                                   # (3,)

        # World vector -> tip frame:  x_local = R^T x_world
        # (3, 3)^T (3,) = (3,)   ->  [forward, left, up] components
        offset_local = R.T @ (q - C) / self.tube.radius     # (3,)
        dir_local = R.T @ (D / np.linalg.norm(D))           # (3,)

        # Points ahead: each row l_k = R^T (p_k - q)
        # in row form: (m, 3) @ (3, 3) = (m, 3)
        ahead = self.tube.points_ahead(i, self.n_ahead)     # (m, 3)
        local = (ahead - q) @ R                             # (m, 3)

        obs = np.concatenate([
            offset_local[1:],        # left, up offset (forward part is ~0)
            dir_local[1:],           # left, up part of the tube direction
            local.ravel(),           # 3m numbers
        ])                                                  # (4 + 3m,)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._fig is None:
            self._fig = plt.figure(figsize=(7, 6))
            self._ax = self._fig.add_subplot(projection="3d")

        ax = self._ax
        ax.clear()
        X, Y, Z = self.tube.wall_surface(n_around=16)
        ax.plot_wireframe(X[::3], Y[::3], Z[::3], color="lightblue", lw=0.5)
        ax.plot(*self.tube.centerline.T, "k--", lw=0.8)
        path = np.array(self.path)                      # (steps+1, 3)
        ax.plot(*path.T, "g-", lw=2)
        ax.scatter(*self.pos, c="red", s=30)

        mid = self.tube.centerline.mean(axis=0)
        half = np.ptp(self.tube.centerline, axis=0).max() / 2 + self.tube.radius
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_title(f"step {self.steps}")

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ----------------------------------------------------------------------
# Hand-made baseline (not RL): turn toward the 2nd point ahead
# ----------------------------------------------------------------------
def make_pursuit_3d(env):
    def pursuit_policy(obs):
        # obs[4:] = points ahead in tip frame, row k = [forward, left, up]
        lx, ly, lz = obs[7], obs[8], obs[9]              # 2nd point ahead
        yaw = np.arctan2(ly, lx)                         # turn left/right
        pitch = np.arctan2(lz, np.hypot(lx, ly))         # turn up/down
        a = np.array([yaw, pitch]) / env.max_turn
        return np.clip(a, -1, 1).astype(np.float32)      # (2,)
    return pursuit_policy


# ----------------------------------------------------------------------
# Quick test: python endoscope_env3d.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = EndoscopeEnv3D(render_mode="human")
    obs, info = env.reset(seed=42)
    print("first observation:", obs.shape)

    pursuit = make_pursuit_3d(env)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(pursuit(obs))
        total += reward
        if terminated or truncated:
            break

    print(f"pursuit: steps={env.steps}  total reward={total:.2f}  info={info}")
    plt.show()
    env.close()
