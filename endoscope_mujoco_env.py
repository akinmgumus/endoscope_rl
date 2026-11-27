"""
endoscope_mujoco_env.py — Gymnasium environment with MuJoCo physics.

A bendable endoscope is pushed from its base at constant speed into a random
curved 3D tube. The body is passive and slides along the wall. The agent bends
only the tip (bending section) and tries to reach the end with little wall contact.

Main difference from endoscope_env3d.py:
    - the tip does not move by a formula, MuJoCo simulates it
    - touching the wall does NOT end the episode; it gives a penalty
      proportional to the contact force. The tip force has a much larger
      weight than the body force: the tip can perforate the wall, and the agent
      can change the tip contact by steering. The body is pushed from the base,
      so steering changes its contact only a little.

Notation (Latin letters only):
    q      : tip position, shape (3,)
    R      : tip orientation (rotation matrix of the last link), shape (3, 3)
    F_tip  : normal contact force between the bending section and the wall, scalar
    F_body : normal contact force between the rest of the endoscope and the wall
    r      : tube radius
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from tube3d import Tube3D
from mujoco_model import ModelParams, build_xml


class EndoscopeMujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        n_ahead: int = 5,
        frame_skip: int = 25,            # physics steps per env step -> 0.05 s
        max_steps: int = 250,            # time limit
        curviness_range=(0.0, 0.08),
        w_progress: float = 10.0,
        w_center: float = 0.1,
        w_tip: float = 2.0,              # penalty weight for tip contact force
        w_body: float = 0.1,             # penalty weight for body contact force
        force_ref: float = 1.0,          # force scale for normalization
        goal_bonus: float = 10.0,
        escape_penalty: float = 100.0,   # large: escaping must never pay off
        params: ModelParams = None,
        render_mode=None,
    ):
        self.p = params or ModelParams()
        self.n_ahead = n_ahead
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.curviness_range = curviness_range
        self.w_progress = w_progress
        self.w_center = w_center
        self.w_tip = w_tip
        self.w_body = w_body
        self.force_ref = force_ref
        self.goal_bonus = goal_bonus
        self.escape_penalty = escape_penalty
        self.render_mode = render_mode

        # Action: [yaw, pitch] target of the bending section, each in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Observation: offset (2) + tube direction (2) + m points ahead (3m)
        #              + bending angles (2) + tip and body contact force (2)
        obs_size = 4 + 3 * n_ahead + 4                 # 23 when m = 5
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Set by reset()
        self.model = None
        self.data = None
        self._renderer = None

    # ------------------------------------------------------------------
    # reset: new tube -> new MuJoCo model
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = self.p
        self.tube = Tube3D(self.np_random, n_points=p.n_points, step=p.tube_step,
                           width=p.tube_width, curviness_range=self.curviness_range)

        self.model = mujoco.MjModel.from_xml_string(build_xml(self.tube, p))
        self.data = mujoco.MjData(self.model)
        if self._renderer is not None:              # model changed -> new renderer
            self._renderer.close()
            self._renderer = None

        # Indices used every step
        self._tip_site = self.model.site("tip").id
        self._tip_body = self.model.body(f"link{p.n_links - 1}").id
        bend_ids = range(p.n_links - p.n_bend, p.n_links)
        self._yaw_q = [self.model.jnt_qposadr[self.model.joint(f"yaw{i}").id] for i in bend_ids]
        self._pitch_q = [self.model.jnt_qposadr[self.model.joint(f"pitch{i}").id] for i in bend_ids]
        self._link_geoms = set(
            g for g in range(self.model.ngeom) if self.model.geom_contype[g] == 1
        )
        tip_bodies = {self.model.body(f"link{i}").id for i in bend_ids}
        self._tip_geoms = set(
            g for g in self._link_geoms if self.model.geom_bodyid[g] in tip_bodies
        )

        self.data.ctrl[0] = p.insert_speed            # constant insertion speed
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        _, _, self.progress, _ = self.tube.project(self._tip_pos())
        self.path = [self._tip_pos()]
        self.f_tip, self.f_body = 0.0, 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        p = self.p
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # 1. Bending targets: same angle on every joint of the bending section
        #    ctrl = [insert, yaw_i, pitch_i, yaw_i+1, pitch_i+1, ...]
        self.data.ctrl[1::2] = a[0] * p.max_bend
        self.data.ctrl[2::2] = a[1] * p.max_bend

        # 2. Simulate frame_skip physics steps; keep the largest contact forces
        f_tip, f_body = 0.0, 0.0
        for _ in range(self.frame_skip):          # loop over physics steps
            mujoco.mj_step(self.model, self.data)
            ft, fb = self._contact_forces()
            f_tip, f_body = max(f_tip, ft), max(f_body, fb)
        self.f_tip, self.f_body = f_tip, f_body
        self.steps += 1

        # 3. Where is the tip now?
        q = self._tip_pos()
        self.path.append(q)
        _, dist, progress, _ = self.tube.project(q)
        r = self.tube.radius

        reached = progress >= self.tube.total_length - 0.05
        escaped = dist > r + 0.1                  # tip pushed through the wall cage

        # 4. Reward
        reward = self.w_progress * (progress - self.progress)
        reward -= self.w_center * (min(dist, r) / r) ** 2
        reward -= self.w_tip * (f_tip / self.force_ref)
        reward -= self.w_body * (f_body / self.force_ref)
        if reached:
            reward += self.goal_bonus
        if escaped:
            reward -= self.escape_penalty
        self.progress = progress

        terminated = reached or escaped
        truncated = (self.steps >= self.max_steps) and not terminated
        info = {"reached": reached, "crashed": escaped, "progress": progress,
                "f_tip": f_tip, "f_body": f_body}
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site].copy()          # (3,)

    def _tip_R(self) -> np.ndarray:
        # xmat: body frame -> world. Column 0 = forward, 1 = left, 2 = up
        return self.data.xmat[self._tip_body].reshape(3, 3).copy()   # (3, 3)

    def _contact_forces(self):
        """Normal contact forces with the wall: (bending section, rest of body)."""
        f_tip, f_body = 0.0, 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):            # loop over active contacts
            c = self.data.contact[i]
            g = c.geom1 if c.geom1 in self._link_geoms else c.geom2
            if (c.geom1 in self._link_geoms) == (c.geom2 in self._link_geoms):
                continue                           # not an endoscope-wall contact
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            if g in self._tip_geoms:
                f_tip += abs(f6[0])                # f6[0] = normal force
            else:
                f_body += abs(f6[0])
        return f_tip, f_body

    def bend_angles(self) -> np.ndarray:
        """Mean yaw and pitch of the bending joints, shape (2,)."""
        qpos = self.data.qpos
        return np.array([np.mean(qpos[self._yaw_q]), np.mean(qpos[self._pitch_q])])

    def _get_obs(self):
        q, R = self._tip_pos(), self._tip_R()
        _, _, _, i = self.tube.project(q)

        A, D = self.tube.centerline[i], self.tube.seg_vec[i]
        u = np.clip(((q - A) @ D) / (D @ D), 0.0, 1.0)
        C = A + u * D                                        # closest centerline point

        offset_local = R.T @ (q - C) / self.tube.radius      # (3,)
        dir_local = R.T @ (D / np.linalg.norm(D))            # (3,)
        ahead = self.tube.points_ahead(i, self.n_ahead)      # (m, 3)
        local = (ahead - q) @ R                              # (m, 3)

        obs = np.concatenate([
            offset_local[1:], dir_local[1:], local.ravel(),
            self.bend_angles() / self.p.max_bend,            # (2,)
            [self.f_tip / self.force_ref, self.f_body / self.force_ref],   # (2,)
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Drawing with the MuJoCo renderer
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            c = self.tube.centerline
            self._cam.lookat[:] = c.mean(axis=0)
            self._cam.distance = 1.6 * np.ptp(c, axis=0).max() + 1.0
            self._cam.azimuth = 90
            self._cam.elevation = -35
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ----------------------------------------------------------------------
# Hand-made baselines
# ----------------------------------------------------------------------
def make_straight(env):
    def policy(obs):
        return np.zeros(2, dtype=np.float32)               # never bend
    return policy


def make_pursuit_mujoco(env, k: int = 2, gain: float = 1.0):
    """Bend the tip toward the k-th point ahead.

    The action sets the bending angle relative to the body, so the rule adds
    the angle error to the current bending angle:
        new target = current bend + gain * (angle to the point) / n_bend
    """
    j = 4 + 3 * (k - 1)

    def policy(obs):
        lx, ly, lz = obs[j], obs[j + 1], obs[j + 2]
        yaw_err = np.arctan2(ly, lx)
        pitch_err = np.arctan2(lz, np.hypot(lx, ly))
        bend_now = obs[-4:-2] * env.p.max_bend              # current mean angles (rad)
        # pitch joint axis is +y: positive angle tilts forward DOWN, so pitch up = negative
        target = bend_now + gain * np.array([yaw_err, -pitch_err]) / env.p.n_bend
        return np.clip(target / env.p.max_bend, -1, 1).astype(np.float32)
    return policy


# ----------------------------------------------------------------------
# Quick test: python endoscope_mujoco_env.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    env = EndoscopeMujocoEnv()
    for name, maker in [("straight", make_straight), ("pursuit", make_pursuit_mujoco)]:
        obs, _ = env.reset(seed=0)
        pol = maker(env)
        total, ft, t0 = 0.0, [], time.time()
        while True:
            obs, r, terminated, truncated, info = env.step(pol(obs))
            total += r
            ft.append(info["f_tip"])
            if terminated or truncated:
                break
        ft = np.array(ft)
        print(f"{name:9}: steps={env.steps} return={total:7.2f} reached={info['reached']} "
              f"tip contact {100 * np.mean(ft > 0.01):.1f} % of steps, "
              f"max tip force={ft.max():.2f} ({time.time() - t0:.2f} s)")
