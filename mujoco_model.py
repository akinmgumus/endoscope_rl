"""
mujoco_model.py — Build a MuJoCo model (MJCF XML) from a random 3D tube.

Parts of the model:
    Tube wall : a cage of small capsules, in rings along the centerline.
                MuJoCo cannot collide with a hollow tube as one shape,
                so the wall is built from many small convex pieces.
    Endoscope : a chain of capsule links. Each link joins the previous one
                with two hinge joints (yaw around z, pitch around y).
                - body joints: passive, with a spring and damping
                - bending section (last links): position actuators, the agent sets the target angles
                - base: a slide joint along +x, pushed at constant speed (insertion)
    Gravity is off: the endoscope floats in the lumen.

Notation (Latin letters only):
    r   : tube radius
    L   : link length
    n   : number of links
    K   : number of capsules per wall ring
"""

from dataclasses import dataclass

import numpy as np

from tube3d import Tube3D


@dataclass
class ModelParams:
    # Tube
    n_points: int = 61              # centerline points -> tube length about 3.0
    tube_step: float = 0.05
    tube_width: float = 0.5         # r = 0.25
    ring_every: int = 2             # one wall ring every 2 centerline points
    ring_capsules: int = 16         # K
    wall_capsule_radius: float = 0.035
    # Endoscope
    n_links: int = 34               # n
    link_length: float = 0.1        # L  -> total length 3.4
    link_radius: float = 0.06
    density: float = 20.0           # light links: total mass about 1.4
    n_bend: int = 3                 # last 3 links form the bending section
    max_bend: float = 0.5           # largest target angle per bending joint (rad)
    body_stiffness: float = 0.3     # spring of the passive joints (N m / rad)
    body_damping: float = 0.05
    bend_kp: float = 3.0            # position actuator gain of the bending joints
    # Insertion
    insert_speed: float = 0.5       # base speed (units / s)
    # Simulation
    timestep: float = 0.002
    friction: float = 0.1           # low friction: wet lumen


def _wall_xml(tube: Tube3D, p: ModelParams) -> str:
    """Capsule cage. Output: XML lines for the worldbody."""
    lines = []
    R = tube.radius + p.wall_capsule_radius       # capsule centers sit outside the lumen
    a = np.linspace(0, 2 * np.pi, p.ring_capsules, endpoint=False)    # (K,)
    idx = np.arange(0, tube.n_points - 1, p.ring_every)

    # Why a loop: one ring per selected centerline point, K capsules per ring.
    for k in idx:
        k2 = min(k + p.ring_every, tube.n_points - 1)
        for j in range(p.ring_capsules):
            # wall point = p_k + R (cos a_j n_k + sin a_j v_k)
            d1 = np.cos(a[j]) * tube.normals[k] + np.sin(a[j]) * tube.binormals[k]
            d2 = np.cos(a[j]) * tube.normals[k2] + np.sin(a[j]) * tube.binormals[k2]
            s = tube.centerline[k] + R * d1
            e = tube.centerline[k2] + R * d2
            lines.append(
                f'<geom type="capsule" size="{p.wall_capsule_radius}" '
                f'fromto="{s[0]:.5f} {s[1]:.5f} {s[2]:.5f} {e[0]:.5f} {e[1]:.5f} {e[2]:.5f}" '
                f'class="wall"/>'
            )
    return "\n      ".join(lines)


def _endoscope_xml(p: ModelParams) -> tuple:
    """Nested chain of links. Output: (body XML, actuator XML)."""
    L = p.link_length
    start_x = -p.n_links * L - 0.05               # tip starts just before the tube entry

    # Build from the tip back to the base, so each link wraps the next one.
    inner = (f'<site name="tip" pos="{L:.4f} 0 0" size="0.01"/>')
    for i in reversed(range(p.n_links)):
        bend = i >= p.n_links - p.n_bend
        cls = "bend" if bend else "body"
        joints = ""
        if i > 0:                                  # link 0 sits on the base slide
            joints = (f'<joint name="yaw{i}" type="hinge" axis="0 0 1" class="{cls}"/>'
                      f'<joint name="pitch{i}" type="hinge" axis="0 1 0" class="{cls}"/>')
        pos = f"{L:.4f} 0 0" if i > 0 else "0 0 0"
        color = ' rgba="0.85 0.1 0.1 1"' if bend else ""        # bending section in red
        inner = (f'<body name="link{i}" pos="{pos}">{joints}'
                 f'<geom type="capsule" fromto="0 0 0 {L:.4f} 0 0" class="link"{color}/>'
                 f'{inner}</body>')

    base = (f'<body name="base" pos="{start_x:.4f} 0 0">'
            f'<joint name="insert" type="slide" axis="1 0 0" damping="0"/>'
            f'{inner}</body>')

    act = [f'<velocity name="insert" joint="insert" kv="200"/>']
    for i in range(p.n_links - p.n_bend, p.n_links):
        act.append(f'<position name="yaw{i}" joint="yaw{i}" kp="{p.bend_kp}" '
                   f'ctrlrange="{-p.max_bend} {p.max_bend}"/>')
        act.append(f'<position name="pitch{i}" joint="pitch{i}" kp="{p.bend_kp}" '
                   f'ctrlrange="{-p.max_bend} {p.max_bend}"/>')
    return base, "\n    ".join(act)


def build_xml(tube: Tube3D, p: ModelParams) -> str:
    """Full MJCF model as a string."""
    walls = _wall_xml(tube, p)
    body, actuators = _endoscope_xml(p)
    return f"""
<mujoco model="endoscope_tube">
  <compiler angle="radian"/>
  <option timestep="{p.timestep}" gravity="0 0 0" integrator="implicitfast"/>
  <visual><global offwidth="800" offheight="600"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.9 0.95" rgb2="0.45 0.55 0.65"
             width="256" height="256"/>
  </asset>
  <default>
    <default class="wall">
      <geom contype="2" conaffinity="1" rgba="0.3 0.5 0.9 0.18" friction="{p.friction} 0.005 0.0001"/>
    </default>
    <default class="link">
      <geom contype="1" conaffinity="2" size="{p.link_radius}" rgba="0.15 0.15 0.15 1"
            density="{p.density}" friction="{p.friction} 0.005 0.0001"/>
    </default>
    <default class="body">
      <joint stiffness="{p.body_stiffness}" damping="{p.body_damping}" armature="0.001"
             range="-0.8 0.8" limited="true"/>
    </default>
    <default class="bend">
      <joint stiffness="0" damping="{p.body_damping}" armature="0.001"
             range="-0.8 0.8" limited="true"/>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <camera name="side" pos="1.5 -4.5 2.5" xyaxes="1 0 0 0 0.5 1"/>
      {walls}
    {body}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


# ----------------------------------------------------------------------
# Physics test without learning:  python mujoco_model.py [--view]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curviness", type=float, default=0.06)
    args = parser.parse_args()

    p = ModelParams()
    rng = np.random.default_rng(args.seed)
    tube = Tube3D(rng, n_points=p.n_points, step=p.tube_step, width=p.tube_width,
                  curviness_range=(args.curviness, args.curviness))

    t0 = time.time()
    xml = build_xml(tube, p)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print(f"model: {model.ngeom} geoms, {model.njnt} joints, {model.nu} actuators, "
          f"compiled in {time.time() - t0:.3f} s")

    data.ctrl[0] = p.insert_speed                 # push the base at constant speed
    n_sim = int(8.0 / p.timestep)                 # 8 s of simulated time

    def tip_progress():
        q = data.site("tip").xpos
        _, dist, prog, _ = tube.project(q.copy())
        return prog, dist

    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for _ in range(n_sim):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(p.timestep)
    else:
        t0 = time.time()
        for k in range(n_sim):
            mujoco.mj_step(model, data)
            if k % 500 == 0:
                prog, dist = tip_progress()
                print(f"t={data.time:5.2f} s  tip progress={prog:5.2f} / "
                      f"{tube.total_length:.2f}  dist to center={dist:.3f}  "
                      f"contacts={data.ncon}")
        print(f"simulated 8 s in {time.time() - t0:.2f} s")
