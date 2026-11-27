"""
tube3d.py — Random curved 3D tube with a circular cross-section.

Same idea as tube.py, with two heading angles instead of one.

Notation (Latin letters only):
    p_k   : k-th centerline point, shape (3,)
    y_k   : yaw angle (left/right, around the z axis), scalar
    e_k   : pitch angle (up/down, elevation), scalar
    s     : step length between two centerline points, scalar
    r     : tube radius (= width / 2), scalar
    c     : curviness = standard deviation of the angle changes, scalar
    b     : smoothing factor in [0, 1), scalar
    N     : number of centerline points

Frame along the curve (one per point, all unit vectors, shape (N, 3)):
    t_k : tangent, the direction of the tube
    n_k : normal, 90 deg from t_k ("left" at the start)
    v_k : binormal, v_k = t_k x n_k ("up" at the start)
"""

import numpy as np


class Tube3D:
    """Random curved 3D tube.

    Main data (all NumPy arrays):
        centerline : (N, 3)  points p_0 ... p_{N-1}
        tangents   : (N, 3)  t_k
        normals    : (N, 3)  n_k
        binormals  : (N, 3)  v_k
        arc_length : (N,)    distance along the centerline from p_0 to p_k
    """

    def __init__(
        self,
        rng: np.random.Generator,
        n_points: int = 200,
        step: float = 0.05,
        width: float = 0.5,
        curviness_range: tuple = (0.0, 0.08),
        smoothing: float = 0.95,
        max_yaw_deg: float = 75.0,
        max_pitch_deg: float = 75.0,
        safety: float = 0.8,
    ):
        self.n_points = n_points
        self.step = step
        self.width = width
        self.radius = width / 2.0

        self.curviness = rng.uniform(*curviness_range)

        # Total turn per step must stay below s / r, so the inner wall does not fold.
        # Each angle gets 1/sqrt(2) of the limit, so the combined turn stays below it.
        self.max_turn = safety * step / self.radius
        self.max_turn_each = self.max_turn / np.sqrt(2.0)

        self.max_yaw = np.deg2rad(max_yaw_deg)
        self.max_pitch = np.deg2rad(max_pitch_deg)

        self.centerline = self._generate_centerline(rng, smoothing)  # (N, 3)
        self._compute_geometry()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_centerline(self, rng: np.random.Generator, b: float) -> np.ndarray:
        """Random walk on two angles (yaw, pitch).

        Output: centerline points, shape (N, 3)

        Same loop as in 2D: each angle depends on the previous one,
        so the steps must run in order.
        """
        n, s, c = self.n_points, self.step, self.curviness

        points = np.zeros((n, 3))           # p_0 = (0, 0, 0)
        y, e = 0.0, 0.0                     # start in +x direction
        dy, de = 0.0, 0.0                   # angle changes

        noise_scale = np.sqrt(1.0 - b**2) * c
        lim = self.max_turn_each

        for k in range(n - 1):
            # Smooth changes, one AR(1) process per angle
            dy = np.clip(b * dy + noise_scale * rng.standard_normal(), -lim, lim)
            de = np.clip(b * de + noise_scale * rng.standard_normal(), -lim, lim)

            # Limit the angles, so the tube always moves forward (+x)
            y_new = np.clip(y + dy, -self.max_yaw, self.max_yaw)
            e_new = np.clip(e + de, -self.max_pitch, self.max_pitch)
            dy, de = y_new - y, e_new - e
            y, e = y_new, e_new

            # Unit direction from the two angles:
            # u = [cos e cos y, cos e sin y, sin e]
            u = np.array([np.cos(e) * np.cos(y), np.cos(e) * np.sin(y), np.sin(e)])
            points[k + 1] = points[k] + s * u

        return points

    def _compute_geometry(self) -> None:
        """Tangents, a parallel-transport frame, and arc length."""
        p = self.centerline                                   # (N, 3)

        self.seg_vec = p[1:] - p[:-1]                         # (N-1, 3)
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)   # (N-1,)
        seg_dir = self.seg_vec / self.seg_len[:, None]        # (N-1, 3)

        # Tangent at each point: average of the neighbouring segments
        t = np.zeros_like(p)
        t[0], t[-1] = seg_dir[0], seg_dir[-1]
        t[1:-1] = seg_dir[:-1] + seg_dir[1:]
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.tangents = t

        # Parallel transport: carry the normal along the curve with as little
        # twist as possible. At each point, remove the part of the previous normal
        # that lies along the new tangent, then normalize:
        #   n_k = normalize(n_{k-1} - (n_{k-1} . t_k) t_k)
        # Why a loop: n_k depends on n_{k-1}.
        n = np.zeros_like(p)
        start = np.array([0.0, 1.0, 0.0])                     # "left" at the start
        n[0] = start - (start @ t[0]) * t[0]
        n[0] /= np.linalg.norm(n[0])
        for k in range(1, len(p)):
            m = n[k - 1] - (n[k - 1] @ t[k]) * t[k]
            n[k] = m / np.linalg.norm(m)
        self.normals = n
        self.binormals = np.cross(t, n)                       # v_k = t_k x n_k, (N, 3)

        self.arc_length = np.concatenate([[0.0], np.cumsum(self.seg_len)])  # (N,)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def total_length(self) -> float:
        return float(self.arc_length[-1])

    def project(self, q: np.ndarray):
        """Find the closest centerline point to q.

        Input : q, tip position, shape (3,)
        Output: (offset, dist, progress, index)
            offset   : (2,) offset from the centerline in the local frame,
                       [component along n_i, component along v_i]
            dist     : distance to the centerline, scalar (= |offset|)
            progress : arc length of the closest point
            index    : index i of the closest segment (p_i -> p_{i+1})

        Same segment math as in 2D, now with 3D vectors.
        """
        A = self.centerline[:-1]                              # (M, 3), M = N-1
        D = self.seg_vec                                      # (M, 3)

        rel = q[None, :] - A                                  # (M, 3)
        u = np.clip(np.sum(rel * D, axis=1) / self.seg_len**2, 0.0, 1.0)  # (M,)
        closest = A + u[:, None] * D                          # (M, 3)
        dist_all = np.linalg.norm(q[None, :] - closest, axis=1)  # (M,)

        i = int(np.argmin(dist_all))
        diff = q - closest[i]                                 # (3,)

        # Local offset: project diff on the frame of the nearer end point
        j = i if u[i] < 0.5 else i + 1
        offset = np.array([diff @ self.normals[j], diff @ self.binormals[j]])  # (2,)

        progress = float(self.arc_length[i] + u[i] * self.seg_len[i])
        return offset, float(dist_all[i]), progress, i

    def is_inside(self, q: np.ndarray) -> bool:
        """True if the distance to the centerline is below the radius."""
        _, dist, _, _ = self.project(q)
        return dist < self.radius

    def points_ahead(self, index: int, m: int) -> np.ndarray:
        """Next m centerline points after segment `index`, shape (m, 3)."""
        idx = np.clip(np.arange(index + 1, index + 1 + m), 0, self.n_points - 1)
        return self.centerline[idx]

    def wall_surface(self, n_around: int = 24):
        """Points on the tube wall, for drawing.

        Output: X, Y, Z, each of shape (N, n_around)
            wall point (k, j) = p_k + r (cos a_j n_k + sin a_j v_k)
        """
        a = np.linspace(0, 2 * np.pi, n_around)               # (n_around,)
        ca, sa = np.cos(a), np.sin(a)
        # (N, 1, 3) + r * ((n_around, 1) * (N, 1, 3) ...) -> (N, n_around, 3)
        ring = (ca[None, :, None] * self.normals[:, None, :]
                + sa[None, :, None] * self.binormals[:, None, :])
        surf = self.centerline[:, None, :] + self.radius * ring
        return surf[..., 0], surf[..., 1], surf[..., 2]


# ----------------------------------------------------------------------
# Checks: run "uv run python tube3d.py"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed=42)
    test_curviness = [0.0, 0.03, 0.06, 0.12]

    fig = plt.figure(figsize=(14, 10))
    for plot_k, c in enumerate(test_curviness):
        tube = Tube3D(rng, curviness_range=(c, c))

        # --- Numeric checks -------------------------------------------------
        # 1. Frame is orthonormal: t.n = 0, t.v = 0, n.v = 0
        dots = np.abs(np.concatenate([
            np.sum(tube.tangents * tube.normals, axis=1),
            np.sum(tube.tangents * tube.binormals, axis=1),
            np.sum(tube.normals * tube.binormals, axis=1),
        ]))

        # 2. Smallest turn radius must be larger than the tube radius
        dirs = tube.seg_vec / tube.seg_len[:, None]
        cos_turn = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1, 1)
        max_turn = np.max(np.arccos(cos_turn))
        min_radius = tube.step / max_turn if max_turn > 0 else np.inf

        # 3. Points at a known offset from the centerline:
        #    q = p_k + d (cos a n_k + sin a v_k). Inside if d < r.
        n_test, correct, max_err = 300, 0, 0.0
        for _ in range(n_test):
            k = rng.integers(5, tube.n_points - 5)
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0, 1.5 * tube.radius)
            q = tube.centerline[k] + d * (np.cos(a) * tube.normals[k]
                                          + np.sin(a) * tube.binormals[k])
            offset, dist, _, _ = tube.project(q)
            correct += (dist < tube.radius) == (d < tube.radius)
            max_err = max(max_err, abs(dist - d))

        print(f"c = {c:4.2f} | frame max |dot| = {dots.max():.1e} | "
              f"min turn radius = {min_radius:.3f} (tube r = {tube.radius}) | "
              f"inside test {correct}/{n_test} | max dist error = {max_err:.4f}")

        # --- Drawing --------------------------------------------------------
        ax = fig.add_subplot(2, 2, plot_k + 1, projection="3d")
        X, Y, Z = tube.wall_surface()
        ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.25, linewidth=0)
        ax.plot(*tube.centerline.T, "k--", lw=1)

        _, _, _, i_mid = tube.project(tube.centerline[100])
        ahead = tube.points_ahead(i_mid, 5)                   # (5, 3)
        ax.scatter(*ahead.T, c="orange", marker="x", s=40)

        # equal axis scale, so circles look like circles
        mid = tube.centerline.mean(axis=0)
        half = np.ptp(tube.centerline, axis=0).max() / 2 + tube.radius
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"curviness c = {c}, length = {tube.total_length:.2f}")

    plt.tight_layout()
    plt.savefig("tube3d_check.png", dpi=110)
    plt.show()
