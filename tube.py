"""
tube.py — Random curved 2D tube (simplified colon model).

The tube is a centerline (a chain of points) plus a corridor of constant width.
Each new Tube object has a different random shape: straight, one bend, S-shape, ...

Notation (Latin letters only):
    p_k   : k-th centerline point, shape (2,)
    a_k   : heading angle of the step that ends at p_k (radians), scalar
    d_k   : heading change at step k (radians), scalar
    s     : step length between two centerline points, scalar
    w     : tube width (wall to wall), scalar
    c     : curviness = standard deviation of d_k, scalar
    b     : smoothing factor in [0, 1), scalar
    N     : number of centerline points
"""

import numpy as np
import matplotlib as plt

class Tube:
    """Random curved 2D tube.

    Main data (all NumPy arrays):
        centerline : (N, 2)  points p_0 ... p_{N-1}
        tangents   : (N, 2)  unit direction of the tube at each point
        normals    : (N, 2)  unit vector 90 deg to the left of the tangent
        left_wall  : (N, 2)  p_k + (w/2) n_k
        right_wall : (N, 2)  p_k - (w/2) n_k
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
        max_heading_deg: float = 80.0,
        safety: float = 0.8,
    ):
        self.n_points = n_points
        self.step = step
        self.width = width

        # c: one random curviness value for this tube.
        self.curviness = rng.uniform(*curviness_range)

        # Max heading change per step, so that the inner wall does not fold.
        # Turn radius r = s / |d| must be > w/2   =>   |d| < 2s / w
        self.max_turn = safety * 2.0 * step / width

        self.max_heading = np.deg2rad(max_heading_deg)

        self.centerline = self._generate_centerline(rng, smoothing)  # (N, 2)
        self._compute_geometry()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_centerline(self, rng: np.random.Generator, b: float) -> np.ndarray:
        """Random walk on the heading angle.

        Input : rng (random generator), b (smoothing factor)
        Output: centerline points, shape (N, 2)

        Why a loop over k: each heading a_{k+1} depends on a_k and d_k,
        and the clipping makes the steps depend on each other.
        So the steps must run in order.
        """
        n, s, c = self.n_points, self.step, self.curviness

        points = np.zeros((n, 2))  # (N, 2), p_0 = (0, 0)
        a = 0.0                    # a_0 = 0  -> start in +x direction
        d = 0.0                    # d_0 = 0

        # sqrt(1 - b^2) keeps the std of d_k equal to c (AR(1) process).
        noise_scale = np.sqrt(1.0 - b**2) * c

        for k in range(n - 1):
            # Smooth heading change: d_k = b d_{k-1} + sqrt(1-b^2) * Normal(0, c)
            d = b * d + noise_scale * rng.standard_normal()
            d = np.clip(d, -self.max_turn, self.max_turn)

            # New heading, limited so the tube always moves forward (+x).
            a_new = np.clip(a + d, -self.max_heading, self.max_heading)
            d = a_new - a          # keep d consistent with the real change
            a = a_new

            # p_{k+1} = p_k + s [cos a_{k+1}, sin a_{k+1}]
            points[k + 1] = points[k] + s * np.array([np.cos(a), np.sin(a)])

        return points

    def _compute_geometry(self) -> None:
        """Compute tangents, normals, walls and arc length from the centerline."""
        p = self.centerline                                # (N, 2)

        # Segment vectors D_k = p_{k+1} - p_k, k = 0 ... N-2
        self.seg_vec = p[1:] - p[:-1]                      # (N-1, 2)
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)  # (N-1,)
        seg_dir = self.seg_vec / self.seg_len[:, None]     # (N-1, 2) unit vectors

        # Tangent at each point: average of the two neighbouring segments.
        t = np.zeros_like(p)                               # (N, 2)
        t[0] = seg_dir[0]
        t[-1] = seg_dir[-1]
        t[1:-1] = seg_dir[:-1] + seg_dir[1:]
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.tangents = t

        # Normal n_k = [-t_y, t_x]  (tangent rotated +90 deg, points left)
        self.normals = np.stack([-t[:, 1], t[:, 0]], axis=1)  # (N, 2)

        half_w = self.width / 2.0
        self.left_wall = p + half_w * self.normals         # (N, 2)
        self.right_wall = p - half_w * self.normals        # (N, 2)

        # arc_length[k] = sum of segment lengths up to point k
        self.arc_length = np.concatenate([[0.0], np.cumsum(self.seg_len)])  # (N,)

    # ------------------------------------------------------------------
    # Queries (the environment will use these)
    # ------------------------------------------------------------------
    @property
    def total_length(self) -> float:
        """Length of the whole centerline."""
        return float(self.arc_length[-1])

    def project(self, q: np.ndarray):
        """Find the closest centerline point to q.

        Input : q, tip position, shape (2,)
        Output: (offset, progress, index)
            offset   : signed distance to the centerline (left > 0, right < 0)
            progress : arc length of the closest point (how far along the tube)
            index    : index i of the closest segment (p_i -> p_{i+1})

        No loop: all N-1 segments are handled at once with NumPy.
        """
        A = self.centerline[:-1]                     # (M, 2), M = N-1, segment starts
        D = self.seg_vec                             # (M, 2), segment vectors

        # Projection parameter u_k = ((q - A_k) . D_k) / |D_k|^2, clipped to [0, 1]
        rel = q[None, :] - A                         # (M, 2)
        u = np.sum(rel * D, axis=1) / self.seg_len**2  # (M,)
        u = np.clip(u, 0.0, 1.0)

        # Closest point on each segment: C_k = A_k + u_k D_k
        closest = A + u[:, None] * D                 # (M, 2)
        dist = np.linalg.norm(q[None, :] - closest, axis=1)  # (M,)

        i = int(np.argmin(dist))

        # Sign from the 2D cross product D_i x (q - A_i): > 0 means q is on the left.
        cross = D[i, 0] * rel[i, 1] - D[i, 1] * rel[i, 0]
        offset = float(np.sign(cross) * dist[i])

        progress = float(self.arc_length[i] + u[i] * self.seg_len[i])
        return offset, progress, i

    def is_inside(self, q: np.ndarray) -> bool:
        """True if q is between the two walls: |offset| < w/2."""
        offset, _, _ = self.project(q)
        return abs(offset) < self.width / 2.0

    def points_ahead(self, index: int, m: int) -> np.ndarray:
        """Next m centerline points after segment `index`.

        Input : index (segment index from project), m (number of points)
        Output: shape (m, 2). Near the end, the last point repeats,
                so the output shape is always the same.
        """
        idx = np.arange(index + 1, index + 1 + m)
        idx = np.clip(idx, 0, self.n_points - 1)
        return self.centerline[idx]


# ----------------------------------------------------------------------
# Visual check: run "uv run python tube.py"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed=42)
    test_curviness = [0.0, 0.03, 0.06, 0.12]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, c in zip(axes.ravel(), test_curviness):
        # (c, c) range -> this tube gets exactly curviness c
        tube = Tube(rng, curviness_range=(c, c))

        # Random test points in the bounding box of the tube.
        lo = tube.centerline.min(axis=0) - tube.width
        hi = tube.centerline.max(axis=0) + tube.width
        pts = rng.uniform(lo, hi, size=(400, 2))          # (400, 2)
        inside = np.array([tube.is_inside(q) for q in pts])  # (400,) bool

        ax.plot(*tube.centerline.T, "k--", lw=1, label="centerline")
        ax.plot(*tube.left_wall.T, "b-", lw=2, label="left wall")
        ax.plot(*tube.right_wall.T, "r-", lw=2, label="right wall")
        ax.scatter(*pts[inside].T, s=6, c="green", label="inside")
        ax.scatter(*pts[~inside].T, s=6, c="lightcoral", label="outside")

        # Check points_ahead from the middle of the tube.
        _, _, i_mid = tube.project(tube.centerline[100])
        ahead = tube.points_ahead(i_mid, 5)
        ax.scatter(*ahead.T, s=40, c="orange", marker="x", label="points ahead")

        ax.set_title(f"curviness c = {c}, length = {tube.total_length:.2f}")
        ax.set_aspect("equal")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig("tube_check.png", dpi=120)
    plt.show()
