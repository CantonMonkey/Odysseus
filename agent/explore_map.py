"""
explore_map.py — Online exploration map for VLN agent.

Replaces the pre-computed semantic_map.  The robot builds this map
incrementally as it moves:

  grid  : 2D uint8 — 0=UNKNOWN, 1=EXPLORED
  value : 2D float32 — accumulated VLM target-confidence scores

Frontier detection returns EXPLORED cells adjacent to UNKNOWN cells.
The agent navigates to the frontier with the highest accumulated VLM value.
"""

import numpy as np
from typing import List, Optional, Tuple

UNKNOWN  = 0
EXPLORED = 1

EYE_HEIGHT   = 1.0   # must match habitat_env.py
MAX_DEPTH    = 5.0   # metres, ignore deeper readings
SCORE_ALPHA  = 0.4   # EMA weight when updating value map

VLM_CALL_INTERVAL = 8  # call VLM every N navigation steps


class ExploreMap:
    """
    Top-down 2D map in the XZ plane.

    Parameters
    ----------
    resolution : metres per cell (default 0.1 m)
    size       : total map width/height in metres (default 40 m)
    """

    def __init__(self, resolution: float = 0.1, size: float = 40.0):
        self.res = resolution
        self.n   = int(size / resolution)
        self.off = size / 2.0            # world origin → map centre

        self.grid  = np.zeros((self.n, self.n), dtype=np.uint8)
        self.value = np.zeros((self.n, self.n), dtype=np.float32)

    # ── coordinate helpers ──────────────────────────────────────────────

    def _w2g(self, wx: float, wz: float) -> Tuple[int, int]:
        i = int((wx + self.off) / self.res)
        j = int((wz + self.off) / self.res)
        return i, j

    def _g2w(self, i: int, j: int) -> Tuple[float, float]:
        wx = i * self.res - self.off + self.res / 2
        wz = j * self.res - self.off + self.res / 2
        return wx, wz

    def _valid(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    # ── map update ──────────────────────────────────────────────────────

    def update(
        self,
        agent_pos: np.ndarray,
        R: np.ndarray,
        vlm_score: float,
        hfov: float = 90.0,
        view_radius: float = 4.0,
    ) -> None:
        """
        Mark the camera's field-of-view footprint as EXPLORED and
        update the VLM value map with *vlm_score*.

        agent_pos  : [x, y, z] robot base position in world frame
        R          : 3×3 rotation matrix (agent-local → world)
        vlm_score  : 0-1, how confident VLM is that target is in current view
        """
        # Forward direction projected onto XZ plane
        fwd = R @ np.array([0.0, 0.0, -1.0])
        fwd_xz = np.array([fwd[0], fwd[2]])
        norm = np.linalg.norm(fwd_xz)
        if norm > 1e-6:
            fwd_xz /= norm

        half_fov = np.radians(hfov / 2.0)
        ri, rj   = self._w2g(agent_pos[0], agent_pos[2])
        cells    = int(view_radius / self.res) + 1

        for di in range(-cells, cells + 1):
            for dj in range(-cells, cells + 1):
                i, j = ri + di, rj + dj
                if not self._valid(i, j):
                    continue
                dx, dz = di * self.res, dj * self.res
                dist   = np.sqrt(dx * dx + dz * dz)
                if dist < 0.01 or dist > view_radius:
                    continue
                # Angle from robot forward to this cell
                cos_a = float(np.dot(fwd_xz, np.array([dx, dz]) / dist))
                if cos_a < np.cos(half_fov):
                    continue
                self.grid[i, j] = EXPLORED
                # Exponential moving average
                self.value[i, j] = (
                    (1.0 - SCORE_ALPHA) * self.value[i, j]
                    + SCORE_ALPHA * vlm_score
                )

        # Robot's immediate neighbourhood is always explored
        for di in range(-2, 3):
            for dj in range(-2, 3):
                i2, j2 = ri + di, rj + dj
                if self._valid(i2, j2):
                    self.grid[i2, j2] = EXPLORED

    # ── frontier detection ───────────────────────────────────────────────

    def _frontier_mask(self) -> np.ndarray:
        """Boolean mask: EXPLORED cells with at least one UNKNOWN 4-neighbour."""
        unknown  = self.grid == UNKNOWN
        explored = self.grid == EXPLORED
        adj = (
            np.roll(unknown,  1, axis=0) | np.roll(unknown, -1, axis=0) |
            np.roll(unknown,  1, axis=1) | np.roll(unknown, -1, axis=1)
        )
        return explored & adj

    def frontiers(self) -> List[Tuple[int, int]]:
        """Return list of (i, j) grid indices that are frontiers."""
        mask = self._frontier_mask()
        rows, cols = np.where(mask)
        return list(zip(rows.tolist(), cols.tolist()))

    def best_frontier(
        self, robot_pos: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Return world-frame [x, y, z] of the most promising frontier.

        Score = VLM value + small proximity bonus (avoid pointlessly distant goals).
        """
        cells = self.frontiers()
        if not cells:
            return None

        ri, rj = self._w2g(robot_pos[0], robot_pos[2])

        best_score = -1.0
        best_ij    = None
        for i, j in cells:
            v    = float(self.value[i, j])
            dist = np.sqrt((i - ri) ** 2 + (j - rj) ** 2) * self.res
            # Proximity bonus: prefer frontiers within 6 m
            prox = max(0.0, (6.0 - dist) / 6.0) * 0.05
            s    = v + prox
            if s > best_score:
                best_score = s
                best_ij    = (i, j)

        if best_ij is None:
            return None

        wx, wz = self._g2w(*best_ij)
        return np.array([wx, robot_pos[1], wz], dtype=np.float32)

    def explored_fraction(self) -> float:
        return float(np.mean(self.grid == EXPLORED))
