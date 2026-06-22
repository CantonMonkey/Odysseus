"""
topo_map.py — Semantic topological map for the Odysseus VLN agent.

Maintains a sparse graph of visited locations, each tagged with a room
label, the objects observed by the VLM, and the navigation step at which
it was created.  Provides:
  - add_node       : conditionally insert a new node
  - get_nodes_by_floor / has_explored_floor
  - find_room_node : nearest node with a matching room keyword
  - suggest_goal_direction : commonsense reasoning over the graph
  - summary        : compact textual summary for logging
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict


# ---------------------------------------------------------------------------
# Commonsense object → room mapping (English room labels to match VLM output)
# ---------------------------------------------------------------------------
OBJECT_ROOM_MAP: Dict[str, str] = {
    "床":    "bedroom",
    "衣柜":  "bedroom",
    "床头柜":"bedroom",
    "沙发":  "living_room",
    "电视":  "living_room",
    "椅子":  "living_room",
    "冰箱":  "kitchen",
    "灶台":  "kitchen",
    "马桶":  "bathroom",
    "浴缸":  "bathroom",
}

VALID_ROOMS = ("living_room", "bedroom", "hallway", "kitchen", "staircase", "bathroom", "other")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TopoEdge:
    """Undirected edge between two node indices."""
    a: int
    b: int
    dist: float


@dataclass
class TopoNode:
    pos: np.ndarray          # 3-D world position (x, y, z)
    floor: int               # 0 = ground (Y < 1.5 m), 1 = second (Y >= 1.5 m)
    room: str                # one of VALID_ROOMS
    objects_seen: List[str]  # object strings reported by the VLM
    step: int                # navigation step when node was created

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float32)


# ---------------------------------------------------------------------------
# TopoMap
# ---------------------------------------------------------------------------

class TopoMap:
    """Sparse semantic topological map built online during navigation."""

    EDGE_RADIUS = 5.0   # connect edges to nodes within this radius (m)
    FLOOR_THRESH = 1.5  # Y < threshold → ground floor (0), else floor 1

    def __init__(self, min_node_dist: float = 2.5):
        """
        Parameters
        ----------
        min_node_dist : float
            Minimum Euclidean distance (metres) between any two nodes.
            A new candidate is discarded if a node already exists within
            this radius.
        """
        self.min_node_dist = min_node_dist
        self.nodes: List[TopoNode] = []
        self.edges: List[TopoEdge] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _floor_from_pos(self, pos: np.ndarray) -> int:
        return 0 if float(pos[1]) < self.FLOOR_THRESH else 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(
        self,
        pos,
        room: str,
        objects_seen: List[str],
        step: int,
    ) -> bool:
        """Add a node at *pos* if no existing node is within min_node_dist.

        Also connects edges to every existing node within EDGE_RADIUS.

        Returns True if the node was inserted, False if it was suppressed.
        """
        pos = np.asarray(pos, dtype=np.float32)

        # Reject if too close to any existing node
        for existing in self.nodes:
            if self._dist(pos, existing.pos) < self.min_node_dist:
                return False

        floor = self._floor_from_pos(pos)
        if room not in VALID_ROOMS:
            room = "other"

        new_idx = len(self.nodes)
        node = TopoNode(
            pos=pos,
            floor=floor,
            room=room,
            objects_seen=list(objects_seen),
            step=step,
        )
        self.nodes.append(node)

        # Connect edges to nearby nodes
        for idx, other in enumerate(self.nodes[:-1]):  # exclude self
            d = self._dist(pos, other.pos)
            if d <= self.EDGE_RADIUS:
                self.edges.append(TopoEdge(a=new_idx, b=idx, dist=d))

        return True

    # ------------------------------------------------------------------

    def get_nodes_by_floor(self, floor: int) -> List[TopoNode]:
        """Return all nodes on *floor* (0 = ground, 1 = second)."""
        return [n for n in self.nodes if n.floor == floor]

    def has_explored_floor(self, floor: int) -> bool:
        """Return True if at least one node exists on *floor*."""
        return any(n.floor == floor for n in self.nodes)

    # ------------------------------------------------------------------

    def find_room_node(self, room_keyword: str) -> Optional[TopoNode]:
        """Return the first node whose room label contains *room_keyword*.

        Among all matching nodes, the one inserted earliest (lowest step)
        is preferred, which approximates the most-visited / most-certain
        location.
        """
        matches = [n for n in self.nodes if room_keyword in n.room]
        if not matches:
            return None
        # Return earliest-visited match as a stable reference
        return min(matches, key=lambda n: n.step)

    # ------------------------------------------------------------------

    def suggest_goal_direction(
        self,
        goal: str,
        robot_pos: np.ndarray,
    ) -> dict:
        """Use object-room commonsense to recommend a navigation action.

        Returns one of:
          {"action": "goto", "pos": np.ndarray}  — head to a known room node
          {"action": "go_upstairs"}               — ascend to unexplored floor 1
          {"action": "explore"}                   — keep exploring current floor
        """
        robot_pos = np.asarray(robot_pos, dtype=np.float32)

        # 1. Map goal object to a target room type
        target_room = OBJECT_ROOM_MAP.get(goal)

        if target_room is not None:
            # 2. Search topo_map for a node whose room matches
            matches = [n for n in self.nodes if target_room in n.room]
            if matches:
                # Pick the closest matching node
                closest = min(matches, key=lambda n: self._dist(robot_pos, n.pos))
                return {"action": "goto", "pos": closest.pos}

            # 3. If the target is typically upstairs and floor 1 is unexplored
            UPSTAIRS_ROOMS = {"bedroom", "bathroom"}
            if target_room in UPSTAIRS_ROOMS and not self.has_explored_floor(1):
                return {"action": "go_upstairs"}

        # 4. Also go upstairs for bedroom-related goals if floor 1 unseen
        bedroom_keywords = {"床", "衣柜", "床头柜"}
        if goal in bedroom_keywords and not self.has_explored_floor(1):
            return {"action": "go_upstairs"}

        # 5. Default: keep exploring
        return {"action": "explore"}

    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary: 'N nodes: 客厅×2, 走廊×1, ...'"""
        if not self.nodes:
            return "0 nodes"
        counts: Dict[str, int] = defaultdict(int)
        for n in self.nodes:
            counts[n.room] += 1
        parts = ", ".join(f"{room}×{cnt}" for room, cnt in sorted(counts.items()))
        return f"{len(self.nodes)} nodes: {parts}"

    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tm = TopoMap(min_node_dist=2.5)

    added = tm.add_node([0.0, 0.16, 0.0], "living_room", ["沙发", "电视"], step=0)
    assert added, "first node must be added"

    added = tm.add_node([1.0, 0.16, 0.0], "living_room", ["椅子"], step=5)
    assert not added, "node within min_node_dist must be rejected"

    added = tm.add_node([3.0, 0.16, 0.0], "hallway", [], step=10)
    assert added, "node beyond min_node_dist must be added"

    added = tm.add_node([0.0, 3.16, 0.0], "bedroom", ["床"], step=20)
    assert added, "upper-floor node must be added"

    assert tm.node_count == 3
    assert len(tm.get_nodes_by_floor(0)) == 2
    assert len(tm.get_nodes_by_floor(1)) == 1
    assert tm.has_explored_floor(0)
    assert tm.has_explored_floor(1)

    node = tm.find_room_node("hallway")
    assert node is not None and node.room == "hallway"

    direction = tm.suggest_goal_direction("床", np.array([0.0, 0.16, 0.0]))
    # floor 1 has a node now, so should suggest goto
    assert direction["action"] == "goto"

    print("summary:", tm.summary())
    print("All assertions passed.")
