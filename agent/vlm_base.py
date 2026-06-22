"""
vlm_base.py — VLMBackend Protocol (PEP 544 structural subtyping).

Any class implementing perceive() + parse_goal() satisfies this interface
without explicitly subclassing — no inheritance required.
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class VLMBackend(Protocol):
    """Interface contract for all VLM backends used by OdysseusAgent."""

    def perceive(
        self,
        frame: np.ndarray,
        goal: str,
        annotated_frame: "np.ndarray | None" = None,
        n_waypoints: int = 0,
        context: "dict | None" = None,
    ) -> dict:
        """Analyse the current RGB frame and return a perception dict.

        Returns keys: target_visible, direction, confidence, room, relevance,
                      skill, reason, waypoint
        """
        ...

    def parse_goal(self, user_input: str) -> "str | None":
        """Extract navigation target from natural language. None on failure."""
        ...
