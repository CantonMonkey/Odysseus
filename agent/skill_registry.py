"""
skill_registry.py — @skill decorator + global registry.

Usage:
    from agent.skill_registry import skill

    @skill("follow_path")
    def follow_path(env, nav_state): ...

    # Custom skill (no core-code change needed):
    @skill("spiral_search", termination="timeout")
    def spiral_search(env, nav_state): ...
"""

_REGISTRY: dict = {}


def skill(name: str, *, termination: str = "waypoint_reached"):
    """Register a navigation skill by name.

    Args:
        name:        Skill name as returned by VLM (must match VLM prompt enum).
        termination: Condition that ends this skill:
                       waypoint_reached | timeout | vlm_call
    """
    def decorator(fn):
        _REGISTRY[name] = {"fn": fn, "termination": termination}
        return fn
    return decorator


def get_skill(name: str):
    """Return skill function for *name*, or None if not registered."""
    entry = _REGISTRY.get(name)
    return entry["fn"] if entry else None


def skill_names() -> list:
    """Return list of registered skill names (for VLM prompt enum)."""
    return list(_REGISTRY.keys())


def registered_skill_map() -> dict:
    """Return {name: fn} dict compatible with loop.py skill_map."""
    return {k: v["fn"] for k, v in _REGISTRY.items()}
