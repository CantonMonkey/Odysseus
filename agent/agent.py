"""
agent/agent.py — OdysseusAgent: unified plug-and-play entry point.

Examples:
    # Zero-config (auto-detect from environment variables)
    from agent import OdysseusAgent
    agent = OdysseusAgent()
    result = agent.run_task(env, "冰箱")

    # Explicit VLM backend (e.g. swap to Qwen-VL)
    from agent.backends import VLLMBackend
    agent = OdysseusAgent(vlm=VLLMBackend(base_url="http://localhost:8000/v1", model="Qwen-VL"))

    # Custom skill — no core-code change needed
    from agent.skill_registry import skill

    @skill("spiral_search", termination="timeout")
    def spiral_search(env, nav_state): ...

    agent = OdysseusAgent(extra_skills=[spiral_search])
"""
import os
from agent.vlm_base import VLMBackend


def _auto_detect_backend() -> VLMBackend:
    """Mirror the priority chain: vLLM → local → API → rule."""
    from agent.backends import VLLMBackend, InternVL3Backend, AnthropicBackend, RuleBasedBackend
    if os.environ.get("VLN_VLLM_BASE"):
        return VLLMBackend()
    if os.environ.get("VLN_LOCAL_MODEL"):
        return InternVL3Backend()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicBackend()
    return RuleBasedBackend()


class OdysseusAgent:
    """Plug-and-play VLN agent — swap VLM or add skills without touching core code."""

    def __init__(self, vlm: VLMBackend = None, extra_skills: list = []):
        self.vlm = vlm or _auto_detect_backend()
        for fn in extra_skills:
            name = getattr(fn, "_skill_name", fn.__name__)
            from agent.skill_registry import _REGISTRY
            _REGISTRY[name] = {"fn": fn, "termination": "waypoint_reached"}

    def run_task(self, env, goal: str, **kwargs) -> dict:
        """Run a navigation episode. Returns nav_state dict with done/step_count."""
        from agent.loop import run_task
        return run_task(env, goal, llm_perceive=self.vlm.perceive, **kwargs)

    def parse_goal(self, user_input: str) -> "str | None":
        """Extract navigation target from natural language using the active backend."""
        return self.vlm.parse_goal(user_input)
