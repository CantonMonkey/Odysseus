"""
agent/backends — VLM backend implementations for OdysseusAgent.

All backends satisfy the VLMBackend Protocol (agent.vlm_base).
"""
from agent.backends.vllm_http     import VLLMBackend
from agent.backends.internvl3     import InternVL3Backend
from agent.backends.anthropic_api import AnthropicBackend
from agent.backends.rule_based    import RuleBasedBackend

__all__ = ["VLLMBackend", "InternVL3Backend", "AnthropicBackend", "RuleBasedBackend"]
