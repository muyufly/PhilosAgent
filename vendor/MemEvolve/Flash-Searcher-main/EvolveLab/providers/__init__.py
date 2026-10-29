# Modified for PhiloAgent (2026-10-29): state integration and runtime compatibility.
"""
Memory providers for different frameworks
"""

from importlib import import_module

__all__ = ["AgentKBProvider", "SkillWeaverProvider", "MobileEProvider", "ExpeLProvider"]

_PROVIDER_MODULES = {
    "AgentKBProvider": ".agent_kb_provider",
    "SkillWeaverProvider": ".skillweaver_provider",
    "MobileEProvider": ".mobilee_provider",
    "ExpeLProvider": ".expel_provider",
}


def __getattr__(name):
    """Load optional provider dependencies only when that export is requested."""
    module_name = _PROVIDER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    provider = getattr(import_module(module_name, __name__), name)
    globals()[name] = provider
    return provider


def __dir__():
    return sorted(set(globals()) | set(__all__))
