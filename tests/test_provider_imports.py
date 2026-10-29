"""Check real lightweight imports without installing unrelated provider stacks."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "MemEvolve" / "Flash-Searcher-main"


class ProviderImportTests(unittest.TestCase):
    def run_isolated(self, script):
        result = subprocess.run(
            [sys.executable, "-I", "-c", "import sys\nsys.path.insert(0, " + repr(str(UPSTREAM)) + ")\n" + textwrap.dedent(script)],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lightweight_import_does_not_touch_optional_provider_dependencies(self):
        self.run_isolated("""
            import importlib.abc
            forbidden = {"sentence_transformers", "torch", "sklearn", "jieba"}
            class RejectOptionalImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in forbidden:
                        raise AssertionError("Unexpected optional dependency: " + fullname)
            sys.meta_path.insert(0, RejectOptionalImports())
            from EvolveLab.providers.lightweight_memory_provider import LightweightMemoryProvider
            from EvolveLab.base_memory import BaseMemoryProvider
            import EvolveLab.providers as providers
            assert issubclass(LightweightMemoryProvider, BaseMemoryProvider)
            for name in ("agent_kb_provider", "skillweaver_provider", "mobilee_provider", "expel_provider"):
                assert "EvolveLab.providers." + name not in sys.modules
            assert not forbidden.intersection(sys.modules)
            assert "AgentKBProvider" in dir(providers)
        """)

    def test_existing_public_exports_resolve_lazily(self):
        self.run_isolated("""
            import types
            import EvolveLab.providers as providers
            exports = {"AgentKBProvider": "agent_kb_provider", "SkillWeaverProvider": "skillweaver_provider",
                       "MobileEProvider": "mobilee_provider", "ExpeLProvider": "expel_provider"}
            assert set(providers.__all__) == set(exports)
            for name, module_name in exports.items():
                assert name not in vars(providers)
                # Stand in only for an optional provider's dependency-heavy module.
                module = types.ModuleType("EvolveLab.providers." + module_name)
                provider_class = type(name, (), {})
                setattr(module, name, provider_class)
                sys.modules[module.__name__] = module
                namespace = {}
                exec("from EvolveLab.providers import " + name, namespace)
                assert namespace[name] is provider_class
                assert getattr(providers, name) is provider_class
            try:
                providers.UnknownProvider
            except AttributeError:
                pass
            else:
                raise AssertionError("Missing export must raise AttributeError")
        """)


if __name__ == "__main__":
    unittest.main()
