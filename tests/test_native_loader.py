"""Exercise the production WebWalkerQA loader without an LLM or network calls."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from philoagent.orientation import bootstrap


class NativeLoaderTest(unittest.TestCase):
    def test_actual_runner_loads_oriented_lightweight_provider(self):
        project = Path(__file__).resolve().parents[1]
        upstream = project / "vendor" / "MemEvolve" / "Flash-Searcher-main"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap(root / "lineage", "lightweight_memory", ["A provisional method."])
            env = dict(os.environ)
            env.update({
                "PHILO_LINEAGE_ROOT": str(root / "lineage"),
                "PHILO_EVIDENCE_DIR": str(root / "evidence"),
                "PYTHONPATH": os.pathsep.join([str(project), str(upstream)]),
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            script = """
import socket
def forbidden(*args, **kwargs):
    raise AssertionError('Network access is forbidden in this offline test')
socket.socket.connect = forbidden
socket.create_connection = forbidden
from run_flash_searcher_webwalkerqa import load_memory_provider
from EvolveLab.memory_types import MemoryRequest, MemoryStatus
provider = load_memory_provider('lightweight_memory')
assert provider.enable_longterm_provision
response = provider.provide_memory(MemoryRequest('offline question', '', MemoryStatus.BEGIN))
assert any('A provisional method.' in str(item.content) for item in response.memories)
assert response.total_count == len(response.memories)
try:
    load_memory_provider('not_a_registered_provider')
except ValueError:
    pass
else:
    raise AssertionError('An experimental run silently dropped its memory provider')
print('NATIVE_LOADER_OK')
"""
            result = subprocess.run([sys.executable, "-c", script], env=env, cwd=root,
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("NATIVE_LOADER_OK", result.stdout)
            events = [json.loads(line) for path in (root / "evidence").glob("*.jsonl")
                      for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["operation"] == "provide_memory" for event in events))
            self.assertTrue(all(event["disk_matches_loaded_code"] for event in events))


if __name__ == "__main__":
    unittest.main()
