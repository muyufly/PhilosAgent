"""The real native creator must not report an empty creation as successful."""

import subprocess
import sys


def test_native_creation_failure_is_reported_at_creation_and_keeps_original_cause(tmp_path):
    script = r'''
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import json, sys
from philoagent.cli import init_run
from philoagent.live_smoke import activate_workspace
from philoagent.hss_experiment import write_json, read_json
root=Path(sys.argv[1])
run=Path(init_run('offline-create', 'none', output_root=root)['run_dir'])
activate_workspace(run)
from MemEvolve.core.memory_evolver import MemoryEvolver
from MemEvolve.phases.memory_creator import MemorySystemCreator
work=run/'evolution/round_00';work.mkdir(parents=True)
config=work/'generated_system_2.json'
write_json(config, {'memory_type_info': {'enum_value': 'covenant_memory'}})
evolver=MemoryEvolver.__new__(MemoryEvolver)
evolver.work_dir=work;evolver.state_file=work/'state.json';evolver.state={'phases': {}}
try:
    from philoagent.hss_creation import guard_creation
    create=guard_creation(MemoryEvolver.create)
except ImportError:
    create=MemoryEvolver.create
with patch.object(MemorySystemCreator,'create_memory_system', return_value={
    'success':False,'error':'Provider file covenant_memory_provider.py already exists'}):
    result=create(evolver,config_path=str(config))
assert result['success'] is False, result
assert result['created']==[]
assert 'already exists' in result['error']
assert evolver.state['phases']['create']['completed'] is False
assert read_json(work/'created_system.json')['failed'][0]['error'].endswith('already exists')
with patch.object(MemorySystemCreator,'create_memory_system', return_value={'success':True}):
    result=create(evolver,config_path=str(config))
assert result['success'] is True and result['created']==['covenant_memory']
assert evolver.state['phases']['create']['completed'] is True
'''
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(tmp_path)],
                               capture_output=True, text=True, encoding="utf8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
