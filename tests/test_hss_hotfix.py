"""A controller-only fix leaves active tasks, inputs, protocol and budget intact."""

import os
import pytest

from philoagent.hss_experiment import prepare, read_json, write_json
from philoagent.hss_budget import process_identity
from philoagent.hss_hotfix import publish
from philoagent.cli import source_hash


def test_hotfix_preserves_live_worker_and_protocol_without_touching_its_files(tmp_path):
    root = tmp_path / "run"
    manifest = prepare(root, values={"DEFAULT_MODEL": "offline", "OPENAI_API_KEY": "not-serialized"})
    original_snapshot = root / "code_snapshot/philoagent/hss_hotfix.py"
    original_snapshot.write_text('# previous adapter version\n', encoding='utf8')
    manifest['adapter_sha256'] = source_hash(root / 'code_snapshot/philoagent')
    write_json(root / 'manifest.json', manifest)
    job_dir = root / 'jobs' / manifest['jobs'][0]['job_id']
    state = {'status': 'running', 'pid': os.getpid(), 'process_identity': process_identity(os.getpid())}
    write_json(job_dir / 'job_status.json', state)
    write_json(job_dir / 'paid_answer.json', {'answer': 'Keep the paid answer.'})
    before_budget = (root / 'budget/budget.json').read_bytes()
    evidence = publish(root, reason='Config parsing fix')
    after = read_json(root / 'manifest.json')
    assert after['protocol'] == manifest['protocol']
    assert after['jobs'] == manifest['jobs']
    assert read_json(job_dir / 'job_status.json') == state
    assert read_json(job_dir / 'paid_answer.json')['answer'] == 'Keep the paid answer.'
    assert (root / 'budget/budget.json').read_bytes() == before_budget
    assert len(evidence['live_workers_preserved']) == 1
    assert 'hss_hotfix.py' in evidence['changed_files']
    assert after['adapter_sha256'] == source_hash(root / after['adapter_snapshot'])
    assert publish(root, reason='Config parsing fix') == evidence


def test_hotfix_refuses_changed_experiment_inputs_before_writing(tmp_path):
    root = tmp_path / 'run'
    manifest = prepare(root, values={'DEFAULT_MODEL': 'offline', 'OPENAI_API_KEY': 'not-serialized'})
    relative = next(iter(manifest['input_hashes']))
    (root / relative).write_text('changed', encoding='utf8')
    with pytest.raises(ValueError, match='input changed'):
        publish(root, reason='Config parsing fix')
    assert not (root / 'revisions/repair-r5').exists()


def test_successive_hotfix_keeps_actual_old_worker_version_and_rejects_pid_reuse(tmp_path):
    from philoagent.hss_hotfix import _loaded_source
    old = {'pid': 42, 'process_identity': {'started_unix': 100},
           'loaded_adapter_revision': 'repair-r4', 'loaded_adapter_sha256': 'old4',
           'source_snapshot': '/snapshot/r4'}
    evidence = tmp_path / 'r5.json'
    write_json(evidence, {'live_workers_preserved': [old]})
    manifest = {'adapter_revision': 'repair-r5', 'adapter_sha256': 'new5', 'protocol': {},
                'amendments': [{'evidence': str(evidence)}]}
    state = {'pid': 42, 'process_identity': {'started_unix': 100}}
    assert _loaded_source(tmp_path, manifest, state, '/snapshot/r5')['loaded_adapter_revision'] == 'repair-r4'
    state['process_identity']['started_unix'] = 200
    assert _loaded_source(tmp_path, manifest, state, '/snapshot/r5')['loaded_adapter_revision'] == 'repair-r5'
