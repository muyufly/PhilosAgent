"""Temporary-directory checks of original native checkpoint recovery."""
from pathlib import Path
import importlib.util
import json
import os
import shutil

import pytest

PROJECT=Path(__file__).resolve().parents[1]
ACTUAL=PROJECT/'runs/hss-business-formal-20261029-r01'
SCRIPT=ACTUAL/'native_recovery_20261030/resume_prepare.py'
FIXTURE=ACTUAL/'native_recovery_20261030/test_fixture'
spec=importlib.util.spec_from_file_location('hss_native_recovery_prepare',SCRIPT)
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

@pytest.fixture
def recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(helper,'process_owned',lambda state:False)
    jobs=['humanities-native-r3','humanities-full-r1']
    (tmp_path/'manifest.json').write_text(json.dumps({'jobs':[{'job_id':j} for j in jobs]}),encoding='utf-8')
    relatives=[
        'jobs/humanities-native-r3/evolution/round_00/base_logs/1.json',
        'jobs/humanities-native-r3/evolution/round_00/base_logs/2.json',
        'jobs/humanities-full-r1/evolution/round_01/eval_finalists/praxis_reliquary/8.json',
        'jobs/humanities-native-r3/workspace/EvolveLab/providers/lightweight_memory_provider.py',
        'jobs/humanities-full-r1/workspace/EvolveLab/providers/praxis_reliquary_provider.py',
        'budget/attempts/009028.json','budget/attempts/009079.json','budget/attempts/009089.json',
    ]
    for job in jobs:
        storage='praxis_reliquary' if job.endswith('r1') else 'lightweight_memory'
        relatives += [str(p.relative_to(FIXTURE)) for p in (FIXTURE/'jobs'/job/'workspace/storage'/storage).glob('*') if p.is_file()]
    for relative in relatives:
        src=FIXTURE/relative;dest=tmp_path/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    return tmp_path

def test_audit_does_not_mutate(recovery):
    before={str(p.relative_to(recovery)):helper.sha(p) for p in recovery.glob('**/*') if p.is_file()}
    result=helper.prepare(recovery,audit_only=True)
    assert result['resume_allowed'] is True
    assert result['phase_counts']=={'completed':1,'blocked_ingestion':2}
    after={str(p.relative_to(recovery)):helper.sha(p) for p in recovery.glob('**/*') if p.is_file()}
    assert before==after

def test_apply_rolls_back_only_credit_tail_and_reuses_paid_work(recovery):
    completed=recovery/'jobs/humanities-native-r3/evolution/round_00/base_logs/1.json'
    original_sha=helper.sha(completed)
    credit=recovery/'jobs/humanities-full-r1/workspace/storage/praxis_reliquary/credit_log.jsonl'
    credit_before=credit.read_bytes()
    result=helper.prepare(recovery)
    assert result['status']=='prepared' and result['resume_allowed']
    assert set(result['prepared_jobs'])=={'humanities-native-r3','humanities-full-r1'}
    assert credit.read_bytes()==b''.join(credit_before.splitlines(keepends=True)[:-1])
    assert helper.sha(completed)==original_sha
    for operation in result['operations']:
        current=json.loads((recovery/operation['path']).read_text(encoding='utf-8'))
        assert current['status']=='graded'
        assert not current.get('memory_ingestion_started')
        assert current['memory_ingestion_complete'] is False
        assert current['answer']==operation['record']['answer']
        receipt=current['native_ingestion_recovery']
        assert receipt['status']=='storage_restored_for_ingestion_retry'
        assert helper.sha(receipt['receipt_path'])==receipt['receipt_sha256']
        detailed=json.loads(Path(receipt['receipt_path']).read_text(encoding='utf-8'))
        assert detailed['task_identity']==current['task_identity']
        assert detailed['saved_grade_sha256']==helper._json_hash(current['grade'])
        for path,digest in detailed['storage_files_after'].items():assert helper.sha(path)==digest
        if current['provider']=='praxis_reliquary':
            assert current['api_usage']['unrecovered_failed_attempt_ids']==[9079]
            assert current['api_usage']['usage_complete'] is False

def test_praxis_newer_ledger_is_blocked_without_mutation(recovery):
    lessons=recovery/'jobs/humanities-full-r1/workspace/storage/praxis_reliquary/lessons.json'
    os.utime(lessons,(1793292000,1793292000))
    credit=recovery/'jobs/humanities-full-r1/workspace/storage/praxis_reliquary/credit_log.jsonl'
    before=credit.read_bytes()
    result=helper.prepare(recovery)
    assert not result['resume_allowed'] and result['blockers']
    assert credit.read_bytes()==before
    assert not (recovery/'native_recovery_20261030/preparations').exists()

def test_unknown_ingestion_is_blocked(recovery):
    task=recovery/'jobs/humanities-native-r3/evolution/round_00/base_logs/2.json'
    record=json.loads(task.read_text(encoding='utf-8'));record['task_execution_id']='different'
    task.write_text(json.dumps(record),encoding='utf-8')
    result=helper.prepare(recovery)
    assert not result['resume_allowed']
    assert any('judge' in e['reason'] for e in result['blockers'])

def test_complete_response_missing_usage_is_accounting_only():
    record={'task_execution_id':'x'}
    event={'task_execution_id':'x','attempt':1,'status':'error','error_type':'MissingOrInvalidUsage',
           'response_accepted_without_usage':True,'response_has_content':True,
           'response_finish_reasons':['stop'],'usage':None,
           'response':{'choices':[{'message':{'content':'A retained complete answer'},'finish_reason':'stop'}]}}
    usage,_=helper._usage(record,[event])
    assert usage['unrecovered_failed_request_count']==0
    assert usage['failed_request_count']==1
    assert usage['usage_complete'] is False

def test_accepted_response_links_prior_transport_error():
    record={'task_execution_id':'x'}
    events=[{'task_execution_id':'x','attempt':1,'status':'error','error_type':'APIConnectionError','usage':None},
            {'task_execution_id':'x','attempt':2,'status':'error','error_type':'MissingOrInvalidUsage',
             'response_accepted_without_usage':True,'response_has_content':True,
             'response_finish_reasons':['stop'],'recovered_attempts':[1],'usage':None,
             'response':{'choices':[{'message':{'content':'A retained complete answer'},'finish_reason':'stop'}]}}]
    usage,_=helper._usage(record,events)
    assert usage['unrecovered_failed_request_count']==0
    assert usage['failed_request_count']==2
    assert usage['unknown_usage_attempts']==2

def test_global_receipt_without_response_has_flags_uses_actual_response_evidence():
    record={'task_execution_id':'global'}
    events=[{'task_execution_id':'global','attempt':1,'status':'error','error_type':'APIConnectionError','usage':None},
            {'task_execution_id':'global','attempt':2,'status':'error','error_type':'MissingOrInvalidUsage',
             'response_accepted_without_usage':True,'response_finish_reasons':['stop'],
             'recovered_attempts':[1],'usage':None,
             'response':{'choices':[{'finish_reason':'stop','message':{'content':'Real retained answer'}}],'usage':None}}]
    before=json.loads(json.dumps(events))
    usage,_=helper._usage(record,events)
    assert usage['unrecovered_failed_attempt_ids']==[]
    assert usage['failed_request_count']==2
    assert usage['unknown_usage_attempts']==2
    assert usage['usage_complete'] is False
    assert events==before

@pytest.mark.parametrize('response',[
    None, {'choices':[]}, {'choices':[{'finish_reason':'stop','message':{'content':'   ','tool_calls':[]}}]},
    {'choices':[{'finish_reason':'length','message':{'content':'Real but truncated answer'}}]},
])
def test_global_receipt_accepted_flag_does_not_prove_usable_response(response):
    event={'task_execution_id':'global','attempt':2,'status':'error','error_type':'MissingOrInvalidUsage',
           'response_accepted_without_usage':True,'response_has_content':True,
           'response_finish_reasons':['stop'],'usage':None,'response':response}
    usage,_=helper._usage({'task_execution_id':'global'},[event])
    assert usage['unrecovered_failed_attempt_ids']==[2]
    assert usage['unknown_usage_attempts']==1
    assert usage['failed_request_count']==1

def test_global_receipt_actual_tool_response_is_usable():
    event={'task_execution_id':'global','attempt':2,'status':'error','error_type':'MissingOrInvalidUsage',
           'response_accepted_without_usage':True,'response_finish_reasons':['tool_calls'],
           'usage':None,'response':{'choices':[{'finish_reason':'tool_calls','message':{'content':None,
           'tool_calls':[{'id':'call_1','type':'function','function':{'name':'search','arguments':'{}'}}]}}]}}
    usage,_=helper._usage({'task_execution_id':'global'},[event])
    assert usage['unrecovered_failed_attempt_ids']==[]
    assert usage['unknown_usage_attempts']==1
    assert usage['failed_request_count']==1

def test_valid_actor_retry_is_explicit(recovery):
    task=recovery/'jobs/humanities-native-r3/evolution/round_00/base_logs/2.json'
    record=json.loads(task.read_text(encoding='utf-8'))
    for key in ('answer','agent_result','grade','memory_ingestion_started'):record.pop(key,None)
    record['status']='started'
    task.write_text(json.dumps(record),encoding='utf-8')
    result=helper.prepare(recovery)
    assert result['resume_allowed']
    current=json.loads(task.read_text(encoding='utf-8'))
    assert current['recovery_authorized']['allow_actor_retry'] is True
    assert current['recovery_authorized']['next_phase']=='actor'
