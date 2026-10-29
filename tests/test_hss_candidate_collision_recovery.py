"""Repair a paid duplicate candidate without overwriting a validated sibling."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r'''
from copy import deepcopy
import json, os, socket, sys
from pathlib import Path
import openai
from philoagent.cli import init_run
from philoagent.ablation_runtime import digest
from philoagent.hss_experiment import write_json, read_json
from philoagent.live_smoke import activate_workspace
from philoagent.orientation import load
from philoagent.hss_candidate_collision_recovery import repair_candidate_collision

def forbidden(*args, **kwargs):
    raise AssertionError("Collision recovery attempted a paid/network call")
openai.OpenAI.__init__ = forbidden
socket.socket.connect = forbidden
experiment=Path(sys.argv[1])
jobdir=Path(init_run('offline-collision','dialectical',output_root=experiment/'jobs')['run_dir'])
workspace=activate_workspace(jobdir)
from MemEvolve.phases.memory_creator import MemorySystemCreator
rounddir=jobdir/'evolution/round_00';rounddir.mkdir(parents=True)
job={'job_id':jobdir.name,'code_evolve':True,'orientation_evolve':False}
write_json(experiment/'manifest.json',{'jobs':[job],'protocol':{'generation_model':'offline'}})
write_json(experiment/'budget/budget.json',{'pending':{},'unknown_usage_attempts':0})
write_json(jobdir/'job_status.json',{'status':'failed','pid':-1})
snapshot=load(jobdir/'lineage','lightweight_memory')
def config(marker):
 return {'provider_code':{'class_name':'CovenantMemoryProvider','module_name':'covenant_memory_provider',
  'code':'class CovenantMemoryProvider:\n    namespace = "covenant_memory"\n    marker = '+repr(marker)+'\n'},
 'memory_type_info':{'enum_name':'COVENANT_MEMORY','enum_value':'covenant_memory'},
 'config_updates':{'storage_dir':'./storage/covenant_memory','database_path':'./storage/covenant_memory/memory.json'},
 '_orientation_parent':{'lineage_root':str(jobdir/'lineage'),'snapshot':snapshot.to_dict()},
 'orientation_items':list(snapshot.items),'_raw_llm_output':'paid generation '+marker,
 '_generation_metadata':{'parent_provider':'lightweight_memory','model_id':'offline'}}
first,second=config('first'),config('second')
one=rounddir/'generated_system_1.json';two=rounddir/'generated_system_2.json'
write_json(one,first);write_json(two,second)
creation=MemorySystemCreator.create_memory_system(first,base_dir=workspace)
assert creation['success'],creation
failed=MemorySystemCreator.create_memory_system(second,base_dir=workspace)
assert not failed['success'] and 'already exists' in failed['error']
source=workspace/'EvolveLab/providers/covenant_memory_provider.py'
firstsource=source.read_bytes();firstbytes=one.read_bytes()
store=workspace/'storage/covenant_memory/memory.json';write_json(store,{'first':'must survive'})
storebytes=store.read_bytes()
write_json(rounddir/'created_system.json',{'created':['covenant_memory'],'failed':[{'system':'covenant_memory','error':failed['error']}]})
validated={'validated':['covenant_memory'],'failed':[],'details':{'covenant_memory':{'verdict':'passed'}}}
write_json(rounddir/'validated_systems.json',validated)
state={'phases':{'analyze':{'completed':True,'default_provider':'lightweight_memory'},
 'generate':{'completed':True,'output':str(two)},'create':{'completed':True,'created_systems':[],
 'failed_systems':[{'system':'covenant_memory','error':failed['error']}]},
 'validate':{'completed':True,'validated_systems':['covenant_memory'],'failed_systems':[]}}}
write_json(rounddir/'state.json',state)
checkpoint={'parent_provider':'lightweight_memory','parent_orientation_sha256':snapshot.sha256,
            'step_completed':1,'selection_metadata':{'preserve':True}}
write_json(rounddir/'checkpoint.json',checkpoint)
write_json(experiment/'budget/attempts/000123.json',{'attempt':123,'job_id':jobdir.name,'role':'generation',
 'provider':'lightweight_memory','round_dir':str(rounddir),'status':'success',
 'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300},
 'response':{'choices':[{'finish_reason':'stop','message':{'content':second['_raw_llm_output']}}]}})
hashes={one.name:digest(one.read_bytes()),two.name:digest(two.read_bytes())}
damage=sys.argv[2]if len(sys.argv)>2 else 'none'
if damage!='none':
 if damage=='bad_hash':hashes[one.name]='wrong'
 elif damage=='unvalidated':write_json(rounddir/'validated_systems.json',{'validated':[],'failed':[]})
 elif damage=='livejob':write_json(jobdir/'job_status.json',{'status':'running','pid':-1})
 elif damage=='pending':
  write_json(experiment/'budget/budget.json',{'pending':{'900':{'job_id':jobdir.name}}})
  write_json(experiment/'budget/attempts/000900.json',{'job_id':jobdir.name})
 before={str(p):p.read_bytes()for p in rounddir.rglob('*')if p.is_file()}
 try:repair_candidate_collision(experiment,jobdir.name,expected_hashes=hashes)
 except ValueError:pass
 else:raise AssertionError('Unsafe collision repair was accepted')
 assert before=={str(p):p.read_bytes()for p in rounddir.rglob('*')if p.is_file()}
 print('COLLISION_REFUSED_WITHOUT_WRITES')
 sys.exit(0)
receipt=repair_candidate_collision(experiment,jobdir.name,expected_hashes=hashes)
assert repair_candidate_collision(experiment,jobdir.name,expected_hashes=hashes)==receipt
repaired=read_json(two)
assert repaired['memory_type_info']['enum_value']!='covenant_memory'
assert repaired['orientation_items']==second['orientation_items']
assert repaired['_raw_llm_output']==second['_raw_llm_output']
assert one.read_bytes()==firstbytes and source.read_bytes()==firstsource and store.read_bytes()==storebytes
assert read_json(rounddir/'validated_systems.json')==validated
assert read_json(rounddir/'checkpoint.json')==checkpoint
assert read_json(rounddir/'state.json')['phases']['create']['completed'] is False
assert read_json(rounddir/'created_system.json')=={'created':['covenant_memory'],'failed':[]}
assert receipt['new_model_requests']==0 and receipt['candidate_index']==2
result=MemorySystemCreator.create_memory_system(repaired,base_dir=workspace)
assert result['success'],result
assert source.read_bytes()==firstsource and store.read_bytes()==storebytes
newsource=workspace/'EvolveLab/providers'/f"{repaired['provider_code']['module_name']}.py"
assert newsource.exists() and newsource!=source
print('COLLISION_REPAIRED_WITHOUT_MODEL',json.dumps(receipt))
'''


def _run_collision(tmp_path, damage='none'):
    script=tmp_path/'collision.py'
    script.write_text(SCRIPT,encoding='utf-8')
    result=subprocess.run([sys.executable,'-X','utf8',str(script),str(tmp_path/'experiment'),damage],
                          cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,
                          encoding='utf-8',timeout=60,
                          env={**{k:os.environ[k]for k in('SystemRoot','WINDIR','PATH','TEMP','TMP')if k in os.environ},
                               'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'PYTHONUTF8':'1'})
    assert result.returncode==0,(result.stdout+result.stderr)[-15000:]
    return result.stdout


def test_actual_native_creator_preserves_first_and_creates_repaired_second(tmp_path):
    output=_run_collision(tmp_path)
    assert 'COLLISION_REPAIRED_WITHOUT_MODEL' in output


@pytest.mark.parametrize('damage',['bad_hash','unvalidated','livejob','pending'])
def test_unsafe_collision_recovery_preserves_all_native_artifacts(tmp_path, damage):
    output=_run_collision(tmp_path, damage)
    assert 'COLLISION_REFUSED_WITHOUT_WRITES' in output
