"""Recover enum folding using real native creation, without paid transport."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r'''
import json, socket, sys
from pathlib import Path
import openai
from philoagent.cli import init_run
from philoagent.live_smoke import activate_workspace
from philoagent.orientation import load
from philoagent.ablation_runtime import digest
from philoagent.hss_experiment import write_json,read_json
from philoagent.hss_enum_collision_recovery import enum_collision_input_hashes,repair_enum_collision
def forbidden(*args,**kwargs):raise AssertionError('Real model/network access forbidden')
openai.OpenAI.__init__=forbidden
socket.socket.connect=forbidden
root=Path(sys.argv[1]);damage=sys.argv[2]
jobdir=Path(init_run('enum-fold','dialectical',output_root=root/'jobs')['run_dir'])
workspace=activate_workspace(jobdir)
from MemEvolve.phases.memory_creator import MemorySystemCreator
rounddir=jobdir/'evolution/round_00';rounddir.mkdir(parents=True)
job={'job_id':jobdir.name,'code_evolve':True,'orientation_evolve':False}
write_json(root/'manifest.json',{'jobs':[job],'protocol':{'generation_model':'offline'}})
write_json(root/'budget/budget.json',{'pending':{}})
write_json(jobdir/'job_status.json',{'status':'failed','pid':-1})
snapshot=load(jobdir/'lineage','lightweight_memory')
def config(classname,module,marker):
 return {'provider_code':{'class_name':classname,'module_name':module,
  'code':'class '+classname+':\n    marker = '+repr(marker)+'\n    namespace = "praxis_loom_memory"\n'},
 'memory_type_info':{'enum_name':'PRAXIS_LOOM_MEMORY','enum_value':'praxis_loom_memory'},
 'config_updates':{'storage_dir':'./storage/praxis_loom_memory','database_path':'./storage/praxis_loom_memory/memory.json'},
 'orientation_items':list(snapshot.items),'_orientation_parent':{'lineage_root':str(jobdir/'lineage'),'snapshot':snapshot.to_dict()},
 '_generation_metadata':{'parent_provider':'lightweight_memory'},'_raw_llm_output':'paid source '+marker}
first=config('PraxisLoomProvider','praxis_loom_provider','first')
second=config('PraxisLoomMemoryProvider','praxis_loom_memory_provider','second')
for i,c in enumerate((first,second),1):
 write_json(rounddir/f'generated_system_{i}.json',c)
 result=MemorySystemCreator.create_memory_system(c,base_dir=workspace)
 assert result['success'],result
registry=workspace/'EvolveLab/memory_types.py'
assert 'MemoryType.PRAXIS_LOOM_MEMORY: ("PraxisLoomProvider", "praxis_loom_provider")'in registry.read_text(encoding='utf-8')
store=workspace/'storage/praxis_loom_memory/memory.json';write_json(store,{'valid_first_experiences':['batch1','batch2','finalist2']})
write_json(rounddir/'created_system.json',{'created':['praxis_loom_memory'],'failed':[]})
validation={'validated':['praxis_loom_memory'],'failed':[],
 'details':{'praxis_loom_memory':{'verdict':'passed','fast_validation':{'tests':{'import':{'class_name':'PraxisLoomProvider','module_name':'praxis_loom_provider'}}}}}}
write_json(rounddir/'validated_systems.json',validation)
state={'phases':{'analyze':{'completed':True,'default_provider':'lightweight_memory'},
 'generate':{'completed':True,'output':str(rounddir/'generated_system_2.json')},
 'create':{'completed':True,'created_systems':['praxis_loom_memory'],'failed_systems':[]},
 'validate':{'completed':True,'validated_systems':['praxis_loom_memory'],'failed_systems':[]}}}
write_json(rounddir/'state.json',state)
checkpoint={'parent_provider':'lightweight_memory','parent_orientation_sha256':snapshot.sha256,'step_completed':4,
 'validated_systems':['praxis_loom_memory','praxis_loom_memory'],'batch_indices':[0,1],
 'eval_results':{'praxis_loom_memory':{'success':True}},'finalist_results':{'praxis_loom_memory':{'success':False}},
 'finalist_tasks':[1,2,3],'evolution_results':{'analysis':{'success':True},'validation':['duplicate']}}
write_json(rounddir/'checkpoint.json',checkpoint)
for folder in ('eval_batch','eval_finalists'):
 write_json(rounddir/folder/'praxis_loom_memory/2.json',{'status':'completed','answer':'already paid work',
  'memory_ingestion_complete':True,'runtime_artifact':{'source_path':str(workspace/'EvolveLab/providers/praxis_loom_provider.py')}})
 write_json(rounddir/folder/'praxis_loom_memory/_quality/summary.json',{'success':True,'summary':{'quality_score':80}})
evidence=jobdir/'evidence/runtime/calls-1.jsonl';evidence.parent.mkdir(parents=True,exist_ok=True)
records=[{'operation':'loaded','code_path':str(workspace/'EvolveLab/providers/praxis_loom_provider.py'),
 'loaded_code_sha256':digest((workspace/'EvolveLab/providers/praxis_loom_provider.py').read_bytes()),'timestamp_utc':str(i)}for i in range(2)]
if damage=='second_ran':records.append({'operation':'loaded','code_path':str(workspace/'EvolveLab/providers/praxis_loom_memory_provider.py')})
evidence.write_text('\n'.join(json.dumps(x)for x in records),encoding='utf-8')
write_json(root/'budget/attempts/000123.json',{'attempt':123,'job_id':jobdir.name,'provider':'lightweight_memory',
 'role':'generation','round_dir':str(rounddir),'status':'success','usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300},
 'response':{'choices':[{'finish_reason':'stop','message':{'content':second['_raw_llm_output']}}]}})
pins=enum_collision_input_hashes(root,jobdir.name)
if damage=='bad_hash':pins['evolution/round_00/generated_system_1.json']='wrong'
elif damage=='pending':write_json(root/'budget/budget.json',{'pending':{'2':{'job_id':'other'}}})
before={str(p):p.read_bytes()for p in jobdir.rglob('*')if p.is_file()and'__pycache__'not in p.parts}
if damage!='none':
 try:repair_enum_collision(root,jobdir.name,expected_hashes=pins)
 except ValueError:pass
 else:raise AssertionError('Unsafe repair accepted')
 assert before=={str(p):p.read_bytes()for p in jobdir.rglob('*')if p.is_file()and'__pycache__'not in p.parts}
 print('ENUM_RECOVERY_REFUSED');sys.exit(0)
receipt=repair_enum_collision(root,jobdir.name,expected_hashes=pins)
assert repair_enum_collision(root,jobdir.name,expected_hashes=pins)==receipt
after=read_json(rounddir/'checkpoint.json')
assert after['step_completed']==2 and after['validated_systems']==['praxis_loom_memory']
assert after['finalist_tasks']==[1,2,3]
assert 'eval_results'not in after and'finalist_results'not in after
assert read_json(rounddir/'generated_system_1.json')==first
assert read_json(rounddir/'validated_systems.json')==validation
assert store.read_bytes()==before[str(store)]
for folder in ('eval_batch','eval_finalists'):
 paid=rounddir/folder/'praxis_loom_memory/2.json'
 assert paid.read_bytes()==before[str(paid)]
 assert (rounddir/'_enum_collision_recovery_2'/(folder+'.before')/'praxis_loom_memory/2.json').exists()
repaired=read_json(rounddir/'generated_system_2.json')
assert repaired['orientation_items']==second['orientation_items']
assert repaired['_raw_llm_output']==second['_raw_llm_output']
assert repaired['memory_type_info']['enum_value']!='praxis_loom_memory'
result=MemorySystemCreator.create_memory_system(repaired,base_dir=workspace)
assert result['success'],result
mapping='MemoryType.'+repaired['memory_type_info']['enum_name']+': ('+repr(repaired['provider_code']['class_name'])+', '+repr(repaired['provider_code']['module_name'])+')'
reg=registry.read_text(encoding='utf-8').replace('"',"'")
assert mapping in reg
assert 'MemoryType.PRAXIS_LOOM_MEMORY: (\'PraxisLoomProvider\', \'praxis_loom_provider\')'in reg
assert store.read_bytes()==before[str(store)]
firstsource=workspace/'EvolveLab/providers/praxis_loom_provider.py'
assert firstsource.read_bytes()==before[str(firstsource)]
assert receipt['new_model_requests']==0 and receipt['generated_code_executed']is False
print('ENUM_COLLISION_REPAIRED')
'''


@pytest.mark.parametrize('damage',['none','bad_hash','second_ran','pending'])
def test_enum_recovery_native_creation_and_guards(tmp_path,damage):
    project=Path(__file__).resolve().parents[1]
    env={key:os.environ[key]for key in('SystemRoot','WINDIR','PATH','TEMP','TMP')if key in os.environ}
    env.update(PYTHONPATH=str(project),PYTHONUTF8='1')
    result=subprocess.run([sys.executable,'-X','utf8','-c',SCRIPT,str(tmp_path/'experiment'),damage],
                          cwd=project,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,(result.stdout+result.stderr)[-12000:]
    assert ('ENUM_COLLISION_REPAIRED'if damage=='none'else'ENUM_RECOVERY_REFUSED')in result.stdout
