"""Offline score-sensitivity audit; never changes experimental records."""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path
from collections import Counter, defaultdict
from philoagent.hss_deadline_scoring import score_work

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'runs/hss-business-formal-20261029-r01/deadline_20261030/results.jsonl'
records=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
def no_model(*args, **kwargs):
    raise AssertionError('Audit must never call a model')
for r in records:
    r['uniform_rule_score']=score_work({'question':r['question'],'domain':r['domain']},r['answer'],no_model,0)['quality_score']

def mean(values): return round(st.mean(values),6) if values else None
def aggregate(rs):
    ds=[r['uniform_rule_score']-r['score'] for r in rs]
    return {'n':len(rs),'original_mean':mean([r['score'] for r in rs]),'uniform_rule_mean':mean([r['uniform_rule_score'] for r in rs]),'rule_minus_original_mean':mean(ds),'mean_absolute_difference':mean([abs(d) for d in ds]),'max_absolute_difference':max(map(abs,ds),default=None)}
def table(rs,fields):
    grouped=defaultdict(list)
    for r in rs: grouped[tuple(r[f] for f in fields)].append(r)
    return [{**dict(zip(fields,k)),**aggregate(v)} for k,v in sorted(grouped.items())]
def compare(rs,left,right,only_both_model=False):
    ix={(r['domain'],r['repetition'],r['task_id'],r['group']):r for r in rs}
    pairs=[]
    for r in rs:
        if r['group']!=left:continue
        b=ix.get((r['domain'],r['repetition'],r['task_id'],right))
        if b is None:continue
        if only_both_model and not(r['score_method']==b['score_method']=='model_quick'):continue
        pairs.append({'domain':r['domain'],'repetition':r['repetition'],'task_id':r['task_id'],'left_method':r['score_method'],'right_method':b['score_method'],'original_difference':r['score']-b['score'],'uniform_rule_difference':r['uniform_rule_score']-b['uniform_rule_score']})
    def sign(x):return (x>0)-(x<0)
    return {'comparison':left+'_minus_'+right,'both_model_only':only_both_model,'n':len(pairs),'original_paired_mean_difference':mean([p['original_difference'] for p in pairs]),'uniform_rule_paired_mean_difference':mean([p['uniform_rule_difference'] for p in pairs]),'strict_positive_negative_direction_reversals':sum(p['original_difference']*p['uniform_rule_difference']<0 for p in pairs),'winner_or_tie_changed':sum(sign(p['original_difference'])!=sign(p['uniform_rule_difference']) for p in pairs),'by_domain':[{ 'domain':d,'n':len(x),'original_difference':mean([p['original_difference'] for p in x]),'uniform_rule_difference':mean([p['uniform_rule_difference'] for p in x])} for d in ['humanities','social','business'] if(x:=[p for p in pairs if p['domain']==d])],'pairs':pairs}

held=[r for r in records if r['stage']=='heldout']
reused=[r for r in records if r.get('reused')]
reuse_differences=[]
source_phases=Counter()
for r in reused:
    source_path=Path(r['source'])
    source=json.loads(source_path.read_text(encoding='utf-8'))
    source_phases[source_path.relative_to(ROOT/'runs/hss-business-formal-20261029-r01/jobs').parts[1]]+=1
    if source.get('answer',source.get('agent_result'))!=r['answer'] or source.get('quality_score',source.get('grade',{}).get('quality_score'))!=r['score']:
        reuse_differences.append({'job_id':r['job_id'],'task_id':r['task_id']})
out={'audit':'offline scoring sensitivity, not truth or quality-loss estimate','input':str(DATA),'never_calls_model':True,'original_records_unchanged':True,'counts':dict(Counter(r['score_method'] for r in records)),'heldout_counts':dict(Counter(r['score_method'] for r in held)),'all':aggregate(records),'heldout':aggregate(held),'all_by_stage_and_method':table(records,['stage','score_method']),'heldout_by_domain_group':table(held,['domain','group']),'heldout_by_method':table(held,['score_method']),'heldout_by_group_method':table(held,['group','score_method']),'heldout_by_domain_method':table(held,['domain','score_method']),'comparisons':[compare(held,l,'native',m) for l in ['full','fixed'] for m in [False,True]],'uniform_rule_all_heldout_groups':table(held,['group']),'largest_score_changes':[{k:r[k] for k in ['job_id','task_id','score','score_method','uniform_rule_score']} for r in sorted(held,key=lambda r:abs(r['uniform_rule_score']-r['score']),reverse=True)[:12]],'note':'69 original-native-judge records reuse the exact same answer and original score, so they are not independent before-after experiments. There are no original-native-judge heldout answers in the deadline set. Removing rule-scored pairs changes the sample as well as the score method. Uniform-rule regrading is a sensitivity analysis on identical works; it does not establish which scoring method is accurate.'}
path=ROOT/'research/compression-measurement-audit-20261030.json'
out['reuse_verification']={'n':len(reused),'answer_or_score_differences_from_source':reuse_differences,'source_phases':dict(source_phases),'original_terminal_directories':len(list((ROOT/'runs/hss-business-formal-20261029-r01/jobs').glob('*/terminal'))),'original_terminal_task_records':len(list((ROOT/'runs/hss-business-formal-20261029-r01/jobs').glob('*/terminal/task_logs/*/[0-9]*.json')))}
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['comparisons','largest_score_changes','heldout_by_domain_method','heldout_by_group_method']},ensure_ascii=True,indent=2))
print('COMPARISONS',json.dumps([{k:v for k,v in x.items() if k!='pairs'} for x in out['comparisons']],ensure_ascii=True,indent=2))
