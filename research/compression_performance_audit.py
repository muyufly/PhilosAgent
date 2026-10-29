"""Read-only artifact/receipt audit; no model calls."""
import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import median, mean
from datetime import datetime, timezone

ROOT=Path('D:/PhiloAgent/runs/hss-business-formal-20261029-r01')
D=ROOT/'deadline_20261030'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def percentile(xs,p):
    xs=sorted(xs)
    if not xs:return None
    return xs[round((len(xs)-1)*p)]
def stats(xs):return {'n':len(xs),'mean':round(mean(xs),3) if xs else None,'median':median(xs) if xs else None,'min':min(xs) if xs else None,'p25':percentile(xs,.25),'p75':percentile(xs,.75),'max':max(xs) if xs else None}
transition=read(D/'transition.json')
cut=datetime.fromisoformat(transition['transition_utc']).timestamp()
roles=defaultdict(list); old=[]; other=[]
for p in (ROOT/'budget/attempts').glob('*.json'):
    a=read(p)
    if str(a.get('role','')).startswith('deadline_'):roles[a['role']].append(a)
    elif a.get('started_unix',0)<cut:old.append(a)
    else:other.append(a)
def receipt_stats(aa):
    known=[a for a in aa if isinstance(a.get('usage'),dict)]
    choices=[(a,a.get('response',{}).get('choices',[{}])[0]) for a in aa if isinstance(a.get('response'),dict) and a['response'].get('choices')]
    return {'attempts':len(aa),'statuses':dict(Counter(a.get('status') for a in aa)),'known_usage':len(known),'unknown_usage':len(aa)-len(known),'known_prompt_tokens':sum(a['usage'].get('prompt_tokens',0) for a in known),'known_completion_tokens':sum(a['usage'].get('completion_tokens',0) for a in known),'known_total_tokens':sum(a['usage'].get('total_tokens',0) for a in known),'request_max_tokens':dict(Counter(a.get('request',{}).get('max_tokens') for a in aa)),'finish_reasons':dict(Counter(c.get('finish_reason') for a,c in choices)),'length_finish_requests':[a.get('attempt') for a,c in choices if c.get('finish_reason')=='length'],'error_types':dict(Counter(a.get('error_type') for a in aa if a.get('status')!='success')),'elapsed_seconds':stats([a['finished_unix']-a['started_unix'] for a in aa if a.get('finished_unix')]),'started_utc':datetime.fromtimestamp(min(a['started_unix'] for a in aa),timezone.utc).isoformat() if aa else None,'last_finished_utc':datetime.fromtimestamp(max(a.get('finished_unix',a['started_unix']) for a in aa),timezone.utc).isoformat() if aa else None}
records=[read(p) for p in (D/'jobs').glob('*/tasks/*.json')]
by_source=defaultdict(list); by_sourcephase=defaultdict(list)
for t in records:
    source='reused_native' if t.get('reused') else 'deadline_generated'
    by_source[source].append(t)
    by_sourcephase[source+':'+t['phase']].append(t)
def task_stats(tt):return {'tasks':len(tt),'characters':stats([len(t.get('answer','')) for t in tt]),'scores':stats([t.get('score',0) for t in tt]),'sources':dict(Counter(t.get('source') for t in tt)),'score_methods':dict(Counter(t.get('score_method') for t in tt)),'phases':dict(Counter(t.get('phase') for t in tt))}
allrapid=[a for aa in roles.values() for a in aa]
matched=[]
for t in records:
    if t.get('reused'):continue
    candidates=[]
    for a in roles['deadline_actor']+roles['deadline_rescue_actor']:
        if a.get('job_id')==t['job_id'] and a.get('task_id')==t['task_id'] and isinstance(a.get('response'),dict):
            for c in a['response'].get('choices',[]):
                content=c.get('message',{}).get('content')
                if content==t.get('answer'):
                    candidates.append((a,c))
    if candidates:
        a,c=max(candidates,key=lambda ac:ac[0]['attempt'])
        matched.append({'job_id':t['job_id'],'task_id':t['task_id'],'phase':t['phase'],'group':t['group'],'domain':t['domain'],'attempt':a['attempt'],'score':t['score'],'score_method':t['score_method'],'chars':len(t['answer']),'finish_reason':c.get('finish_reason'),'max_tokens':a.get('request',{}).get('max_tokens'),'completion_tokens':a.get('usage',{}).get('completion_tokens'),'reasoning_chars':len(c.get('message',{}).get('reasoning_content') or '')})
out={'generated_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'transition':transition,'old_pretransition_receipts':receipt_stats(old),'other_nonrapid_posttransition':receipt_stats(other),'rapid_all':receipt_stats(allrapid),'rapid_by_role':{k:receipt_stats(v) for k,v in roles.items()},'supervisor_started_utc':read(D/'supervisor.json')['started_utc'],'summary_completed_utc':read(D/'summary.json')['completed_utc'],'completion_finalized_utc':read(D/'completion.json')['finalized_utc'],'task_stats_by_source':{k:task_stats(v) for k,v in by_source.items()},'task_stats_by_source_phase':{k:task_stats(v) for k,v in by_sourcephase.items()},'new_final_answer_receipt_match':{'matched':len(matched),'finish_reasons':dict(Counter(m['finish_reason'] for m in matched)),'by_phase':{p:dict(Counter(m['finish_reason'] for m in matched if m['phase']==p)) for p in ['learning','heldout']},'by_domain':{p:dict(Counter(m['finish_reason'] for m in matched if m['domain']==p)) for p in ['humanities','social','business']},'by_group':{p:dict(Counter(m['finish_reason'] for m in matched if m['group']==p)) for p in ['native','fixed','full']},'length_score':stats([m['score'] for m in matched if m['finish_reason']=='length']),'stop_score':stats([m['score'] for m in matched if m['finish_reason']=='stop']),'lengths':matched},'limitations':['No controlled rerun of the same learned state under both protocols; protocol compression has no identified causal score delta.','Character lengths compare different tasks, groups and stages; not a matched causal comparison.','Unknown-usage requests may have incurred tokens not in confirmed totals.','finish_reason=length indicates token-limit termination, not proof every such final answer is unusable.','No empty answer or failed_execution does not imply fulfillment of all requested sections and length.']}
(Path('D:/PhiloAgent/research')/'compression-performance-audit-20261030.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
small={k:v for k,v in out.items() if k not in ['new_final_answer_receipt_match']};small['new_final_answer_receipt_match']={k:v for k,v in out['new_final_answer_receipt_match'].items() if k!='lengths'}
print(json.dumps(small,ensure_ascii=False,indent=2))
