"""Assert HW11 acceptance criteria against preserved real Kubernetes evidence."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hw11.client import summarize


def read(path):return json.loads(Path(path).read_text())
def records(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def require(condition,message):
    if not condition:raise AssertionError(message)


def accounting(rows):
    ids=[r['attempt_id'] for r in rows]
    require(len(ids)==len(set(ids)),'duplicate attempt IDs')
    summary=summarize(rows)
    require(summary['planned']==summary['sent']+summary['counts']['unsent'],'planned conservation')
    require(sum(summary['counts'].values())==summary['planned'],'terminal conservation')
    for row in rows:
        if row['outcome']=='unsent':continue
        require(row['end_monotonic']>=row['sent_monotonic'],'negative duration')
        if row['outcome']=='completed':
            require(row['status']==200 and row['final_marker'] and row['chunks']>0,'false completed response')
            require(row.get('error') is None,'completed response has a transport/protocol error')
            payload,done=row['payload'],row['done']
            require(row['chunks']==payload['max_chunks']==done['chunks'],'response chunk count differs from requested limit')
            require(done['request_id']==payload['request_id']==row['attempt_id'],'response is bound to another request')
            require(done['session_id']==payload['session_id'],'response is bound to another dialogue')
            require(all(done.get(key)==row.get(key) for key in
                ('pod','model_version','artifact_sha256','backend_mode')),'response identity differs from completion marker')
            require(row['backend_mode']=='simulated' and row['pod'] and row['model_version'],'missing actual backend identity')
            require(row['ttfc_seconds']>=0,'negative TTFC')
            require(len(row['inter_chunk_seconds'])==row['chunks']-1,'wrong inter-chunk denominator')
    return summary


def histogram_count(snapshot,metric):
    result={}
    for pod,text in snapshot['pods'].items():
        matches=[float(line.split()[-1]) for line in text.splitlines() if line.startswith(metric+' ')]
        require(len(matches)==1,f'missing or duplicate metric {metric} on {pod}')
        result[pod]=matches[0]
    return result


def check_a(root):
    base=root/'a';faulty=records(base/'faulty/requests.jsonl');fixed=records(base/'fixed/requests.jsonl')
    early=[r for r in faulty if r['status']==503 and 'backend_not_ready' in r.get('error_body','')]
    require(early,'original faulty probes did not deliver traffic to warming backend')
    require(not [r for r in fixed if r['status']==503 and 'backend_not_ready' in r.get('error_body','')],
            'fixed probes still delivered traffic to warming backend')
    timeline=records(base/'fixed/timeline.jsonl')
    require(any(x.get('kind')=='EndpointSlice' for t in timeline for x in t['data']['items']),'missing EndpointSlice evidence')
    long=read(base/'rollout/long-stream.json');row=long['request'];start=long['rollout_started_utc']
    require(row['outcome']=='completed' and row['first_chunk_utc']<start<row['end_utc'],'rollout stream did not cross update')
    rollout=records(base/'rollout/background/requests.jsonl')
    require(rollout and all(r['outcome']=='completed' and r['status']==200 for r in rollout),'nonzero errors in normal rollout')
    bad=records(base/'bad-rollout/requests.jsonl')
    require(bad and all(r['model_version']=='A' for r in bad if r['status']==200),'never-ready B got working traffic')
    bad_values=read(base/'bad-rollout/bad-values.json')
    bad_runtime=read(base/'bad-rollout/bad-runtime-config.json')['data']
    require(bad_values.get('neverReady') is True and bad_values.get('modelVersion')=='B'
            and bad_runtime.get('NEVER_READY')=='true' and bad_runtime.get('MODEL_VERSION')=='B',
            'bad rollout configuration was not preserved')
    checksum=read(base/'bad-rollout/bad-deployment.json')['spec']['template']['metadata']['annotations']['checksum/config']
    bad_timeline=records(base/'bad-rollout/timeline.jsonl');bad_pods=set()
    for sample in bad_timeline:
        for item in sample['data']['items']:
            if item.get('kind')=='Pod' and item['metadata'].get('annotations',{}).get('checksum/config')==checksum:
                bad_pods.add(item['metadata']['name'])
                require(not any(c.get('type')=='Ready' and c.get('status')=='True' for c in
                    item.get('status',{}).get('conditions',[])),'never-ready B became Ready')
    require(bad_pods,'never-ready B Pod creation not observed')
    for sample in bad_timeline:
        for item in sample['data']['items']:
            if item.get('kind')=='EndpointSlice':
                require(not any(e.get('targetRef',{}).get('name') in bad_pods and
                    e.get('conditions',{}).get('ready') is True for e in item.get('endpoints',[])),
                    'never-ready B entered ready EndpointSlice')
    rollback=records(base/'rollback/requests.jsonl')
    artifact=hashlib.sha256((ROOT/'hw11/artifacts/model-a.json').read_bytes()).hexdigest()
    require(rollback and all(r['outcome']=='completed' and r['model_version']=='A' and r['artifact_sha256']==artifact for r in rollback),
            'rollback did not serve verified A')
    killed=read(base/'forced-kill/forced-stream.json')['request']
    require(killed['status']==200 and killed['chunks']>0 and not killed['final_marker'] and killed['outcome']=='interrupted',
            'forced-kill stream interruption not captured')
    before=records(base/'replacement/before/requests.jsonl');after=records(base/'replacement/after/requests.jsonl')
    require(before and after,'missing replacement requests')
    replacement=read(base/'replacement/replacement-identity.json')
    require(replacement['old_pod']!=replacement['new_pod'] and
            all(row.get('pod')==replacement['new_pod'] for row in after),'continued dialog did not reach replacement Pod')
    for rows in (before,after):
        for i,row in enumerate(sorted(rows,key=lambda r:r['id'])):
            marker=f'ISOLATED_DIALOG_{i+1}';foreign=f'ISOLATED_DIALOG_{2-i}'
            require(row['outcome']=='completed' and row['model_version']=='A' and row['artifact_sha256']==artifact,'replacement identity failure')
            require(marker in row.get('text','') and foreign not in row.get('text',''),'dialog leak or lost context')
    for path in base.rglob('requests.jsonl'):accounting(records(path))
    drain=check_drain(base/'drain-proof')
    return dict(warmup_original_failure=True,warmup_fixed_pass=True,rollback_loaded_version=True,
        replacement_and_dialog_recovery=True,rejections_and_interrupted_stream=True,dialog_isolation=True,
        rollout_zero_errors=True,active_stream_graceful_drain=drain)


def check_drain(base):
    """Read actual Pod deletion metadata and completed streams, not a proof flag alone."""
    proof=read(base/'proof.json');long_rows=records(base/'requests.jsonl')
    before=read(base/'deployment-before.json');values=read(base/'values-before.json')
    require(values.get('modelVersion')=='A' and values.get('warmupSeconds')==10
            and values.get('hpa',{}).get('enabled') is False,'supplemental drain setup must be explicit A/warmup10/HPAoff')
    grace=before['spec']['template']['spec']['terminationGracePeriodSeconds']
    require(grace==60,'drain proof grace configuration changed')
    pre_stop=before['spec']['template']['spec']['containers'][0]['lifecycle']['preStop']['exec']['command']
    require('time.sleep(5)' in ' '.join(pre_stop),'drain proof preStop delay is not five seconds')
    old={pod['metadata']['name'] for pod in read(base/'old-pods.json')}
    require(len(old)==2 and len(long_rows)==2 and {row['pod'] for row in long_rows}==old,
            'drain proof must start one stream on each old Pod')
    accounting(long_rows)
    require(all(row['outcome']=='completed' and row['first_chunk_utc']<proof['upgrade_utc']<row['end_utc']
                for row in long_rows),'both drain proof streams must cross update and complete')
    background=records(base/'background/requests.jsonl');accounting(background)
    require(background and all(row['outcome']=='completed' and row['status']==200 for row in background),
            'supplemental rollout has HTTP errors or unfinished background streams')
    metadata={}
    for sample in records(base/'timeline.jsonl'):
        for pod in sample['data'].get('items',[]):
            if (pod.get('kind')=='Pod' and pod['metadata']['name'] in old
                    and pod['metadata'].get('deletionTimestamp') and pod['metadata']['name'] not in metadata):
                meta=pod['metadata'];deadline=datetime.fromisoformat(meta['deletionTimestamp'].replace('Z','+00:00')).timestamp()
                metadata[meta['name']]=dict(deadline=deadline,
                    start=deadline-meta['deletionGracePeriodSeconds'],observed=sample['utc'])
    overlapping=[]
    for row in long_rows:
        meta=metadata.get(row['pod'])
        if meta and max(meta['start'],meta['observed'])+6<row['end_utc']<=meta['deadline']:
            overlapping.append(dict(pod=row['pod'],termination_started_utc=meta['start'],
                first_observed_terminating_utc=meta['observed'],deletion_deadline_utc=meta['deadline'],
                stream_finished_utc=row['end_utc'],seconds_after_termination_start=row['end_utc']-meta['start']))
    require(overlapping,'no actual old stream completed after preStop+one-second margin within its deletion deadline')
    require(proof.get('passed') is True,'drain orchestrator reported failure')
    return dict(long_streams_completed=len(long_rows),background_completed=len(background),
                warmup_seconds=10,overlapping_streams=overlapping)


def check_b(root):
    base=root/'b';hashes={};runs=[]
    for policy in ('normal','clientip'):
        for traffic in ('uniform','skewed'):
            for repeat in range(1,4):
                path=base/policy/traffic/f'run-{repeat}';config=read(path/'config.json');rows=records(path/'requests.jsonl')
                require(config['duration_seconds']>=120,'B measured arrival phase shorter than120s')
                require(config['actual_phase_seconds']>=120,'B actual measured arrival phase shorter than120s')
                require(len(set(config['clients'].values()))==4,'B needs four distinct client IPs')
                require(config['keep_alive'] is False,'B keepalive contract differs')
                summary=accounting(rows);require(summary['planned']==len(read(path/'schedule.json')),'unsent schedule rows missing')
                hashes.setdefault(traffic,set()).add(config['schedule_sha256'])
                for metric in ('hw11_ttfc_seconds_count','hw11_request_duration_seconds_count'):
                    before=histogram_count(read(path/'metrics-before.json'),metric)
                    after=histogram_count(read(path/'metrics-after.json'),metric)
                    require(before.keys()==after.keys(),'metric cohort Pod set changed')
                    require(all(after[p]>=before[p] for p in before),'histogram counter reset')
                    require(sum(after[p]-before[p] for p in before)==summary['successful'],'histogram/client cohort count differs')
                    api_counts={}
                    for boundary,expected in [('before',before),('after',after)]:
                        document=read(path/f'prom-{boundary}.json')
                        require(document.get('status')==200 and document.get('matched_direct_counts') is True,
                                'Prometheus API cohort was not matched to exporter snapshot')
                        vector=document['data']['data']['result']
                        observed={entry['metric'].get('pod'):float(entry['value'][1]) for entry in vector
                                  if entry['metric'].get('__name__')==metric and entry['metric'].get('pod') in expected}
                        require(observed==expected,'actual Prometheus API count differs from exporter snapshot')
                        api_counts[boundary]=observed
                    require(sum(api_counts['after'][p]-api_counts['before'][p] for p in before)==summary['successful'],
                            'Prometheus API/client cohort count differs')
                runs.append(dict(policy=policy,traffic=traffic,repeat=repeat,**summary))
    require(all(len(values)==1 for values in hashes.values()),'workload schedule differs across policies/repeats')
    require((base/'no-skew/requests.jsonl').exists(),'missing diagnostic without expected skew')
    return dict(full_runs=len(runs),schedule_hashes={k:next(iter(v)) for k,v in hashes.items()},runs=runs)


def check_c(root):
    base=root/'c';timeline=records(base/'hpa/timeline.jsonl');series=[];ready_counts=[];cpu_requests=[]
    hpa=read(base/'hpa-before.json')
    require(hpa.get('apiVersion')=='autoscaling/v2' and hpa['spec'].get('minReplicas')==2
            and hpa['spec'].get('maxReplicas')==3,'HPA must use autoscaling/v2 with min2/max3')
    require(any(metric.get('type')=='Resource' and metric.get('resource',{}).get('name')=='cpu'
            for metric in hpa['spec'].get('metrics',[])),'HPA is not configured for CPU')
    metrics=read(base/'metrics-api.json')
    require(metrics.get('apiVersion')=='metrics.k8s.io/v1beta1' and metrics.get('kind')=='PodMetricsList',
            'resource metrics API evidence is missing')
    api_metrics=[item for item in metrics.get('items',[]) if item['metadata'].get('labels',{}).get('app')=='hw11-api']
    require(len(api_metrics)>=2 and all(item.get('timestamp') and item.get('window') and
            any(container.get('name')=='api' and all(container.get('usage',{}).get(key)
                for key in ('cpu','memory')) for container in item.get('containers',[])) for item in api_metrics),
            'resource metrics API did not expose actual usage for both API Pods')
    for sample in timeline:
        ready_count=0
        for item in sample['data']['items']:
            if item.get('kind')=='HorizontalPodAutoscaler':series.append(item.get('status',{}).get('currentReplicas'))
            if (item.get('kind')=='Pod' and item['metadata'].get('labels',{}).get('app')=='hw11-api'
                    and not item['metadata'].get('deletionTimestamp')):
                ready_count+=any(condition.get('type')=='Ready' and condition.get('status')=='True'
                    for condition in item.get('status',{}).get('conditions',[]))
            if item.get('kind')=='Deployment' and item['metadata'].get('name')=='hw11-api':
                for container in item['spec']['template']['spec']['containers']:
                    value=container.get('resources',{}).get('requests',{}).get('cpu')
                    require(value is not None,'CPU request missing from HPA target container')
                    value=str(value)
                    cpu_requests.append(float(value[:-1])/1000 if value.endswith('m') else float(value))
        ready_counts.append(ready_count)
    require(2 in series and 3 in series and series.index(3)>series.index(2) and 2 in series[series.index(3)+1:],
            'HPA evidence lacks actual 2→3→2')
    require(max(ready_counts,default=0)>=3,'HPA target3 was never observed as three actually Ready API Pods')
    require(cpu_requests and all(value>0 for value in cpu_requests),'HPA CPU request must be positive')
    scale_down=read(base/'hpa/scale-down.json')
    elapsed=scale_down['minimum_observed_utc']-scale_down['load_stopped_utc']
    require(elapsed>=0 and abs(elapsed-scale_down['scale_down_seconds'])<.01,
            'scale-down duration is not measured from actual arrival stop')
    for path in base.rglob('requests.jsonl'):accounting(records(path))
    states=read(base/'alerts/alert-states.json');observed={};cycles={}
    for sample in states:
        groups=sample.get('data',{}).get('data',{}).get('groups',[])
        for group in groups:
            for rule in group.get('rules',[]):
                if rule.get('type')=='alerting':
                    name,state=rule['name'],rule.get('state')
                    observed.setdefault(name,set()).add(state)
                    cycle=cycles.setdefault(name,dict(next=0,transitions=[]))
                    expected=('inactive','pending','firing','inactive')
                    if cycle['next']<len(expected) and state==expected[cycle['next']]:
                        cycle['transitions'].append(dict(state=state,utc=sample['utc']))
                        cycle['next']+=1
    require(len([name for name,values in observed.items() if {'pending','firing'}<=values])>=2,
            'both alerts need actual pending and firing states')
    final=[rule for group in states[-1].get('data',{}).get('data',{}).get('groups',[])
           for rule in group.get('rules',[]) if rule.get('type')=='alerting'] if states else []
    require(len(final)>=2 and all(rule.get('state')=='inactive' for rule in final),'alerts did not return inactive')
    require(len(cycles)>=2 and all(cycle['next']==4 for cycle in cycles.values()),
            'both alerts must show ordered inactive→pending→firing→inactive')
    return dict(hpa_two_three_two=True,maximum_ready_api_pods=max(ready_counts),
        cpu_request_cores=sorted(set(cpu_requests)),resource_metrics_api_pods=len(api_metrics),
        scale_down_seconds=scale_down['scale_down_seconds'],alerts={k:sorted(v) for k,v in observed.items()},
        alert_cycles={name:cycle['transitions'] for name,cycle in cycles.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/hw11')
    parser.add_argument('--experiment',choices=['a','b','c'])
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args();checks={};failed=False
    for name,fn in [('a',check_a),('b',check_b),('c',check_c)]:
        if args.experiment and args.experiment!=name:continue
        try:checks[name]=dict(passed=True,evidence=fn(args.results))
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as exc:
            checks[name]=dict(passed=False,error=str(exc));failed=True
    output=dict(passed=not failed,checks=checks)
    (args.results/'checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
