"""Execute and preserve real Kubernetes HW11 experiments, without external models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hw11.client import attempt,summarize
TOOLS=Path('/private/tmp/mlops-hw11-tools')
KUBECTL=os.environ.get('HW11_KUBECTL',str(TOOLS/'kubectl'))
HELM=os.environ.get('HW11_HELM',str(TOOLS/'helm'))
KUBECONFIG=os.environ.get('HW11_KUBECONFIG',str(TOOLS/'kubeconfig'))
NAMESPACE='hw11'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def command(args,timeout=180,check=True,**kwargs):
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout,**kwargs)
    if check and result.returncode:
        raise RuntimeError(f"Command failed {args[:5]}: {result.stderr[-1500:]}")
    return result


def kube(*args,**kwargs):
    return command([KUBECTL,'--kubeconfig',KUBECONFIG,'-n',NAMESPACE,*args],**kwargs)


def get(kind,selector=None):
    args=['get',kind,'-o','json']
    if selector:args+=['-l',selector]
    return json.loads(kube(*args).stdout)


def pods():return get('pods','app=hw11-api')['items']


def ready_pods():
    return [p for p in pods() if not p['metadata'].get('deletionTimestamp') and
        any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[]))]


def wait_ready(count=2,timeout=180):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        current=ready_pods()
        if len(current)==count:return current
        time.sleep(1)
    raise RuntimeError(f'Expected {count} Ready API Pods')


def upgrade(**values):
    args=[HELM,'--kubeconfig',KUBECONFIG,'upgrade','hw11',str(ROOT/'deploy/hw11/chart'),'-n',NAMESPACE,'--reuse-values']
    for key,value in values.items():args+=['--set',f'{key}={str(value).lower() if isinstance(value,bool) else value}']
    return command(args,timeout=90)


def rollout(timeout=180):
    kube('rollout','status','deployment/hw11-api',f'--timeout={timeout}s',timeout=timeout+10)
    wait_ready(2,timeout)


def http(url,payload=None,timeout=5):
    req=urllib.request.Request(url,data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as exc:
        body=exc.read().decode();return exc.code,json.loads(body) if body else {}
    except Exception as exc:return None,{'error':type(exc).__name__}


@contextmanager
def forward(service='service/hw11-api',remote=8000):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    proc=subprocess.Popen([KUBECTL,'--kubeconfig',KUBECONFIG,'-n',NAMESPACE,'port-forward',service,f'{port}:{remote}'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    start=time.monotonic()
    try:
        while time.monotonic()-start<15:
            if proc.poll() is not None:raise RuntimeError('port-forward failed')
            with socket.socket() as listener:
                if listener.connect_ex(('127.0.0.1',port))==0:break
            time.sleep(.1)
        yield f'http://127.0.0.1:{port}'
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


class Collector:
    def __init__(self,out):
        self.out=Path(out);self.stop=threading.Event();self.thread=None;self.error=None
    def __enter__(self):
        self.out.mkdir(parents=True,exist_ok=True)
        self.started_utc=time.time()
        self.prom_context=forward('service/hw11-prometheus',9090)
        self.prom_url=self.prom_context.__enter__()
        self.thread=threading.Thread(target=self.guarded_collect,daemon=True);self.thread.start();return self
    def guarded_collect(self):
        try:
            self.collect()
        except Exception as exc:
            self.error=type(exc).__name__
            save(self.out/'collector-error.json',dict(utc=time.time(),error=self.error,
                message=str(exc),passed=False))
    def collect(self):
        with (self.out/'timeline.jsonl').open('w',buffering=1) as stream, (self.out/'prom-runtime.jsonl').open('w',buffering=1) as metrics:
            iteration=0
            while not self.stop.is_set():
                started=time.monotonic()
                result=kube('get','pods,endpointslices,hpa,deployment','-o','json',check=False,timeout=15)
                try:data=json.loads(result.stdout)
                except json.JSONDecodeError:data={'error':result.stderr,'items':[]}
                stream.write(json.dumps(dict(utc=time.time(),monotonic=started,data=data))+'\n');stream.flush()
                query=('{__name__=~"hw11_running|hw11_waiting|hw11_requests_total|hw11_cache_total"}'
                       ' or container_cpu_usage_seconds_total{namespace="hw11",container="api"}'
                       ' or container_memory_working_set_bytes{namespace="hw11",container="api"}')
                metrics.write(json.dumps(dict(utc=time.time(),query=query,**prometheus(self.prom_url,query)))+'\n');metrics.flush()
                if iteration%3==0:
                    for pod in data.get('items',[]):
                        if pod.get('kind')=='Pod' and pod['metadata'].get('labels',{}).get('app')=='hw11-api':
                            name=pod['metadata']['name'];log=kube('logs',name,'--timestamps',check=False,timeout=10)
                            if log.returncode==0:(self.out/f'{name}.log').write_text(log.stdout)
                iteration+=1
                self.stop.wait(max(0,1-(time.monotonic()-started)))
    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=40)
        if self.thread.is_alive():
            self.error='CollectorShutdownTimeout'
            save(self.out/'collector-error.json',dict(utc=time.time(),error=self.error,passed=False))
        ended_utc=time.time()
        from urllib.parse import urlencode
        queries={
            'running':'hw11_running', 'waiting':'hw11_waiting',
            'ttfc_p95':'histogram_quantile(0.95,sum by (le,pod) (rate(hw11_ttfc_seconds_bucket[30s])))',
            'full_p95':'histogram_quantile(0.95,sum by (le,pod) (rate(hw11_request_duration_seconds_bucket[30s])))',
            'cpu':'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[30s]))',
            'memory':'container_memory_working_set_bytes{namespace="hw11",container="api"}'}
        for name,query in queries.items():
            parameters=dict(query=query,start=self.started_utc,end=ended_utc,step=1)
            status,data=http(self.prom_url+'/api/v1/query_range?'+urlencode(parameters))
            save(self.out/f'prom-{name}-range.json',dict(status=status,**parameters,data=data))
        self.prom_context.__exit__(None,None,None)
        save(self.out/'deployment.json',get('deployment/hw11-api'))
        save(self.out/'services.json',get('services'))
        result=command([HELM,'--kubeconfig',KUBECONFIG,'get','values','hw11','-n',NAMESPACE,'-o','json'])
        (self.out/'values.json').write_text(result.stdout)
        if self.error and _[0] is None:
            raise RuntimeError(f'Collector failed: {self.error}; inspect preserved collector-error.json')


def ensure_clients():
    image=get('deployment/hw11-api')['spec']['template']['spec']['containers'][0]['image']
    for index in range(4):
        name=f'hw11-client-{index}'
        exists=kube('get','pod',name,check=False)
        if exists.returncode:
            spec=dict(apiVersion='v1',kind='Pod',metadata=dict(name=name,labels={'app':'hw11-client'}),
                spec=dict(restartPolicy='Never',containers=[dict(name='client',image=image,imagePullPolicy='IfNotPresent',
                    command=['python','-c','import time; time.sleep(86400)'],
                    resources={'requests':{'cpu':'25m','memory':'48Mi'},'limits':{'cpu':'500m','memory':'192Mi'}})]))
            kube('apply','-f','-',input=json.dumps(spec))
    kube('wait','--for=condition=Ready','pod','-l','app=hw11-client','--timeout=120s',timeout=130)
    return {p['metadata']['name']:p['status']['podIP'] for p in get('pods','app=hw11-client')['items']}


def schedule(duration,rate,traffic,chunks=4,delay=.2):
    rows=[]
    for i in range(int(duration*rate)):
        client=i%4 if traffic=='uniform' else (0 if i%10<8 else 1+(i//10+i%2)%3)
        rows.append(dict(id=f'planned-{i:06}',offset_seconds=i/rate,client=client,
            payload=dict(prompt=f'prefix-client-{client}: movie plot',session_id=f'dialog-{client}',
                messages=[dict(role='user',content=f'CONTROL_DIALOG_{client}')],max_chunks=chunks,chunk_delay_seconds=delay)))
    return rows


def run_clients(out,rows,service='hw11-api',duration=None,epoch=None,timeout=45):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    clients=ensure_clients();epoch=epoch or time.time()+8
    run_id=str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else out.name
    save(out/'schedule.json',rows)
    save(out/'config.json',dict(run_id=run_id,service=service,epoch_utc=epoch,duration_seconds=duration,
        schedule_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        clients=clients,keep_alive=False,max_outstanding=128,timeout_seconds=timeout))
    processes=[]
    for index in range(4):
        name=f'hw11-client-{index}';subset=[dict(r,attempt_id=f'{hashlib.sha256(run_id.encode()).hexdigest()[:16]}-{r["id"]}') for r in rows if r['client']==index]
        for row in subset:row['payload']={**row['payload'],'request_id':row['attempt_id']}
        config=dict(url=f'http://{service}:8000',epoch_utc=epoch,schedule=subset,client=name,
            client_ip=clients[name],run_id=run_id,timeout_seconds=timeout,max_outstanding=128)
        stream=(out/f'client-{index}.jsonl').open('w')
        errors=(out/f'client-{index}.stderr').open('w')
        process=subprocess.Popen([KUBECTL,'--kubeconfig',KUBECONFIG,'-n',NAMESPACE,'exec','-i',name,'--',
            'python','-m','hw11.client','--config','-','--out',f'/tmp/{out.name}-{index}.jsonl'],
            stdin=subprocess.PIPE,stdout=stream,stderr=errors,text=True)
        process.stdin.write(json.dumps(config));process.stdin.close()
        processes.append((process,stream,errors))
    deadline=time.monotonic()+(duration or max([r['offset_seconds'] for r in rows],default=0))+timeout+40
    try:
        for process,_,_ in processes:
            code=process.wait(timeout=max(1,deadline-time.monotonic()))
            if code:raise RuntimeError(f'Client exec exit {code}; see {out}')
    finally:
        for process,stream,errors in processes:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            stream.close();errors.close()
    # An arrival phase remains open for the full declared duration even when
    # its final fast request finishes before the last scheduled second.
    if duration is not None:
        remaining=epoch+duration-time.time()
        if remaining>0:time.sleep(remaining)
    config_path=out/'config.json'
    config=json.loads(config_path.read_text())
    config.update(measurement_started_utc=epoch,measurement_ended_utc=time.time(),
                  actual_phase_seconds=time.time()-epoch)
    save(config_path,config)
    records=[]
    for index in range(4):
        for line in (out/f'client-{index}.jsonl').read_text().splitlines():
            records.append(json.loads(line))
    with (out/'requests.jsonl').open('w') as stream:
        for row in sorted(records,key=lambda r:r['offset_seconds']):stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary=summarize(records);save(out/'summary.json',summary)
    return records


def pod_http(ip,path,payload=None):
    script='import json,urllib.request; r=urllib.request.urlopen(urllib.request.Request('+repr('http://'+ip+':8000'+path)+',data='+repr(None if payload is None else json.dumps(payload).encode())+",headers={'Content-Type':'application/json'}),timeout=5); print(r.read().decode())"
    result=kube('exec','hw11-client-0','--','python','-c',script,timeout=15)
    return result.stdout


def metrics_snapshot(out,name):
    snapshot={}
    for pod in ready_pods():
        podname=pod['metadata']['name'];snapshot[podname]=pod_http(pod['status']['podIP'],'/metrics')
    captured=time.time()
    save(Path(out)/f'metrics-{name}.json',dict(utc=captured,pods=snapshot))
    expected={}
    for pod,text in snapshot.items():
        for line in text.splitlines():
            metric=line.split(' ',1)[0]
            if metric in ('hw11_ttfc_seconds_count','hw11_request_duration_seconds_count'):
                expected[(pod,metric)]=float(line.split()[-1])
    query='{__name__=~"hw11_ttfc_seconds_bucket|hw11_ttfc_seconds_count|hw11_request_duration_seconds_bucket|hw11_request_duration_seconds_count"}'
    with forward('service/hw11-prometheus',9090) as url:
        deadline=time.monotonic()+15
        while True:
            result=prometheus(url,query)
            vector=result.get('data',{}).get('data',{}).get('result',[])
            observed={(entry['metric'].get('pod'),entry['metric'].get('__name__')):float(entry['value'][1])
                      for entry in vector if entry['metric'].get('__name__','').endswith('_count')}
            matched=bool(expected) and all(observed.get(key)==value for key,value in expected.items())
            save(Path(out)/f'prom-{name}.json',dict(utc=time.time(),direct_snapshot_utc=captured,
                query=query,matched_direct_counts=matched,expected_counts=[dict(pod=p,metric=m,value=v)
                    for (p,m),v in expected.items()],**result))
            if matched:break
            if time.monotonic()>deadline:raise RuntimeError('Prometheus quiet cohort did not match exporter counters')
            time.sleep(.5)
    return snapshot


def reset_cache(out):
    states={}
    for pod in ready_pods():states[pod['metadata']['name']]=json.loads(pod_http(pod['status']['podIP'],'/reset-cache',{}))
    save(Path(out)/'cache-initial.json',dict(utc=time.time(),reset=states))


def smoke(out):
    wait_ready();ensure_clients()
    with Collector(out):
        rows=run_clients(Path(out)/'clients',schedule(4,2,'uniform',2,.05),duration=4)
    if not rows or any(r['outcome']!='completed' for r in rows):raise RuntimeError('Smoke did not complete')
    print(json.dumps({'smoke':'PASS','requests':len(rows)}))


def experiment_b(out,duration=120,repeats=3,rate=2):
    out=Path(out);upgrade(**{'hpa.enabled':False,'modelVersion':'A','neverReady':False,'brokenProbes':False,'warmupSeconds':20})
    rollout();ensure_clients()
    save(out/'network.json',dict(nodes=get('nodes'),clients=get('pods','app=hw11-client'),
        services=get('services'),kube_proxy=kube('get','configmap','kube-proxy','-n','kube-system','-o','yaml',check=False).stdout))
    # Same heavy-client profile below capacity: affinity need not cause queuing.
    for diagnostic,traffic,rps,seconds in [('no-skew','skewed',.5,20),('calibration','skewed',rate,15)]:
        target=out/diagnostic;target.mkdir(parents=True,exist_ok=True);reset_cache(target)
        with Collector(target):run_clients(target,schedule(seconds,rps,traffic),service='hw11-affinity',duration=seconds)
        if diagnostic=='calibration':
            names={p['metadata']['name'] for p in ready_pods()}
            examples=[]
            for line in (target/'prom-runtime.jsonl').read_text().splitlines():
                sample=json.loads(line);gauges={}
                for entry in sample.get('data',{}).get('data',{}).get('result',[]):
                    labels=entry['metric'];pod=labels.get('pod')
                    if pod in names and labels.get('__name__') in ('hw11_running','hw11_waiting'):
                        gauges.setdefault(pod,{})[labels['__name__']]=float(entry['value'][1])
                hot=[pod for pod,g in gauges.items() if g.get('hw11_waiting',0)>0]
                idle=[pod for pod,g in gauges.items() if g.get('hw11_running',1)==0]
                if any(a!=b for a in hot for b in idle):examples.append(dict(utc=sample['utc'],gauges=gauges))
            save(target/'calibration-decision.json',dict(passed=bool(examples),rate=rate,
                criterion='one current Pod waiting>0 while another current Pod running=0',examples=examples))
            if not examples:
                raise RuntimeError('Calibration did not show hot queue and idle peer; preserve this run and adjust rate before 12 runs')
    for policy,service in [('normal','hw11-api'),('clientip','hw11-affinity')]:
        for traffic in ('uniform','skewed'):
            rows=schedule(duration,rate,traffic)
            for repeat in range(1,repeats+1):
                target=out/policy/traffic/f'run-{repeat}';target.mkdir(parents=True,exist_ok=True)
                wait_ready();reset_cache(target)
                with Collector(target):
                    metrics_snapshot(target,'before')
                    run_clients(target,rows,service=service,duration=duration)
                    metrics_snapshot(target,'after')
                print(f'B {policy}/{traffic}/{repeat} completed',flush=True)


def wait_recovered(old_name,out,timeout=180):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        current=ready_pods()
        if len(current)>=2 and any(p['metadata']['name']!=old_name for p in current):
            save(Path(out)/'recovered.json',dict(utc=time.time(),elapsed_seconds=time.monotonic()-start,pods=current));return
        time.sleep(1)
    raise RuntimeError('API did not recover')


def stream_for_pod(out,pod,seconds=15):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Port-forward addresses one actual Pod and exposes a first-chunk sentinel.
    return forward('pod/'+pod['metadata']['name'])


def experiment_a(out):
    out=Path(out);ensure_clients()
    upgrade(**{'hpa.enabled':False,'modelVersion':'A','neverReady':False,'warmupSeconds':20,'brokenProbes':True})
    with Collector(out/'faulty'):
        faulty_rows=run_clients(out/'faulty',schedule(35,2,'uniform',2,.05),duration=35)
    early_faulty=[row for row in faulty_rows if row['status']==503 and 'backend_not_ready' in row.get('error_body','')]
    save(out/'faulty/warmup-assertion.json',dict(name='working_traffic_excludes_unready_backend',
        passed=not bool(early_faulty),expected_failure=True,unready_attempts=len(early_faulty)))
    if not early_faulty:raise RuntimeError('Original faulty probe failure was not reproduced')
    rollout()
    upgrade(brokenProbes=False)
    with Collector(out/'fixed'):
        fixed_rows=run_clients(out/'fixed',schedule(35,2,'uniform',2,.05),duration=35)
    early_fixed=[row for row in fixed_rows if row['status']==503 and 'backend_not_ready' in row.get('error_body','')]
    save(out/'fixed/warmup-assertion.json',dict(name='working_traffic_excludes_unready_backend',
        passed=not bool(early_fixed),expected_failure=False,unready_attempts=len(early_fixed)))
    if early_fixed:raise RuntimeError('Corrected readiness still delivered traffic before warmup')
    rollout()
    # Long stream is opened directly to an old Pod; background traffic uses Service.
    pod=ready_pods()[0];target=out/'rollout'
    with Collector(target),forward('pod/'+pod['metadata']['name']) as url,ThreadPoolExecutor(max_workers=2) as pool:
        item=dict(id='long-stream',attempt_id='a-rollout-long',offset_seconds=0,
            payload=dict(request_id='a-rollout-long',session_id='long-rollout',prompt='long rollout',max_chunks=28,chunk_delay_seconds=1))
        future=pool.submit(attempt,url,item,time.time(),50)
        time.sleep(2);started=time.time();upgrade(modelVersion='B')
        background=pool.submit(run_clients,target/'background',schedule(60,1,'uniform',2,.05),duration=60)
        rollout();row=future.result();background.result()
        save(target/'long-stream.json',dict(request=row,rollout_started_utc=started,pod=pod['metadata']['name']))
        if row['outcome']!='completed' or row['first_chunk_utc'] is None or row['first_chunk_utc']>=started or row['end_utc']<=started:raise RuntimeError('Long stream failed rollout boundary')
    upgrade(modelVersion='A');rollout()
    # Record stable A revision, deploy deliberately never-ready B, then actual helm rollback.
    history=json.loads(command([HELM,'--kubeconfig',KUBECONFIG,'history','hw11','-n',NAMESPACE,'-o','json']).stdout)
    revision=history[-1]['revision'];target=out/'bad-rollout'
    with Collector(target):
        upgrade(modelVersion='B',neverReady=True)
        save(target/'bad-values.json',json.loads(command([HELM,'--kubeconfig',KUBECONFIG,
            'get','values','hw11','-n',NAMESPACE,'-o','json']).stdout))
        save(target/'bad-deployment.json',get('deployment/hw11-api'))
        save(target/'bad-runtime-config.json',get('configmap/hw11-runtime'))
        rows=run_clients(target,schedule(35,1,'uniform',2,.05),duration=35)
        command([HELM,'--kubeconfig',KUBECONFIG,'rollback','hw11',str(revision),'-n',NAMESPACE],timeout=90)
        rollout()
        save(target/'rollback-revision.json',dict(revision=revision))
    with Collector(out/'rollback'):
        run_clients(out/'rollback',schedule(5,2,'uniform',2,.05),duration=5)
    # Kill the container that owns an active response, not an arbitrary replica.
    pod=ready_pods()[0];target=out/'forced-kill'
    with Collector(target),forward('pod/'+pod['metadata']['name']) as url,ThreadPoolExecutor(max_workers=1) as pool:
        item=dict(id='forced-stream',attempt_id='a-forced-stream',offset_seconds=0,
            payload=dict(request_id='a-forced-stream',session_id='forced',prompt='forced stream',max_chunks=24,chunk_delay_seconds=1))
        future=pool.submit(attempt,url,item,time.time(),40);time.sleep(2)
        container_id=pod['status']['containerStatuses'][0]['containerID'].split('://',1)[1]
        killed=time.time()
        kill_command=['docker','--context','colima-mlops-hw11','exec',pod['spec']['nodeName'],
                      'ctr','--namespace','k8s.io','tasks','kill','--signal','SIGKILL',container_id]
        result=command(kill_command,check=False)
        row=future.result();save(target/'forced-stream.json',dict(request=row,kill_utc=killed,
            kill_exit=result.returncode,kill_command=kill_command,kill_stderr=result.stderr,
            container_id=container_id,old_container_status=pod['status']['containerStatuses']))
        if result.returncode:raise RuntimeError('Container runtime SIGKILL failed')
        # Container can restart in the same Pod; wait readiness and restart count increase.
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            current=ready_pods()
            if len(current)==2 and any(p['metadata']['name']==pod['metadata']['name'] and
                any(c.get('restartCount',0)>pod['status']['containerStatuses'][0].get('restartCount',0)
                    and c.get('containerID','').split('://')[-1]!=container_id
                    for c in p['status'].get('containerStatuses',[])) for p in current):break
            time.sleep(1)
        else:raise RuntimeError('Killed container did not recover')
        run_clients(target/'recovery',schedule(5,2,'uniform',2,.05),duration=5)
        save(target/'recovered-containers.json',dict(utc=time.time(),pods=current))
        if not (row['outcome']=='interrupted' and row['chunks']>0 and not row['final_marker']
                and row['first_chunk_utc']<killed<row['end_utc']):
            raise RuntimeError('Forced kill did not interrupt an already started stream')
    # Actual replacement (not only restart) and stateless dialog continuation.
    target=out/'replacement';initial_pods=ready_pods();pod=initial_pods[0];before=pod['metadata']['name']
    initial_names={p['metadata']['name'] for p in initial_pods}
    with Collector(target):
        dialog_payloads=[dict(prompt='continue',messages=[dict(role='user',content=f'ISOLATED_DIALOG_{i}'),
            dict(role='assistant',content=f'ACK_{i}')],session_id=f'isolation-{i}',max_chunks=2,chunk_delay_seconds=.02) for i in (1,2)]
        rows=[dict(id=f'before-{i}',offset_seconds=i*.2,client=i,payload=p) for i,p in enumerate(dialog_payloads)]
        run_clients(target/'before',rows,duration=1)
        kube('delete','pod',before,'--wait=false');wait_recovered(before,target)
        replacement=next(p for p in ready_pods() if p['metadata']['name'] not in initial_names)
        save(target/'replacement-identity.json',dict(old_pod=before,new_pod=replacement['metadata']['name']))
        for i,payload in enumerate(dialog_payloads):rows[i]=dict(id=f'after-{i}',offset_seconds=i*.2,client=i,payload=payload)
        run_clients(target/'after',rows,service=replacement['status']['podIP'],duration=1)
    experiment_drain(out/'drain-proof')
    print('A completed',flush=True)


def experiment_drain(out):
    """Prove that a real terminating old Pod finishes an active stream after preStop.

    The original A warmup20 evidence stays unchanged. This separate experiment
    uses warmup10 so the first old Pod reaches SIGTERM during a 28-second stream.
    Starting streams on both old Pods removes dependence on deletion selection.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    upgrade(**{'hpa.enabled':False,'modelVersion':'A','neverReady':False,
               'brokenProbes':False,'warmupSeconds':10})
    rollout();ensure_clients();old=ready_pods()
    if len(old)!=2:raise RuntimeError('Drain proof needs exactly two old Ready A Pods')
    before=get('deployment/hw11-api')
    save(out/'deployment-before.json',before)
    save(out/'values-before.json',json.loads(command([HELM,'--kubeconfig',KUBECONFIG,
        'get','values','hw11','-n',NAMESPACE,'-o','json']).stdout))
    save(out/'old-pods.json',old)
    grace=before['spec']['template']['spec']['terminationGracePeriodSeconds']
    with Collector(out),forward('pod/'+old[0]['metadata']['name']) as first_url,\
            forward('pod/'+old[1]['metadata']['name']) as second_url,ThreadPoolExecutor(max_workers=3) as pool:
        futures=[]
        for index,url in enumerate((first_url,second_url)):
            identifier=f'drain-proof-long-{index}'
            item=dict(id=identifier,attempt_id=identifier,offset_seconds=0,
                payload=dict(request_id=identifier,session_id=identifier,prompt='drain proof long answer',
                             max_chunks=28,chunk_delay_seconds=1))
            futures.append(pool.submit(attempt,url,item,time.time(),50))
        time.sleep(2)
        upgrade_utc=time.time();upgrade(modelVersion='B')
        background=pool.submit(run_clients,out/'background',schedule(60,1,'uniform',2,.05),duration=60)
        rollout();long_rows=[future.result() for future in futures];background_rows=background.result()
        with (out/'requests.jsonl').open('w') as stream:
            for row in long_rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        save(out/'summary.json',summarize(long_rows))
    deleted={}
    for line in (out/'timeline.jsonl').read_text().splitlines():
        sample=json.loads(line)
        for pod in sample['data'].get('items',[]):
            if (pod.get('kind')=='Pod' and pod['metadata'].get('deletionTimestamp')
                    and pod['metadata']['name'] not in deleted):
                metadata=pod['metadata']
                deadline=datetime.fromisoformat(metadata['deletionTimestamp'].replace('Z','+00:00')).timestamp()
                deletion_grace=metadata.get('deletionGracePeriodSeconds',grace)
                # Kubernetes records the deletion deadline, not the instant
                # the graceful deletion request began.
                deleted[metadata['name']]=dict(termination_started_utc=deadline-deletion_grace,
                    deletion_deadline_utc=deadline,deletion_grace_seconds=deletion_grace,
                    first_observed_terminating_utc=sample['utc'])
    streams=[]
    for row in long_rows:
        metadata=deleted.get(row['pod'],{})
        deletion=metadata.get('termination_started_utc')
        observed=metadata.get('first_observed_terminating_utc')
        streams.append(dict(request_id=row['attempt_id'],pod=row['pod'],outcome=row['outcome'],
            first_chunk_utc=row['first_chunk_utc'],end_utc=row['end_utc'],deletion_utc=deletion,
            expected_sigterm_utc=None if deletion is None else deletion+5,
            deletion_to_finish_seconds=None if deletion is None else row['end_utc']-deletion,
            finished_after_prestop=deletion is not None and max(deletion,observed)+6<row['end_utc']<=metadata['deletion_deadline_utc'],
            **metadata))
    long_pass=all(row['outcome']=='completed' and row['error'] is None and row['first_chunk_utc']<upgrade_utc<row['end_utc']
                  for row in long_rows)
    background_pass=bool(background_rows) and all(row['outcome']=='completed' and row['status']==200 for row in background_rows)
    passed=long_pass and background_pass and any(item['finished_after_prestop'] for item in streams)
    save(out/'proof.json',dict(passed=passed,warmup_seconds=10,upgrade_utc=upgrade_utc,
        termination_grace_seconds=grace,prestop_seconds=5,uvicorn_grace_seconds=50,
        streams=streams,background=summarize(background_rows)))
    if not passed:raise RuntimeError('Drain proof failed; preserve actual streams/timeline and inspect proof.json')
    print('Drain proof completed',flush=True)


def prometheus(url,query):
    from urllib.parse import urlencode
    status,data=http(url+'/api/v1/query?'+urlencode({'query':query}))
    return dict(status=status,data=data)


def experiment_c(out):
    """Observe real CPU autoscaling and Prometheus rule state transitions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Preserve the exact B backend. CPU pressure is changed through arrivals and
    # inter-chunk delay, not through different hit/miss implementations.
    upgrade(**{'hpa.enabled': True, 'modelVersion': 'A', 'neverReady': False,
               'brokenProbes': False, 'hitIterations': 75000,
               'missIterations': 700000, 'chunkIterations': 25000})
    wait_ready(2, 180)
    ensure_clients()
    save(out/'metrics-api.json', json.loads(kube('get', '--raw',
         '/apis/metrics.k8s.io/v1beta1/pods').stdout))
    save(out/'hpa-before.json', get('hpa/hw11-api'))
    target = out/'hpa'
    with Collector(target):
        rows = run_clients(target/'load', schedule(120, 30, 'uniform', 4, .005),
                           duration=120, timeout=45)
        config = json.loads((target/'load/config.json').read_text())
        arrivals_stopped = config['epoch_utc'] + config['duration_seconds']
        save(target/'load-window.json', dict(start_utc=config['epoch_utc'],
             stop_utc=arrivals_stopped, clients_drained_utc=time.time()))
        deadline = time.monotonic()+240
        observed_three = False
        minimum_at = None
        while time.monotonic() < deadline:
            # The collector preserves creation, readiness, recommendations and
            # conditions continuously, including events before the load drains.
            for line in (target/'timeline.jsonl').read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    # A final JSONL record can still be in flight in the
                    # collector thread. The completed file is checked later.
                    continue
                for item in sample['data'].get('items', []):
                    if (item.get('kind') == 'HorizontalPodAutoscaler'
                            and item.get('status', {}).get('currentReplicas') == 3):
                        observed_three = True
            hpa = get('hpa/hw11-api')
            if (observed_three and hpa.get('status', {}).get('currentReplicas') == 2
                    and len(ready_pods()) == 2):
                minimum_at = time.time()
                break
            time.sleep(2)
        if minimum_at is None:
            raise RuntimeError('HPA did not complete measured 2->3->2; inspect preserved timeline')
        save(target/'scale-down.json', dict(observed_three=True,
             load_stopped_utc=arrivals_stopped, minimum_observed_utc=minimum_at,
             scale_down_seconds=minimum_at-arrivals_stopped,
             stabilization_seconds=30, hpa=hpa))
    with Collector(out/'hot-replica'):
        run_clients(out/'hot-replica', schedule(75, 30, 'skewed', 4, .005),
                    service='hw11-affinity', duration=75, timeout=45)
    target = out/'alerts'
    target.mkdir(parents=True, exist_ok=True)
    with Collector(target), forward('service/hw11-prometheus', 9090) as url, ThreadPoolExecutor(max_workers=1) as pool:
        observations = []
        # Record inactive rules before admission pressure. /alerts omits inactive;
        # /rules includes each alert's state (inactive/pending/firing).
        def sample_rules():
            status, data = http(url+'/api/v1/rules')
            entry = dict(utc=time.time(), status=status, data=data)
            observations.append(entry)
            save(target/'alert-states.json', observations)
            return [r for g in data.get('data', {}).get('groups', [])
                    for r in g.get('rules', []) if r.get('type') == 'alerting']
        baseline_deadline = time.monotonic() + 120
        while True:
            rules = sample_rules()
            states = [json.loads(pod_http(p['status']['podIP'], '/debug')) for p in ready_pods()]
            idle = bool(states) and all(state['running'] == 0 and state['waiting'] == 0 for state in states)
            if idle and len(rules) >= 2 and all(r['state'] == 'inactive' for r in rules):
                save(target/'baseline.json', dict(utc=time.time(), idle=True, rules=rules))
                break
            if time.monotonic() >= baseline_deadline:
                raise RuntimeError('Previous hot load did not clear; alerts need an inactive baseline')
            time.sleep(2)
        future = pool.submit(run_clients, target/'load',
                             schedule(100, 24, 'uniform', 8, .15),
                             duration=100, timeout=45)
        deadline = time.monotonic()+260
        seen = {}
        while time.monotonic() < deadline:
            rules = sample_rules()
            for rule in rules:
                seen.setdefault(rule['name'], set()).add(rule['state'])
            all_cycles = len(seen) >= 2 and all({'pending', 'firing'} <= states for states in seen.values())
            if future.done() and all_cycles and rules and all(r['state'] == 'inactive' for r in rules):
                break
            time.sleep(1)
        future.result()
        rules = sample_rules()
        cycles = {name: sorted(states) for name, states in seen.items()}
        save(target/'transitions.json', dict(states_seen=cycles,
             all_inactive=bool(rules) and all(r['state'] == 'inactive' for r in rules),
             passed=len(seen) >= 2 and all({'pending', 'firing'} <= states for states in seen.values())
                    and bool(rules) and all(r['state'] == 'inactive' for r in rules)))
        save(target/'rules.json', http(url+'/api/v1/rules')[1])
        if not json.loads((target/'transitions.json').read_text())['passed']:
            raise RuntimeError('Both actual alerts must show pending, firing and recovery; preserved raw states')
    save(out/'hpa-after.json', get('hpa/hw11-api'))
    print('C completed', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['smoke','a','b','c','drain'])
    parser.add_argument('--results',type=Path,default=ROOT/'results/hw11')
    parser.add_argument('--duration',type=float,default=120)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--rate',type=float,default=2)
    args=parser.parse_args();args.results.mkdir(parents=True,exist_ok=True)
    if args.experiment=='smoke':smoke(args.results/'smoke')
    elif args.experiment=='a':experiment_a(args.results/'a')
    elif args.experiment=='b':experiment_b(args.results/'b',args.duration,args.repeats,args.rate)
    elif args.experiment=='drain':experiment_drain(args.results/'a/drain-proof')
    else:experiment_c(args.results/'c')


if __name__=='__main__':main()
