"""Deterministic open-loop HTTP streaming client (also runs inside client Pods)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from urllib.parse import urlsplit


def classify(status, completed, chunks, error=None):
    if status is not None and status >= 400:
        return 'rejected' if status in (429, 503) else 'error'
    if error == 'timeout':
        return 'timeout'
    if error:
        return 'interrupted' if status == 200 or chunks else 'connection_error'
    if completed:
        return 'completed'
    if status == 200 or chunks:
        return 'interrupted'
    return 'connection_error'


def attempt(base_url, item, epoch, timeout=45):
    parsed = urlsplit(base_url)
    planned = epoch + item['offset_seconds']
    sent_wall, sent_mono = time.time(), time.monotonic()
    payload = item['payload']
    row = dict(item, planned_utc=planned, sent_utc=sent_wall, sent_monotonic=sent_mono,
               send_lag_seconds=sent_wall-planned, status=None, chunks=0, final_marker=False,
               first_chunk_monotonic=None, first_chunk_utc=None, error=None,
               inter_chunk_seconds=[], pod=None, model_version=None, artifact_sha256=None,
               backend_mode=None, queue_seconds=None, cache_hit=None)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    previous = None
    try:
        connection.request('POST', (parsed.path.rstrip('/') or '')+'/stream',
            body=json.dumps(payload).encode(), headers={'Content-Type':'application/json','Connection':'close'})
        response = connection.getresponse()
        row['status'] = response.status
        for field, header in (('pod','X-Pod-Name'),('model_version','X-Model-Version'),
                              ('artifact_sha256','X-Artifact-Sha256'),('backend_mode','X-Backend-Mode')):
            row[field] = response.getheader(header)
        if response.status != 200:
            row['error_body'] = response.read(4096).decode(errors='replace')
        else:
            while True:
                line = response.readline()
                if not line:
                    break
                event = json.loads(line)
                now = time.monotonic()
                kind = event.get('type', event.get('event'))
                if kind == 'chunk':
                    if previous is None:
                        row['first_chunk_monotonic'], row['first_chunk_utc'] = now,time.time()
                    else:
                        row['inter_chunk_seconds'].append(now-previous)
                    previous = now
                    row['chunks'] += 1
                    row.setdefault('text','')
                    row['text'] += str(event.get('text',event.get('content','')))
                    for field in ('pod','model_version','artifact_sha256','backend_mode'):
                        row[field] = event.get(field,row[field])
                elif kind == 'done':
                    row['final_marker'] = event.get('chunks') == row['chunks'] and row['chunks'] > 0
                    row['done'] = event
                    for field in ('pod','model_version','artifact_sha256','backend_mode','queue_seconds','cache_hit'):
                        row[field] = event.get(field,row[field])
                    # Read through clean HTTP EOF. Closing immediately after
                    # the final JSON line can cancel server completion accounting.
    except OSError as exc:
        # OSError includes connection reset; only actual timeout is a timeout.
        row['error'] = 'timeout' if isinstance(exc,TimeoutError) else type(exc).__name__
    except Exception as exc:
        row['error'] = type(exc).__name__
    finally:
        connection.close()
    row['end_monotonic'],row['end_utc'] = time.monotonic(),time.time()
    row['full_seconds'] = row['end_monotonic']-sent_mono
    row['ttfc_seconds'] = None if row['first_chunk_monotonic'] is None else row['first_chunk_monotonic']-sent_mono
    row['inter_chunk_latency_seconds'] = statistics.mean(row['inter_chunk_seconds']) if row['inter_chunk_seconds'] else None
    row['outcome'] = classify(row['status'],row['final_marker'],row['chunks'],row['error'])
    return row


def run(config, output):
    """Do not block arrivals on workers; explicitly record excess backlog as unsent."""
    lock = threading.Lock()
    maximum = config.get('max_outstanding',128)
    slots = threading.BoundedSemaphore(maximum)
    epoch = config['epoch_utc']
    origin_monotonic = time.monotonic() + epoch - time.time()
    items = sorted(config['schedule'],key=lambda x:x['offset_seconds'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',buffering=1) as stream, ThreadPoolExecutor(max_workers=maximum) as pool:
        def save(row):
            row.update(client=config.get('client'),client_ip=config.get('client_ip'),
                       run_id=config.get('run_id'))
            with lock:
                line=json.dumps(row,ensure_ascii=False,allow_nan=False)
                stream.write(line+'\n'); stream.flush()
                print(line,flush=True)
        def work(item):
            try:
                save(attempt(config['url'],item,epoch,config.get('timeout_seconds',45)))
            finally:
                slots.release()
        for item in items:
            remaining=origin_monotonic+item['offset_seconds']-time.monotonic()
            if remaining>0:
                time.sleep(remaining)
            if not slots.acquire(blocking=False):
                save(dict(item,planned_utc=epoch+item['offset_seconds'],sent_utc=None,
                          outcome='unsent',error='client_backlog_limit'))
            else:
                pool.submit(work,item)


def summarize(rows):
    good=[r for r in rows if r['outcome']=='completed']
    counts={key:sum(r['outcome']==key for r in rows) for key in
            ('completed','rejected','interrupted','timeout','connection_error','error','unsent')}
    def quantile(key,p):
        values=sorted(r[key] for r in good if r.get(key) is not None)
        if not values:return None
        position=(len(values)-1)*p;lo=int(position);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return dict(planned=len(rows),sent=len(rows)-counts['unsent'],successful=len(good),counts=counts,
        ttfc_p50_seconds=quantile('ttfc_seconds',.5),ttfc_p95_seconds=quantile('ttfc_seconds',.95),
        full_p50_seconds=quantile('full_seconds',.5),full_p95_seconds=quantile('full_seconds',.95),
        pods={pod:sum(r.get('pod')==pod for r in rows) for pod in sorted({r.get('pod') for r in rows if r.get('pod')})})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',type=Path,default=Path('/tmp/hw11-client.jsonl'))
    args=parser.parse_args()
    config=json.load(sys.stdin) if args.config=='-' else json.loads(Path(args.config).read_text())
    run(config,args.out)


if __name__=='__main__':main()
