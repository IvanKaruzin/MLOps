"""Client accounting and streaming protocol checks without real model execution."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hw11.client import attempt,classify,run,summarize


class Response:
    def __init__(self,status,events):
        self.status=status
        self.body=io.BytesIO(b''.join(json.dumps(event).encode()+b'\n' for event in events))
    def getheader(self,key):
        return {'X-Pod-Name':'pod-a','X-Model-Version':'A','X-Artifact-Sha256':'a'*64,'X-Backend-Mode':'simulated'}.get(key)
    def readline(self):return self.body.readline()
    def read(self,*args):return self.body.read(*args)


class ClientTests(unittest.TestCase):
    def item(self):return dict(id='one',attempt_id='a',offset_seconds=0,payload={'prompt':'test'})
    def call(self,response):
        with patch('hw11.client.http.client.HTTPConnection') as connection:
            connection.return_value.getresponse.return_value=response
            return attempt('http://service:8000',self.item(),time.time())
    def test_complete_and_single_chunk_interval(self):
        row=self.call(Response(200,[{'type':'chunk','content':'hello'},{'type':'done','chunks':1}]))
        self.assertEqual(row['outcome'],'completed')
        self.assertEqual(row['text'],'hello')
        self.assertIsNone(row['inter_chunk_latency_seconds'])
        self.assertGreaterEqual(row['full_seconds'],row['ttfc_seconds'])
    def test_eof_without_marker_is_interrupted(self):
        row=self.call(Response(200,[{'type':'chunk','content':'part'}]))
        self.assertEqual(row['outcome'],'interrupted')
        self.assertFalse(row['final_marker'])
    def test_wrong_chunk_count_cannot_be_completed(self):
        row=self.call(Response(200,[{'type':'chunk','content':'part'},{'type':'done','chunks':2}]))
        self.assertEqual(row['outcome'],'interrupted')
    def test_transport_error_after_marker_is_still_an_interrupted_stream(self):
        response=Response(200,[])
        events=[{'type':'chunk','content':'part'},{'type':'done','chunks':1}]
        with patch.object(response,'readline',side_effect=[
                *(json.dumps(event).encode()+b'\n' for event in events),ConnectionResetError()]):
            row=self.call(response)
        self.assertTrue(row['final_marker'])
        self.assertEqual(row['outcome'],'interrupted')
        self.assertEqual(row['error'],'ConnectionResetError')
    def test_rejection_and_timeout_remain_distinct(self):
        self.assertEqual(self.call(Response(429,[]))['outcome'],'rejected')
        self.assertEqual(classify(None,False,0,'timeout'),'timeout')
        self.assertEqual(classify(None,False,0,'ConnectionRefusedError'),'connection_error')
        self.assertEqual(classify(500,False,0),'error')
    def test_identical_schedule_preserves_planned_and_actual_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'requests.jsonl';epoch=time.time()-.01
            with patch('hw11.client.attempt',side_effect=lambda url,item,epoch,timeout:dict(item,outcome='completed')):
                run(dict(url='http://service',epoch_utc=epoch,schedule=[self.item()],max_outstanding=1),target)
            row=json.loads(target.read_text())
            self.assertEqual(row['offset_seconds'],0)
            self.assertEqual(row['attempt_id'],'a')
    def test_full_client_backlog_is_recorded_unsent_without_delaying_arrival(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'requests.jsonl'
            items=[dict(self.item(),id=str(i),attempt_id=str(i)) for i in range(2)]
            def slow(url,item,epoch,timeout):
                time.sleep(.03)
                return dict(item,outcome='completed')
            with patch('hw11.client.attempt',side_effect=slow):
                run(dict(url='http://service',epoch_utc=time.time(),schedule=items,max_outstanding=1),target)
            rows=[json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual({r['outcome'] for r in rows},{'completed','unsent'})
            self.assertEqual(len(rows),2)


if __name__=='__main__':unittest.main()
