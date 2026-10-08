import argparse, json, time
from pathlib import Path
import httpx

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://127.0.0.1:8000');ap.add_argument('--model',default='tinyllama')
    a=ap.parse_args();results=[]
    with httpx.Client(base_url=a.url.rstrip('/'),timeout=300) as client:
        r=client.get('/health');r.raise_for_status()
        r=client.get('/v1/models');r.raise_for_status()
        assert a.model in [m['id'] for m in r.json()['data']],r.text
        print('PASS health and model listing')
        for name,body in [('single-turn',{'messages':[{'role':'user','content':'Explain what a container is in one sentence.'}]}),
           ('multi-turn',{'messages':[{'role':'user','content':'My name is Gaurav.'},{'role':'assistant','content':'Hello Gaurav.'},
              {'role':'user','content':'What name did I tell you?'}]})]:
            start=time.perf_counter()
            r=client.post('/v1/chat/completions',json={'model':a.model,'temperature':0,'max_tokens':64,**body});r.raise_for_status()
            data=r.json();content=data['choices'][0]['message']['content'];assert content and content.strip(),data
            elapsed=time.perf_counter()-start;tokens=data['usage']['completion_tokens']
            result={'case':name,'latency_seconds':round(elapsed,3),'completion_tokens':tokens,
              'end_to_end_tokens_per_second':round(tokens/elapsed,3),'response':content}
            results.append(result);print(json.dumps(result,indent=2))
        start=time.perf_counter();first=None;chunks=[]
        with client.stream('POST','/v1/chat/completions',json={'model':a.model,'stream':True,'temperature':0,'max_tokens':32,
              'messages':[{'role':'user','content':'Say hello in one sentence.'}]}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):continue
                payload=line[6:]
                if payload=='[DONE]':break
                event=json.loads(payload)
                for choice in event.get('choices',[]):
                    content=choice.get('delta',{}).get('content')
                    if content:
                        if first is None:first=time.perf_counter()-start
                        chunks.append(content)
        assert first is not None and chunks,'No streamed content'
        results.append({'case':'streaming','ttft_seconds':round(first,3),'latency_seconds':round(time.perf_counter()-start,3),'response':''.join(chunks)})
        print('PASS streaming:',results[-1])
        r=client.get('/metrics');r.raise_for_status();assert 'vllm:' in r.text,'No vLLM metrics found'
        print('PASS metrics')
    out=Path(__file__).resolve().parents[1]/'results';out.mkdir(exist_ok=True)
    (out/'smoke-test.json').write_text(json.dumps(results,indent=2))
    print('All API smoke tests passed. Review response quality separately.')
if __name__=='__main__':main()
