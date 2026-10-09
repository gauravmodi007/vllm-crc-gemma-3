"""Cross-platform orchestration. oc commands target the CURRENT kube context."""
import argparse, json, platform, shutil, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.json"


def ensure_command(name: str):
    if shutil.which(name):
        return
    if name == 'oc':
        raise RuntimeError(
            "Missing required command 'oc' on PATH. Install the OpenShift CLI or load CRC's environment, then confirm the cluster context is your CRC cluster. "
            "On Windows PowerShell: 'crc oc-env | Invoke-Expression'. On Bash/zsh: 'eval \"$(crc oc-env)\"'."
        )
    if name == 'podman':
        raise RuntimeError(
            "Missing required command 'podman' on PATH. Install Podman or set a verified image digest in config.json and skip pin-image."
        )
    raise RuntimeError(f"Missing required command '{name}' on PATH.")


def run(args, capture=False, check=True):
    print("+", " ".join(args), flush=True)
    try:
        return subprocess.run(args, check=check, text=True, capture_output=capture)
    except FileNotFoundError as exc:
        tool = args[0]
        ensure_command(tool)
        raise RuntimeError(f"Could not execute '{tool}'.") from exc

def config():
    c=json.loads(CONFIG.read_text())
    if not c['namespace'] or c['namespace'] in ('default','openshift','kube-system'):
        raise ValueError('Use a dedicated POC namespace.')
    return c

def generate(c):
    ns=c['namespace']; labels={'app':c['deployment_name']}
    def obj(kind,name,spec=None,api='v1'):
        r={'apiVersion':api,'kind':kind,'metadata':{'name':name,'namespace':ns}}
        if spec is not None:r['spec']=spec
        return r
    args=[c['model'],'--host','0.0.0.0','--port','8000','--served-model-name',c['served_model_name'],
          '--dtype','bfloat16','--max-model-len',str(c['max_model_len']),
          '--quantization',c['quantization'],
          '--kv-cache-memory-bytes',str(c['kv_cache_memory_bytes']),
          '--max-num-seqs','1','--max-num-batched-tokens','128','--enforce-eager']
    if c.get('model_revision'):args += ['--revision',c['model_revision'],'--tokenizer-revision',c['model_revision']]
    env={'HOME':'/tmp','HF_HOME':'/cache/huggingface','VLLM_CACHE_ROOT':'/cache/vllm',
         'XDG_CACHE_HOME':'/cache','VLLM_CPU_OMP_THREADS_BIND':'nobind',
         'OMP_NUM_THREADS':c['cpu_limit'],'TOKENIZERS_PARALLELISM':'false','VLLM_NO_USAGE_STATS':'1'}
    env_vars=[{'name':k,'value':v} for k,v in env.items()]
    container={'name':'vllm','image':c['image'],'imagePullPolicy':'IfNotPresent',
      'command':['vllm','serve'],'args':args,'env':env_vars,
      'ports':[{'name':'http','containerPort':8000}],
      'resources':{'requests':{'cpu':c['cpu_request'],'memory':c['memory']},'limits':{'cpu':c['cpu_limit'],'memory':c['memory']}},
      'securityContext':{'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']},'runAsNonRoot':True},
      'volumeMounts':[{'name':'cache','mountPath':'/cache'},{'name':'shm','mountPath':'/dev/shm'}],
      'startupProbe':{'httpGet':{'path':'/health','port':'http'},'periodSeconds':10,'timeoutSeconds':5,'failureThreshold':180},
      'readinessProbe':{'httpGet':{'path':'/health','port':'http'},'periodSeconds':10,'timeoutSeconds':5},
      'livenessProbe':{'httpGet':{'path':'/health','port':'http'},'periodSeconds':30,'timeoutSeconds':10,'failureThreshold':10}}
    pod={'serviceAccountName':c['service_account_name'],'automountServiceAccountToken':False,
         'enableServiceLinks':False,'nodeSelector':{'kubernetes.io/arch':'amd64'},
         'securityContext':{'seccompProfile':{'type':'RuntimeDefault'}},
         'containers':[container],'volumes':[{'name':'cache','persistentVolumeClaim':{'claimName':c['cache_claim_name']}},
          {'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'1Gi'}}]}
    pvc={'accessModes':['ReadWriteOnce'],'resources':{'requests':{'storage':c['storage_size']}}}
    if c['storage_class'] is not None:pvc['storageClassName']=c['storage_class']
    if c.get('existing_pv'):pvc['volumeName']=c['existing_pv']
    resources=[obj('ServiceAccount',c['service_account_name']),obj('PersistentVolumeClaim',c['cache_claim_name'],pvc),
      obj('Deployment',c['deployment_name'],{'replicas':1,'strategy':{'type':'Recreate'},'selector':{'matchLabels':labels},
          'template':{'metadata':{'labels':labels},'spec':pod}},'apps/v1'),
      obj('Service',c['service_name'],{'selector':labels,'ports':[{'name':'http','port':8000,'targetPort':'http'}]})]
    # Preflight uses same UID policy and storage as serving; catches image/permissions/CPU issues.
    probe=r"""
import os, platform, subprocess
from pathlib import Path
print('Architecture:', platform.machine(), 'UID:', os.getuid(), flush=True)
assert platform.machine() == 'x86_64', 'CRC node must be amd64'
flags=Path('/proc/cpuinfo').read_text()
print('CPU flags:', next((x for x in flags.splitlines() if x.startswith('flags')), 'not found'), flush=True)
print('Allowed CPUs:', sorted(os.sched_getaffinity(0)), flush=True)
for directory in ('/cache/huggingface','/cache/vllm'):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    f=d/'preflight-write-test';f.write_text('ok');f.unlink()
import torch, vllm
print('vLLM:', vllm.__version__, 'PyTorch:', torch.__version__, flush=True)
a=torch.ones((64,64),dtype=torch.bfloat16)
assert float((a@a)[0,0]) == 64
subprocess.run(['vllm','serve','--help'],check=True)
from huggingface_hub import hf_hub_download
checkpoint=hf_hub_download(repo_id=os.environ['GEMMA_MODEL_ID'],
    filename='model.safetensors',revision=os.environ['GEMMA_MODEL_REVISION'])
print('Gemma INT8 W8A8 safetensors checkpoint download passed; cached bytes:',
    Path(checkpoint).stat().st_size, flush=True)
print('PREFLIGHT PASSED',flush=True)
"""
    import copy
    probe_pod=copy.deepcopy(pod)
    probe_pod['restartPolicy']='Never'
    pc=probe_pod['containers'][0]
    for k in ('startupProbe','readinessProbe','livenessProbe','ports'):pc.pop(k,None)
    pc.update(name='preflight',command=['python3','-u','-c'],args=[probe])
    pc['env'].extend([
        {'name':'GEMMA_MODEL_ID','value':c['model']},
        {'name':'GEMMA_MODEL_REVISION','value':c['model_revision']}])
    job=obj('Job',c['preflight_job_name'],{'backoffLimit':0,'activeDeadlineSeconds':1800,
      'template':{'metadata':{'labels':{'app':c['preflight_job_name']}},'spec':probe_pod}},'batch/v1')
    route=obj('Route',c['service_name'],{'to':{'kind':'Service','name':c['service_name']},
       'port':{'targetPort':'http'},'tls':{'termination':'edge','insecureEdgeTerminationPolicy':'Redirect'}},'route.openshift.io/v1')
    route['metadata']['annotations']={'haproxy.router.openshift.io/timeout':'300s'}
    out=ROOT/'rendered';out.mkdir(exist_ok=True)
    for name,items in [('app.yaml',resources),('preflight.yaml',[resources[0],resources[1],job]),('route.yaml',[route])]:
        (out/name).write_text(yaml.safe_dump_all(items,sort_keys=False))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['render','pin-image','inspect','preflight','deploy','status','diagnose','port-forward','route','stop'])
    a=ap.parse_args();c=config();ns=c['namespace']
    if a.action=='pin-image':
        ensure_command('podman')
        run(['podman','pull','--platform','linux/amd64',c['image']])
        data=json.loads(run(['podman','image','inspect',c['image']],True).stdout)[0]
        if data['Architecture'] != 'amd64':raise ValueError('Wrong image architecture')
        digests=data.get('RepoDigests',[])
        if not digests:raise ValueError('No registry digest found; do not deploy a guessed digest.')
        c['image']=digests[0];CONFIG.write_text(json.dumps(c,indent=2)+'\n')
        print('Pinned image:',c['image']);generate(c);return
    out=generate(c)
    if a.action=='render':print(out);return
    ensure_command('oc')
    run(['oc','whoami']);run(['oc','config','current-context'])
    def oc(*args,**kw):return run(['oc','-n',ns,*args],**kw)
    if a.action=='inspect':
        run(['oc','get','nodes','-o','wide']);run(['oc','get','storageclass']);run(['oc','get','pv']);
        run(['oc','get','nodes','-o','json']);return
    if a.action in ('preflight','deploy'):
        if '@sha256:' not in c['image']:raise ValueError('Pin image first (pin-image), or set a verified registry digest in config.json.')
        if a.action=='preflight':
            # Create a dedicated project only if it does not already exist.
            found=run(['oc','get','namespace',ns,'--ignore-not-found','-o','name'],True)
            if not found.stdout.strip():run(['oc','new-project',ns])
            oc('delete','job',c['preflight_job_name'],'--ignore-not-found=true')
            oc('apply','--dry-run=server','-f',str(out/'preflight.yaml'))
            oc('apply','-f',str(out/'preflight.yaml'))
            try:oc('wait','--for=condition=complete',f"job/{c['preflight_job_name']}",'--timeout=1800s')
            finally:oc('logs',f"job/{c['preflight_job_name']}",check=False)
        else:
            j=json.loads(oc('get','job',c['preflight_job_name'],'-o','json',capture=True).stdout)
            template=j['spec']['template']['spec']['containers'][0]
            if j.get('status',{}).get('succeeded',0)<1 or template['image']!=c['image']:
                raise ValueError('Run a successful preflight with this image first.')
            oc('apply','--dry-run=server','-f',str(out/'app.yaml'))
            oc('apply','-f',str(out/'app.yaml'))
            oc('rollout','status',f"deployment/{c['deployment_name']}",'--timeout=1800s')
    elif a.action=='port-forward':oc('port-forward',f"service/{c['service_name']}",'8000:8000','--address=127.0.0.1')
    elif a.action=='route':
        oc('apply','--dry-run=server','-f',str(out/'route.yaml'));oc('apply','-f',str(out/'route.yaml'))
        oc('get','route',c['service_name'])
    elif a.action=='stop':oc('scale',f"deployment/{c['deployment_name']}",'--replicas=0')
    elif a.action=='status':
        oc('get','deployment,pod,svc,pvc,job');oc('get','events','--sort-by=.lastTimestamp')
    elif a.action=='diagnose':
        for args in [('get','pods','-o','wide'),('describe','pods'),('describe','pvc',c['cache_claim_name']),
                     ('get','events','--sort-by=.lastTimestamp'),
                     ('logs',f"deployment/{c['deployment_name']}",'--tail=200'),
                     ('logs',f"deployment/{c['deployment_name']}",'--previous','--tail=200')]:oc(*args,check=False)
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,subprocess.CalledProcessError,FileNotFoundError) as e:
        print('ERROR:',e,file=sys.stderr);sys.exit(1)
