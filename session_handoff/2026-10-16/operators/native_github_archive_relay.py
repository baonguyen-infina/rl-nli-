"""Relay a Colab archive through a capped buffer into an owned private draft.

Checkpoint bytes are never written to this Mac. The GitHub credential remains
local, and the Colab credential is sent only to its original Colab host.
"""
import argparse,hashlib,json,pathlib,re,subprocess,time
from urllib.parse import quote,urlparse

BUFFER_BYTES=1024*1024
SESSIONS={
    'N15R18':'rl-nli-resume-r18-g4',
    'N16R19':'rl-nli-r19-native-swap-2b-g4',
    'N20R20':'rl-nli-resume-r18-g4',
    'N21R20':'rl-nli-r19-native-swap-2b-g4',
}

class BoundedReader:
    def __init__(self,raw,size,buffer_bytes=BUFFER_BYTES):
        self.raw=raw;self.size=size;self.buffer_bytes=buffer_bytes
        self.position=0;self.max_chunk=0;self.sha256=hashlib.sha256()
    def __len__(self):return self.size
    def tell(self):return self.position
    def read(self,n=-1):
        remaining=self.size-self.position
        if remaining==0:
            if self.raw.read(1):raise ValueError('Source archive exceeds declared size')
            return b''
        count=min(remaining,self.buffer_bytes,n if n is not None and n>=0 else self.buffer_bytes)
        if count==0:return b''
        data=self.raw.read(count)
        if not data:raise ValueError('Source archive ended early')
        if len(data)>count:raise ValueError('Source violated bounded read')
        self.position+=len(data);self.max_chunk=max(self.max_chunk,len(data));self.sha256.update(data)
        return data

def validate_spec(spec):
    if not re.fullmatch(r'(N15R18|N16R19|N20R20|N21R20)_s0_(checkpoint-(100|200|300|400)|final|logs-[0-9]+)\.tar\.gz',str(spec.get('name',''))):
        raise ValueError('Unexpected archive name')
    path=pathlib.PurePosixPath(str(spec.get('path','')))
    if str(path.parent)!='/content/native_github_staging' or path.name!=spec['name'] or '..' in path.parts:
        raise ValueError('Unexpected source path')
    if type(spec.get('bytes')) is not int or not 0<spec['bytes']<=2*1024**3:
        raise ValueError('Archive size outside release-asset bounds')
    if not re.fullmatch('[0-9a-f]{64}',str(spec.get('sha256',''))):raise ValueError('Invalid archive digest')
    return spec

def verify_asset(asset,spec):
    if (asset.get('name')!=spec['name'] or asset.get('size')!=spec['bytes'] or
        asset.get('state')!='uploaded' or asset.get('digest')!='sha256:'+spec['sha256']):
        raise ValueError('Remote archive identity or integrity mismatch')
    return True

def gh_json(endpoint,method='GET'):
    result=subprocess.run(['gh','api','--method',method,endpoint],capture_output=True,text=True,timeout=40)
    if result.returncode:raise RuntimeError('GitHub metadata request failed with exit '+str(result.returncode))
    return json.loads(result.stdout) if result.stdout.strip() else None

def verify_destination(state):
    repo=state['repo'];repository=gh_json('repos/'+repo)
    if not repository.get('private') or repository['owner']['login']!=state['account_login']:
        raise ValueError('Backup destination must be owned and private')
    release=gh_json('repos/'+repo+'/releases/'+str(state['release_id']))
    if not release.get('draft') or release.get('published_at') is not None:
        raise ValueError('Backup release must remain unpublished')
    return release

def relay(spec,state):
    import requests
    from colab_cli.common import State
    validate_spec(spec)
    repo=state['repo'];release=verify_destination(state)
    assets=[];page=1
    while True:
        batch=gh_json('repos/'+repo+'/releases/'+str(state['release_id'])+'/assets?per_page=100&page='+str(page))
        assets.extend(batch)
        if len(batch)<100:break
        page+=1
    matching=[value for value in assets if value['name']==spec['name']]
    if matching:
        if len(matching)!=1:raise ValueError('Ambiguous archive asset')
        if matching[0].get('state')=='starter' and matching[0].get('size')==0:
            # GitHub can leave this empty task-owned asset after a failed upload.
            gh_json('repos/'+repo+'/releases/assets/'+str(matching[0]['id']),method='DELETE')
            matching=[]
    if matching:
        verify_asset(matching[0],spec)
        verify_destination(state)
        return {'provider':'github-private-draft-release','id':matching[0]['id'],
            'name':spec['name'],'size':str(spec['bytes']),'sha256':spec['sha256'],
            'github_digest':matching[0]['digest'],'repo':repo,'release_id':state['release_id'],
            'repo_private':True,'release_draft':True,'reused':True,'model_bytes_written_locally':0,
            'bytes_relayed_through_memory':0,'max_buffer_bytes':0,'verified_at':time.time()}
    run=spec['name'].split('_',1)[0];session=State().get_session(SESSIONS[run])
    parsed=urlparse(session.url)
    if parsed.scheme!='https' or not (parsed.hostname.endswith('.colab.dev') or parsed.hostname.endswith('.colab.googleusercontent.com')):
        raise ValueError('Unexpected Colab proxy host')
    upload_url=release['upload_url'].split('{',1)[0]
    parsed_upload=urlparse(upload_url)
    expected='/repos/'+repo+'/releases/'+str(state['release_id'])+'/assets'
    if parsed_upload.scheme!='https' or parsed_upload.hostname!='uploads.github.com' or parsed_upload.path!=expected:
        raise ValueError('Unexpected GitHub asset endpoint')
    token=subprocess.run(['gh','auth','token'],capture_output=True,text=True,timeout=20)
    if token.returncode:raise RuntimeError('GitHub credential unavailable')
    credential=token.stdout.strip()
    source=None
    try:
        source=requests.get(session.url.rstrip('/')+'/files'+quote(spec['path'],safe='/'),
            params={'authuser':'0','colab-runtime-proxy-token':session.token},stream=True,
            allow_redirects=False,timeout=(15,120))
        if source.status_code!=200:raise RuntimeError('Colab archive HTTP '+str(source.status_code))
        if source.headers.get('Content-Length') and int(source.headers['Content-Length'])!=spec['bytes']:
            raise ValueError('Source content length changed')
        reader=BoundedReader(source.raw,spec['bytes'])
        uploaded=requests.post(upload_url,params={'name':spec['name']},data=reader,
            headers={'Authorization':'Bearer '+credential,'Accept':'application/vnd.github+json',
                'X-GitHub-Api-Version':'2026-03-10','Content-Type':'application/gzip',
                'Content-Length':str(spec['bytes'])},allow_redirects=False,timeout=(15,180))
        if uploaded.status_code!=201:raise RuntimeError('GitHub archive upload HTTP '+str(uploaded.status_code))
        if reader.position!=spec['bytes'] or reader.sha256.hexdigest()!=spec['sha256']:
            raise ValueError('Relayed archive digest changed')
        asset=uploaded.json()
    except requests.RequestException as exc:
        # Requests exception text can contain the private Colab proxy token.
        raise RuntimeError('Archive transport error: '+type(exc).__name__) from None
    finally:
        credential=None
        if source is not None:source.close()
    actual=gh_json('repos/'+repo+'/releases/assets/'+str(asset['id']))
    verify_asset(actual,spec)
    verify_destination(state)
    return {'provider':'github-private-draft-release','id':actual['id'],'name':spec['name'],
        'size':str(spec['bytes']),'sha256':spec['sha256'],'github_digest':actual['digest'],
        'repo':repo,'release_id':state['release_id'],'repo_private':True,'release_draft':True,
        'model_bytes_written_locally':0,'bytes_relayed_through_memory':reader.position,
        'max_buffer_bytes':reader.max_chunk,'verified_at':time.time()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    parser.add_argument('--receipt',required=True);options=parser.parse_args()
    state=json.loads(pathlib.Path('/private/tmp/native_github_archive_state.json').read_text())
    spec=json.loads(pathlib.Path(options.spec).read_text())
    result=relay(spec,state)
    destination=pathlib.Path(options.receipt);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(destination)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
