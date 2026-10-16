"""Lightweight G4 status loop and final private archive relay."""
import datetime,json,pathlib,subprocess,sys,time,traceback
root=pathlib.Path('/Users/infina/Documents/rl-nli')
cli='/Users/infina/.local/share/uv/tools/google-colab-cli/bin/colab'
python='/Users/infina/.local/share/uv/tools/google-colab-cli/bin/python'
lanes={'N20R20':'rl-nli-resume-r18-g4','N21R20':'rl-nli-r19-native-swap-2b-g4'}
out=root/'artifacts/analysis/native_r20_live_status.json'
archived=set();last={}
def save(value):
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out)
while True:
    record={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lanes':{}}
    for run,session in lanes.items():
        try:
            result=subprocess.run([cli,'exec','--session',session,'--file','/private/tmp/native_r20_remote_probe.py',
                '--timeout','30','--env','NATIVE_RUN_ID='+run],capture_output=True,text=True,timeout=45)
            if result.returncode:raise RuntimeError('Colab probe exited '+str(result.returncode))
            record['lanes'][run]={'session':session,'probe':result.stdout[-4000:]}
            last[run]=result.stdout
            if '"stage": "test_verified_backup_pending"' in result.stdout and run not in archived:
                save(record)
                backup=subprocess.run([python,'/private/tmp/native_r20_archive.py',run],capture_output=True,text=True,timeout=2400)
                if backup.returncode:raise RuntimeError('Private archive relay failed with exit '+str(backup.returncode))
                record['lanes'][run]['archive']={'completed':True,'result':backup.stdout[-1000:]}
                archived.add(run)
            if '"stage": "test_verified_archived"' in result.stdout:archived.add(run)
        except Exception as exc:
            record['lanes'][run]={'session':session,'poll_error':type(exc).__name__}
    save(record)
    if len(archived)==len(lanes):
        save({**record,'complete':True});break
    time.sleep(120)
