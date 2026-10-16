"""Upload registered lightweight run files and execute one G4 smoke gate."""
import pathlib,subprocess,sys
root=pathlib.Path('/Users/infina/Documents/rl-nli')
cli='/Users/infina/.local/share/uv/tools/google-colab-cli/bin/colab'
run=sys.argv[1]
session={'N20R20':'rl-nli-resume-r18-g4','N21R20':'rl-nli-r19-native-swap-2b-g4'}[run]
suffix='N20R20' if run=='N20R20' else 'N21R20'
files=[
 (root/'configs/matrix.py','/content/rl-nli/configs/matrix.py'),
 (root/f'configs/runs/{run}.yaml',f'/content/rl-nli/configs/runs/{run}.yaml'),
 (root/f'configs/runs/G0{run}P.yaml',f'/content/rl-nli/configs/runs/G0{run}P.yaml'),
 (root/f'artifacts/protocols/20261015/{run}_registration.json',f'/content/{run}_test_registration.json'),
 (root/'artifacts/analysis/r18_prepared_data_hashes.json','/content/native_r18_prepared_data_hashes.json'),
 (pathlib.Path('/private/tmp/native_r20_pipeline.py'),'/content/native_r20_pipeline.py'),
 (pathlib.Path('/private/tmp/native_r20_test_worker.py'),'/content/native_r20_test_worker.py'),
 (pathlib.Path('/private/tmp/native_r20_launch.py'),'/content/native_r20_launch.py'),
]
for source,destination in files:
 subprocess.run([cli,'upload','--session',session,str(source),destination],check=True)
print('uploaded',run,session,flush=True)
subprocess.run([cli,'exec','--session',session,'--file','/private/tmp/native_r20_launch.py',
                '--timeout','1800','--env','NATIVE_RUN_ID='+run],check=True)
