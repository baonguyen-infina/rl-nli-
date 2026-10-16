import csv,json,pathlib,math
root=pathlib.Path('/content/rl-nli')
runs=['N15R18'] if __import__('os').environ.get('NATIVE_RUN_ID')=='N15R18' else ['N16R19']
for run in runs:
 path=root/'results'/(run+'_Qwen3.5-2B_s0')/'train_log.csv'
 if not path.exists():
  print(json.dumps({'run':run,'missing':str(path)}),flush=True);continue
 rows=list(csv.DictReader(path.read_text(errors='replace').splitlines()))
 rows=[r for r in rows if (r.get('step') or '').isdigit()]
 fields=sorted({k for r in rows for k in r})
 selected=['native/accuracy','rewards/<lambda>/mean','reward','frac_reward_zero_std','entropy','grad_norm','sampling/importance_sampling_ratio/min','sampling/sampling_logp_difference/mean']
 def vals(key,part):
  out=[]
  for row in part:
   try:
    x=float(row.get(key,''))
    if math.isfinite(x):out.append(x)
   except (ValueError,TypeError):pass
  return round(sum(out)/len(out),6) if out else None
 early=rows[:50];late=rows[-50:]
 print(json.dumps({'run':run,'rows':len(rows),'first_step':rows[0].get('step') if rows else None,'last_step':rows[-1].get('step') if rows else None,'early_mean':{k:vals(k,early) for k in selected},'late_mean':{k:vals(k,late) for k in selected},'fields_present':[k for k in selected if k in fields],'columns':fields[:15]},ensure_ascii=False),flush=True)
