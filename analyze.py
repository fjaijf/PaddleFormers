import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
T=np.load(O/'torch-inference.npz');P=np.load(O/'paddle-inference.npz')
assert np.array_equal(T['input_ids'],P['input_ids'])
a,b=T['logits'],P['logits'];assert a.shape==b.shape and a.shape[1]==20
D=np.abs(a.astype('float64')-b.astype('float64'))
r={'model':'openbmb/MiniCPM4.1-8B','commit':'729cca11dc5663b00f67f91c62fb69724cf54c10','dtype':'float32','logits_shape':list(a.shape),'log_diff_mean_abs':float(D.mean()),'logits_max_abs_diff':float(D.max()),'top1_matches':int((a.argmax(-1)==b.argmax(-1)).sum()),'generated_10_ids_match':bool(np.array_equal(T['generated_ids'],P['generated_ids'])),'torch_generated_ids':T['generated_ids'].tolist(),'paddle_generated_ids':P['generated_ids'].tolist(),'per_position_mean_abs_diff':D.mean(axis=(0,2)).tolist(),'per_position_max_abs_diff':D.max(axis=(0,2)).tolist()}
assert T['generated_ids'].shape==P['generated_ids'].shape==(1,10)
rows=[]
for backend in ['paddle','swift']:
 rows.append([json.loads(s) for s in (O/f'{backend}-trace/losses.jsonl').read_text().splitlines()])
assert len(rows[0])==len(rows[1])==20
x=np.arange(1,21);p=np.array([s['loss'] for s in rows[0]]);t=np.array([s['loss'] for s in rows[1]]);d=p-t
r['training']={'steps':20,'paddle_first':float(p[0]),'swift_first':float(t[0]),'paddle_last':float(p[-1]),'swift_last':float(t[-1]),'paddle_mean':float(p.mean()),'swift_mean':float(t.mean()),'loss_mean_abs_diff':float(np.abs(d).mean()),'loss_max_abs_diff':float(np.abs(d).max())}
checks=[]
for i in range(20):
 pa=np.load(O/f'paddle-trace/batch-{i:03d}.npz');sw=np.load(O/f'swift-trace/batch-{i:03d}.npz')
 ids=pa['input_ids'];sids=sw['input_ids'];lab=pa['labels'];slab=sw['labels']
 same_ids=np.array_equal(ids,sids) or np.array_equal(ids,sids[:,:-1])
 same_labels=np.array_equal(lab,slab[:,1:]) or np.array_equal(lab[:,:-1],slab[:,1:])
 ppos=pa['position_ids'] if 'position_ids' in pa else np.arange(ids.shape[-1])[None,:]
 spos=sw['position_ids'] if 'position_ids' in sw else np.arange(sids.shape[-1])[None,:]
 position=bool(np.array_equal(ppos,spos) or np.array_equal(ppos,spos[...,:-1]))
 checks.append({'step':i+1,'input_ids_equal_with_causal_shift':same_ids,'supervision_labels_equal_after_shift':same_labels,'position_ids_equal':position,'paddle_keys':pa.files,'swift_keys':sw.files})
 if not same_ids or not same_labels:print('BATCH_MISMATCH',i,ids.tolist(),sids.tolist(),lab.tolist(),slab.tolist())
r['batch_checks']=checks
r['model_info']={k:json.loads((O/f'{k}-trace/model-info.json').read_text()) for k in ['paddle','swift']}
assert r['model_info']['paddle']['parameter_dtypes']==['paddle.float32']
assert r['model_info']['swift']['parameter_dtypes']==['torch.float32']
assert r['model_info']['paddle']['trainable_parameters']==r['model_info']['swift']['trainable_parameters']==22151168
assert all(v['length']<512 for group in rows for v in group)
pi=json.loads((O/'paddle-trace/initial-lora.json').read_text());si=json.loads((O/'swift-trace/initial-lora.json').read_text());r['lora_initialization_equal']=pi==si
assert pi==si
assert all(c['input_ids_equal_with_causal_shift'] and c['supervision_labels_equal_after_shift'] for c in checks)
(O/'summary.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
with (O/'losses.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['step','paddleformers_loss','ms_swift_loss','loss_diff','abs_loss_diff']);w.writerows(zip(x,p,t,d,np.abs(d)))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.plot(x,p,'o-',label='PaddleFormers',lw=1.8,ms=5);ax.plot(x,t,'x--',label='ms-swift',lw=1.2,ms=6);ax.set(xlabel='Optimizer step',ylabel='Training loss',title='MiniCPM4.1-8B | FP32 LoRA | A100 80GB',xticks=[1,5,10,15,20]);ax.grid(alpha=.2);ax.legend();fig.savefig(O/'loss-comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.axhline(0,color='gray',lw=.8);ax.plot(x,d,'o-',color='#b34e18',lw=1.5,ms=4);ax.set(xlabel='Optimizer step',ylabel='PaddleFormers loss - ms-swift loss',title=f'Loss difference | mean abs = {np.abs(d).mean():.3e} | max abs = {np.abs(d).max():.3e}',xticks=[1,5,10,15,20]);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2);fig.savefig(O/'loss-diff.png',dpi=180);plt.close(fig)
print(json.dumps({k:v for k,v in r.items() if k not in ['batch_checks','per_position_mean_abs_diff','per_position_max_abs_diff']},indent=2))
