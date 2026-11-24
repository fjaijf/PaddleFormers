"""Audit actual batches and plot losses captured from both official trainers."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
p=root/'paddle-trace';s=root/'swift-trace'
pl=[json.loads(x) for x in (p/'losses.jsonl').read_text().splitlines()]
sl=[json.loads(x) for x in (s/'losses.jsonl').read_text().splitlines()]
assert len(pl)==len(sl)==20
assert json.loads((p/'initial-lora.json').read_text())==json.loads((s/'initial-lora.json').read_text())
audit=[]
for i in range(20):
 a=np.load(p/f'batch-{i:03d}.npz');b=np.load(s/f'batch-{i:03d}.npz')
 same_ids=np.array_equal(a['input_ids'],b['input_ids'])
 same_labels=np.array_equal(a['labels'][:,:-1],b['labels'][:,1:]) and (a['labels'][:,-1]==-100).all()
 same_positions=np.array_equal(a['position_ids'],b.get('position_ids',np.arange(a['input_ids'].shape[-1],dtype=np.int64)[None]))
 assert same_ids and same_labels and same_positions,(i,same_ids,same_labels,same_positions)
 audit.append({'step':i+1,'input_ids_equal':same_ids,'shifted_labels_equal':bool(same_labels),'position_ids_equal':same_positions,'input_sha256':hashlib.sha256(a['input_ids'].tobytes()).hexdigest(),'supervised_tokens':pl[i]['supervised_tokens'],'length':pl[i]['length']})
(root/'batch-audit.json').write_text(json.dumps(audit,indent=2))
x=np.arange(1,21);a=np.array([z['loss'] for z in pl]);b=np.array([z['loss'] for z in sl]);d=a-b
summary={'steps':20,'samples':10,'epochs':2,'trainable_parameters':19988480,'lora_modules':224,'initial_lora_equal':True,'all_actual_batches_equal':True,'first_step':{'paddle':a[0],'swift':b[0],'abs_diff':abs(d[0])},'last_step':{'paddle':a[-1],'swift':b[-1],'abs_diff':abs(d[-1])},'mean_loss':{'paddle':a.mean(),'swift':b.mean()},'max_abs_loss_diff':abs(d).max(),'mean_abs_loss_diff':abs(d).mean(),'rmse_loss_diff':np.sqrt((d*d).mean())}
(root/'training-summary.json').write_text(json.dumps(summary,indent=2))
with (root/'loss-comparison.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['step','paddle_loss','ms_swift_loss','signed_diff','abs_diff'])
 for row in zip(x,a,b,d,abs(d)):w.writerow(row)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
ax.plot(x,a,'o-',color='#167DAD',label='PaddleFormers official SFT',markersize=5,linewidth=1.7)
ax.plot(x,b,'x--',color='#E57E25',label='ms-swift sft',markersize=6,linewidth=1.3)
ax.set(title='InternLM-7B: FP32 LoRA training loss',xlabel='Optimizer step',ylabel='Supervised next-token cross-entropy')
ax.set_xticks(x[::2]);ax.grid(alpha=.22);ax.legend();ax.text(.99,.97,'10 samples × 2 epochs | rank 8, alpha 16 | AdamW, LR 1e−4',ha='right',va='top',transform=ax.transAxes,fontsize=9)
fig.savefig(root/'loss-comparison.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
ax.plot(x,d,'o-',color='#7A43A3',markersize=5,linewidth=1.5);ax.axhline(0,color='#6B7280',linewidth=.8)
ax.set(title='InternLM-7B: training loss difference',xlabel='Optimizer step',ylabel='PaddleFormers loss − ms-swift loss')
ax.set_xticks(x[::2]);ax.grid(alpha=.22);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.text(.02,.98,f'Max |diff| = {abs(d).max():.6g}\nMean |diff| = {abs(d).mean():.6g}',transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'#D1D5DB','alpha':.9})
fig.savefig(root/'loss-diff.png');plt.close(fig)
print(json.dumps(summary,indent=2))
# Inference comparison can be regenerated from the two inference script outputs.
if (root/'paddle-inference.npz').exists() and (root/'torch-inference.npz').exists():
 pa=np.load(root/'paddle-inference.npz');ta=np.load(root/'torch-inference.npz')
 delta=pa['logits'].astype('float64')-ta['logits'].astype('float64')
 inf={'max_abs':float(abs(delta).max()),'mean_abs':float(abs(delta).mean()),'rmse':float(np.sqrt((delta*delta).mean())),'top1_equal':int((pa['logits'].argmax(-1)==ta['logits'].argmax(-1)).sum()),'top1_total':20,'generation_equal':bool(np.array_equal(pa['generated_ids'],ta['generated_ids'])),'per_position_max_abs':abs(delta).max(-1).tolist()}
 (root/'inference-summary.json').write_text(json.dumps(inf,indent=2))
 print(json.dumps(inf,indent=2))
