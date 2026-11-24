from pathlib import Path
import json,csv,hashlib
import numpy as np
root=Path('/root/pr4416-validation/framework-training')
paths={'paddle':root/'paddle-fp32-trace','torch':root/'swift-trace'}
rows={key:[json.loads(line) for line in (path/'losses.jsonl').read_text().splitlines()] for key,path in paths.items()}
assert len(rows['paddle'])==len(rows['torch'])==20
comparison=[];batch_checks=[]
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for i,(pa,pt) in enumerate(zip(rows['paddle'],rows['torch'])):
 a=np.load(paths['paddle']/f'batch-{i:03d}.npz');b=np.load(paths['torch']/f'batch-{i:03d}.npz')
 canonical_labels=np.concatenate([b['labels'][:,1:],np.full_like(b['labels'][:,:1],-100)],axis=1)
 checks={'step':i+1,'sequence_length':a['input_ids'].shape[-1],'supervised_tokens':int((a['labels']!=-100).sum()),'input_ids_equal':bool(np.array_equal(a['input_ids'],b['input_ids'])),'labels_equal_after_reference_shift':bool(np.array_equal(a['labels'],canonical_labels)),'pixel_values_equal':bool(np.array_equal(a['pixel_values'],b['pixel_values'])),'reference_attention_mask_all_ones':bool(np.all(b['attention_mask']==1)),'input_ids_sha256':sha(a['input_ids']),'canonical_labels_sha256':sha(a['labels']),'pixel_values_sha256':sha(a['pixel_values'])}
 assert all(checks[k] for k in ['input_ids_equal','labels_equal_after_reference_shift','pixel_values_equal','reference_attention_mask_all_ones'])
 batch_checks.append(checks)
 x,y=pa['loss'],pt['loss'];assert np.isfinite(x) and np.isfinite(y)
 comparison.append({'step':i+1,'paddle_loss':x,'torch_loss':y,'signed_diff':x-y,'abs_diff':abs(x-y)})
summary={'model':'InternVL3-1B','code_commit':'abaa61909ab2328500e5d4bde41a447c82c3ccdc','steps':20,'dtype':'float32','batch_size':1,'gradient_accumulation_steps':1,'trainable_parameters':629697920,'learning_rate':1e-6,'optimizer':'AdamW','scheduler':'constant','weight_decay':0.0,'gradient_clipping':False,'freeze_vision':True,'freeze_aligner':True,'vision_drop_path_rate':0.0,'tf32':False,'flash_attention':False,'mean_absolute_loss_diff':float(np.mean([r['abs_diff'] for r in comparison])),'max_absolute_loss_diff':float(np.max([r['abs_diff'] for r in comparison])),'all_batches_equal':True,'validation_scope':'20-step native-framework training smoke test, not full convergence'}
for key in ['paddle','torch']:
 values=np.array([r['loss'] for r in rows[key]])
 summary[key]={'first_loss':float(values[0]),'last_loss':float(values[-1]),'mean_loss':float(values.mean()),'first_five_mean_loss':float(values[:5].mean()),'last_five_mean_loss':float(values[-5:].mean()),'all_losses_finite':True,'model_info':json.loads((paths[key]/'model-info.json').read_text())}
summary['exit_codes']={key:int((root/file).read_text().strip()) for key,file in [('paddle','paddle-fp32.exit'),('torch','swift-full.exit')]}
assert summary['exit_codes']=={'paddle':0,'torch':0}
(root/'framework-training-summary.json').write_text(json.dumps(summary,indent=2))
(root/'batch-equivalence.json').write_text(json.dumps(batch_checks,indent=2))
with (root/'framework-loss-comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
print(json.dumps(summary,indent=2))
