import os
if os.environ.get('ALIGN_BACKEND')=='paddle':
 import functools,inspect,json
 from pathlib import Path
 import numpy as np
 from alignment_common import initialize
 import paddleformers.cli.train.sft.workflow as workflow
 from paddleformers.transformers.minicpm4_1.modeling import MiniCPM4_1ForCausalLM
 trace=Path(os.environ['ALIGN_TRACE']);trace.mkdir(parents=True,exist_ok=True)
 original_create=workflow.create_peft_model
 def create(*args,**kwargs):
  model=original_create(*args,**kwargs)
  (trace/'initial-lora.json').write_text(json.dumps(initialize(model,'paddle'),indent=2))
  (trace/'model-info.json').write_text(json.dumps({'parameter_dtypes':sorted(set(str(p.dtype) for p in model.parameters())),'trainable_parameters':sum(int(p.numel()) for p in model.parameters() if not p.stop_gradient)},indent=2))
  return model
 workflow.create_peft_model=create
 original=MiniCPM4_1ForCausalLM.forward;signature=inspect.signature(original);count=0
 @functools.wraps(original)
 def forward(self,*args,**kwargs):
  global count
  values=signature.bind(self,*args,**kwargs).arguments
  if values.get('labels') is None:return original(self,*args,**kwargs)
  arrays={k:v.detach().cpu().numpy().copy() for k,v in values.items() if hasattr(v,'detach') and k!='self'}
  np.savez_compressed(trace/f'batch-{count:03d}.npz',**arrays)
  out=original(self,*args,**kwargs);loss=out.loss if hasattr(out,'loss') else out[0]
  if count==0:
   logits=out.logits if hasattr(out,'logits') else out[1]
   np.savez_compressed(trace/'first-logits.npz',logits=logits.detach().astype("float32").cpu().numpy())
   (trace/'model-config.json').write_text(json.dumps(self.config.to_dict(),indent=2))
  with (trace/'losses.jsonl').open('a') as f:f.write(json.dumps({'step':count+1,'loss':float(loss.item()),'supervised_tokens':int((arrays['labels']!=-100).sum()),'length':int(arrays['input_ids'].shape[-1])})+'\n')
  count+=1
  return out
 MiniCPM4_1ForCausalLM.forward=forward
