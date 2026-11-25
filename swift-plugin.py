import functools,json,os
from pathlib import Path
import numpy as np
from swift.llm.template import register_template,TemplateMeta
from swift.trainers import Seq2SeqTrainer
from alignment_common import initialize
# Mirror PaddleFormers' built-in default template, including its terminal EOS.
register_template(TemplateMeta('minicpm41_paddle_alignment',prefix=[],prompt=['Human: {{QUERY}}',[73440],'\nAssistant:'],chat_sep=[],suffix=[[73440],'\n',[73440]],auto_add_bos=False))
original=Seq2SeqTrainer.compute_loss;count=0
@functools.wraps(original)
def observed(self,model,inputs,*args,**kwargs):
 global count
 trace=Path(os.environ['ALIGN_TRACE']);trace.mkdir(parents=True,exist_ok=True)
 if count==0:
  (trace/'initial-lora.json').write_text(json.dumps(initialize(model,'torch'),indent=2))
  (trace/'model-info.json').write_text(json.dumps({'parameter_dtypes':sorted(set(str(p.dtype) for p in model.parameters())),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)},indent=2))
 arrays={k:v.detach().cpu().numpy().copy() for k,v in inputs.items() if hasattr(v,'detach')}
 np.savez_compressed(trace/f'batch-{count:03d}.npz',**arrays)
 if count==0:
  def capture(module,args,output):
   np.savez_compressed(trace/'first-logits.npz',logits=output.logits.detach().float().cpu().numpy())
  handle=model.register_forward_hook(capture)
 result=original(self,model,inputs,*args,**kwargs);loss=result[0] if isinstance(result,tuple) else result
 if count==0:handle.remove()
 with (trace/'losses.jsonl').open('a') as f:f.write(json.dumps({'step':count+1,'loss':float(loss.item()),'supervised_tokens':int((arrays['labels'][:,1:]!=-100).sum()),'length':int(arrays['input_ids'].shape[-1])})+'\n')
 count+=1
 return result
Seq2SeqTrainer.compute_loss=observed
