import functools,json,os
from pathlib import Path
import numpy as np
from swift.llm.template import register_template
from swift.llm.template.template.internvl import Internvl2Template
from swift.llm.template.template.utils import ChatmlTemplateMeta
from swift.trainers import Seq2SeqTrainer
SYSTEM="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
class AlignedInternvlTemplate(Internvl2Template):
 def replace_tag(self,media_type,index,inputs):
  if media_type=="image":return ["<img>",[-100],"</img>"]
  return super().replace_tag(media_type,index,inputs)
register_template(ChatmlTemplateMeta("internvl3_paddle_alignment",template_cls=AlignedInternvlTemplate,default_system=SYSTEM,suffix=["<|im_end|>\n"]))
original=Seq2SeqTrainer.compute_loss
count=0
@functools.wraps(original)
def observed(self,model,inputs,*args,**kwargs):
 global count
 trace=Path(os.environ["PRECISION_TRACE_DIR"]);trace.mkdir(parents=True,exist_ok=True)
 arrays={k:v.detach().cpu().numpy().copy() for k,v in inputs.items() if hasattr(v,"detach")}
 np.savez_compressed(trace/f"batch-{count:03d}.npz",**arrays)
 result=original(self,model,inputs,*args,**kwargs)
 loss=result[0] if isinstance(result,tuple) else result
 row={"step":count+1,"loss":float(loss.item()),"keys":list(arrays),"sequence_length":int(arrays["input_ids"].shape[-1]),"supervised_tokens":int((arrays["labels"][:,1:]!=-100).sum())}
 with (trace/"losses.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
 if count==0:
  (trace/"model-info.json").write_text(json.dumps({"parameter_dtypes":sorted(set(str(p.dtype) for p in model.parameters())),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"vision_drop_path_rate":model.config.vision_config.drop_path_rate},indent=2))
 count+=1
 return result
Seq2SeqTrainer.compute_loss=observed
