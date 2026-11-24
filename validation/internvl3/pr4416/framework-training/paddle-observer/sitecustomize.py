import os
if os.environ.get("PRECISION_TRACE_BACKEND")=="paddle":
 import functools,inspect,json,hashlib
 from pathlib import Path
 import numpy as np
 from paddleformers.transformers.internvl3.modeling_internvl_chat import InternVLChatModel
 original=InternVLChatModel.forward
 signature=inspect.signature(original)
 count=0
 @functools.wraps(original)
 def observed(self,*args,**kwargs):
  global count
  bound=signature.bind(self,*args,**kwargs);values=bound.arguments
  extra=values.get("kwargs",{});values={**values,**extra}
  if values.get("labels") is None:return original(self,*args,**kwargs)
  trace=Path(os.environ["PRECISION_TRACE_DIR"]);trace.mkdir(parents=True,exist_ok=True)
  if count==0:
   buffers={k:v.detach().cpu().numpy().copy() for k,v in self.named_buffers()}
   np.savez_compressed(trace/"initial-buffers.npz",**buffers)
   active={}
   def hook(name):
    def capture(layer,inputs,output):
     value=output if hasattr(output,"detach") else (output.last_hidden_state if hasattr(output,"last_hidden_state") else output[0])
     active[name]=value.detach().cpu().numpy().copy()
    return capture
   if os.environ.get("CAPTURE_ACTIVATIONS")=="1":
    for name,layer in self.named_sublayers():
     if name in ["vision_model.embeddings","vision_model","mlp1","language_model.model.embed_tokens","language_model.model.layers.0","language_model.model.norm"]:
      layer.register_forward_post_hook(hook(name))
   fingerprints={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in self.state_dict().items()}
   (trace/"initial-weights.json").write_text(json.dumps(fingerprints,indent=2))
  arrays={k:v.detach().cpu().numpy().copy() for k,v in values.items() if hasattr(v,"detach") and k!="self"}
  np.savez_compressed(trace/f"batch-{count:03d}.npz",**arrays)
  output=original(self,*args,**kwargs)
  loss=output.loss if hasattr(output,"loss") else output[0]
  if count==0 and active:np.savez_compressed(trace/"initial-activations.npz",**active)
  row={"step":count+1,"loss":float(loss.item()),"keys":list(arrays),"sequence_length":int(arrays["input_ids"].shape[-1]),"supervised_tokens":int((arrays["labels"]!=-100).sum())}
  with (trace/"losses.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
  if count==0:
   (trace/"model-config.json").write_text(json.dumps(self.config.to_dict(),indent=2))
   (trace/"model-info.json").write_text(json.dumps({"parameter_dtypes":sorted(set(str(p.dtype) for p in self.parameters())),"trainable_parameters":sum(int(p.numel()) for p in self.parameters() if not p.stop_gradient),"vision_drop_path_rate":self.config.vision_config.drop_path_rate},indent=2))
  count+=1
  return output
 InternVLChatModel.forward=observed
