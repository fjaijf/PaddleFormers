"""Shared deterministic LoRA initialization; does not change either training loop."""
import re,hashlib,math
import numpy as np

def initial_a(key,in_features,rank):
 seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
 return np.random.default_rng(seed).uniform(-1/math.sqrt(in_features),1/math.sqrt(in_features),(in_features,rank)).astype('float32')

def canonical(name):
 match=re.search(r'model\.layers\.\d+\.(?:self_attn|mlp)\.[^.]+',name)
 if not match:raise ValueError(name)
 return match.group()

def initialize(model,backend):
 rows={}
 if backend=='paddle':
  import paddle
  for name,layer in model.named_sublayers():
   if not hasattr(layer,'lora_A'):continue
   key=canonical(name);a=initial_a(key,int(layer.lora_A.shape[0]),int(layer.lora_A.shape[1]))
   layer.lora_A.set_value(a);layer.lora_B.set_value(paddle.zeros_like(layer.lora_B))
   rows[key]={'A_shape':list(a.shape),'A_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'B_zero':True,'scaling':layer.scaling}
 else:
  import torch
  with torch.no_grad():
   for name,layer in model.named_modules():
    if not hasattr(layer,'lora_A') or 'default' not in layer.lora_A:continue
    key=canonical(name);param=layer.lora_A['default'].weight;a=initial_a(key,int(param.shape[1]),int(param.shape[0]))
    param.copy_(torch.from_numpy(a.T.copy()).to(param.device));layer.lora_B['default'].weight.zero_()
    rows[key]={'A_shape':list(a.shape),'A_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'B_zero':True,'scaling':layer.scaling['default']}
 if len(rows)!=224:raise ValueError(f'Expected 32*7 LoRA modules; got {len(rows)}')
 return rows
