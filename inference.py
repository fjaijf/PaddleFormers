import os,sys,json,time
from pathlib import Path
import numpy as np
ROOT='/root/baidu_98_minicpm4_1/PaddleFormers'
OUT=Path('/root/pr4840-validation');MODEL='/data/minicpm4.1-8b'
sys.path.insert(0,ROOT)
backend=sys.argv[1]
prompt='The history of artificial intelligence began with the study of human reasoning. Today, researchers build language models that can understand and generate text. In the future,'
if backend=='torch':
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from transformers.generation import GenerationMixin
 from transformers import PreTrainedModel
 torch.manual_seed(23);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
 ids=np.asarray([tok.encode(prompt,add_special_tokens=True)],dtype='int64')
 np.savez(OUT/'shared-inputs.npz',input_ids=ids)
 (OUT/'input.json').write_text(json.dumps({'prompt':prompt,'input_ids':ids.tolist()},indent=2))
 model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').eval().cuda()
 kw={'input_ids':torch.from_numpy(ids).cuda(),'attention_mask':torch.ones(ids.shape,dtype=torch.long,device='cuda')}
 with torch.inference_mode():
  logits=model(**kw,use_cache=False,return_dict=True).logits[:,:20,:].float().cpu().numpy()
  generated=model.generate(**kw,do_sample=False,max_new_tokens=10,min_new_tokens=10,use_cache=True,eos_token_id=2).cpu().numpy()[:,ids.shape[1]:]
 version={'torch':torch.__version__}
else:
 import paddle
 from paddleformers.transformers import MiniCPM4_1ForCausalLM,AutoTokenizer
 paddle.set_device('gpu:0');paddle.seed(23)
 ids=np.load(OUT/'shared-inputs.npz')['input_ids']
 model=MiniCPM4_1ForCausalLM.from_pretrained(MODEL,dtype='float32',convert_from_hf=True,load_checkpoint_format='normal',load_via_cpu=True,_attn_implementation='eager').eval()
 from transformers import AutoTokenizer as HFTokenizer
 tok=HFTokenizer.from_pretrained(MODEL,trust_remote_code=True)
 kw={'input_ids':paddle.to_tensor(ids),'attention_mask':paddle.ones(ids.shape,dtype='int64')}
 with paddle.no_grad():
  logits=model(**kw,use_cache=False,return_dict=True).logits[:,:20,:].numpy()
  out=model.generate(**kw,decode_strategy='greedy_search',max_new_tokens=10,min_new_tokens=10,use_cache=True,eos_token_id=2)
  print('GENERATION_RAW',repr(out),flush=True)
  generated=(out[0] if isinstance(out,tuple) else out).numpy()
  if generated.shape[-1]==ids.shape[-1]+10:generated=generated[:,ids.shape[-1]:]
 version={'paddle':paddle.__version__}
np.savez(OUT/f'{backend}-inference.npz',logits=logits,generated_ids=generated,input_ids=ids)
result={'backend':backend,'dtype':'float32','input_length':int(ids.shape[-1]),'logits_shape':list(logits.shape),'generated_ids':generated.tolist(),'generated_text':tok.batch_decode(generated,skip_special_tokens=False),'versions':version,'parameter_dtypes':sorted(set(str(p.dtype) for p in model.parameters()))}
(OUT/f'{backend}-inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
