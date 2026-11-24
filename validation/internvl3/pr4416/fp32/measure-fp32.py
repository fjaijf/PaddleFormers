import os,sys,json,importlib.util
import numpy as np
ROOT="/root/baidu_98_internvl/PaddleFormers"
OUT="/root/pr4416-validation/fp32"
sys.path.insert(0,ROOT)
spec=importlib.util.spec_from_file_location("compare",ROOT+"/examples/inference/compare_internvl3_logits.py")
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
backend=sys.argv[1]
sys.argv=[sys.argv[0],"--backend",backend,"--model_name_or_path","/data/models/internvl3_1b","--image_path","/data/models/internvl3_1b/examples/image1.jpg","--num_tokens","20","--dtype","float32"]
a=c.parse_args()
if backend=="paddle":
 import paddle
 paddle.set_device("gpu:0");paddle.seed(23)
 print("NVIDIA_TF32_OVERRIDE",os.environ.get("NVIDIA_TF32_OVERRIDE"),flush=True)
 from paddleformers.transformers import AutoModelForConditionalGeneration,AutoTokenizer
 inputs=c.load_inputs_from_npz("/root/pr4416-validation/shared-inputs.npz")
 np.savez(OUT+"/shared-inputs.npz",**inputs)
 model=AutoModelForConditionalGeneration.from_pretrained(a.model_name_or_path,dtype=a.dtype,convert_from_hf=True,load_checkpoint_format="flex_checkpoint",load_via_cpu=True,use_flash_attn=False)
 model.eval()
 kw={k:paddle.to_tensor(inputs[k]) for k in ["input_ids","attention_mask","position_ids","image_flags"]}
 kw["pixel_values"]=paddle.to_tensor(inputs["pixel_values"]).astype(a.dtype)
 with paddle.no_grad():
  pred=model(**kw,use_cache=False,return_dict=True)
  logits=pred.logits[:,:20,:].astype("float32").numpy()
  del pred
  out=model.generate(input_ids=kw["input_ids"],pixel_values=kw["pixel_values"],attention_mask=kw["attention_mask"],do_sample=False,max_new_tokens=10,min_new_tokens=10,use_cache=True)
  print("GENERATION_OUTPUT",repr(out),flush=True)
  gen=model._generation_ids(out)
  ids=gen.numpy().astype("int64") if hasattr(gen,"numpy") else np.asarray([x.numpy() if hasattr(x,"numpy") else x for x in gen],dtype="int64").reshape(1,-1)
 tok=AutoTokenizer.from_pretrained(a.model_name_or_path)
else:
 import torch
 from transformers import AutoModel,AutoTokenizer
 torch.manual_seed(23)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision("highest")
 # Torch 2.0 CUDA lacks BF16 triangular-mask kernels. The mask only contains
 # zeros and dtype-min/inf values; computing its triangle in FP32 is exact.
 original_triu=torch.triu
 def compatible_triu(x,diagonal=0,**kwargs):
  if x.is_cuda and x.dtype==torch.bfloat16:
   return original_triu(x.float(),diagonal=diagonal,**kwargs).to(x.dtype)
  return original_triu(x,diagonal=diagonal,**kwargs)
 torch.triu=compatible_triu
 inputs=c.load_inputs_from_npz(OUT+"/shared-inputs.npz")
 model=AutoModel.from_pretrained(a.model_name_or_path,torch_dtype=torch.float32,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True).eval().cuda()
 model.img_context_token_id=int(inputs["image_token_id"])
 kw={k:torch.from_numpy(inputs[k]).cuda() for k in ["input_ids","attention_mask","position_ids","image_flags"]}
 kw["pixel_values"]=torch.from_numpy(inputs["pixel_values"]).to(device="cuda",dtype=torch.float32)
 with torch.inference_mode():
  pred=model(**kw,use_cache=False,return_dict=True)
  logits=pred.logits[:,:20,:].float().cpu().numpy()
  del pred
  out=model.generate(input_ids=kw["input_ids"],pixel_values=kw["pixel_values"],attention_mask=kw["attention_mask"],do_sample=False,max_new_tokens=10,min_new_tokens=10)
  ids=out.cpu().numpy().astype("int64")
 tok=AutoTokenizer.from_pretrained(a.model_name_or_path,trust_remote_code=True)
np.savez(OUT+"/"+backend+"-results.npz",logits=logits,generated_ids=ids)
result={"backend":backend,"dtype":"float32","logit_shape":list(logits.shape),"input_length":int(inputs["input_ids"].shape[-1]),"num_patches":int(inputs["num_patches"]),"generated_ids":ids.tolist(),"generated_text":tok.batch_decode(ids,skip_special_tokens=True)}
with open(OUT+"/"+backend+"-results.json","w") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print("VALIDATION_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
