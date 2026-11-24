"""Replay identical exported SFT batches in Paddle and HF for 20 FP32 AdamW steps."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get("PADDLEFORMERS_ROOT","/root/baidu_98_internvl/PaddleFormers"))
OUT=Path(os.environ.get("VALIDATION_DIR","/root/pr4416-validation/training-alignment"))
MODEL=os.environ.get("MODEL_PATH","/data/models/internvl3_1b")
sys.path.insert(0,str(ROOT))
backend=sys.argv[1]
SEED=23;LR=1e-6;STEPS=20

def batch_at(i):
    path=OUT/"batches"/f"{i:03d}.npz"
    data=np.load(path,allow_pickle=False)
    ids=data["input_ids"].astype("int64")
    if "position_ids" in data:
        pos=data["position_ids"].astype("int64")
    else:
        # The exported single, unpacked sequence has no trailing padding.
        # A constant end index verifies the original causal attention region.
        ends=data["attn_mask_startend_row_indices"]
        assert np.all(ends==ids.shape[-1]),"Packed/padded regions need explicit handling"
        pos=np.arange(ids.shape[-1],dtype="int64")[None,:]
    assert ids.shape[0]==1 and pos.ndim==2,(ids.shape,pos.shape)
    n=int(np.flatnonzero(pos[0]>0)[-1])+1
    assert np.array_equal(pos[0,:n],np.arange(n)),"Only single unpacked sequences are supported"
    labels=data["labels"].astype("int64")[:,:n]
    assert (labels!=-100).sum()>0
    pixel=data["pixel_values"].astype("float32")
    arrays=dict(input_ids=ids[:,:n],position_ids=pos[:,:n],labels=labels,
                attention_mask=np.ones((1,n),dtype="int64"),pixel_values=pixel,
                image_flags=np.ones((pixel.shape[0],1),dtype="int64"))
    h=hashlib.sha256()
    for k,v in sorted(arrays.items()):h.update(k.encode());h.update(v.tobytes())
    return arrays,h.hexdigest()

if backend=="paddle":
    import paddle
    from paddleformers.transformers import AutoModelForConditionalGeneration
    paddle.set_device("gpu:0");paddle.seed(SEED)
    model=AutoModelForConditionalGeneration.from_pretrained(MODEL,dtype="float32",convert_from_hf=True,load_checkpoint_format="flex_checkpoint",load_via_cpu=True,use_flash_attn=False)
    for p in model.vision_model.parameters():p.stop_gradient=True
    for p in model.mlp1.parameters():p.stop_gradient=True
    model.train();model.vision_model.eval();model.mlp1.eval()
    params=[p for p in model.parameters() if not p.stop_gradient]
    optimizer=paddle.optimizer.AdamW(learning_rate=LR,beta1=0.9,beta2=0.999,epsilon=1e-8,weight_decay=0.0,parameters=params)
    versions={"paddle":paddle.__version__}
else:
    import torch,transformers
    from transformers import AutoModel
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    model=AutoModel.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True).cuda()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    model.img_context_token_id=tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    for p in model.vision_model.parameters():p.requires_grad_(False)
    for p in model.mlp1.parameters():p.requires_grad_(False)
    model.train();model.vision_model.eval();model.mlp1.eval()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=LR,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,foreach=False)
    versions={"torch":torch.__version__,"transformers":transformers.__version__}

records=[]
for i in range(STEPS):
    arrays,digest=batch_at(i)
    start=time.time()
    if backend=="paddle":
        batch={k:paddle.to_tensor(v) for k,v in arrays.items()}
        optimizer.clear_grad()
        output=model(**batch,use_cache=False,return_dict=True)
        loss=output.loss
        loss.backward()
        grad=model.language_model.lm_head.weight.grad
        grad_norm=float(paddle.linalg.norm(grad).item())
        value=float(loss.item())
        optimizer.step()
        del output,loss,batch,grad
    else:
        batch={k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
        labels=batch.pop("labels")
        optimizer.zero_grad(set_to_none=True)
        output=model(**batch,use_cache=False,return_dict=True)
        # PaddleFormers SFT labels are already shifted. Do not use the HF
        # wrapper's internal label shift; apply CE directly to these labels.
        loss=torch.nn.functional.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction="mean")
        loss.backward()
        grad_norm=float(model.language_model.lm_head.weight.grad.norm().item())
        value=float(loss.item())
        optimizer.step()
        del output,loss,batch,labels
    assert np.isfinite(value) and np.isfinite(grad_norm) and grad_norm>0
    row=dict(step=i+1,loss=value,lm_head_grad_norm=grad_norm,batch_sha256=digest,
             sequence_length=int(arrays["input_ids"].shape[-1]),supervised_tokens=int((arrays["labels"]!=-100).sum()),seconds=time.time()-start)
    records.append(row)
    print(json.dumps(row),flush=True)
    result=dict(backend=backend,versions=versions,model=MODEL,dtype="float32",seed=SEED,steps=STEPS,
                optimizer="AdamW",lr=LR,betas=[0.9,0.999],epsilon=1e-8,weight_decay=0.0,
                batch_size=1,gradient_accumulation=1,vision_frozen=True,projector_frozen=True,
                dropout="Frozen vision/projector in eval mode; LM attention dropout is 0",
                labels="SFTDataset pre-shifted, -100 ignored, CE mean over supervised tokens",records=records)
    (OUT/f"{backend}-train.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print("TRAIN_ALIGNMENT_COMPLETE",backend,flush=True)
