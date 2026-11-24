import os,sys,json
import numpy as np
import paddle
ROOT="/root/baidu_98_internvl/PaddleFormers"
sys.path.insert(0,ROOT)
from paddleformers.cli.train.sft.sft_trainer import SFTTrainer
OUT="/root/pr4416-validation/training-alignment"
os.makedirs(OUT+"/batches",exist_ok=True)
def export_batches(self,*args,**kwargs):
 manifest=[]
 for i,batch in enumerate(self.get_train_dataloader()):
  payload={k:(v.numpy() if isinstance(v,paddle.Tensor) else np.asarray(v)) for k,v in batch.items() if v is not None}
  np.savez(OUT+f"/batches/{i:03d}.npz",**payload)
  record={k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in payload.items()}
  record["supervised_tokens"]=int((payload["labels"]!=-100).sum())
  print("EXPORTED",i,json.dumps(record),flush=True)
  manifest.append(record)
  if i==19:break
 json.dump(manifest,open(OUT+"/batch-manifest.json","w"),indent=2)
 raise SystemExit(0)
SFTTrainer.train=export_batches
from paddleformers.cli.train.tuner import run_tuner
run_tuner()
