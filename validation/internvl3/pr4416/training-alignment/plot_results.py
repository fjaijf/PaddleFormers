import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/"paddle-train.json").read_text())
r=json.loads((ROOT/"torch-train.json").read_text())
a=p["records"];b=r["records"]
assert len(a)==len(b)==20
assert all(x["step"]==y["step"] and x["batch_sha256"]==y["batch_sha256"] for x,y in zip(a,b))
x=np.array([v["step"] for v in a]);lp=np.array([v["loss"] for v in a]);lr=np.array([v["loss"] for v in b]);delta=lp-lr;ad=np.abs(delta)
result={"steps":20,"precision":"float32","threshold":0.01,"all_steps_pass":bool((ad<0.01).all()),"mean_abs_loss_diff":float(ad.mean()),"max_abs_loss_diff":float(ad.max()),"max_diff_step":int(x[ad.argmax()]),"paddle_first_loss":float(lp[0]),"reference_first_loss":float(lr[0]),"paddle_last_loss":float(lp[-1]),"reference_last_loss":float(lr[-1]),"paddle_mean_loss":float(lp.mean()),"reference_mean_loss":float(lr.mean()),"identical_batch_hashes":True}
(ROOT/"training-summary.json").write_text(json.dumps(result,indent=2))
with (ROOT/"loss-comparison.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["step","paddle_loss","reference_loss","signed_loss_diff","absolute_loss_diff","sequence_length","supervised_tokens","batch_sha256"])
 for i,(aa,bb) in enumerate(zip(a,b)):w.writerow([aa["step"],aa["loss"],bb["loss"],float(delta[i]),float(ad[i]),aa["sequence_length"],aa["supervised_tokens"],aa["batch_sha256"]])
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
fig,ax=plt.subplots(figsize=(10,4.6),layout="constrained")
ax.plot(x,lr,color="#1d4ed8",linewidth=2,label="HF reference (FP32)")
ax.plot(x,lp,color="#e67700",linestyle="--",marker="o",markersize=4,linewidth=1.5,label="Paddle (FP32)")
ax.set(title="InternVL3-1B training loss alignment",xlabel="Optimizer step (same batch order)",ylabel="Cross entropy / supervised token",xticks=np.arange(1,21))
ax.grid(alpha=.18);ax.legend();fig.savefig(ROOT/"loss-comparison.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
axes[0].plot(x,delta,color="#7c3aed",marker="o",markersize=4);axes[0].axhline(0,color="#64748b",linewidth=1,linestyle="--")
axes[0].set(title="Signed loss difference",xlabel="Optimizer step",ylabel="Paddle loss - reference loss")
axes[0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
axes[1].semilogy(x,np.maximum(ad,1e-12),color="#0f766e",marker="o",markersize=4,label="Absolute difference")
axes[1].axhline(.01,color="#dc2626",linestyle="--",label="Acceptance threshold: 0.01")
axes[1].set(title=f"Absolute loss difference (max {ad.max():.3e})",xlabel="Optimizer step",ylabel="Absolute loss difference (log scale)")
axes[1].legend(fontsize=9)
for ax in axes:ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.18)
fig.suptitle("InternVL3-1B | identical SFT batches | FP32 | AdamW",fontsize=13)
fig.savefig(ROOT/"loss-diff.png",dpi=180);plt.close(fig)
print(json.dumps(result,indent=2))
