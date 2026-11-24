from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
with (root/'framework-loss-comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
x=np.array([int(r['step']) for r in rows]);pa=np.array([float(r['paddle_loss']) for r in rows]);pt=np.array([float(r['torch_loss']) for r in rows]);diff=np.abs(pa-pt)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
ax.plot(x,pt,color='#e58428',linewidth=2.8,label='ms-swift / PyTorch',alpha=.85)
ax.plot(x,pa,color='#2465ae',linewidth=1.5,linestyle='--',marker='o',markersize=3.5,label='PaddleFormers / Paddle')
ax.set(title='InternVL3-1B: native framework training (FP32)',xlabel='Optimizer step',ylabel='Training loss',xticks=[1,5,10,15,20]);ax.grid(alpha=.18);ax.legend(frameon=False)
fig.savefig(root/'framework-loss-comparison.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
ax.plot(x,diff,color='#8054b7',marker='o',markersize=4,linewidth=1.5)
ax.set(title='Absolute training loss difference',xlabel='Optimizer step',ylabel='|PaddleFormers loss - ms-swift loss|',xticks=[1,5,10,15,20],ylim=(0,float(diff.max())*1.15));ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.18)
fig.savefig(root/'framework-loss-diff.png',dpi=200);plt.close(fig)
print('Plots saved')
