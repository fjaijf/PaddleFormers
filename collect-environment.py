import sys,subprocess,json,hashlib,platform,importlib.metadata
from pathlib import Path
out=Path('/root/pr4840-validation')
r={'python':sys.version,'platform':platform.platform(),'packages':{}}
for p in ['paddlepaddle-gpu','torch','transformers','ms-swift','peft','numpy','paddlefleet']:
 try:r['packages'][p]=importlib.metadata.version(p)
 except importlib.metadata.PackageNotFoundError:pass
r['installed_packages']=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)
r['source_commit']=subprocess.check_output(['git','-C','/root/baidu_98_minicpm4_1/PaddleFormers','rev-parse','HEAD'],text=True).strip()
(out/('environment-'+sys.argv[1]+'.json')).write_text(json.dumps(r,indent=2))
