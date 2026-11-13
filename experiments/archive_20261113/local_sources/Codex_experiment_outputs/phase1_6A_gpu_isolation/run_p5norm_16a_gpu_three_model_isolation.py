#!/usr/bin/env python3
"""Read-only memory gate only; intentionally does not create models."""
import os, subprocess
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import torch
s=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
smi=int(s.splitlines()[0])*1024**2
host=int(next(line.split()[1] for line in open('/proc/meminfo') if line.startswith('MemAvailable:')))*1024
print('nvidia_smi_free',smi,'torch_free',torch.cuda.mem_get_info(0)[0],'host_available',host)
if smi<6.5*1024**3 or host<5.5*1024**3:raise SystemExit('RESOURCE BLOCKED; no model created')
raise SystemExit('STOP: this gate does not implement the three-model test')
