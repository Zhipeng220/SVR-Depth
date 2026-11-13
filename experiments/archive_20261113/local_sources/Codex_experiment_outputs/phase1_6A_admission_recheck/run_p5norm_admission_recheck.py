#!/usr/bin/env python3
"""One-process ten-sample admission check. No model, tensor or CUDA compute."""
import os,time,subprocess,json,datetime
from pathlib import Path
SMI='/usr/lib/wsl/lib/nvidia-smi'; OUT=Path('/tmp/p5norm_admission_recheck_20261110.json')
def gpu():
 fields='index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu'
 v=subprocess.check_output([SMI,'-i','0','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip().split(', ')
 return {'index':int(v[0]),'uuid':v[1],'name':v[2],'total_mib':int(v[3]),'free_mib':int(v[4]),'used_mib':int(v[5]),'util_pct':int(v[6])}
initial=gpu()
import torch
data={'initial_before_torch_cuda':initial,'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'python':os.sys.executable,'rounds':[]}
if not torch.cuda.is_available():
 data['error']='CUDA unavailable';OUT.write_text(json.dumps(data,indent=2));raise SystemExit('CUDA unavailable')
for i in range(10):
 t=time.monotonic();before=gpu();free,total=torch.cuda.mem_get_info(0);after=gpu()
 mm={line.split(':')[0]:int(line.split()[1])/1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:','SwapTotal:'))}
 rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))/1024
 processes=subprocess.check_output([SMI,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 full_smi=subprocess.check_output([SMI],text=True)
 row={'round':i+1,'timestamp':datetime.datetime.now().astimezone().isoformat(),'smi_before_mib':before['free_mib'],'torch_free_mib':free/1048576,'torch_total_mib':total/1048576,'smi_after_mib':after['free_mib'],'conservative_free_mib':min(before['free_mib'],free/1048576,after['free_mib']),'host_available_mib':mm['MemAvailable'],'swap_free_mib':mm['SwapFree'],'swap_total_mib':mm['SwapTotal'],'rss_mib':rss,'gpu_util_before_pct':before['util_pct'],'gpu_util_after_pct':after['util_pct'],'gpu_compute_processes':processes,'nvidia_smi_full':full_smi,'duration_s':time.monotonic()-t}
 data['rounds'].append(row)
 if i==0:
  p=torch.cuda.get_device_properties(0);data['torch_device']={'name':p.name,'uuid':str(getattr(p,'uuid','UNKNOWN')),'count':torch.cuda.device_count(),'allocated_mib':torch.cuda.memory_allocated(0)/1048576,'reserved_mib':torch.cuda.memory_reserved(0)/1048576}
 OUT.write_text(json.dumps(data,indent=2))
 if i<9:time.sleep(max(0,1-(time.monotonic()-t)))
print('OUTPUT',OUT)
