#!/usr/bin/env python3
"""Ten read-only GPU resource samples; no tensors, models or CUDA kernels."""
import os,time,datetime,subprocess,json,csv,statistics
from pathlib import Path
import torch
SMI='/usr/lib/wsl/lib/nvidia-smi'; OUT=Path('/tmp/p5norm_resource_reconciliation_20261110.json')
def smi():
 fields='index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu'
 s=subprocess.check_output([SMI,'--query-gpu='+fields,'--format=csv,noheader,nounits','-i','0'],text=True).strip().split(', ')
 return dict(index=int(s[0]),uuid=s[1],pci=s[2],name=s[3],total_mib=int(s[4]),used_mib=int(s[5]),free_mib=int(s[6]),util=int(s[7]))
def mem():
 d={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':',1)
  if k in ('MemTotal','MemAvailable','SwapFree','SwapTotal'):d[k]=int(v.split()[0])/1024
 d['rss_mib']=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))/1024
 return d
p=torch.cuda.get_device_properties(0);initial=smi()
env={'hostname':os.uname().nodename,'user':os.getenv('USER'),'python':os.sys.executable,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'name':p.name,'uuid':str(getattr(p,'uuid','UNKNOWN')),'pci':str(getattr(p,'pci_bus_id','UNKNOWN')),'smi':initial,'allocated_mib':torch.cuda.memory_allocated(0)/1048576,'reserved_mib':torch.cuda.memory_reserved(0)/1048576,'cgroup_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'cgroup_current':int(Path('/sys/fs/cgroup/memory.current').read_text())}
if not env['available'] or env['device_count']!=1 or env['uuid'].lower().removeprefix('gpu-')!=initial['uuid'].lower().removeprefix('gpu-'):raise SystemExit('GPU identity mismatch or unavailable')
rows=[]
for i in range(10):
 start=time.monotonic(); before=smi();free,total=torch.cuda.mem_get_info(0);after=smi();host=mem();tf=free/1048576
 rows.append(dict(round=i+1,timestamp=datetime.datetime.now().astimezone().isoformat(),smi_before_mib=before['free_mib'],torch_free_mib=tf,torch_total_mib=total/1048576,smi_after_mib=after['free_mib'],gpu_util_before=before['util'],gpu_util_after=after['util'],host_available_mib=host['MemAvailable'],host_total_mib=host['MemTotal'],rss_mib=host['rss_mib'],swap_free_mib=host['SwapFree'],swap_total_mib=host['SwapTotal'],delta_before_mib=tf-before['free_mib'],delta_after_mib=tf-after['free_mib'],conservative_free_mib=min(before['free_mib'],tf,after['free_mib']),sampling_seconds=time.monotonic()-start))
 if i<9:time.sleep(max(0,1-(time.monotonic()-start)))
OUT.write_text(json.dumps({'env':env,'rows':rows},indent=2))
print('OUTPUT',OUT,flush=True)
