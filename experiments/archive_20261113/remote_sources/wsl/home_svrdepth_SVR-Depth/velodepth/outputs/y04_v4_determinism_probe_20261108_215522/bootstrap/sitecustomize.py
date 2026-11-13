import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.use_deterministic_algorithms(True)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
print('[DETERMINISM] ENABLED',flush=True)
