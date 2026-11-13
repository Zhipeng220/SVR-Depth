"""9244-parameter shared four-scale FiLM micro-readout; current frame only."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class PhaseMicroReadout(nn.Module):
    def __init__(self,conditioned=True):
        super().__init__();self.conditioned=bool(conditioned)
        self.phase_mlp=nn.Sequential(nn.Linear(2,8),nn.GELU(),nn.Linear(8,1024))
        self.spatial=nn.Conv2d(3,1,1)
        nn.init.zeros_(self.phase_mlp[-1].weight);nn.init.zeros_(self.phase_mlp[-1].bias)
        nn.init.zeros_(self.spatial.weight);nn.init.zeros_(self.spatial.bias)
    def forward(self,features,causal,h,compute_regularizer=True):
        if not isinstance(features,(tuple,list)) or len(features)!=4:raise ValueError('Expected four C* feature levels')
        if any(x.ndim!=4 or x.shape[1]!=512 or x.shape[0]!=1 for x in features):raise ValueError('Expected B=1, C=512 at each native scale')
        if not 0<=int(h)<=7:raise ValueError('S8 local phase must be 0..7')
        if int(h)==0:return features,features[0].new_zeros((),dtype=torch.float32)
        if causal is None or causal.ndim!=4 or causal.shape[1]!=3 or causal.shape[0]!=1:raise ValueError('Z_fast must contain exactly flow x/y + image residual, no age channel')
        vector=[h/8.,math.sin(math.pi*h/8.)] if self.conditioned else [0.,0.]
        e=torch.tensor([vector],device=features[0].device,dtype=torch.float32)
        gb=self.phase_mlp(e);gamma,beta=gb.chunk(2,dim=1);gamma=gamma[:,:,None,None];beta=beta[:,:,None,None]
        outputs=[];deltas=[]
        for x in features:
            z=F.interpolate(causal.detach().float(),size=x.shape[-2:],mode='bilinear',align_corners=False)
            scale=1.+.1*torch.tanh(self.spatial(z))
            delta=scale*(gamma*x.detach()+beta);outputs.append(x.detach()+delta);deltas.append(delta)
        regularizer=(sum(d.float().square().sum() for d in deltas)/sum(x.detach().float().square().sum() for x in features).clamp_min(1e-12)) if compute_regularizer else features[0].new_zeros((),dtype=torch.float32)
        return outputs,regularizer
