import hashlib
def fingerprint(t):
 x=t.detach().float().cpu().contiguous()
 return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'mean':float(x.mean()),'maxabs':float(x.abs().max()),'finite':bool(x.isfinite().all())}
def pack(xs):return [fingerprint(x) for x in xs]
