#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, os, sys
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="outputs/D13_engineering_audit")
    a=ap.parse_args()
    checks = {
      "velodepth/models/velodepth.py": ROOT/"velodepth/models/velodepth.py",
      "velodepth/models/decoder_velo.py": ROOT/"velodepth/models/decoder_velo.py",
      "experiments/p1/gt4a2_y08cal_y09pareto.py": ROOT/"experiments/p1/gt4a2_y08cal_y09pareto.py",
      "experiments/p1/y08y09_models.py": ROOT/"experiments/p1/y08y09_models.py",
      "experiments/p1/y08y09_common.py": ROOT/"experiments/p1/y08y09_common.py",
      "experiments/p1/gt4a2_y04a_native_pre_readout_eval.py": ROOT/"experiments/p1/gt4a2_y04a_native_pre_readout_eval.py",
    }
    result={"package":"D13","status":"AUDIT_ONLY_NOT_METHOD_VALIDATION","repo":str(ROOT),"files":{}}
    missing=[]
    for k,p in checks.items():
        if p.is_file(): result["files"][k]={"exists":True,"sha256":sha(p),"bytes":p.stat().st_size}
        else: result["files"][k]={"exists":False}; missing.append(k)
    snap=os.environ.get("VDEEP_SNAPSHOT","")
    result["snapshot"]={"path":snap,"exists":bool(snap and Path(snap).is_dir())}
    result["cuda_visible_devices"]=os.environ.get("CUDA_VISIBLE_DEVICES","unset")
    result["missing_required_sources"]=missing
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
    (out/"source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if missing: raise SystemExit("Required source files missing; no model training was started.")
if __name__=="__main__": main()
