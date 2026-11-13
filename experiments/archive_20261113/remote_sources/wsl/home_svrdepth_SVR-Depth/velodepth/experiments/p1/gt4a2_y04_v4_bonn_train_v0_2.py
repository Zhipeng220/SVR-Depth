#!/usr/bin/env python3
"""Y04-V4 Bonn3/Bonn13 standalone persistent-head trainer.

Purpose: development of the original pre_readout Y04 head with a new, auditable
WSL manifest. DOES NOT change historical Y04 trainers/checkpoints.

check: CPU manifest validation.  dev: short end-to-end GPU runs with recoveries.
formal: only after independent physical-scene split review, 1000 updates.
adapt: explicitly disclosed same-environment Bonn domain adaptation, 1000 updates.

Each optimizer step takes one complete K32 cycle with TBPTT=4. The C3 cache
builder is invoked separately per sequence with <= 24 cycles per sequence;
no 24-cycle silent truncation of the global slot list.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

VERSION = "Y04-V4-Bonn-Adapt-0.2"
TRAIN3 = ["crowd2", "crowd3", "balloon2"]
NEEDED_FILES = [
    "experiments/p1/gt4a2_y04_fair_multiscene.py",
    "experiments/p1/gt4a2_c3_closed_loop.py",
    "experiments/p1/gt4a2_exp1_train_eval.py",
    "experiments/p1/gt2_fixed_clip_repair_train.py",
    "velodepth/models/decoder_velo.py",
    "velodepth/models/persistent_feature_repair.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_curve(path, rows):
    tmp = Path(str(path) + ".tmp")
    if rows:
        with tmp.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["check", "dev", "formal", "adapt"])
    p.add_argument("--profile", required=True, choices=["bonn3", "bonn13"])
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    p.add_argument("--init-head", default="outputs/gt4a2_c3/gt4a2_c3_initial_head.pt")
    p.add_argument("--cache-root", default="outputs/y04_v4_bonn_cache")
    p.add_argument("--run-dir", default="")
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--slots-per-seq", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--stop-after", type=int, default=0,
                   help="Development only: stop after this *total* step and write a recovery")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--protocol-file", default="", help="Reviewed in-domain protocol JSON (adapt required)")
    p.add_argument("--deterministic", action="store_true", help="Use strict CUDA determinism and disable TF32")
    p.add_argument("--preflight-only", action="store_true", help="Verify selected training protocol without CUDA, cache, or optimizer")
    p.add_argument("--scene-review", default="",
                   help="Manually audited scene mapping; required for formal mode")
    return p.parse_args()


def load_manifests(a):
    from gt4a2_exp1_train_eval import read_manifest
    folder = Path(a.manifest_dir).resolve()
    meta_file = folder / "manifest_identity.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("experiment") != "Y04-V4-Bonn13-Pmulti":
        raise RuntimeError("Unexpected manifest identity")
    all_train = list(meta["train_sequences"])
    all_dev = list(meta["dev_sequences"])
    if len(all_train) != 13 or len(all_dev) != 5 or set(all_train) & set(all_dev):
        raise RuntimeError("Expected 13 disjoint training and five development sequences")
    names = TRAIN3 if a.profile == "bonn3" else all_train
    if any(s not in all_train for s in names):
        raise RuntimeError("Bonn3 names do not agree with the manifest")
    if not (1 <= a.slots_per_seq <= 24):
        raise RuntimeError("C3 cache limit is 24 cycles per sequence; choose 1..24")
    rows_by_seq, csv_by_seq, hashes, gt_counts = {}, {}, {}, {}
    for seq in names:
        fp = folder / f"{seq}_pairs_v4_candidate.csv"
        digest = sha(fp)
        if meta["manifests"][seq]["manifest_sha256"] != digest:
            raise RuntimeError(f"Manifest SHA changed: {seq}")
        rows = read_manifest(fp, training=False)  # unmatched GT allowed during propagation
        if len(rows) < 32 * a.slots_per_seq:
            raise RuntimeError(f"Insufficient K32 cycles for {seq}")
        n = []
        for cyc in range(a.slots_per_seq):
            segment = rows[cyc * 32:(cyc + 1) * 32]
            count = sum(x.get("gt_match_status", "").upper() == "MATCHED"
                        for x in segment[1:])
            if count <= 0:
                raise RuntimeError(f"Cycle has no supervised propagation: {seq}/{cyc}")
            n.append(count)
        rows_by_seq[seq] = rows
        csv_by_seq[seq] = fp
        hashes[seq] = digest
        gt_counts[seq] = n
    return meta, names, rows_by_seq, csv_by_seq, hashes, gt_counts


def check_scene_review(a, meta):
    if not a.scene_review:
        raise RuntimeError("Formal training requires --scene-review from an actual manual scene audit")
    path = Path(a.scene_review).resolve()
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("verified") is not True or not review.get("evidence"):
        raise RuntimeError("Scene review not verified / missing evidence")
    ids = review.get("physical_scene_ids", {})
    names = list(meta["train_sequences"]) + list(meta["dev_sequences"])
    if any(not str(ids.get(seq, "")).strip() for seq in names):
        raise RuntimeError("All 18 physical scene IDs must be populated")
    if set(ids[s] for s in meta["train_sequences"]) & set(ids[s] for s in meta["dev_sequences"]):
        raise RuntimeError("Train/dev physical scene overlap; split must be revised first")
    if review.get("frozen_train_sequences") != meta["train_sequences"]:
        raise RuntimeError("Reviewed training list differs from candidate manifest")
    if review.get("frozen_dev_sequences") != meta["dev_sequences"]:
        raise RuntimeError("Reviewed development list differs from candidate manifest")
    if review.get("acknowledge_previous_y11_dev_exposure") is not True:
        raise RuntimeError("Historical Y11 development exposure needs acknowledgement")
    return sha(path)



def check_indomain_protocol(a, meta):
    if not a.protocol_file:
        raise RuntimeError("Adaptation requires --protocol-file (explicit shared-scene disclosure)")
    path = Path(a.protocol_file).resolve()
    p = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": "Y04-V4-Bonn-InDomain-1.0",
        "experiment_purpose": "bonn_in_domain_head_training",
        "split_type": "sequence_disjoint_same_environment",
        "physical_scene_independence": False,
        "shared_environment_declared": True,
        "bonn_dataset_zero_shot": False,
        "historical_y11_dev_exposure_acknowledged": True,
        "checkpoint_selection": "final_update_only",
    }
    for key, value in required.items():
        if p.get(key) != value:
            raise RuntimeError(f"In-domain protocol mismatch: {key}: {p.get(key)!r}")
    if p.get("train_sequences") != meta["train_sequences"] or p.get("dev_sequences") != meta["dev_sequences"]:
        raise RuntimeError("In-domain protocol train/dev lists do not match frozen manifest")
    if p.get("external_evaluation_datasets") != ["TUM-RGBD", "ScanNet", "Sintel"]:
        raise RuntimeError("Unexpected external evaluation plan")
    evidence = p.get("evidence", {})
    if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("sha256"):
        raise RuntimeError("In-domain protocol missing original review evidence")
    audit = (path.parent / evidence["path"]).resolve()
    if not audit.is_file() or sha(audit) != evidence["sha256"]:
        raise RuntimeError(f"In-domain scene audit evidence missing or changed: {audit}")
    print("[IN-DOMAIN] Bonn TRAIN/DEV share physical environment; not a scene-level generalization test", flush=True)
    print("[IN-DOMAIN] Bonn dataset-level zero-shot claim forbidden for this candidate", flush=True)
    return sha(path)


def configure_determinism():
    # This function must run before the first CUDA operation.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG. Expected :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print("[DETERMINISM] strict algorithms enabled; TF32 disabled; CUBLAS_WORKSPACE_CONFIG=:4096:8", flush=True)



def install_deterministic_backward():
    import torch
    import torch.nn.functional as F
    _pad=F.pad
    _grid=F.grid_sample
    class _Reflection(torch.autograd.Function):
     @staticmethod
     def forward(ctx, x, pad):
      ctx.pad=pad; ctx.shape=x.shape; ctx.device=x.device; ctx.dtype=x.dtype
      return _pad(x,pad,mode='reflect')
     @staticmethod
     def backward(ctx, grad):
      with torch.enable_grad():
       x=torch.zeros(ctx.shape,device='cpu',dtype=ctx.dtype,requires_grad=True)
       y=_pad(x,ctx.pad,mode='reflect')
       g=torch.autograd.grad(y,x,grad.to('cpu'))[0]
      return g.to(ctx.device),None
    def patched(x,pad,mode='constant',value=None):
     if mode=='reflect' and x.is_cuda and x.requires_grad and len(pad)==4 and x.ndim==4:
      return _Reflection.apply(x,tuple(pad))
     return _pad(x,pad,mode,value)
    F.pad=patched
    class _Grid(torch.autograd.Function):
     @staticmethod
     def forward(ctx,x,grid,mode,padding_mode,align_corners):
      ctx.save_for_backward(x,grid); ctx.options=(mode,padding_mode,align_corners)
      return _grid(x,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
     @staticmethod
     def backward(ctx,grad):
      x,grid=ctx.saved_tensors; mode,padding,align=ctx.options
      dtype=torch.float32 if x.dtype in (torch.float16,torch.bfloat16) else x.dtype
      with torch.enable_grad():
       xc=x.detach().to('cpu',dtype=dtype).requires_grad_(True); gc=grid.detach().to('cpu',dtype=dtype).requires_grad_(True)
       y=_grid(xc,gc,mode=mode,padding_mode=padding,align_corners=align)
       gx,gg=torch.autograd.grad(y,(xc,gc),grad.to('cpu',dtype=dtype))
      return gx.to(x.device,dtype=x.dtype),gg.to(grid.device,dtype=grid.dtype),None,None,None
    def patched_grid(x,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
     if x.is_cuda and (x.requires_grad or grid.requires_grad): return _Grid.apply(x,grid,mode,padding_mode,align_corners)
     return _grid(x,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
    F.grid_sample=patched_grid


def main():
    a = get_args()
    if not a.snapshot:
        raise RuntimeError("Set VDEEP_SNAPSHOT or pass --snapshot")
    snapshot = Path(a.snapshot).resolve()
    init_path = Path(a.init_head).resolve()
    if a.steps < 1 or a.save_every < 1 or a.stop_after < 0:
        raise RuntimeError("Invalid steps/save interval")
    if a.mode == "formal":
        if a.steps != 1000 or a.slots_per_seq != 8 or a.stop_after:
            raise RuntimeError("Formal protocol fixes 1000 updates, eight K32 slots/sequence")
    elif a.mode == "adapt":
        if a.steps != 1000 or a.slots_per_seq != 8 or a.stop_after:
            raise RuntimeError("In-domain adaptation fixes 1000 updates, eight K32 slots/sequence")
        if not a.deterministic:
            raise RuntimeError("In-domain adaptation requires --deterministic (embedded, no sitecustomize)")
    elif a.mode == "dev" and a.steps > 25:
        raise RuntimeError("Development mode restricted to <=25 steps")
    if a.mode == "check" and a.resume:
        raise RuntimeError("Cannot resume a check")
    if a.preflight_only and a.resume:
        raise RuntimeError("Cannot resume a preflight")
    if a.deterministic:
        configure_determinism()
    if a.resume and a.mode == "check":
        raise RuntimeError("Cannot resume a check")
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise RuntimeError("Missing VeloDepth snapshot")
    if not init_path.is_file():
        raise RuntimeError(f"Missing shared initial head: {init_path}")
    import gt4a2_exp1_train_eval as exp1
    import gt4a2_c3_closed_loop as c3
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_y04_fair_multiscene as y04
    if a.deterministic:
        install_deterministic_backward()
    meta, names, rows, csvs, hashes, gt_counts = load_manifests(a)
    slots = [(seq, i) for seq in names for i in range(a.slots_per_seq)]
    order = y04.balanced_order(len(slots), a.steps, a.seed)
    if a.mode == "formal":
        freeze_sha = check_scene_review(a, meta)
    elif a.mode == "adapt" or a.protocol_file:
        freeze_sha = check_indomain_protocol(a, meta)
    else:
        freeze_sha = "UNVERIFIED_DEVELOPMENT"
    identity = {
        "version": VERSION, "profile": a.profile, "mode": a.mode,
        "training_sequences": names, "development_sequences": list(meta["dev_sequences"]),
        "candidate_manifest_sha256": sha(Path(a.manifest_dir) / "manifest_identity.json"),
        "pair_sha256": hashes, "slots": slots, "order_sha256": stable_sha(order),
        "seed": a.seed, "updates": a.steps, "K": 32, "tbptt": 4,
        "init_sha256": sha(init_path), "base_sha256": sha(snapshot / "model.safetensors"),
        "snapshot_config_sha256": sha(snapshot / "config.json"),
        "source_sha256": {p: sha(ROOT / p) for p in NEEDED_FILES},
        "trainer_sha256": sha(Path(__file__)), "scene_review_sha256": freeze_sha,
        "deterministic_algorithms": bool(a.deterministic),
        "tf32_disabled": bool(a.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") if a.deterministic else "UNCONTROLLED",
        "experiment_claim": ("Bonn single-domain head training; shared-scene sequence-held-out development; "
                             "external test on TUM-RGBD/ScanNet/Sintel") if a.mode == "adapt" else "DEVELOPMENT_OR_FORMAL",
        "checkpoint_policy": "final optimizer step only; no validation selection",
    }
    print(f"[PREFLIGHT PASS] profile={a.profile}, slots={len(slots)}, steps={a.steps}")
    print("[GT EXPOSURE] unique matched propagation frames:", sum(map(sum, gt_counts.values())))
    print("[GT EXPOSURE] actual scheduled matched propagation frames:",
          sum(gt_counts[slots[i][0]][slots[i][1]] for i in order))
    print("[IDENTITY]", stable_sha(identity))
    if a.mode == "check" or a.preflight_only:
        print("[CHECK COMPLETE] no CUDA, no cache, no training")
        return
    if not a.run_dir:
        raise RuntimeError("Specify --run-dir for GPU runs")
    run = Path(a.run_dir).resolve()
    if a.resume:
        if not (run / "run_identity.json").is_file():
            raise RuntimeError("No run identity to resume")
        old = json.loads((run / "run_identity.json").read_text(encoding="utf-8"))
        if stable_sha(old) != stable_sha(identity):
            raise RuntimeError("Run identity mismatch: refuse resume")
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError("Output run directory not empty; use --resume or new path")
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "run_identity.json", identity)
    cache_root = Path(a.cache_root).resolve()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    readers = {}
    for seq in names:
        paths = {"snapshot": snapshot, "crowd2": csvs[seq]}
        print(f"[CACHE] {seq}: {a.slots_per_seq} K32 cycles", flush=True)
        cache = c3.build_cycle_cache(paths, rows[seq], cache_root / seq,
                                     device, limit=a.slots_per_seq)
        readers[seq] = c3.CycleReader(cache, max_items=0)  # Bound aggregate RAM across sequences.
        for cyc in range(a.slots_per_seq):
            y04.validate_cycle_geometry(readers[seq].load(cyc), f"{seq}/{cyc}")
    reader = y04.SceneCycleReader(readers, slots)
    # Seed *after* cache creation so fresh-cache vs warm-cache does not change training RNG.
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    model = gt2.load_model(snapshot, "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    head.load_state_dict(initial, strict=True)
    model.eval()
    head.train()
    params = [p for p in head.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 170496:
        raise RuntimeError("Unexpected number of repair-head params")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 170496:
        raise RuntimeError("Backbone not frozen")
    aligned = y04.GeometryAlignedReader(reader, model, a.profile)
    print("[WARMUP]", c3.warmup_shapes(model, aligned, device))
    optim = torch.optim.AdamW(params, lr=c3.LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    rec_file = run / "recovery.pt"
    final_file = run / "final.pt"
    begin, history = 0, []
    if a.resume:
        if final_file.exists():
            final = torch.load(final_file, map_location="cpu", weights_only=False)
            if final["identity_sha256"] != stable_sha(identity):
                raise RuntimeError("Final checkpoint belongs to another run")
            print("[ALREADY COMPLETE]", final_file)
            return
        if not rec_file.is_file():
            raise RuntimeError("Recovery missing")
        obj = torch.load(rec_file, map_location="cpu", weights_only=False)
        if obj["identity_sha256"] != stable_sha(identity):
            raise RuntimeError("Recovery identity mismatch")
        begin = int(obj["completed_updates"])
        history = obj["history"]
        if len(history) != begin or any(r["step"] != i + 1 for i, r in enumerate(history)):
            raise RuntimeError("Recovery history invalid")
        head.load_state_dict(obj["repair_head_state_dict"], strict=True)
        optim.load_state_dict(obj["optimizer_state_dict"])
        scaler.load_state_dict(obj["scaler_state_dict"])
        torch.set_rng_state(obj["torch_rng_state"])
        torch.cuda.set_rng_state_all(obj["cuda_rng_states"])
        np.random.set_state(obj["numpy_rng_state"])
        random.setstate(obj["python_rng_state"])
        write_curve(run / "training_curve.csv", history)
        print(f"[RESUME] restored exactly through step {begin}")
    def save_recovery(step):
        obj = {
            "identity_sha256": stable_sha(identity), "completed_updates": step,
            "repair_head_state_dict": {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            "optimizer_state_dict": optim.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "history": history,
        }
        temporary = Path(str(rec_file) + ".tmp")
        torch.save(obj, temporary)
        os.replace(temporary, rec_file)
        write_curve(run / "training_curve.csv", history)
    finish = a.steps
    if a.mode == "dev" and a.stop_after:
        finish = min(finish, a.stop_after)
    if finish < begin:
        raise RuntimeError("stop-after precedes recovered step")
    for step in range(begin + 1, finish + 1):
        slot = order[step - 1]
        seq, local_cycle = slots[slot]
        item = aligned.load(slot)
        t0 = time.perf_counter()
        metrics = c3.train_cycle(model, head, optim, scaler, item, device)
        record = {"step": step, "slot": slot, "sequence": seq,
                  "local_cycle": local_cycle, "wall_sec": round(time.perf_counter() - t0, 5),
                  **metrics}
        history.append(record)
        if step == 1 or step % 5 == 0 or step == finish:
            print(f"[TRAIN] {step}/{a.steps} seq={seq} cycle={local_cycle} "
                  f"loss={metrics['mean_log_depth_loss_before_update']:.6f} "
                  f"grad={metrics['gradient_norm_before_clip']:.6f}", flush=True)
        if (step % (1 if a.mode == "dev" else a.save_every) == 0
                or step == finish):
            save_recovery(step)
    if finish != a.steps:
        print(f"[INTERRUPTED BY DESIGN] step {finish}/{a.steps}; "
              "use identical args and --resume without --stop-after")
        return
    result = {"experiment": VERSION, "identity_sha256": stable_sha(identity),
              "mode": a.mode, "profile": a.profile, "updates": finish,
              "repair_head_state_dict": {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
              "final_head_state_sha256": exp1.state_sha256(head.state_dict()),
              "pretrained_weights_sha256": identity["base_sha256"],
              "training_pair_sha256_by_sequence": hashes,
              "optimizer_state_dict": optim.state_dict(),
              "checkpoint_selection": "final update; no validation selection",
              "training_purpose": identity["experiment_claim"],
              "determinism": {"enabled": bool(a.deterministic),
                              "cublas_workspace_config": identity["cublas_workspace_config"],
                              "tf32_disabled": bool(a.deterministic)}}
    tmp = Path(str(final_file) + ".tmp")
    torch.save(result, tmp)
    os.replace(tmp, final_file)
    verify = torch.load(final_file, map_location="cpu", weights_only=False)
    if verify["final_head_state_sha256"] != result["final_head_state_sha256"]:
        raise RuntimeError("Final checkpoint reload mismatch")
    atomic_json(run / "completion.json", {
        "status": ("DEV_COMPLETE" if a.mode == "dev" else
                   "BONN_IN_DOMAIN_ADAPT_COMPLETE" if a.mode == "adapt" else
                   "FORMAL_SCENE_INDEPENDENT_COMPLETE"),
        "head_sha256": result["final_head_state_sha256"],
        "checkpoint_sha256": sha(final_file), "training_updates": finish,
        "unique_slots": len(slots), "scene_review_sha256": freeze_sha,
        "evaluation_performed": False,
    })
    print("[COMPLETE]", run / "completion.json", flush=True)

if __name__ == "__main__":
    main()
