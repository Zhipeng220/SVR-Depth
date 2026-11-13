"""Y12-D paired recovery-space diagnostic on C* own S8 histories."""
from y12d_common import *
from y05_engine import Engine, detached_state, numpy_metrics, tsave
import contextlib, hashlib, json, math, time
from pathlib import Path


def _clone_cpu(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, list):
        return [_clone_cpu(torch, x) for x in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(torch, x) for x in value)
    if isinstance(value, dict):
        return {k: _clone_cpu(torch, v) for k, v in value.items()}
    return value


@contextlib.contextmanager
def capture_forward_fast(decoder, torch):
    """Capture only tensors already produced by the native fast path; no extra encoder pass."""
    capture = {"calls": 0, "inputs": None}
    original = decoder.forward_fast
    wanted = {
        "features_teacher", "features_residual", "flow_features", "flow_bwd_refined",
        "flow_bwd", "grid_sample", "image_residual", "validity_mask"
    }

    def wrapped(inputs, image_metas):
        capture["calls"] += 1
        if capture["inputs"] is None:
            capture["inputs"] = {k: _clone_cpu(torch, inputs[k]) for k in wanted if k in inputs}
            capture["input_keys"] = sorted(str(k) for k in inputs.keys())
        return original(inputs, image_metas)

    decoder.forward_fast = wrapped
    try:
        yield capture
    finally:
        decoder.forward_fast = original


def _tensor_leaves(torch, value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for x in value:
            yield from _tensor_leaves(torch, x)
    elif isinstance(value, dict):
        for x in value.values():
            yield from _tensor_leaves(torch, x)


def tensor_stats(torch, value):
    vals = [x.detach().float().reshape(-1) for x in _tensor_leaves(torch, value)]
    vals = [x for x in vals if x.numel()]
    if not vals:
        return {"shapes": "[]", "finite_ratio": None, "mean": None, "std": None,
                "p10": None, "p50": None, "p90": None, "numel": 0}
    flat = torch.cat(vals)
    finite = flat[torch.isfinite(flat)]
    result = {"shapes": json.dumps([list(x.shape) for x in _tensor_leaves(torch, value)], separators=(",", ":")),
              "finite_ratio": float(finite.numel() / max(flat.numel(), 1)), "numel": int(flat.numel())}
    if finite.numel():
        q = torch.quantile(finite, torch.tensor([0.10, 0.50, 0.90], device=finite.device))
        result.update(mean=float(finite.mean().item()), std=float(finite.std(unbiased=False).item()),
                      p10=float(q[0].item()), p50=float(q[1].item()), p90=float(q[2].item()))
    else:
        result.update(mean=None, std=None, p10=None, p50=None, p90=None)
    return result


def build_teacher_conditioned_features(engine, current, rays_base):
    """Same-ray teacher-conditioned feature construction used by E2-S4."""
    torch = engine.rt.torch
    model = engine.model
    decoder = model.pixel_decoder
    # The Y05-R pinned decoder (SHA 35ca1262...) does NOT expose
    # ``readout_fast_features``. Use a verified reproduction of the original
    # radial forward_fast readout, without mutating/patching the model source.
    require(hasattr(decoder, "radial_module"), "Decoder lacks native radial module")
    require(hasattr(decoder, "radial_module") and hasattr(decoder.radial_module, "condition"),
            "Reviewed decoder lacks radial same-ray conditioning")
    image = normalized_image_from_frame(current)
    require(isinstance(image, torch.Tensor),
            "Y05 Engine frame['img'] is not a torch.Tensor: " + type(image).__name__)
    require(image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3,
            "Y05 Engine frame['img'] must be normalized BCHW with B=1,C=3; "
            f"got shape={tuple(image.shape)}")
    require(bool(torch.isfinite(image).all().item()),
            "Y05 Engine frame['img'] contains nonfinite values")
    # load_frame already returns ImageNet-normalized/padded/resized BCHW; do not normalize again.
    # Match native keyframe_state: encoder and decoder adapter run under CUDA FP16 autocast.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        features, tokens = model.pixel_encoder(image)
        features = [model.stacking_fn(features[i:j]).contiguous()
                    for i, j in model.slices_encoder_range]
        tokens = [model.stacking_fn(tokens[i:j]).contiguous()
                  for i, j in model.slices_encoder_range]
        metas = [{"T": 1, "B": 1}]
        base_inputs = {
            "image": image,
            "image_original": image,
            "validity_mask": engine.shape["validity_mask"],
            "features": features,
            "tokens": tokens,
        }
        decoder.set_shapes(base_inputs, metas)
        adapted = decoder.input_adapter(features)
        radial = decoder.radial_module
        ray_embedding = radial.embed_rays(rays_base)
        conditioned = radial.condition(adapted, ray_embedding)
        h_common, w_common = radial.shapes
        donor = [x.reshape(x.shape[0], h_common, w_common, x.shape[-1])
                   .permute(0, 3, 1, 2).contiguous() for x in conditioned]
    return donor


def native_persistent_readout(engine, features):
    """Reproduce original Y05-R fast readout exactly, without changing decoder.

    The locked decoder SHA=35ca12629acc... contains this block inside
    RadialModule.forward_fast, but has NO readout_fast_features API.
    A pinned C* propagated state is passed through this helper at runtime and
    its radius/confidence are checked against the original forward_fast result.
    """
    torch = engine.rt.torch
    radial = engine.model.pixel_decoder.radial_module
    require(len(features) == 4 and len(radial.process_features) == len(radial.ups) == 3,
            "Unexpected native radial fast readout structure")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        lateral = [features[0]] + [radial.process_features[i](features[i + 1])
                                    for i in range(len(radial.ups))]
        latents = lateral[0]
        decoded = []
        for i, up in enumerate(radial.ups):
            latents = latents + lateral[i + 1]
            latents = up(latents)
            decoded.append(latents)
        lograd = radial.depth_proj(decoded)
        # The original reviewed forward_fast uses logconf_student = logdepth_student.
        radius = torch.exp(lograd.clip(min=-8.0, max=8.0) + 2.0)
        confidence = torch.exp(lograd.clip(min=-8.0, max=10.0))
    return radius, confidence


def check_native_readout_identity(engine, propagated_state, case_id):
    torch = engine.rt.torch
    radius, confidence = native_persistent_readout(engine, propagated_state[2])
    checks = {}
    for key, actual, expected in (("radius", radius, propagated_state[0]),
                                  ("confidence", confidence, propagated_state[1])):
        require(tuple(actual.shape) == tuple(expected.shape),
                f"{case_id}: native fast-readout {key} shape mismatch")
        diff = (actual.float() - expected.float()).abs()
        max_abs = float(diff.max().item())
        denom = expected.float().abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item())
        # Exact same frozen modules / same autocast; tolerate limited FP16
        # nondeterminism but fail hard on state-contract discrepancies.
        require(max_abs <= 5e-3 and max_rel <= 5e-3,
                f"{case_id}: reproduced native fast readout differs for {key}: abs={max_abs}, rel={max_rel}")
        checks[key + "_max_abs"] = max_abs
        checks[key + "_max_rel"] = max_rel
    return checks


def _as_b1hw(torch, value, device, h, w, default=True):
    if not isinstance(value, torch.Tensor):
        return torch.full((1, 1, h, w), 1.0 if default else 0.0, device=device)
    x = value.detach().to(device=device, dtype=torch.float32)
    while x.ndim > 4:
        x = x[:, 0]
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[:, None] if x.shape[0] == 1 else x[None]
    if x.ndim != 4:
        raise RuntimeError(f"Expected validity mask rank 2-4, got {tuple(x.shape)}")
    if x.shape[1] != 1:
        x = x[:, :1]
    if tuple(x.shape[-2:]) != (h, w):
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="nearest")
    return (x > 0.5).float()


def _flow_grad(torch, flow):
    f = flow.detach().float()
    du_dy, du_dx = torch.gradient(f[:, 0], dim=(1, 2))
    dv_dy, dv_dx = torch.gradient(f[:, 1], dim=(1, 2))
    return torch.sqrt(du_dx.square() + du_dy.square() + dv_dx.square() + dv_dy.square() + 1e-12)


def make_region_masks(engine, capture_inputs, previous, current):
    """Create predeclared warp-valid / invalid-exposure / flow-boundary proxy masks."""
    torch = engine.rt.torch
    F = torch.nn.functional
    from velodepth.models.decoder_velo import flow2grid_sample
    flow = capture_inputs.get("flow_bwd_refined")
    require(isinstance(flow, torch.Tensor),
            "Fast-path capture did not expose flow_bwd_refined; cannot form the required warp-region diagnostics")
    flow = flow.detach().to(device="cuda", dtype=torch.float32)
    require(flow.ndim == 4 and flow.shape[1] == 2, f"Unexpected refined flow shape: {tuple(flow.shape)}")
    h, w = flow.shape[-2:]
    grid = flow2grid_sample(flow.clone(), mask=False)
    in_bounds = ((grid[..., 0] >= -1) & (grid[..., 0] <= 1)
                 & (grid[..., 1] >= -1) & (grid[..., 1] <= 1))[:, None]
    cur_valid = _as_b1hw(torch, current.get("validity_mask"), flow.device, h, w)
    prev_valid = _as_b1hw(torch, previous.get("validity_mask"), flow.device, h, w)
    warped_prev_valid = F.grid_sample(prev_valid, grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
    valid_warp = in_bounds & (cur_valid > 0.5) & warped_prev_valid
    invalid_exposure = (cur_valid > 0.5) & ~valid_warp
    grad = _flow_grad(torch, flow)
    valid2d = valid_warp[0, 0]
    valid_grad = grad[0][valid2d]
    if valid_grad.numel():
        # Stable top-decile selection avoids marking every cell as a boundary when gradients tie.
        k = max(1, int(math.ceil(0.10 * valid_grad.numel())))
        order = torch.argsort(valid_grad, descending=True, stable=True)
        selected_flat = torch.nonzero(valid2d.reshape(-1), as_tuple=False).reshape(-1)[order[:k]]
        motion2d = torch.zeros_like(valid2d).reshape(-1)
        motion2d[selected_flat] = True
        motion = motion2d.reshape(1, 1, h, w)
        q90_value = float(valid_grad[order[k - 1]].item())
    else:
        motion = torch.zeros_like(valid_warp)
        q90_value = None
    normal = valid_warp & ~motion
    masks = {
        "normal_warp": normal[0, 0].detach().cpu().numpy().astype(bool),
        "invalid_or_exposure_proxy": invalid_exposure[0, 0].detach().cpu().numpy().astype(bool),
        "motion_boundary_proxy": motion[0, 0].detach().cpu().numpy().astype(bool),
        "current_valid": (cur_valid[0, 0] > 0.5).detach().cpu().numpy().astype(bool),
    }
    return masks, {"flow_shape": [int(x) for x in flow.shape], "flow_grad_q90": q90_value,
                   "valid_warp_cells": int(valid_warp.sum().item()),
                   "invalid_exposure_cells": int(invalid_exposure.sum().item()),
                   "motion_boundary_cells": int(motion.sum().item()),
                   "current_validity_mask_present": isinstance(current.get("validity_mask"), torch.Tensor),
                   "previous_validity_mask_present": isinstance(previous.get("validity_mask"), torch.Tensor),
                   "mask_to_gt_mapping": "crop to current-valid bounding box, then nearest-neighbor resize to GT shape",
                   "region_proxy_note": "Flow-gradient and warp-validity proxies only; not semantic/occlusion ground truth."}


def _mask_to_gt(cv2, mask, current_valid, shape):
    gh, gw = shape
    valid = current_valid.astype("uint8")
    ys, xs = __import__("numpy").where(valid > 0)
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = mask[y0:y1, x0:x1]
    else:
        crop = mask
    if crop.size == 0:
        crop = mask
    return cv2.resize(crop.astype("uint8"), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)


def _pixel_error(np, pred, gt):
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    abs_err = np.zeros(gt.shape, dtype=np.float32)
    abs_err[valid] = np.abs(pred[valid].astype(np.float32) - gt[valid].astype(np.float32))
    rel_err = np.zeros(gt.shape, dtype=np.float32)
    rel_err[valid] = abs_err[valid] / gt[valid].astype(np.float32)
    return rel_err, abs_err, valid


def region_error_rows(ctx, case, branch_predictions, gt, masks, proxy_meta):
    np = ctx.rt.np
    cv2 = ctx.rt.cv2
    if gt is None:
        return [], None
    errors, abs_errors = {}, {}
    valid = None
    for branch, pred in branch_predictions.items():
        errors[branch], abs_errors[branch], branch_valid = _pixel_error(np, pred, gt)
        valid = branch_valid if valid is None else (valid & branch_valid)
    mapped = {}
    for name in ("normal_warp", "invalid_or_exposure_proxy", "motion_boundary_proxy"):
        mapped[name] = _mask_to_gt(cv2, masks[name], masks["current_valid"], gt.shape) & valid
    rows = []
    for region, mask in mapped.items():
        count = int(mask.sum())
        row = {"case_id": case["case_id"], "sequence": case["sequence"], "anchor": int(case["anchor"]),
               "region": region, "valid_pixels": count, "flow_grad_q90": proxy_meta.get("flow_grad_q90"),
               "flow_shape": json.dumps(proxy_meta.get("flow_shape")),
               "valid_warp_cells": proxy_meta.get("valid_warp_cells"),
               "invalid_exposure_cells": proxy_meta.get("invalid_exposure_cells"),
               "motion_boundary_cells": proxy_meta.get("motion_boundary_cells"),
               "current_validity_mask_present": proxy_meta.get("current_validity_mask_present"),
               "previous_validity_mask_present": proxy_meta.get("previous_validity_mask_present"),
               "mask_to_gt_mapping": proxy_meta.get("mask_to_gt_mapping"),
               "proxy_note": proxy_meta["region_proxy_note"]}
        for branch in ("SKIP", "C_STAR", "TEACHER_STATE_COMMIT"):
            row[branch + "_AbsRel"] = float(errors[branch][mask].mean()) if count else None
            row[branch + "_RMSE"] = float(np.sqrt(np.mean(abs_errors[branch][mask] ** 2))) if count else None
        row["teacher_minus_cstar_AbsRel"] = (row["TEACHER_STATE_COMMIT_AbsRel"] - row["C_STAR_AbsRel"]
                                                if count else None)
        row["cstar_minus_skip_AbsRel"] = (row["C_STAR_AbsRel"] - row["SKIP_AbsRel"] if count else None)
        row["teacher_minus_cstar_RMSE"] = (row["TEACHER_STATE_COMMIT_RMSE"] - row["C_STAR_RMSE"] if count else None)
        row["cstar_minus_skip_RMSE"] = (row["C_STAR_RMSE"] - row["SKIP_RMSE"] if count else None)
        rows.append(row)
    arrays = {"valid": valid.astype(bool)}
    for branch in ("SKIP", "C_STAR", "TEACHER_STATE_COMMIT"):
        arrays["err_" + branch.lower()] = errors[branch].astype(np.float16)
        arrays["abs_depth_err_" + branch.lower()] = abs_errors[branch].astype(np.float16)
    arrays["delta_teacher_minus_cstar"] = (errors["TEACHER_STATE_COMMIT"] - errors["C_STAR"]).astype(np.float16)
    arrays["delta_cstar_minus_skip"] = (errors["C_STAR"] - errors["SKIP"]).astype(np.float16)
    for name, mask in mapped.items():
        arrays["mask_" + name] = mask.astype(bool)
    return rows, arrays


def feature_gap_rows(ctx, case, parent_features, donor_features, capture_inputs, masks):
    torch = ctx.rt.torch
    F = torch.nn.functional
    from velodepth.models.decoder_velo import flow2grid_sample
    flow = capture_inputs["flow_bwd_refined"].detach().to(device="cuda", dtype=torch.float32)
    rows = []
    for level, (base, donor) in enumerate(zip(parent_features, donor_features)):
        require(tuple(base.shape) == tuple(donor.shape),
                f"{case['case_id']}: teacher/base feature shape mismatch at level {level}: {tuple(base.shape)} vs {tuple(donor.shape)}")
        f = F.interpolate(flow, size=base.shape[-2:], mode="bilinear", align_corners=True)
        grid = flow2grid_sample(f.clone(), mask=False)
        warped = F.grid_sample(base.detach().float(), grid, mode="bilinear", align_corners=True)
        donor_f = donor.detach().float()
        gap = (donor_f - warped).abs().mean(dim=1)[0]
        base_gap = (donor_f - base.detach().float()).abs().mean(dim=1)[0]
        denom = warped.square().sum().sqrt().clamp_min(1e-12)
        rel_l2 = float((donor_f - warped).square().sum().sqrt().div(denom).item())
        cosine = float(F.cosine_similarity(donor_f.flatten(), warped.flatten(), dim=0).item())
        # Reuse the exact disjoint input-resolution proxy masks, downsampled to this feature grid.
        feat_masks = {}
        for region in ("normal_warp", "invalid_or_exposure_proxy", "motion_boundary_proxy"):
            source_mask = torch.as_tensor(masks[region], device=gap.device, dtype=torch.float32)[None, None]
            feat_masks[region] = F.interpolate(source_mask, size=gap.shape, mode="nearest")[0, 0] > 0.5
        for region, mask in feat_masks.items():
            n = int(mask.sum().item())
            rows.append({"case_id": case["case_id"], "sequence": case["sequence"],
                "anchor": int(case["anchor"]), "feature_level": level, "region": region,
                "height": int(gap.shape[-2]), "width": int(gap.shape[-1]), "cells": n,
                "donor_minus_warped_mean_abs": float(gap[mask].mean().item()) if n else None,
                "donor_minus_unwarped_mean_abs": float(base_gap[mask].mean().item()) if n else None,
                "warped_F_mean": float(warped[0].mean().item()),
                "warped_F_std": float(warped[0].std(unbiased=False).item()),
                "donor_F_mean": float(donor_f[0].mean().item()),
                "donor_F_std": float(donor_f[0].std(unbiased=False).item()),
                "global_donor_minus_warped_rel_l2": rel_l2,
                "global_donor_warped_cosine": cosine})
    return rows


def captured_input_rows(torch, case, capture_inputs):
    rows = []
    for name in ("features_teacher", "image_residual", "features_residual", "flow_features", "flow_bwd_refined", "flow_bwd", "grid_sample"):
        if name not in capture_inputs:
            continue
        s = tensor_stats(torch, capture_inputs[name])
        rows.append({"case_id": case["case_id"], "sequence": case["sequence"], "anchor": int(case["anchor"]),
                     "signal": name, **s})
    return rows


def _run_future(engine, state, frames, statics, branch, case):
    rows, predictions = [], {}
    torch = engine.rt.torch
    for h in range(1, HORIZON + 1):
        state, pred = engine.eval_step(state, frames[h - 1], frames[h], K, "S8", apply=False, static=statics[h])
        mm = numpy_metrics(engine.rt.np, pred, frames[h])
        row = {"case_id": case["case_id"], "sequence": case["sequence"], "anchor": int(case["anchor"]),
               "age": int(case["age"]), "branch": branch, "h": h, "frame_index": int(frames[h]["index"]),
               "gt_matched": bool(frames[h].get("gt_np") is not None), **mm}
        rows.append(row)
        predictions[h] = pred
    return rows, predictions


def _case_metric_row(case, branch, h, frame, pred, mm):
    return {"case_id": case["case_id"], "sequence": case["sequence"], "anchor": int(case["anchor"]),
            "age": int(case["age"]), "branch": branch, "h": h, "frame_index": int(frame["index"]),
            "gt_matched": bool(frame.get("gt_np") is not None), **mm,
            "prediction_sha256": hashlib.sha256(pred.astype("float32", copy=False).tobytes()).hexdigest()}


def run_case(ctx, engine, case, parent, out_dir, signature):
    torch = ctx.rt.torch
    np = ctx.rt.np
    engine.configure(case["sequence"])
    frames, statics = engine.context(case, horizon=HORIZON)
    require(int(frames[-1]["index"]) == int(case["anchor"]) - 1, "Parent/previous RGB index mismatch")
    require(int(frames[0]["index"]) == int(case["anchor"]), "Anchor RGB index mismatch")
    parent_gpu = detached_state(parent, "cuda")
    require(len(parent_gpu) == 4 and isinstance(parent_gpu[2], (list, tuple)) and len(parent_gpu[2]) == 4,
            "Y05-R C* parent state contract differs from (radius, confidence, 4 feature levels, rays)")
    require(all(bool(torch.isfinite(x).all().item()) for x in (parent_gpu[0], parent_gpu[1], *parent_gpu[2], parent_gpu[3])),
            "Nonfinite C* own-history parent state")

    # D0 anchor propagation, while capturing the exact already-computed fast-path signals.
    with capture_forward_fast(engine.model.pixel_decoder, torch) as capture:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        state_skip, pred_skip = engine.eval_step(detached_state(parent_gpu, "cuda"), frames[-1], frames[0], K, "S8", apply=False, static=statics[0])
        torch.cuda.synchronize(); skip_anchor_ms = (time.perf_counter() - t0) * 1000.0
    require(capture["calls"] == 1 and capture["inputs"] is not None,
            f"Expected exactly one native pixel_decoder.forward_fast call on the anchor, observed {capture['calls']}")
    cap = capture["inputs"]
    require("flow_bwd_refined" in cap, "Captured fast path lacks flow_bwd_refined")
    require("features_residual" in cap and "flow_features" in cap,
            "Captured fast path lacks residual/flow feature evidence required by Y12-D")
    captured_parent_features = cap.get("features_teacher")
    require(isinstance(captured_parent_features, (list, tuple)) and len(captured_parent_features) == 4,
            "Fast-path features_teacher input is missing or does not expose all four persistent levels")
    max_fast_input_parent_gap = 0.0
    for level, (parent_feature, captured_feature) in enumerate(zip(parent_gpu[2], captured_parent_features)):
        require(tuple(parent_feature.shape) == tuple(captured_feature.shape),
                f"{case['case_id']}: captured fast-path parent feature shape differs at level {level}")
        diff = (parent_feature.detach().float().cpu() - captured_feature.detach().float()).abs().max().item()
        max_fast_input_parent_gap = max(max_fast_input_parent_gap, float(diff))
    require(max_fast_input_parent_gap <= 1e-5,
            f"{case['case_id']}: captured fast-path feature input is not the pinned C* parent (max abs gap {max_fast_input_parent_gap})")

    # D1: one C* repair at the same anchor and from the identical parent state.
    torch.cuda.synchronize(); t0 = time.perf_counter()
    state_cstar, pred_cstar = engine.eval_step(detached_state(parent_gpu, "cuda"), frames[-1], frames[0], K, "S8", apply=True, static=statics[0])
    torch.cuda.synchronize(); cstar_anchor_ms = (time.perf_counter() - t0) * 1000.0

    # D2: full current-frame pixel encoder, conditioned on the same persistent ray field.
    require(torch.equal(state_skip[3], parent_gpu[3]),
            f"{case['case_id']}: native propagation unexpectedly changed rays_base; same-ray donor contract is not valid")
    # Contract canary: locally reproduced fast readout must agree with native
    # forward_fast on a real propagated C* state before using it for the donor.
    native_readout_identity = check_native_readout_identity(engine, state_skip, case['case_id'])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    donor_features = build_teacher_conditioned_features(engine, frames[0], state_skip[3])
    torch.cuda.synchronize(); teacher_encode_ms = (time.perf_counter() - t0) * 1000.0
    for level, (base, donor) in enumerate(zip(state_skip[2], donor_features)):
        require(tuple(base.shape) == tuple(donor.shape),
                f"{case['case_id']}: donor feature level {level} shape mismatch {tuple(base.shape)} vs {tuple(donor.shape)}")
        require(bool(torch.isfinite(donor).all().item()), f"{case['case_id']}: nonfinite teacher donor at level {level}")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        donor_radius, donor_confidence = native_persistent_readout(engine, donor_features)
    require(bool(torch.isfinite(donor_radius).all().item()) and bool((donor_radius > 0).all().item()),
            f"{case['case_id']}: teacher donor readout radius is invalid")
    require(bool(torch.isfinite(donor_confidence).all().item()),
            f"{case['case_id']}: teacher donor readout confidence is invalid")
    torch.cuda.synchronize(); teacher_fast_readout_ms = (time.perf_counter() - t0) * 1000.0
    # The Y05 native tuple is (radius, confidence, persistent feature levels, rays).
    # Commit donor readout geometry and every feature level together; keep rays unchanged
    # because the donor was explicitly conditioned in this same ray field.
    state_teacher = (donor_radius.detach().clone(), donor_confidence.detach().clone(),
                     [x.detach().clone() for x in donor_features], state_skip[3].detach().clone())
    require(tuple(state_teacher[0].shape) == tuple(state_skip[0].shape)
            and tuple(state_teacher[1].shape) == tuple(state_skip[1].shape),
            "Teacher readout geometry shape differs from native propagated state")
    require(all(torch.equal(a, b) for a, b in zip((state_teacher[3],), (state_skip[3],))),
            "Same-ray teacher state commit changed rays_base")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        pred_teacher_anchor = engine.read_depth(state_teacher)
    torch.cuda.synchronize(); teacher_depth_conversion_ms = (time.perf_counter() - t0) * 1000.0

    # Engine.eval_step returns CPU NumPy; Engine.read_depth returns a CUDA Tensor.
    # Canonicalize every h0 branch before metrics, hashing and pixel diagnostics.
    anchor_predictions = {
        "SKIP": prediction_to_numpy(np, pred_skip),
        "C_STAR": prediction_to_numpy(np, pred_cstar),
        "TEACHER_STATE_COMMIT": prediction_to_numpy(np, pred_teacher_anchor),
    }

    # Anchor metrics: h0 is deliberately separate from H4 future scores.
    anchor_rows = []
    for branch, pred in anchor_predictions.items():
        mm = numpy_metrics(np, pred, frames[0])
        anchor_rows.append(_case_metric_row(case, branch, 0, frames[0], pred, mm))

    region_masks, proxy_meta = make_region_masks(engine, cap, frames[-1], frames[0])
    region_rows, pixel_arrays = region_error_rows(ctx, case, anchor_predictions, frames[0].get("gt_np"), region_masks, proxy_meta)
    feature_rows = feature_gap_rows(ctx, case, parent_gpu[2], donor_features, cap, region_masks)
    signal_rows = captured_input_rows(torch, case, cap)

    # Future H4: all three branches start from their own anchor state; no further writeback.
    future_rows = []
    for branch, state in (("SKIP", state_skip), ("C_STAR", state_cstar), ("TEACHER_STATE_COMMIT", state_teacher)):
        rows, _ = _run_future(engine, detached_state(state, "cuda"), frames, statics, branch, case)
        future_rows.extend(rows)

    pixel_rel = None
    pixel_sha = None
    if pixel_arrays is not None:
        pixel_dir = Path(out_dir) / "pixel_diagnostics"
        pixel_dir.mkdir(parents=True, exist_ok=True)
        pixel_path = pixel_dir / (case["case_id"] + ".npz")
        np.savez_compressed(pixel_path, **pixel_arrays)
        pixel_rel = str(pixel_path.relative_to(out_dir))
        pixel_sha = sha(pixel_path)

    timing_row = {"case_id": case["case_id"], "sequence": case["sequence"], "anchor": int(case["anchor"]),
                  "skip_anchor_ms": skip_anchor_ms, "cstar_anchor_ms": cstar_anchor_ms,
                  "teacher_encoder_condition_ms": teacher_encode_ms, "teacher_fast_readout_ms": teacher_fast_readout_ms,
                  "teacher_depth_conversion_ms": teacher_depth_conversion_ms,
                  "fast_forward_calls_captured": int(capture["calls"]),
                  "fast_forward_input_keys": json.dumps(capture.get("input_keys", [])),
                  "max_abs_fast_input_parent_feature_gap": max_fast_input_parent_gap,
                  "native_fast_readout_identity": native_readout_identity,
                  "timing_scope_note": "Per-anchor fast step and teacher encoder/conditioning; DIS/static precomputation and RGB/GT I/O are outside these intervals.",
                  "teacher_cost_note": "Full-frame current pixel encoder + radial same-ray conditioning; diagnostic reference only, not low-cost deployment timing."}
    return {"signature": signature, "case_id": case["case_id"], "sequence": case["sequence"],
            "anchor": int(case["anchor"]), "age": int(case["age"]),
            "anchor_rows": anchor_rows, "future_rows": future_rows, "region_rows": region_rows,
            "feature_rows": feature_rows, "signal_rows": signal_rows, "timing_row": timing_row,
            "proxy_meta": proxy_meta, "pixel_npz": pixel_rel, "pixel_npz_sha256": pixel_sha,
            "parent_snapshot_index": int(case["snapshot_index"]),
            "parent_state_contract": "Y05 Engine tuple (radius, confidence, 4 persistent feature levels, rays); D2 replaces radius, confidence and all 4 feature levels from the same-ray donor readout; rays_base is unchanged.",
            "teacher_semantics": "Same-ray radial-conditioned full-frame pixel-encoder donor with matching fast readout geometry; not GT and not a full native keyframe reset."}


def aggregate_case_results(ctx, out_dir, cases, signature, partial_smoke=False):
    out_dir = Path(out_dir)
    case_files = sorted((out_dir / "cases").glob("*.json"))
    all_anchor, all_future, all_region, all_feature, all_signal, all_timing = [], [], [], [], [], []
    for p in case_files:
        obj = jread(p)
        require(obj.get("signature") == signature, "Foreign case result signature: " + str(p))
        if obj.get("pixel_npz"):
            pixel = out_dir / obj["pixel_npz"]
            require(pixel.is_file() and sha(pixel) == obj["pixel_npz_sha256"], "Pixel diagnostic archive missing/changed: " + str(pixel))
        all_anchor.extend(obj["anchor_rows"]); all_future.extend(obj["future_rows"])
        all_region.extend(obj["region_rows"]); all_feature.extend(obj["feature_rows"])
        all_signal.extend(obj["signal_rows"]); all_timing.append(obj["timing_row"])
    require(len({r["case_id"] for r in all_anchor}) == len(cases), "Not all planned anchor cases are complete")
    cwrite(out_dir / "y12d_anchor_frames.csv", all_anchor)
    cwrite(out_dir / "y12d_h4_frames.csv", all_future)
    anchor_by_case_branch = {(r["case_id"], r["branch"]): r for r in all_anchor}
    anchor_seq_rows = []
    for seq in SEQS:
        local_cases = [c for c in cases if c["sequence"] == seq]
        paired = []
        for c in local_cases:
            d0 = anchor_by_case_branch.get((c["case_id"], "SKIP"))
            d1 = anchor_by_case_branch.get((c["case_id"], "C_STAR"))
            d2 = anchor_by_case_branch.get((c["case_id"], "TEACHER_STATE_COMMIT"))
            if d0 and d1 and d2 and all(x.get("AbsRel") not in (None, "") for x in (d0, d1, d2)):
                paired.append((d0, d1, d2))
        if paired:
            anchor_seq_rows.append({"sequence": seq, "n_anchors": len(paired),
                "SKIP_h0_AbsRel": sum(float(x[0]["AbsRel"]) for x in paired) / len(paired),
                "C_STAR_h0_AbsRel": sum(float(x[1]["AbsRel"]) for x in paired) / len(paired),
                "TEACHER_h0_AbsRel": sum(float(x[2]["AbsRel"]) for x in paired) / len(paired),
                "teacher_minus_cstar_h0_AbsRel": sum(float(x[2]["AbsRel"]) - float(x[1]["AbsRel"]) for x in paired) / len(paired),
                "teacher_better_anchors": sum(float(x[2]["AbsRel"]) < float(x[1]["AbsRel"]) for x in paired),
                "teacher_worse_anchors": sum(float(x[2]["AbsRel"]) > float(x[1]["AbsRel"]) for x in paired),
                "teacher_severe_harmed_anchors_gt0p01": sum(float(x[2]["AbsRel"]) - float(x[1]["AbsRel"]) > SEVERE_HARM_THRESHOLD_ABSREL for x in paired)})
    if anchor_seq_rows:
        anchor_seq_rows.append({"sequence": "MACRO_EQUAL_VIDEO", "n_anchors": sum(r["n_anchors"] for r in anchor_seq_rows),
            "SKIP_h0_AbsRel": sum(r["SKIP_h0_AbsRel"] for r in anchor_seq_rows) / len(anchor_seq_rows),
            "C_STAR_h0_AbsRel": sum(r["C_STAR_h0_AbsRel"] for r in anchor_seq_rows) / len(anchor_seq_rows),
            "TEACHER_h0_AbsRel": sum(r["TEACHER_h0_AbsRel"] for r in anchor_seq_rows) / len(anchor_seq_rows),
            "teacher_minus_cstar_h0_AbsRel": sum(r["teacher_minus_cstar_h0_AbsRel"] for r in anchor_seq_rows) / len(anchor_seq_rows),
            "teacher_better_anchors": sum(r["teacher_better_anchors"] for r in anchor_seq_rows),
            "teacher_worse_anchors": sum(r["teacher_worse_anchors"] for r in anchor_seq_rows),
            "teacher_severe_harmed_anchors_gt0p01": sum(r["teacher_severe_harmed_anchors_gt0p01"] for r in anchor_seq_rows)})
        cwrite(out_dir / "y12d_anchor_summary.csv", anchor_seq_rows)
    if all_region: cwrite(out_dir / "y12d_region_diagnostics.csv", all_region)
    if all_feature: cwrite(out_dir / "y12d_feature_gap.csv", all_feature)
    if all_signal: cwrite(out_dir / "y12d_fast_signal_stats.csv", all_signal)
    cwrite(out_dir / "y12d_timing.csv", all_timing)

    # H4 window summaries: each anchor is one paired window; h0 is excluded.
    windows = []
    by_case_branch = {}
    for row in all_future:
        if row.get("AbsRel") in (None, ""):
            continue
        by_case_branch.setdefault((row["case_id"], row["sequence"], row["branch"]), []).append(row)
    case_lookup = {c["case_id"]: c for c in cases}
    for case_id, case in case_lookup.items():
        seq = case["sequence"]
        branch_metrics = {}
        for branch in BRANCHES:
            rows = sorted(by_case_branch.get((case_id, seq, branch), []), key=lambda r: int(r["h"]))
            if not rows:
                branch_metrics[branch] = {k: None for k in ("AbsRel", "RMSE", "delta1")}
            else:
                branch_metrics[branch] = {k: sum(float(r[k]) for r in rows) / len(rows) for k in ("AbsRel", "RMSE", "delta1")}
        d0, d1, d2 = (branch_metrics[x] for x in BRANCHES)
        matched_h = sum(1 for r in by_case_branch.get((case_id, seq, "SKIP"), []) if r.get("AbsRel") not in (None, ""))
        windows.append({"case_id": case_id, "sequence": seq, "anchor": int(case["anchor"]), "age": int(case["age"]),
            "matched_future_frames": matched_h,
            "SKIP_H4_AbsRel": d0["AbsRel"], "C_STAR_H4_AbsRel": d1["AbsRel"], "TEACHER_H4_AbsRel": d2["AbsRel"],
            "cstar_minus_skip_H4_AbsRel": (d1["AbsRel"] - d0["AbsRel"]) if d0["AbsRel"] is not None and d1["AbsRel"] is not None else None,
            "teacher_minus_skip_H4_AbsRel": (d2["AbsRel"] - d0["AbsRel"]) if d0["AbsRel"] is not None and d2["AbsRel"] is not None else None,
            "teacher_minus_cstar_H4_AbsRel": (d2["AbsRel"] - d1["AbsRel"]) if d1["AbsRel"] is not None and d2["AbsRel"] is not None else None,
            "SKIP_H4_RMSE": d0["RMSE"], "C_STAR_H4_RMSE": d1["RMSE"], "TEACHER_H4_RMSE": d2["RMSE"],
            "SKIP_H4_delta1": d0["delta1"], "C_STAR_H4_delta1": d1["delta1"], "TEACHER_H4_delta1": d2["delta1"]})
    cwrite(out_dir / "y12d_h4_windows.csv", windows)

    seq_rows = []
    for seq in SEQS:
        rows = [r for r in windows if r["sequence"] == seq and r["teacher_minus_cstar_H4_AbsRel"] is not None]
        if not rows:
            continue
        vals = [float(r["teacher_minus_cstar_H4_AbsRel"]) for r in rows]
        seq_rows.append({"sequence": seq, "n_windows": len(rows),
            "SKIP_H4_AbsRel": sum(float(r["SKIP_H4_AbsRel"]) for r in rows) / len(rows),
            "C_STAR_H4_AbsRel": sum(float(r["C_STAR_H4_AbsRel"]) for r in rows) / len(rows),
            "TEACHER_H4_AbsRel": sum(float(r["TEACHER_H4_AbsRel"]) for r in rows) / len(rows),
            "teacher_minus_cstar_H4_AbsRel": sum(vals) / len(vals),
            "teacher_better_windows": sum(v < 0 for v in vals),
            "teacher_worse_windows": sum(v > 0 for v in vals),
            "teacher_severe_harmed_windows_gt0p01": sum(v > SEVERE_HARM_THRESHOLD_ABSREL for v in vals)})
    macro = None
    if len(seq_rows) == len(SEQS):
        macro = {"sequence": "MACRO_EQUAL_VIDEO", "n_windows": sum(r["n_windows"] for r in seq_rows),
            "SKIP_H4_AbsRel": sum(r["SKIP_H4_AbsRel"] for r in seq_rows) / len(seq_rows),
            "C_STAR_H4_AbsRel": sum(r["C_STAR_H4_AbsRel"] for r in seq_rows) / len(seq_rows),
            "TEACHER_H4_AbsRel": sum(r["TEACHER_H4_AbsRel"] for r in seq_rows) / len(seq_rows),
            "teacher_minus_cstar_H4_AbsRel": sum(r["teacher_minus_cstar_H4_AbsRel"] for r in seq_rows) / len(seq_rows),
            "teacher_better_windows": sum(r["teacher_better_windows"] for r in seq_rows),
            "teacher_worse_windows": sum(r["teacher_worse_windows"] for r in seq_rows),
            "teacher_severe_harmed_windows_gt0p01": sum(r["teacher_severe_harmed_windows_gt0p01"] for r in seq_rows)}
        seq_rows.append(macro)
    cwrite(out_dir / "y12d_h4_summary.csv", seq_rows)

    # One row per anchor joins the teacher's H4 effect to observable fast-path evidence.
    case_diag_rows = []
    for case in cases:
        cid = case["case_id"]
        win = next((r for r in windows if r["case_id"] == cid), {})
        a0 = anchor_by_case_branch.get((cid, "SKIP"), {})
        a1 = anchor_by_case_branch.get((cid, "C_STAR"), {})
        a2 = anchor_by_case_branch.get((cid, "TEACHER_STATE_COMMIT"), {})
        row = {"case_id": cid, "sequence": case["sequence"], "anchor": int(case["anchor"]), "age": int(case["age"]),
               "teacher_minus_cstar_h0_AbsRel": (float(a2["AbsRel"]) - float(a1["AbsRel"])) if a1.get("AbsRel") not in (None, "") and a2.get("AbsRel") not in (None, "") else None,
               "teacher_minus_cstar_H4_AbsRel": win.get("teacher_minus_cstar_H4_AbsRel"),
               "cstar_minus_skip_H4_AbsRel": win.get("cstar_minus_skip_H4_AbsRel")}
        for region in ("normal_warp", "invalid_or_exposure_proxy", "motion_boundary_proxy"):
            rr = next((x for x in all_region if x["case_id"] == cid and x["region"] == region), None)
            row[region + "_teacher_minus_cstar_h0_AbsRel"] = rr.get("teacher_minus_cstar_AbsRel") if rr else None
            fr = [x for x in all_feature if x["case_id"] == cid and x["region"] == region and x.get("donor_minus_warped_mean_abs") not in (None, "")]
            row[region + "_donor_minus_warped_mean_abs"] = (sum(float(x["donor_minus_warped_mean_abs"]) for x in fr) / len(fr)) if fr else None
        for signal in ("flow_bwd_refined", "flow_features", "features_residual", "image_residual"):
            sr = next((x for x in all_signal if x["case_id"] == cid and x["signal"] == signal), None)
            row[signal + "_mean"] = sr.get("mean") if sr else None
            row[signal + "_std"] = sr.get("std") if sr else None
        case_diag_rows.append(row)
    if case_diag_rows:
        cwrite(out_dir / "y12d_case_diagnostics.csv", case_diag_rows)

        def average_ranks(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values); pos = 0
            while pos < len(order):
                end = pos + 1
                while end < len(order) and values[order[end]] == values[order[pos]]:
                    end += 1
                rank = (pos + 1 + end) / 2.0
                for j in range(pos, end): ranks[order[j]] = rank
                pos = end
            return ranks

        def spearman(xs, ys):
            if len(xs) < 5: return None
            rx, ry = average_ranks(xs), average_ranks(ys)
            mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
            num = sum((x-mx)*(y-my) for x,y in zip(rx,ry))
            denx = math.sqrt(sum((x-mx)**2 for x in rx)); deny = math.sqrt(sum((y-my)**2 for y in ry))
            return num / (denx * deny) if denx and deny else None

        target = "teacher_minus_cstar_H4_AbsRel"
        corr_rows = []
        candidate_features = [
            "normal_warp_donor_minus_warped_mean_abs",
            "invalid_or_exposure_proxy_donor_minus_warped_mean_abs",
            "motion_boundary_proxy_donor_minus_warped_mean_abs",
            "flow_bwd_refined_std", "flow_features_std", "features_residual_std", "image_residual_std"]
        for feature in candidate_features:
            pairs = [(finite_float(r.get(feature)), finite_float(r.get(target))) for r in case_diag_rows]
            pairs = [(x,y) for x,y in pairs if x is not None and y is not None]
            rho = spearman([x for x,_ in pairs], [y for _,y in pairs])
            corr_rows.append({"observable": feature, "target": target, "n_cases": len(pairs),
                "spearman_rho": rho, "note": "Descriptive only; 27 correlated development anchors, not causal evidence or a trained gate."})
        if corr_rows: cwrite(out_dir / "y12d_observable_correlations.csv", corr_rows)

    # H4 frame-level severe-harm count is more direct than window-mean threshold.
    severe = 0; paired = 0
    by_case_h_branch = {(r["case_id"], int(r["h"]), r["branch"]): r for r in all_future}
    for case in cases:
        for h in range(1, HORIZON + 1):
            c = by_case_h_branch.get((case["case_id"], h, "C_STAR"))
            t = by_case_h_branch.get((case["case_id"], h, "TEACHER_STATE_COMMIT"))
            if c and t and c.get("AbsRel") not in (None, "") and t.get("AbsRel") not in (None, ""):
                paired += 1
                if float(t["AbsRel"]) - float(c["AbsRel"]) > SEVERE_HARM_THRESHOLD_ABSREL:
                    severe += 1
    macro_row = next((r for r in seq_rows if r["sequence"] == "MACRO_EQUAL_VIDEO"), None)
    anchor_macro_row = next((r for r in anchor_seq_rows if r["sequence"] == "MACRO_EQUAL_VIDEO"), None)
    anchor_severe = int(anchor_macro_row.get("teacher_severe_harmed_anchors_gt0p01", 0)) if anchor_macro_row else 0
    severe_total = severe + anchor_severe
    if partial_smoke:
        decision = "SMOKE_ONLY_NOT_A_METHOD_RESULT"
        jwrite(out_dir / "y12d_gate.json", {"experiment": "Y12-D", "status": decision, "Y11_pass": False, "method_pass": False,
            "note": "Three-anchor real-CUDA smoke only; no recovery-space decision is permitted."})
    else:
        decision = classify_recovery([r for r in seq_rows if r["sequence"] in SEQS], macro_row or {}, severe_total) if macro_row else "INCONCLUSIVE_MISSING_METRICS"
        gate = {"experiment": "Y12-D", "status": decision, "Y11_pass": False, "method_pass": False,
                "teacher_severe_harmed_anchor_frames_gt0p01_vs_C_star": anchor_severe,
                "teacher_severe_harmed_future_frames_gt0p01_vs_C_star": severe,
                "teacher_severe_harmed_anchor_plus_future_frames_gt0p01_vs_C_star": severe_total,
                "paired_future_frames": paired,
                "practical_screen_floor_AbsRel": PRACTICAL_SCREEN_FLOOR_ABSREL,
                "practical_screen_floor_note": "Implementation-defined engineering floor for triage only; not a statistical significance threshold and not stated as such in the source report.",
                "sequence_macro_rows": [r for r in seq_rows if r["sequence"] in SEQS],
                "macro_equal_video": macro_row,
                "anchor_macro_equal_video": anchor_macro_row,
                "decision_limits": ["Only three previously used development videos and 27 correlated anchors.",
                    "Teacher donor is a same-ray full-frame pixel-encoder diagnostic, not GT or a theoretical upper bound.",
                    "Teacher compute is not a low-cost deployment proposal.",
                    "Flow/warp regions are proxies, not semantic or occlusion labels.",
                    "A positive teacher residual only justifies reviewing Y12-Z; it does not establish a method or Y11 pass."]}
        jwrite(out_dir / "y12d_gate.json", gate)
    summary = {"experiment": "Y12-D", "version": VERSION, "signature": signature,
        "status": "SMOKE_PASS_NOT_RESULT" if partial_smoke else "Y12D_COMPLETE_DIAGNOSTIC_NOT_METHOD_PASS", "partial_smoke": bool(partial_smoke),
        "planned_anchors": len(cases), "completed_anchors": len(all_anchor) // 3,
        "anchor_frame_records": len(all_anchor), "h4_frame_records": len(all_future),
        "region_records": len(all_region), "feature_gap_records": len(all_feature),
        "fast_signal_records": len(all_signal), "pixel_archives": sum(1 for p in (out_dir / "pixel_diagnostics").glob("*.npz")) if (out_dir / "pixel_diagnostics").exists() else 0,
        "teacher_severe_harmed_anchor_frames_gt0p01_vs_C_star": anchor_severe,
        "teacher_severe_harmed_future_frames_gt0p01_vs_C_star": severe,
        "teacher_severe_harmed_anchor_plus_future_frames_gt0p01_vs_C_star": severe_total,
        "anchor_h0_teacher_minus_cstar_AbsRel_macro_equal_video": (anchor_macro_row.get("teacher_minus_cstar_h0_AbsRel") if anchor_macro_row else None),
        "h4_teacher_minus_cstar_AbsRel_macro_equal_video": (macro_row.get("teacher_minus_cstar_H4_AbsRel") if macro_row else None),
        "gate_status": decision, "Y11_pass": False, "method_pass": False,
        "source_Y05R_signature": ctx.y05r_signature}
    jwrite(out_dir / "summary.json", summary)
    report = ["# Y12-D 1.0.2 结果摘要", "",
        "- 目的：在 Y05-R C* 自身 S8 历史上，确认当前强基线之后是否仍有较充分当前观测可恢复的误差。",
        "- 三臂：SKIP、C* 单次写回、same-ray teacher-conditioned state commit。",
        "- h0 单独报告；H4 期间关闭后续修复；K32/S8 与未来 RGB/Flow 配对。",
        "- 教师供体不是 GT 或理论上界；教师编码成本单列。",
        "- 区域为 warp-validity / flow-gradient 代理，不是语义/遮挡真值。", "",
        f"- h0 等视频宏平均 teacher−C* AbsRel：{anchor_macro_row['teacher_minus_cstar_h0_AbsRel'] if anchor_macro_row else 'NA'}（负值表示 teacher 较好）。",
        f"- H4 等视频宏平均 teacher−C* AbsRel：{macro_row['teacher_minus_cstar_H4_AbsRel'] if macro_row else 'NA'}（负值表示 teacher 较好）。",
        f"- teacher−C* 严重受损计数（anchor+H1–H4，单帧 AbsRel 增量>{SEVERE_HARM_THRESHOLD_ABSREL}）：{severe_total}。",
        f"- Y12-D 筛查状态：{decision}。Y11 PASS：False。", "",
        "## 解释边界", "",
        ("本结果仅为三锚点真实 CUDA Smoke，不得用于恢复空间判断。" if partial_smoke else
         "只有三个已用于开发的视频与 27 个相关 anchor；本结果是机制诊断，不是跨场景统计泛化、部署成本优势或 CVPR 方法验证。") ]
    (out_dir / "Y12D_RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def evaluate_cases(ctx, engine, cases, out_dir, signature, resume=False, partial_smoke=False):
    out_dir = Path(out_dir)
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    parents = {}
    for seq in SEQS:
        base = Path(ctx.y05r_dir) / "evaluation" / "own_parents" / "C_star" / seq
        meta = jread(base / "complete.json")
        parent_path = base / "parents.pt"
        require(meta.get("signature") == ctx.y05r_signature and meta.get("parent_sha256") == sha(parent_path),
                f"C* own-history parent cache identity/SHA mismatch: {seq}")
        obj = ctx.rt.torch.load(parent_path, map_location="cpu", weights_only=False)
        require(obj.get("signature") == ctx.y05r_signature and obj.get("arm") == "C_star"
                and obj.get("sequence") == seq and obj.get("head_sha256") == ctx.cstar_head_sha256,
                f"Loaded C* parent object identity mismatch: {seq}")
        parents[seq] = obj["snapshots"]
    for i, case in enumerate(cases, 1):
        result_path = case_dir / (case["case_id"] + ".json")
        if resume and result_path.is_file():
            try:
                existing = jread(result_path)
                if existing.get("signature") == signature:
                    if not existing.get("pixel_npz") or ((out_dir / existing["pixel_npz"]).is_file()
                            and sha(out_dir / existing["pixel_npz"]) == existing.get("pixel_npz_sha256")):
                        print(f"[RESUME SKIP] {case['case_id']} already complete", flush=True)
                        continue
            except Exception:
                pass
        seq = case["sequence"]
        snapshot_index = int(case["snapshot_index"])
        require(snapshot_index in parents[seq], f"Missing C* own-history parent snapshot {seq}/{snapshot_index}")
        print(f"[Y12-D {i}/{len(cases)}] {case['case_id']}", flush=True)
        result = run_case(ctx, engine, case, parents[seq][snapshot_index], out_dir, signature)
        jwrite(result_path, result)
        del result
        ctx.rt.torch.cuda.empty_cache()
    return aggregate_case_results(ctx, out_dir, cases, signature, partial_smoke=partial_smoke)
