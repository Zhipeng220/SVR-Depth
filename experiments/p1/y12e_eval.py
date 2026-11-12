"""Y12-E paired recovery-space diagnostic on C* own S8 histories."""
from y12e_common import *
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


def build_native_keyframe_donor(engine, current, rays_base):
    """Build a same-ray donor through the pinned native RadialModule.forward_keyframe path.

    The persistent Y05 tuple stores rays as BCHW (B,3,H,W), while the radial
    keyframe embedding expects flattened Bx(HW)x3. Convert with channel-last
    permutation before flattening; a direct view/reshape would scramble XYZ.
    The experiment deliberately holds the C* parent's rays fixed, then runs the
    native radial keyframe path (embed_rays -> process/to_latents -> decode with
    depth_proj and confidence_proj). This is not a top-level camera re-estimation.
    """
    torch = engine.rt.torch
    model = engine.model
    decoder = model.pixel_decoder
    radial = decoder.radial_module
    require(callable(getattr(radial, "forward_keyframe", None)),
            "Pinned decoder lacks RadialModule.forward_keyframe")
    image = normalized_image_from_frame(current)
    require(isinstance(image, torch.Tensor),
            "Y05 Engine frame['img'] is not a torch.Tensor: " + type(image).__name__)
    require(image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3,
            "Y05 Engine frame['img'] must be normalized BCHW with B=1,C=3; "
            f"got shape={tuple(image.shape)}")
    require(bool(torch.isfinite(image).all().item()),
            "Y05 Engine frame['img'] contains nonfinite values")
    require(isinstance(rays_base, torch.Tensor) and rays_base.ndim == 4
            and rays_base.shape[0] == image.shape[0] and rays_base.shape[1] == 3
            and tuple(rays_base.shape[-2:]) == tuple(image.shape[-2:]),
            "Y05 C* rays_base must be a Bx3xHxW map aligned with current frame; "
            f"got rays={tuple(rays_base.shape)}, image={tuple(image.shape)}")
    rays_hr = rays_base.permute(0, 2, 3, 1).reshape(
        rays_base.shape[0], -1, 3
    ).contiguous()

    # load_frame already returns ImageNet-normalized/padded/resized BCHW; do not normalize again.
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

        # Exact native radial keyframe path from the pinned decoder source.
        logdepth_teacher, logconf_teacher, conditioned_features = radial.forward_keyframe(
            adapted, rays_hr
        )

        # Exact geometry transforms used by Decoder.forward_keyframe.
        donor_radius = torch.exp(logdepth_teacher.clip(min=-8.0, max=8.0) + 2.0)
        donor_confidence = torch.exp(logconf_teacher.clip(min=-8.0, max=10.0))

    require(isinstance(conditioned_features, (list, tuple)) and len(conditioned_features) == 4,
            "Native keyframe path did not return four persistent conditioned feature levels")
    require(bool(torch.isfinite(logdepth_teacher).all().item())
            and bool(torch.isfinite(logconf_teacher).all().item()),
            "Native keyframe path returned nonfinite logdepth/logconfidence")
    require(bool(torch.isfinite(donor_radius).all().item()) and bool((donor_radius > 0).all().item()),
            "Native keyframe donor radius is invalid")
    require(bool(torch.isfinite(donor_confidence).all().item()) and bool((donor_confidence > 0).all().item()),
            "Native keyframe donor confidence is invalid")
    for level, feature in enumerate(conditioned_features):
        require(isinstance(feature, torch.Tensor) and bool(torch.isfinite(feature).all().item()),
                f"Native keyframe donor feature level {level} is invalid")
    return donor_radius, donor_confidence, list(conditioned_features)


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
            "Captured fast path lacks residual/flow feature evidence required by Y12-E")
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

    # D2: full current-frame pixel encoder + native radial keyframe readout,
    # conditioned on the identical fixed C* parent ray field.
    require(torch.equal(state_skip[3], parent_gpu[3]),
            f"{case['case_id']}: native propagation unexpectedly changed rays_base; same-ray donor contract is not valid")
    # Independent canary: fast-readout reproduction must still match native
    # forward_fast on the C* propagated state. It is not used for D2 geometry.
    native_readout_identity = check_native_readout_identity(engine, state_skip, case['case_id'])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    donor_radius, donor_confidence, donor_features = build_native_keyframe_donor(
        engine, frames[0], state_skip[3]
    )
    torch.cuda.synchronize(); teacher_keyframe_donor_ms = (time.perf_counter() - t0) * 1000.0
    for level, (base, donor) in enumerate(zip(state_skip[2], donor_features)):
        require(tuple(base.shape) == tuple(donor.shape),
                f"{case['case_id']}: native keyframe donor feature level {level} shape mismatch {tuple(base.shape)} vs {tuple(donor.shape)}")
        require(bool(torch.isfinite(donor).all().item()),
                f"{case['case_id']}: nonfinite native keyframe donor at level {level}")
    # The Y05 native tuple is (radius, confidence, persistent feature levels, rays).
    # Commit donor readout geometry and every feature level together; keep rays unchanged
    # because the donor was explicitly conditioned in this same ray field.
    # Y12-E controlled component interventions. Confidence and rays are fixed to C*.
    state_e0 = (state_cstar[0].detach().clone(), state_cstar[1].detach().clone(),
                [x.detach().clone() for x in state_cstar[2]], state_cstar[3].detach().clone())
    state_e1 = (state_cstar[0].detach().clone(), state_cstar[1].detach().clone(),
                [x.detach().clone() for x in donor_features], state_cstar[3].detach().clone())
    state_e2 = (donor_radius.detach().clone(), state_cstar[1].detach().clone(),
                [x.detach().clone() for x in state_cstar[2]], state_cstar[3].detach().clone())
    state_e3 = (donor_radius.detach().clone(), state_cstar[1].detach().clone(),
                [x.detach().clone() for x in donor_features], state_cstar[3].detach().clone())
    states = {
        "E0_CSTAR": state_e0,
        "E1_FEATURE_ONLY": state_e1,
        "E2_GEOMETRY_ONLY": state_e2,
        "E3_FULL_TEACHER": state_e3,
    }
    for name, state in states.items():
        require(tuple(state[0].shape) == tuple(state_cstar[0].shape), f"{name}: radius shape mismatch")
        require(tuple(state[1].shape) == tuple(state_cstar[1].shape), f"{name}: confidence shape mismatch")
        require(len(state[2]) == 4, f"{name}: expected four persistent feature levels")
        require(all(tuple(a.shape) == tuple(b.shape) for a,b in zip(state[2], state_cstar[2])),
                f"{name}: persistent feature shape mismatch")
        require(torch.equal(state[3], state_cstar[3]), f"{name}: rays changed")
        require(all(bool(torch.isfinite(x).all().item()) for x in (state[0], state[1], *state[2], state[3])),
                f"{name}: nonfinite mixed state")

    # No Teacher h0 readout is evaluated: all four displayed h0 rows are the common C* output.
    # Keep the legacy timing column explicit rather than leaving an unbound value.
    teacher_depth_conversion_ms = 0.0
    # The displayed h0 is deliberately identical for all arms: use the same C* prediction.
    common_h0 = prediction_to_numpy(np, pred_cstar)
    anchor_rows = []
    for branch in BRANCHES:
        mm = numpy_metrics(np, common_h0, frames[0])
        anchor_rows.append(_case_metric_row(case, branch, 0, frames[0], common_h0, mm))
    anchor_predictions = {branch: common_h0 for branch in BRANCHES}

    region_masks, proxy_meta = make_region_masks(engine, cap, frames[-1], frames[0])
    region_anchor_predictions = {
        "SKIP": common_h0, "C_STAR": common_h0, "TEACHER_STATE_COMMIT": common_h0
    }
    region_rows, pixel_arrays = region_error_rows(ctx, case, region_anchor_predictions, frames[0].get("gt_np"), region_masks, proxy_meta)
    feature_rows = feature_gap_rows(ctx, case, parent_gpu[2], donor_features, cap, region_masks)
    signal_rows = captured_input_rows(torch, case, cap)

    # All four future branches start from their component-intervened state; no further writeback.
    future_rows = []
    for branch, state in states.items():
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
                  "teacher_native_keyframe_donor_ms": teacher_keyframe_donor_ms,
                  "teacher_depth_conversion_ms": teacher_depth_conversion_ms,
                  "fast_forward_calls_captured": int(capture["calls"]),
                  "fast_forward_input_keys": json.dumps(capture.get("input_keys", [])),
                  "max_abs_fast_input_parent_feature_gap": max_fast_input_parent_gap,
                  "native_fast_readout_identity": native_readout_identity,
                  "teacher_radius_mean": float(donor_radius.detach().float().mean().item()),
                  "cstar_radius_mean": float(state_skip[0].detach().float().mean().item()),
                  "teacher_radius_std": float(donor_radius.detach().float().std(unbiased=False).item()),
                  "cstar_radius_std": float(state_skip[0].detach().float().std(unbiased=False).item()),
                  "teacher_confidence_mean": float(donor_confidence.detach().float().mean().item()),
                  "cstar_confidence_mean": float(state_skip[1].detach().float().mean().item()),
                  "teacher_confidence_std": float(donor_confidence.detach().float().std(unbiased=False).item()),
                  "cstar_confidence_std": float(state_skip[1].detach().float().std(unbiased=False).item()),
                  "teacher_radius_cstar_rel_l1": float((donor_radius.detach().float() - state_skip[0].detach().float()).abs().mean().div(state_skip[0].detach().float().abs().mean().clamp_min(1e-6)).item()),
                  "teacher_confidence_cstar_rel_l1": float((donor_confidence.detach().float() - state_skip[1].detach().float()).abs().mean().div(state_skip[1].detach().float().abs().mean().clamp_min(1e-6)).item()),
                  "timing_scope_note": "Per-anchor fast step and complete same-ray native radial keyframe donor (pixel encoder + input adapter + forward_keyframe); DIS/static precomputation and RGB/GT I/O are outside these intervals.",
                  "teacher_cost_note": "Full-frame current pixel encoder + native radial keyframe process/decode under fixed C* rays; diagnostic reference only, not low-cost deployment timing."}
    return {"signature": signature, "case_id": case["case_id"], "sequence": case["sequence"],
            "anchor": int(case["anchor"]), "age": int(case["age"]),
            "anchor_rows": anchor_rows, "future_rows": future_rows, "region_rows": region_rows,
            "feature_rows": feature_rows, "signal_rows": signal_rows, "timing_row": timing_row,
            "proxy_meta": proxy_meta, "pixel_npz": pixel_rel, "pixel_npz_sha256": pixel_sha,
            "parent_snapshot_index": int(case["snapshot_index"]),
            "parent_state_contract": "Y05 Engine tuple (radius, confidence, 4 persistent feature levels, BCHW rays); D2 replaces radius, confidence and all 4 feature levels from one native RadialModule.forward_keyframe call under the fixed C* rays converted to Bx(HW)x3; original BCHW rays_base is unchanged.",
            "teacher_semantics": "Same-ray native radial keyframe donor: full current-frame pixel encoder, input adapter and RadialModule.forward_keyframe (native to_latents/process/depth_proj/confidence_proj) under fixed C* rays_base after BCHW-to-B(HW)3 layout conversion; exact Decoder.forward_keyframe geometry transforms; not GT, not camera-ray re-estimation, and not a full top-level keyframe reset."}



def aggregate_case_results(ctx, out_dir, cases, signature, partial_smoke=False):
    """Y12-E component-attribution aggregation; h0 is common C* output by design."""
    out_dir = Path(out_dir)
    case_files = sorted((out_dir / "cases").glob("*.json"))
    all_anchor, all_future, all_region, all_feature, all_signal, all_timing = [], [], [], [], [], []
    for path in case_files:
        obj = jread(path)
        require(obj.get("signature") == signature, "Foreign case result signature: " + str(path))
        if obj.get("pixel_npz"):
            pixel = out_dir / obj["pixel_npz"]
            require(pixel.is_file() and sha(pixel) == obj["pixel_npz_sha256"],
                    "Pixel diagnostic archive missing/changed: " + str(pixel))
        all_anchor.extend(obj["anchor_rows"]); all_future.extend(obj["future_rows"])
        all_region.extend(obj.get("region_rows", [])); all_feature.extend(obj.get("feature_rows", []))
        all_signal.extend(obj.get("signal_rows", [])); all_timing.append(obj.get("timing_row", {}))
    completed = len({r["case_id"] for r in all_anchor})
    require(completed == len(cases), "Not all planned Y12-E anchor cases are complete")
    require(len(all_anchor) == len(cases) * len(BRANCHES), "Unexpected Y12-E h0 record count")
    cwrite(out_dir / "y12e_anchor_frames.csv", all_anchor)
    cwrite(out_dir / "y12e_h4_frames.csv", all_future)
    if all_region: cwrite(out_dir / "y12e_region_diagnostics.csv", all_region)
    if all_feature: cwrite(out_dir / "y12e_feature_gap.csv", all_feature)
    if all_signal: cwrite(out_dir / "y12e_fast_signal_stats.csv", all_signal)
    cwrite(out_dir / "y12e_timing.csv", all_timing)

    by_case_branch = {}
    for row in all_future:
        if row.get("AbsRel") not in (None, ""):
            by_case_branch.setdefault((row["case_id"], row["branch"]), []).append(row)
    windows = []
    for case in cases:
        item = {"case_id": case["case_id"], "sequence": case["sequence"],
                "anchor": int(case["anchor"]), "age": int(case["age"])}
        for branch in BRANCHES:
            rows = sorted(by_case_branch.get((case["case_id"], branch), []), key=lambda x: int(x["h"]))
            item[branch + "_matched_h"] = len(rows)
            for metric in ("AbsRel", "RMSE", "delta1"):
                item[branch + "_H4_" + metric] = (
                    sum(float(r[metric]) for r in rows) / len(rows) if rows else None)
        windows.append(item)
    cwrite(out_dir / "y12e_h4_windows.csv", windows)

    sequence_rows = []
    for seq in SEQS:
        local = [r for r in windows if r["sequence"] == seq]
        if not local: continue
        row = {"sequence": seq, "n_anchors": len(local)}
        for branch in BRANCHES:
            vals = [r[branch + "_H4_AbsRel"] for r in local if r[branch + "_H4_AbsRel"] is not None]
            row[branch + "_H4_AbsRel"] = sum(vals)/len(vals) if vals else None
            row[branch + "_better_vs_E0"] = sum(
                r[branch + "_H4_AbsRel"] < r["E0_CSTAR_H4_AbsRel"]
                for r in local if r[branch + "_H4_AbsRel"] is not None and r["E0_CSTAR_H4_AbsRel"] is not None)
        sequence_rows.append(row)
    if len(sequence_rows) == len(SEQS):
        macro = {"sequence": "MACRO_EQUAL_VIDEO", "n_anchors": sum(x["n_anchors"] for x in sequence_rows)}
        for branch in BRANCHES:
            vals = [x[branch + "_H4_AbsRel"] for x in sequence_rows if x[branch + "_H4_AbsRel"] is not None]
            macro[branch + "_H4_AbsRel"] = sum(vals)/len(vals) if vals else None
        sequence_rows.append(macro)
    cwrite(out_dir / "y12e_h4_summary.csv", sequence_rows)

    # Paired component effects are computed per anchor and then summarized by video.
    paired_rows = []
    for r in windows:
        e0 = r.get("E0_CSTAR_H4_AbsRel")
        if e0 is None: continue
        paired_rows.append({"case_id": r["case_id"], "sequence": r["sequence"],
            "F_only_minus_Cstar": (r.get("E1_FEATURE_ONLY_H4_AbsRel") - e0) if r.get("E1_FEATURE_ONLY_H4_AbsRel") is not None else None,
            "Geometry_only_minus_Cstar": (r.get("E2_GEOMETRY_ONLY_H4_AbsRel") - e0) if r.get("E2_GEOMETRY_ONLY_H4_AbsRel") is not None else None,
            "Full_minus_Cstar": (r.get("E3_FULL_TEACHER_H4_AbsRel") - e0) if r.get("E3_FULL_TEACHER_H4_AbsRel") is not None else None})
    cwrite(out_dir / "y12e_paired_effects.csv", paired_rows)

    summary = {"experiment": "Y12-E", "version": VERSION, "signature": signature,
        "status": "SMOKE_PASS_NOT_RESULT" if partial_smoke else "Y12E_COMPLETE_COMPONENT_DIAGNOSTIC_NOT_METHOD_PASS",
        "partial_smoke": bool(partial_smoke), "planned_anchors": len(cases),
        "completed_anchors": completed, "anchor_frame_records": len(all_anchor),
        "future_frame_records": len(all_future), "branches": list(BRANCHES),
        "h0_policy": "All four arms report the identical C* current-frame prediction; only committed future state differs.",
        "confidence_policy": "C* confidence is fixed in all four arms; current fast transition does not consume confidence.",
        "rays_policy": "C* rays are fixed in all four arms.",
        "sequence_h4_summary": sequence_rows, "Y11_pass": False, "method_pass": False,
        "source_Y05R_signature": ctx.y05r_signature}
    jwrite(out_dir / "summary.json", summary)
    report = ["# Y12-E 1.0.0 结果摘要", "",
        "- 目的：分离原生 same-ray donor 的几何分量与四层 persistent feature 分量对未来传播的影响。",
        "- E0：C* radius + C* confidence + C* features。",
        "- E1：C* radius + C* confidence + Teacher features。",
        "- E2：Teacher radius + C* confidence + C* features。",
        "- E3：Teacher radius + C* confidence + Teacher features。",
        "- 四臂 h0 使用同一个 C* 输出；H1-H4 才比较各自提交状态的后效应。",
        "- Confidence 与 rays 固定为 C*；不训练、不新增网络、不做门控。",
        "- 仅三个既有开发视频与 27 个相关 anchor；本实验是机制归因诊断，不是泛化或方法通过证明。", ""]
    (out_dir / "Y12E_RESULT_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    jwrite(out_dir / "y12e_gate.json", {"experiment":"Y12-E",
        "status":"SMOKE_ONLY_NOT_A_METHOD_RESULT" if partial_smoke else "COMPONENT_ATTRIBUTION_DIAGNOSTIC_ONLY",
        "Y11_pass":False,"method_pass":False,
        "decision_note":"Review per-sequence and paired component effects; no automatic method pass."})
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
        print(f"[Y12-E {i}/{len(cases)}] {case['case_id']}", flush=True)
        result = run_case(ctx, engine, case, parents[seq][snapshot_index], out_dir, signature)
        jwrite(result_path, result)
        del result
        ctx.rt.torch.cuda.empty_cache()
    return aggregate_case_results(ctx, out_dir, cases, signature, partial_smoke=partial_smoke)
