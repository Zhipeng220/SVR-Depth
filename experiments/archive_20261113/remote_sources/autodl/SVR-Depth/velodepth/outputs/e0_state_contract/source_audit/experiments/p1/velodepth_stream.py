"""P1: VeloDepth original-policy, frame-at-a-time inference adapter.

Scope:
  * model inputs are the *packed, preprocessed* tensors produced inside
    VeloDepth.infer() immediately before encode_decode_test().
  * B=1, eval mode. It does NOT load weights, resize RGB, or compute DIS flow.
  * Preserves original auto-keyframe decisions, overlap transition and
    the important 'discard candidate propagation then keyframe SAME frame' rule.
  * No B1/B2 counterfactual branches or local refresh are implemented here.

This is a research adapter, NOT a substitute for model.infer. Always run
`test_stream_equivalence.py` against the exact checkout before experiments.
"""


from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from stream_state import StreamState
from velodepth.models.decoder_velo import flow2grid_sample
from velodepth.models.encoder_velo import rgb2luma


@dataclass
class StepOutput:
    radius: torch.Tensor
    rays: torch.Tensor
    confidence: torch.Tensor
    kind: str
    frame: int


class VeloDepthStream:
    """Model-specific stepper preserving encode_decode_test() semantics."""

    def __init__(self, model):
        self.model = model
        if model.training:
            raise ValueError("Call model.eval() before constructing VeloDepthStream")
        self.overlap = 1
        self.schedule = np.linspace(0, 1, self.overlap + 1, endpoint=True)
        self.forget = 0.5
        self.trace = []

    def _frame(self, x, t, T):
        return self.model._bt(x, 1, T)[:, t]

    def _rays_bchw(self, state, H, W):
        return rearrange(state.rays_base, "b (h w) c -> b c h w", h=H, w=W)

    @torch.no_grad()
    def keyframe_step(self, inputs, t: int, state: StreamState, T: int):
        """Run the real keyframe path once, including original overlap logic."""
        model = self.model
        H, W = inputs["image"].shape[-2:]

        # This reset happens at the start of each outer KF iteration upstream.
        state.total_luma_change = 0.0
        state.flow_magnitude = 0.0
        state.distance_kf = 1

        image = self._frame(inputs["image"], t, T)
        base_inputs = {
            "image": image,
            "image_original": image,
            "validity_mask": self._frame(inputs["validity_mask"], t, T),
        }
        if "rays" in inputs:
            base_inputs["rays"] = self._frame(inputs["rays"], t, T)
        metas = [{"T": 1, "B": 1}]

        features, tokens = model.pixel_encoder(base_inputs["image"])
        features = [
            model.stacking_fn(features[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        tokens = [
            model.stacking_fn(tokens[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_inputs["features"], base_inputs["tokens"] = features, tokens
        model.pixel_decoder.set_shapes(base_inputs, metas)
        rays_base, radius, confidence, features_teacher = (
            model.pixel_decoder.forward_keyframe(base_inputs, metas)
        )

        if not state.first_keyframe and self.overlap > 0:
            if state.fast_tail_radius is None or state.fast_tail_confidence is None:
                raise RuntimeError("Missing fast-tail state before restarting keyframe")
            radius = (self.schedule[0] * radius
                      + (1 - self.schedule[0]) * state.fast_tail_radius)
            confidence = (self.schedule[0] * confidence
                          + (1 - self.schedule[0]) * state.fast_tail_confidence)
            state.fast_tail_features = features_teacher

        state.features_teacher = features_teacher
        state.radius_teacher = radius
        state.confidence_teacher = confidence
        state.rays_base = rays_base
        state.current_t = t + 1

        return StepOutput(
            radius=state.radius_teacher,
            rays=self._rays_bchw(state, H, W),
            confidence=state.confidence_teacher,
            kind="keyframe",
            frame=t,
        ), state

    @torch.no_grad()
    def propagate_step(self, inputs, t: int, state: StreamState, T: int):
        """Return (candidate output, updated state, must_restart_keyframe).

        When restart=True, caller must NOT append candidate output, and must
        run keyframe_step on the SAME frame index t.
        """
        if t <= 0 or state.radius_teacher is None:
            raise ValueError("Propagation requires a prior keyframe")
        model = self.model
        H, W = inputs["image"].shape[-2:]
        curr = self._frame(inputs["image"], t, T)
        prev = self._frame(inputs["image"], t - 1, T)
        fast_in = {
            "grid_sample": self._frame(inputs["grid_sample"], t - 1, T),
            "image": curr,
            "image_original": curr,
            "validity_mask": self._frame(inputs["validity_mask"], t, T),
            "flow_bwd": self._frame(inputs["flow_bwd"], t - 1, T),
            "features_teacher": state.features_teacher,
        }
        fast_in["image_residual"] = (
            rgb2luma(curr).unsqueeze(1)
            - F.grid_sample(
                rgb2luma(prev).unsqueeze(1),
                fast_in["grid_sample"], mode="bilinear"
            )
        )
        if "rays" in inputs:
            fast_in["rays"] = self._frame(inputs["rays"], t, T)
        fast_metas = [{"T": 1, "B": 1}]

        flow = fast_in["flow_bwd"].clone()
        residual = fast_in["image_residual"].clone()
        feats_res, refined_flow, flow_features = model.small_encoder(
            rgb_t=curr, depth_t=state.radius_teacher,
            flow_t=flow, raft_t=None, residuals_t=residual
        )
        if len(feats_res) > 4:
            feats_res = [
                x.permute(0, 3, 1, 2)
                for i, x in enumerate(feats_res) if i in [2, 5, 8, 11]
            ]
        fast_in["features_residual"] = feats_res
        fast_in["flow_features"] = flow_features
        fast_in["flow_bwd_refined"] = refined_flow
        radius_fast, confidence_fast, feature_fast = (
            model.pixel_decoder.forward_fast(fast_in, fast_metas)
        )

        distance = state.distance_kf
        if distance - 1 < self.overlap and not state.first_keyframe:
            fast_in["features_teacher"] = state.fast_tail_features
            radius_overlap, confidence_overlap, _ = (
                model.pixel_decoder.forward_fast(fast_in, fast_metas)
            )
            radius_fast = (self.schedule[distance] * radius_fast
                           + (1 - self.schedule[distance]) * radius_overlap)
            confidence_fast = (self.schedule[distance] * state.confidence_teacher
                               + (1 - self.schedule[distance]) * confidence_overlap)

        state.features_teacher = feature_fast
        state.radius_teacher = radius_fast
        state.fast_tail_radius = radius_fast
        state.confidence_teacher = confidence_fast
        state.fast_tail_confidence = confidence_fast
        state.fast_tail_features = state.features_teacher

        state.flow_magnitude = (
            state.flow_magnitude * self.forget
            + flow.norm(dim=1).quantile(0.95).item()
        )
        flow_heur = state.flow_magnitude > (
            model.alpha_flow + model.beta_flow * (0.9 ** (distance - 1))
        )
        mask_current = (
            F.grid_sample(
                torch.ones_like(fast_in["validity_mask"]).float(),
                flow2grid_sample(refined_flow.clone(), mask=False),
                mode="nearest",
            ) < 0.5
        )
        flowmask = mask_current & fast_in["validity_mask"].bool()
        state.total_luma_change = (
            state.total_luma_change * self.forget
            + flowmask.float().mean().item()
        )
        luma_heur = state.total_luma_change > (
            model.alpha_luma + model.beta_luma * (0.9 ** (distance - 1))
        )
        restart = bool(flow_heur or luma_heur)
        result = StepOutput(
            radius=state.radius_teacher,
            rays=self._rays_bchw(state, H, W),
            confidence=state.confidence_teacher,
            kind="propagation",
            frame=t,
        )
        if restart:
            # Matches _first=False; break (without current_t += 1).
            state.first_keyframe = False
        else:
            state.distance_kf += 1
            state.current_t = t + 1
        return result, state, restart

    @torch.no_grad()
    def run(self, packed_inputs: dict, image_metas: list):
        """Return internal output dict like original encode_decode_test().

        Input must be the ORIGINAL PRE-NORMALIZATION packed payload captured
        at the entry to encode_decode_test(), with B=1.
        """
        B, T = image_metas[0]["B"], image_metas[0]["T"]
        if B != 1:
            raise NotImplementedError("P1 adapter currently supports B=1 only")
        if T < 1:
            raise ValueError("At least one frame is required")
        model = self.model
        inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in packed_inputs.items()}
        model._set_image_paddings(inputs)
        inputs["flow_bwd"], inputs["grid_sample"] = (
            model._normalize_and_grid_from_flow(
                inputs["flow_bwd"], inputs["image"],
                inputs["image_paddings"], inputs["depth_paddings"]
            )
        )
        old_camera_flag = model.pixel_decoder.test_gt_camera
        if "rays" in inputs:
            model.pixel_decoder.test_gt_camera = True

        self.trace = []
        state = StreamState()
        outs = []
        try:
            for t in range(T):
                if t == 0:
                    result, state = self.keyframe_step(inputs, t, state, T)
                    self.trace.append({"frame": t, "action": "keyframe"})
                else:
                    result, state, restart = self.propagate_step(inputs, t, state, T)
                    if restart:
                        self.trace.append({"frame": t, "action": "propagation_rejected"})
                        result, state = self.keyframe_step(inputs, t, state, T)
                        self.trace.append({"frame": t, "action": "keyframe"})
                    else:
                        self.trace.append({"frame": t, "action": "propagation"})
                outs.append(result)

            radius = torch.cat([x.radius for x in outs], dim=0)
            rays = torch.cat([x.rays for x in outs], dim=0)
            confidence = torch.cat([x.confidence for x in outs], dim=0)
            return {
                "radius": radius,
                "rays": rays,
                "confidence": confidence,
                "points": radius * rays,
            }
        finally:
            # Source encode_decode_test() always restores to False at return.
            # Preserving the previous flag is safer for an experimental adapter.
            model.pixel_decoder.test_gt_camera = old_camera_flag
