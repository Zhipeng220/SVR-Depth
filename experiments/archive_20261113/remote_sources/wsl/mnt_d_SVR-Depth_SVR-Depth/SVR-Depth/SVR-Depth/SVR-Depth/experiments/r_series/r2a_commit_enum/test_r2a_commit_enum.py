"""CPU tests for diagnostic comparisons and the state-machine integration."""
import copy
import math
import types
import unittest
from dataclasses import dataclass

import r2a_commit_enum as r


@dataclass
class FakeState:
    features_teacher: list = None
    radius_teacher: float = 0.0
    confidence_teacher: float = 1.0
    rays_base: float = 1.0
    fast_tail_features: list = None
    fast_tail_radius: float = 0.0
    fast_tail_confidence: float = 1.0
    current_t: int = 0
    distance_kf: int = 1
    flow_magnitude: float = 0.0
    total_luma_change: float = 0.0
    first_keyframe: bool = True


class FakeStream:
    def __init__(self, experiment, restarts=()):
        self.experiment = experiment
        self.restarts = set(restarts)

    def keyframe_step(self, inputs, t, state, total):
        self.experiment.runtime = 100 + t
        state.features_teacher = [20.0 + t]
        state.radius_teacher = 10.0 - 0.5*t
        state.rays_base = 2.0
        state.distance_kf = 1
        state.current_t = t + 1
        return types.SimpleNamespace(radius=state.radius_teacher), state

    def propagate_step(self, inputs, t, state, total):
        self.experiment.runtime += 1
        state.radius_teacher += 1
        state.features_teacher = [state.radius_teacher]
        state.fast_tail_radius = state.radius_teacher
        state.fast_tail_features = state.features_teacher
        state.distance_kf += 1
        state.current_t = t + 1
        return types.SimpleNamespace(radius=state.radius_teacher), state, t in self.restarts


class StateMachineTests(unittest.TestCase):
    def experiment(self, restarts=()):
        e = r.GPUExperiment.__new__(r.GPUExperiment)
        e.T, e.inputs, e.runtime = 10, {}, 0
        e.h = types.SimpleNamespace(clone_payload=copy.deepcopy)
        e.torch = types.SimpleNamespace(Tensor=type("UnusedTensor", (), {}))
        e.timed = lambda fn: (fn(), 0.0)
        e.capture = lambda state: (copy.deepcopy(state), e.runtime)
        def restore(snapshot):
            e.runtime = snapshot[1]
            return copy.deepcopy(snapshot[0])
        e.restore = restore
        e.stream = FakeStream(e, restarts)
        return e

    def test_candidate_runtime_does_not_leak_into_discard_state(self):
        e = self.experiment()
        st = FakeState(features_teacher=[2.0], radius_teacher=2.0, distance_kf=7)
        out, prop, event, candidate = e.prepare_step(st, 3, True)
        self.assertEqual(e.runtime, 1)  # propagation runtime restored, not candidate runtime 103
        self.assertEqual(prop.radius_teacher, 3.0)
        self.assertEqual(out.radius, 8.5)
        self.assertEqual(prop.distance_kf, 8)
        e.commit(prop, candidate)
        self.assertEqual(prop.radius_teacher, 8.5)
        self.assertEqual(prop.fast_tail_radius, 3.0)
        self.assertEqual(prop.distance_kf, 8)
        self.assertEqual(event["optional_candidate_computed"], 1)

    def test_native_restart_takes_priority_and_does_not_spend_optional_candidate(self):
        e = self.experiment(restarts=(3,))
        st = FakeState(features_teacher=[2.0], radius_teacher=2.0)
        out, state, event, candidate = e.prepare_step(st, 3, True)
        self.assertIsNone(candidate)
        self.assertEqual(event["native_restart"], 1)
        self.assertEqual(event["scheduled_native_collision"], 1)
        self.assertEqual(event["optional_candidate_computed"], 0)
        self.assertEqual(state.distance_kf, 1)

    def test_discard_and_geom_commit_preserve_keyframe_age(self):
        c = r.Clocks()
        c.advance(0, native=True)
        c.advance(1)
        c.advance(2, commit=True)
        self.assertEqual((c.steps_since_kf_reset, c.steps_since_geom_commit), (2, 0))
        c.advance(3)
        self.assertEqual((c.steps_since_kf_reset, c.steps_since_geom_commit), (3, 1))
        c.advance(4, native=True)
        self.assertEqual((c.steps_since_kf_reset, c.steps_since_geom_commit), (0, 0))

    def test_eight_plans_recurse_through_their_selected_state(self):
        import itertools
        e = self.experiment()
        e.seed = lambda _: None
        e.sync = lambda: None
        e.args = types.SimpleNamespace(seed=0)
        e.base = e.capture(FakeState())
        e.model = object()
        e.records = [{"gt_available": True} for _ in range(e.T)]
        e.gt = [1] * e.T
        e.paddings = e.padded_shape = ()
        e.cf = types.SimpleNamespace(eval_output=lambda out, *_: {
            "absrel": out.radius/100, "rmse": out.radius/10, "delta1": 1-out.radius/100})
        plans = {}
        for bits in itertools.product("01", repeat=3):
            p = "".join(bits)
            plans[p], _ = e.run_plan(p, [2, 4, 6])
        summaries, diagnostic, pairs = r.summarize(plans, [2, 4, 6])
        self.assertEqual(len(summaries), 8)
        self.assertEqual(len(pairs), 12)
        self.assertTrue(diagnostic["all_plans_have_K_optional_candidates"])
        self.assertGreater(plans["000"][-1]["absrel"], plans["111"][-1]["absrel"])
        self.assertEqual(plans["111"][-1]["steps_since_kf_reset"], 9)
        self.assertEqual(plans["111"][-1]["steps_since_geom_commit"], 3)


class DiagnosticTests(unittest.TestCase):
    def rows(self, plan, error):
        return [{"absrel": x, "rmse": x, "delta1": 1-x,
                 "optional_candidate_computed": int(t in (1, 2, 3)),
                 "geom_commit": int(1 <= t <= len(plan) and plan[t-1] == "1"),
                 "native_restart": 0, "scheduled_native_collision": 0,
                 "propagation_ms": 0, "keyframe_ms": 0, "snapshot_restore_commit_ms": 0}
                for t, x in enumerate(error)]

    def test_mixed_plan_full_sequence_oracle_can_beat_both_endpoints(self):
        import itertools
        data = {}
        for bits in itertools.product("01", repeat=3):
            plan = "".join(bits)
            value = 0.1 if plan == "010" else 0.3
            data[plan] = self.rows(plan, [0.2, 0.2, 0.2, 0.2, value])
        summaries, diagnostic, pairs = r.summarize(data, [1, 2, 3])
        self.assertEqual(diagnostic["best_plan_by_full_clip_absrel"], "010")
        self.assertAlmostEqual(diagnostic["gain_over_better_unconditional_absrel"], 0.04)
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(x["same_future_planned_bits"] for x in pairs))

    def test_missing_frames_keep_original_discount_exponents(self):
        a = self.rows("0", [0.2, 0.2, 0.4, float("nan"), 0.4])
        b = self.rows("1", [0.2, 0.2, 0.3, float("nan"), 0.3])
        _, _, pairs = r.summarize({"0": a, "1": b}, [1], gamma=0.9, horizon=3)
        self.assertAlmostEqual(pairs[0]["discounted_future_absrel_gain_sum"], 0.9*0.1+0.9**3*0.1)
        self.assertEqual(pairs[0]["future_pairs_valid"], 2)

    def test_unequal_valid_frames_are_rejected(self):
        a = self.rows("0", [0.2, 0.2, float("nan"), 0.2])
        b = self.rows("1", [0.2, 0.2, 0.1, 0.2])
        with self.assertRaisesRegex(RuntimeError, "Different valid metric"):
            r.summarize({"0": a, "1": b}, [1])


if __name__ == "__main__":
    unittest.main()
