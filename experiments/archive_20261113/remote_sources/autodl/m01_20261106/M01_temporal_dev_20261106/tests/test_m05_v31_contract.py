from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/p1/cvpr_m05_formal.py"


def _source():
    return RUNNER.read_text(encoding="utf-8")


def test_runner_parses():
    ast.parse(_source())


def test_pnative_replay_uses_native_repair_mode():
    src = _source()
    assert 'replay_mode = "pre_readout" if branch == "P_native" else "off"' in src
    assert 'set_mode(model, replay_mode)' in src
    assert '"repair_mode_after_intervention": replay_mode' in src


def test_core_branch_contract_retained():
    src = _source()
    assert 'CORE_BRANCHES = ("A", "O", "F", "T", "P_native")' in src
    assert 'DIAGNOSTIC_BRANCHES = ("R_only", "C_only")' in src


def test_protocol_marks_unimplemented_validation_gaps():
    import json
    cfg = json.loads((ROOT / "configs/eval/m05_v31_protocol.json").read_text(encoding="utf-8"))
    assert cfg["protocol"] == "M05-V3.1-dev"
    assert any("boundary" in x for x in cfg["known_gaps"])
