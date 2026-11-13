import pytest

from svr_depth.config import configure_persistent_feature_repair


def test_config_is_copied_and_repair_enabled():
    original = {"model": {"name": "VeloDepth"}, "data": {"x": 1}}
    updated = configure_persistent_feature_repair(original, mode="pre_readout", bottleneck=16)
    assert original["model"] == {"name": "VeloDepth"}
    assert updated["model"]["persistent_feature_repair"] == {
        "enabled": True, "mode": "pre_readout", "bottleneck": 16
    }


def test_disable_removes_existing_repair_config():
    original = {"model": {"persistent_feature_repair": {"enabled": True}}}
    updated = configure_persistent_feature_repair(original, enabled=False)
    assert "persistent_feature_repair" not in updated["model"]
    assert "persistent_feature_repair" in original["model"]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        configure_persistent_feature_repair({"model": {}}, mode="unknown")
