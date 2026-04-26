# Unit tests for subset_coverage_check thresholds.
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))


def _setup_progress(tmp_path, source, payload):
    os.environ["DM0_VL_ROOT"] = str(tmp_path)
    for mod in ("recipe", "subset_coverage_check"):
        if mod in sys.modules:
            del sys.modules[mod]
    import recipe
    recipe.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    (recipe.PROGRESS_DIR / f"{source}.json").write_text(json.dumps(payload))


def test_scp_pass_when_all_subsets_meet_min(tmp_path):
    _setup_progress(tmp_path, "self_collected_proxy", {
        "rows": 100,
        "image_bytes": 8 * 1024**3,
        "next_episode": 1,
        "seen_image_keys": [],
        "subset_bytes_record": {
            "uground": 600 * 1024**2,
            "embspatial": 100 * 1024**2,
            "synthdog_en": 200 * 1024**2,
            "cord_v2": 20 * 1024**2,
        },
    })
    import subset_coverage_check
    ok, fails = subset_coverage_check.check_source_coverage("self_collected_proxy")
    assert ok, f"unexpected failures: {fails}"


def test_scp_fail_when_embspatial_below_min(tmp_path):
    _setup_progress(tmp_path, "self_collected_proxy", {
        "rows": 100,
        "image_bytes": 8 * 1024**3,
        "next_episode": 1,
        "seen_image_keys": [],
        "subset_bytes_record": {
            "uground": 600 * 1024**2,
            "embspatial": 10 * 1024**2,  # below 50 MB threshold
            "synthdog_en": 200 * 1024**2,
            "cord_v2": 20 * 1024**2,
        },
    })
    import subset_coverage_check
    ok, fails = subset_coverage_check.check_source_coverage("self_collected_proxy")
    assert not ok
    assert any("embspatial" in f for f in fails)


def test_llava_pass_when_5_subsets_above_per_min(tmp_path):
    _setup_progress(tmp_path, "llava_onevision", {
        "rows": 1000,
        "image_bytes": 5 * 1024**3,
        "next_episode": 1,
        "seen_image_keys": [],
        "subset_bytes_record": {f"sub_{i}": 200 * 1024**2 for i in range(5)},
    })
    import subset_coverage_check
    ok, fails = subset_coverage_check.check_source_coverage("llava_onevision")
    assert ok, f"unexpected failures: {fails}"


def test_llava_fail_when_only_3_subsets_above_per_min(tmp_path):
    _setup_progress(tmp_path, "llava_onevision", {
        "rows": 1000,
        "image_bytes": 5 * 1024**3,
        "next_episode": 1,
        "seen_image_keys": [],
        "subset_bytes_record": {
            "sub_0": 200 * 1024**2,
            "sub_1": 200 * 1024**2,
            "sub_2": 200 * 1024**2,
            "sub_3": 50 * 1024**2,  # below per-subset min
            "sub_4": 50 * 1024**2,
        },
    })
    import subset_coverage_check
    ok, fails = subset_coverage_check.check_source_coverage("llava_onevision")
    assert not ok
    assert any("3 subsets" in f or "need 5" in f for f in fails)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
