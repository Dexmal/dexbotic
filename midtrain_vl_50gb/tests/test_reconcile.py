# Unit tests for reconcile_progress alias canonicalisation + on-disk rebuild.
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))


def test_scp_subset_alias_canonicalisation(tmp_path, monkeypatch):
    """synthdog/cord directory names should map to canonical synthdog_en/cord_v2."""
    os.environ["DM0_VL_ROOT"] = str(tmp_path)
    # Re-import so recipe picks up the new ROOT
    if "recipe" in sys.modules:
        del sys.modules["recipe"]
    if "reconcile_progress" in sys.modules:
        del sys.modules["reconcile_progress"]
    import recipe
    import reconcile_progress

    ann = recipe.ANNOTATIONS_DIR / "self_collected_proxy"
    img = recipe.IMAGES_DIR / "self_collected_proxy"
    ann.mkdir(parents=True)
    (img / "uground").mkdir(parents=True)
    (img / "embspatial").mkdir(parents=True)
    (img / "synthdog").mkdir(parents=True)
    (img / "cord").mkdir(parents=True)
    recipe.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    img_data = b"x" * 2048

    def write_record(subdir, idx):
        rel = f"self_collected_proxy/{subdir}/{idx:06d}.jpg"
        (recipe.IMAGES_DIR / rel).write_bytes(img_data)
        return {"images_1": {"type": "image", "url": rel}, "conversations": [], "is_robot": False}

    with (ann / "episode_000000.jsonl").open("w") as f:
        for i in range(3):
            f.write(json.dumps(write_record("uground", i)) + "\n")
            f.write(json.dumps(write_record("embspatial", i)) + "\n")
            f.write(json.dumps(write_record("synthdog", i)) + "\n")
            f.write(json.dumps(write_record("cord", i)) + "\n")

    new_progress = reconcile_progress.reconcile_one("self_collected_proxy")
    assert new_progress is not None
    rec = new_progress["subset_bytes_record"]

    # Aliases applied
    assert "synthdog_en" in rec
    assert "cord_v2" in rec
    assert "synthdog" not in rec
    assert "cord" not in rec
    # No alias for these
    assert "uground" in rec
    assert "embspatial" in rec
    # Each subdir wrote 3 × 2048 bytes
    assert rec["uground"] == 6144
    assert rec["synthdog_en"] == 6144

    assert new_progress["rows"] == 12
    assert new_progress["progress_version"] == 2


def test_reconcile_drops_legacy_extra_for_scp(tmp_path):
    os.environ["DM0_VL_ROOT"] = str(tmp_path)
    for mod in ("recipe", "reconcile_progress"):
        if mod in sys.modules:
            del sys.modules[mod]
    import recipe
    import reconcile_progress

    (recipe.ANNOTATIONS_DIR / "self_collected_proxy").mkdir(parents=True)
    recipe.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = recipe.PROGRESS_DIR / "self_collected_proxy.json"
    progress_path.write_text(json.dumps({
        "rows": 1234,
        "image_bytes": 5000,
        "next_episode": 0,
        "seen_image_keys": [],
        "completed_subsets": ["uground", "embspatial"],
        "subset_offsets": {"uground": 999},
        "subset_bytes_record": {"uground": 9999999},
    }))

    new_progress = reconcile_progress.reconcile_one("self_collected_proxy")
    # SCP forces reset of completed_subsets / subset_offsets / old subset_bytes_record
    assert "completed_subsets" not in new_progress
    assert "subset_offsets" not in new_progress
    # rows/bytes are recalculated (empty annotations dir → 0)
    assert new_progress["rows"] == 0
    assert new_progress["image_bytes"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
