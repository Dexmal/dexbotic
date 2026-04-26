# Unit tests for DurableEpisodeWriter resume / next_episode / pending-seen guard.
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(ROOT))
from _collector_base import DurableEpisodeWriter, coco_url, normalize_conversations


def _make_row(i):
    return {
        "images_1": {"type": "image", "url": f"src/{i:06d}.jpg"},
        "conversations": [{"from": "human", "value": "<image>\nQ"}, {"from": "gpt", "value": "A"}],
        "is_robot": False,
    }


def test_flush_writes_atomic_episode(tmp_path):
    writer = DurableEpisodeWriter(tmp_path / "ann", tmp_path / ".prog/src.json", "src", episode_batch_size=3)
    for i in range(3):
        writer.add(_make_row(i), 1024, {"id": i}, f"img_{i}.jpg")
    out = writer.flush()
    assert out.name == "episode_000000.jsonl"
    assert writer.flushed_rows == 3
    assert writer.flushed_bytes == 3072
    progress = json.loads((tmp_path / ".prog/src.json").read_text())
    assert progress["rows"] == 3
    assert progress["next_episode"] == 1


def test_resume_picks_up_flushed_state(tmp_path):
    w1 = DurableEpisodeWriter(tmp_path / "ann", tmp_path / ".prog/src.json", "src", episode_batch_size=2)
    for i in range(2):
        w1.add(_make_row(i), 100, {"id": i}, f"k_{i}")
    w1.flush()
    w1.final_save()

    w2 = DurableEpisodeWriter(tmp_path / "ann", tmp_path / ".prog/src.json", "src", episode_batch_size=2)
    assert w2.flushed_rows == 2
    assert w2.flushed_bytes == 200
    assert "k_0" in w2.seen_keys


def test_pending_seen_keys_dedup(tmp_path):
    w = DurableEpisodeWriter(tmp_path / "ann", tmp_path / ".prog/src.json", "src", episode_batch_size=10)
    w.add(_make_row(0), 100, {}, "img_a")
    w.add(_make_row(1), 100, {}, "img_a")  # duplicate within pending — should be dropped
    assert len(w._pending_rows) == 1


def test_next_episode_max_existing(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "episode_000000.jsonl").write_text("{}\n")
    (ann / "episode_000005.jsonl").write_text("{}\n")
    w = DurableEpisodeWriter(ann, tmp_path / ".prog/src.json", "src", episode_batch_size=10)
    assert w._next_episode == 6


def test_normalize_conversations_injects_image_token():
    result = normalize_conversations([
        {"from": "human", "value": "What is in the image?"},
        {"from": "gpt", "value": "A bus."},
    ])
    assert "<image>" in result[0]["value"]


def test_coco_url_parses_train_val():
    assert coco_url("coco/train2017/000000033471.jpg") == "https://images.cocodataset.org/train2017/000000033471.jpg"
    assert coco_url("coco/val2017/000000000139.jpg") == "https://images.cocodataset.org/val2017/000000000139.jpg"
    assert coco_url("coco/train2014/COCO_train2014_000000033471.jpg") == "https://images.cocodataset.org/train2014/000000033471.jpg"
    assert coco_url("non/coco/path.jpg") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
