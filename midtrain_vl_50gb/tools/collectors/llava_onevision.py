# LLaVA-OneVision-Data multi-subset collector with dynamic per-subset quota.
# Paper §3.2 (3) — broad multimodal understanding.
import sys
from pathlib import Path

from datasets import load_dataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from _collector_base import (
    CollectorReport,
    DurableEpisodeWriter,
    MAX_IMAGE_BYTES,
    MIN_IMAGE_BYTES,
    normalize_conversations,
)
from disk_guard import DiskFuse
from recipe import RECIPE
try:
    from subset_coverage_check import MIN_SUBSET_BYTES as _COVERAGE_MIN
except Exception:
    _COVERAGE_MIN = {}

REPO = "lmms-lab/LLaVA-OneVision-Data"
SOURCE_NAME = "llava_onevision"
SUBSETS = RECIPE[SOURCE_NAME]["subsets"]
MIN_PER_SUBSET_BYTES = 64 * 1024 * 1024


def _safe_subset_dirname(name):
    return name.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "_")


def _save_image(pil_image, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    img = pil_image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(tmp, format="JPEG", quality=92)
    size = tmp.stat().st_size
    if size < MIN_IMAGE_BYTES or size > MAX_IMAGE_BYTES:
        tmp.unlink()
        return None
    tmp.replace(dest_path)
    return size


def collect_at_scale(
    target_image_bytes,
    annotations_dir,
    images_dir,
    raw_dir,
    progress_path,
    episode_batch_size=10000,
    disk_check_every=200,
    disk_floor_gb=100,
    seen_image_keys=None,
):
    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    progress_path = Path(progress_path)
    (images_dir / SOURCE_NAME).mkdir(parents=True, exist_ok=True)

    fuse = DiskFuse(annotations_dir, disk_floor_gb)
    writer = DurableEpisodeWriter(annotations_dir, progress_path, SOURCE_NAME, episode_batch_size)
    print(f"[llava_ov] resume rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    completed_subsets = set(writer.get_extra("completed_subsets", []))
    subset_offsets = dict(writer.get_extra("subset_offsets", {}))
    subset_bytes_record = dict(writer.get_extra("subset_bytes_record", {}))

    def coverage_done():
        cfg = _COVERAGE_MIN.get(SOURCE_NAME, {})
        min_count = cfg.get("_min_subset_count", 0)
        min_per = cfg.get("_min_per_subset_bytes", 0)
        if min_count <= 0 or min_per <= 0:
            return True
        valid = sum(1 for s, b in subset_bytes_record.items() if not s.startswith("_") and b >= min_per)
        return valid >= min_count

    report = CollectorReport(source=SOURCE_NAME)
    for k in ("no_image", "save_failed", "no_conversations", "ok", "subset_failed"):
        report.skipped.setdefault(k, 0)

    images_since_check = 0

    for subset in SUBSETS:
        if writer.total_bytes >= target_image_bytes and coverage_done():
            print(f"[llava_ov] target reached and coverage satisfied, stop", flush=True)
            break

        # Skip-by-actual: if reconcile already counts enough bytes for this subset,
        # don't re-stream. Avoids duplicate jsonl rows on hybrid resume.
        subset_id = _safe_subset_dirname(subset)
        actual_pre = subset_bytes_record.get(subset_id, 0)
        progress_v = int(writer.get_extra("progress_version", 0))
        cfg_min = _COVERAGE_MIN.get(SOURCE_NAME, {})
        skip_threshold = cfg_min.get("_min_per_subset_bytes", 100 * 1024 * 1024)
        if subset in completed_subsets:
            print(f"[llava_ov] skip completed: {subset}", flush=True)
            continue
        if actual_pre >= skip_threshold and (progress_v >= 2 or writer.total_bytes >= target_image_bytes):
            print(f"[llava_ov] skip {subset} (bytes={actual_pre/1024**2:.1f}MB >= {skip_threshold/1024**2:.0f}MB)", flush=True)
            completed_subsets.add(subset)
            writer.update_extra(
                completed_subsets=list(completed_subsets),
                subset_offsets=subset_offsets,
                subset_bytes_record=subset_bytes_record,
            )
            continue

        open_subsets = [s for s in SUBSETS if s not in completed_subsets]
        remaining = max(0, target_image_bytes - writer.total_bytes)
        per_subset_target = max(remaining // max(1, len(open_subsets)), MIN_PER_SUBSET_BYTES)

        (images_dir / SOURCE_NAME / subset_id).mkdir(parents=True, exist_ok=True)
        subset_start_bytes = writer.total_bytes
        subset_skip_offset = int(subset_offsets.get(subset, 0))
        print(f"[llava_ov] subset={subset} skip_offset={subset_skip_offset} per_subset={per_subset_target/1024**3:.2f}GB", flush=True)

        try:
            ds = load_dataset(REPO, subset, split="train", streaming=True)
        except Exception as e:
            print(f"[llava_ov] load_dataset failed for {subset}: {e}", flush=True)
            report.skipped["subset_failed"] += 1
            continue

        idx = 0
        last_committed_idx = subset_skip_offset
        subset_exhausted = True
        subset_failed = False
        try:
            for item in ds:
                idx += 1
                if idx <= subset_skip_offset:
                    continue
                if writer.total_bytes - subset_start_bytes >= per_subset_target:
                    subset_exhausted = False
                    break
                if writer.total_bytes >= target_image_bytes:
                    subset_exhausted = False
                    break
                image = item.get("image")
                conversations = item.get("conversations") or []
                if image is None:
                    report.skipped["no_image"] += 1
                    continue
                if not conversations:
                    report.skipped["no_conversations"] += 1
                    continue
                local_rel = f"{SOURCE_NAME}/{subset_id}/{idx:08d}.jpg"
                local_path = images_dir / local_rel
                if local_path.exists() and local_path.stat().st_size >= MIN_IMAGE_BYTES:
                    size = local_path.stat().st_size
                else:
                    try:
                        size = _save_image(image, local_path)
                    except Exception:
                        size = None
                    if size is None:
                        report.skipped["save_failed"] += 1
                        continue
                row = {
                    "images_1": {"type": "image", "url": local_rel},
                    "conversations": normalize_conversations(conversations),
                    "is_robot": False,
                }
                meta = {"source": SOURCE_NAME, "repo": REPO, "subset": subset, "row": idx}
                writer.add(row, size, meta, None)
                subset_bytes_record[subset_id] = subset_bytes_record.get(subset_id, 0) + size
                last_committed_idx = idx
                report.skipped["ok"] += 1
                images_since_check += 1
                if images_since_check >= disk_check_every:
                    fuse.check()
                    images_since_check = 0
                if writer.should_flush():
                    subset_offsets[subset] = last_committed_idx
                    writer.update_extra(
                        completed_subsets=list(completed_subsets),
                        subset_offsets=subset_offsets,
                        subset_bytes_record=subset_bytes_record,
                    )
                    out = writer.flush()
                    if out:
                        print(f"[llava_ov] flushed {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
        except Exception as e:
            print(f"[llava_ov] subset {subset} stream error: {e}", flush=True)
            subset_failed = True
            subset_exhausted = False

        subset_offsets[subset] = last_committed_idx
        if not subset_failed and subset_exhausted:
            completed_subsets.add(subset)
        writer.update_extra(
            completed_subsets=list(completed_subsets),
            subset_offsets=subset_offsets,
            subset_bytes_record=subset_bytes_record,
        )

    if writer._pending_rows:
        out = writer.flush()
        if out:
            print(f"[llava_ov] final flush {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
    writer.final_save(
        completed_subsets=list(completed_subsets),
        subset_offsets=subset_offsets,
        subset_bytes_record=subset_bytes_record,
    )

    report.rows = writer.flushed_rows
    report.image_bytes = writer.flushed_bytes
    report.jsonl_files = writer.jsonl_files
    report.items_meta = writer._pending_meta_history
    print(f"[llava_ov] DONE rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB "
          f"completed={len(completed_subsets)}/{len(SUBSETS)} skipped={dict(report.skipped)}", flush=True)
    return report
