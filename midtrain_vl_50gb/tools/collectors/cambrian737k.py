# Cambrian-737k: full COCO download with shuffle for diversity.
# Paper §3.2 (1) — high-quality multi-turn VQA on COCO images.
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from _collector_base import (
    CollectorReport,
    DurableEpisodeWriter,
    SSL_CTX,
    download_coco_record,
    normalize_conversations,
)
from disk_guard import DiskFuse

REPO = "LanguageBind/Cambrian737k"
JSON_URL = (
    "https://huggingface.co/datasets/LanguageBind/Cambrian737k/"
    "resolve/main/Cambrian737k/Cambrian737k.json"
)
SOURCE_NAME = "cambrian737k"
META_TIMEOUT = 1200
META_MIN_BYTES = 1_000_000_000
SHUFFLE_SEED = 20260425
DOWNLOAD_BATCH = 256


def _download_metadata(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "Cambrian737k.json"
    if dest.exists() and dest.stat().st_size >= META_MIN_BYTES:
        return dest
    tmp = dest.with_suffix(".json.tmp")
    print(f"[cambrian737k] downloading meta JSON (~1.08GB) to {dest}", flush=True)
    with urllib.request.urlopen(JSON_URL, timeout=META_TIMEOUT, context=SSL_CTX) as resp, tmp.open("wb") as fh:
        chunk = 1024 * 1024
        total = 0
        last_log = time.time()
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            total += len(buf)
            if time.time() - last_log > 5.0:
                print(f"[cambrian737k]   meta {total/1024**3:.2f}GB", flush=True)
                last_log = time.time()
    tmp.replace(dest)
    return dest


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
    raw_dir = Path(raw_dir)
    (images_dir / SOURCE_NAME).mkdir(parents=True, exist_ok=True)

    fuse = DiskFuse(annotations_dir, disk_floor_gb)
    writer = DurableEpisodeWriter(annotations_dir, progress_path, SOURCE_NAME, episode_batch_size)
    if seen_image_keys:
        for k in seen_image_keys:
            writer.seen_keys.add(k)
    print(f"[cambrian737k] resume rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    json_path = _download_metadata(raw_dir / SOURCE_NAME)
    print(f"[cambrian737k] loading meta JSON ({json_path.stat().st_size/1024**3:.2f}GB)", flush=True)
    t0 = time.time()
    with json_path.open() as f:
        records = json.load(f)
    print(f"[cambrian737k] {len(records)} candidates, load took {time.time()-t0:.1f}s", flush=True)

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(records)

    report = CollectorReport(source=SOURCE_NAME)
    for k in ("invalid_path", "http_failed", "too_small", "too_large", "pil_error",
              "already_present", "no_conversations", "ok_cached", "ok"):
        report.skipped.setdefault(k, 0)

    images_since_check = 0
    print(f"[cambrian737k] target={target_image_bytes/1024**3:.2f}GB current={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch_start in range(0, len(records), DOWNLOAD_BATCH):
            if writer.total_bytes >= target_image_bytes:
                break
            batch = records[batch_start:batch_start + DOWNLOAD_BATCH]
            futures = {pool.submit(download_coco_record, r, images_dir, SOURCE_NAME, writer.seen_keys): r for r in batch}
            for fut in as_completed(futures):
                if writer.total_bytes >= target_image_bytes:
                    break
                try:
                    rec_result, status = fut.result()
                except Exception:
                    report.skipped["http_failed"] += 1
                    continue
                report.skipped[status] = report.skipped.get(status, 0) + 1
                if status not in ("ok", "ok_cached"):
                    continue
                rec, local_rel, size, image_path = rec_result
                conversations = rec.get("conversations") or []
                if not conversations:
                    report.skipped["no_conversations"] += 1
                    continue
                row = {
                    "images_1": {"type": "image", "url": local_rel},
                    "conversations": normalize_conversations(conversations),
                    "is_robot": False,
                }
                meta = {"source": SOURCE_NAME, "repo": REPO, "id": rec.get("id"), "image": image_path}
                writer.add(row, size, meta, Path(image_path).name)
                images_since_check += 1
                if images_since_check >= disk_check_every:
                    fuse.check()
                    images_since_check = 0
                if writer.should_flush():
                    out = writer.flush()
                    if out:
                        print(f"[cambrian737k] flushed {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    if writer._pending_rows:
        out = writer.flush()
        if out:
            print(f"[cambrian737k] final flush {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
    writer.final_save()

    report.rows = writer.flushed_rows
    report.image_bytes = writer.flushed_bytes
    report.jsonl_files = writer.jsonl_files
    report.items_meta = writer._pending_meta_history
    print(f"[cambrian737k] DONE rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB skipped={dict(report.skipped)}", flush=True)
    return report
