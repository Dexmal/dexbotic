# Cambrian-10M filtered: streaming JSONL with paper §3.2 quality filters.
# Removes math-heavy, non-English, writing-centric content; dedupes COCO images
# already pulled by cambrian737k.
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from _collector_base import (
    CollectorReport,
    DurableEpisodeWriter,
    coco_url,
    download_coco_record,
    normalize_conversations,
    stream_jsonl,
)
from disk_guard import DiskFuse

REPO = "nyu-visionx/Cambrian-10M"
JSONL_URL = (
    "https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/"
    "resolve/main/jsons/Cambrian10M.jsonl"
)
SOURCE_NAME = "cambrian10m_filtered"
DOWNLOAD_BATCH = 256

MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 6000
MIN_ASCII_RATIO = 0.7

BANNED_KEYWORDS = [
    "solve", "equation", "proof", "calculate", "derive", "integrate", "differentiate",
    "write an essay", "write a poem", "poem", "sonnet", "haiku", "novel",
    "translate to", "translate from", "translate the",
    "latex", "formula", "theorem", "lemma", "matrix", "polynomial",
    "中文", "日本語", "한국어", "français", "español", "deutsch", "русский",
    "代码", "算法", "函数",
]


def _filter_keep(obj):
    if not obj.get("image"):
        return False, "no_image"
    if not coco_url(str(obj["image"])):
        return False, "non_coco"
    convs = obj.get("conversations") or []
    if not convs:
        return False, "no_conversations"
    text = json.dumps(convs, ensure_ascii=False).lower()
    n = len(text)
    if n < MIN_TEXT_LEN:
        return False, "too_short"
    if n > MAX_TEXT_LEN:
        return False, "too_long"
    ascii_count = sum(1 for c in text if ord(c) < 128)
    if ascii_count / n < MIN_ASCII_RATIO:
        return False, "low_ascii"
    for w in BANNED_KEYWORDS:
        if w in text:
            return False, "banned_keyword"
    return True, "ok"


def _load_cambrian737k_seen(progress_dir):
    p = Path(progress_dir) / "cambrian737k.json"
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).get("seen_image_keys", []))
    except Exception:
        return set()


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
    if seen_image_keys:
        for k in seen_image_keys:
            writer.seen_keys.add(k)
    cambrian737k_seen = _load_cambrian737k_seen(progress_path.parent)
    for k in cambrian737k_seen:
        writer.seen_keys.add(k)
    print(f"[cambrian10m] resume rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB "
          f"dedupe seen={len(writer.seen_keys)} keys (incl {len(cambrian737k_seen)} from cambrian737k)", flush=True)

    jsonl_offset = int(writer.get_extra("jsonl_offset", 0))

    report = CollectorReport(source=SOURCE_NAME)
    for k in ("no_image", "non_coco", "no_conversations", "too_short", "too_long",
              "low_ascii", "banned_keyword", "invalid_path", "http_failed",
              "too_small", "too_large", "pil_error", "already_present",
              "ok_cached", "ok"):
        report.skipped.setdefault(k, 0)

    pending_candidates = []
    images_since_check = 0
    last_log = 0.0
    line_count = 0
    last_offset_save = jsonl_offset

    print(f"[cambrian10m] streaming {JSONL_URL} from offset={jsonl_offset}", flush=True)

    def _drain(candidates):
        nonlocal images_since_check
        if not candidates:
            return
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(download_coco_record, r, images_dir, SOURCE_NAME, writer.seen_keys): r for r in candidates}
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
                row = {
                    "images_1": {"type": "image", "url": local_rel},
                    "conversations": normalize_conversations(conversations),
                    "is_robot": False,
                }
                meta = {
                    "source": SOURCE_NAME, "repo": REPO,
                    "id": rec.get("id"), "image": image_path,
                    "filter": "local_filter_v2", "dedupe_against": "cambrian737k",
                }
                writer.add(row, size, meta, Path(image_path).name)
                images_since_check += 1
                if images_since_check >= disk_check_every:
                    fuse.check()
                    images_since_check = 0
                if writer.should_flush():
                    out = writer.flush()
                    if out:
                        print(f"[cambrian10m] flushed {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    try:
        for obj, byte_offset in stream_jsonl(JSONL_URL, start_offset=jsonl_offset):
            line_count += 1
            if writer.total_bytes >= target_image_bytes:
                break
            keep, reason = _filter_keep(obj)
            if not keep:
                report.skipped[reason] = report.skipped.get(reason, 0) + 1
                continue
            pending_candidates.append(obj)
            if len(pending_candidates) >= DOWNLOAD_BATCH:
                _drain(pending_candidates)
                pending_candidates = []
                writer.update_extra(jsonl_offset=byte_offset)
                last_offset_save = byte_offset
            if time.time() - last_log > 30:
                print(f"[cambrian10m] scanned={line_count} pending={len(pending_candidates)} "
                      f"rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
                last_log = time.time()
    except Exception as e:
        print(f"[cambrian10m] stream error: {e}, will rely on resume", flush=True)

    if pending_candidates:
        _drain(pending_candidates)
    if writer._pending_rows:
        out = writer.flush()
        if out:
            print(f"[cambrian10m] final flush {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
    writer.final_save(jsonl_offset=last_offset_save)

    report.rows = writer.flushed_rows
    report.image_bytes = writer.flushed_bytes
    report.jsonl_files = writer.jsonl_files
    report.items_meta = writer._pending_meta_history
    print(f"[cambrian10m] DONE rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB skipped={dict(report.skipped)}", flush=True)
    return report
