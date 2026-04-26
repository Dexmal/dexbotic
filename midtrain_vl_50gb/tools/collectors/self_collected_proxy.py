# Self-collected multimodal: 4 subsets covering paper §3.2 (4) intent —
# embodied-spatial QA, GUI grounding, OCR, and receipt/regions.
# embspatial uses base64-encoded JPEG bytes (not file paths!) — see _collect_embspatial.
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from _collector_base import (
    CollectorReport,
    DurableEpisodeWriter,
    MAX_IMAGE_BYTES,
    MIN_IMAGE_BYTES,
    SSL_CTX,
    normalize_conversations,
)
from disk_guard import DiskFuse
try:
    from subset_coverage_check import MIN_SUBSET_BYTES as _COVERAGE_MIN
except Exception:
    _COVERAGE_MIN = {}

SOURCE_NAME = "self_collected_proxy"
HF_BASE = "https://huggingface.co/datasets"

UGROUND_REPO = "zonghanHZH/UGround-V1-8k"
EMBSPATIAL_REPO = "Phineas476/EmbSpatial-Bench"
SYNTHDOG_REPO = "naver-clova-ix/synthdog-en"
CORD_REPO = "naver-clova-ix/cord-v2"


def _save_pil(pil_image, dest_path, fmt="JPEG"):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    img = pil_image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(tmp, format=fmt, quality=92)
    size = tmp.stat().st_size
    if size < MIN_IMAGE_BYTES or size > MAX_IMAGE_BYTES:
        tmp.unlink()
        return None
    tmp.replace(dest_path)
    return size


def _hf_url(repo, path):
    return f"{HF_BASE}/{repo}/resolve/main/{urllib.request.pathname2url(path)}"


def _http_get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout, context=SSL_CTX) as resp:
        return resp.read()


def _collect_uground(images_dir, on_record, start_offset=0):
    meta_url = _hf_url(UGROUND_REPO, "metadata/hf_train.json")
    print(f"[scp/uground] fetching metadata: {meta_url}", flush=True)
    items = json.loads(_http_get(meta_url).decode("utf-8"))
    print(f"[scp/uground] {len(items)} items, start_offset={start_offset}", flush=True)
    last_idx = start_offset
    for idx, item in enumerate(items):
        if idx < start_offset:
            continue
        last_idx = idx
        elements = item.get("element") or []
        if not elements:
            continue
        element = elements[0]
        img_path = item.get("img_url")
        if not img_path:
            continue
        try:
            img_bytes = _http_get(_hf_url(UGROUND_REPO, f"images/{img_path}"))
        except Exception:
            continue
        local_rel = f"{SOURCE_NAME}/uground/{Path(img_path).name}"
        local_full = images_dir / local_rel
        local_full.parent.mkdir(parents=True, exist_ok=True)
        if local_full.exists() and local_full.stat().st_size >= MIN_IMAGE_BYTES:
            size = local_full.stat().st_size
        else:
            try:
                Image.open(io.BytesIO(img_bytes)).verify()
                pil = Image.open(io.BytesIO(img_bytes))
            except Exception:
                continue
            ext = local_full.suffix.lower()
            fmt = "PNG" if ext == ".png" else "JPEG"
            size = _save_pil(pil, local_full, fmt=fmt)
            if size is None:
                continue
        point = element.get("point", [0, 0])
        instruction = element.get("instruction", "the target element")
        conversations = [
            {"from": "human", "value": f"<image>\nLocate the GUI element: {instruction}"},
            {"from": "gpt", "value": f"The target point is approximately [{point[0]:.1f}, {point[1]:.1f}]."},
        ]
        meta = {"source": SOURCE_NAME, "repo": UGROUND_REPO, "subset": "uground", "image": img_path, "row": idx}
        if not on_record(local_rel, conversations, size, meta, idx):
            return last_idx, False
    return last_idx + 1, True


def _collect_embspatial(images_dir, on_record, start_offset=0):
    """The image field is base64-encoded JPEG bytes, not a path. Decode in-process."""
    print(f"[scp/embspatial] hf_hub_download metadata from {EMBSPATIAL_REPO}", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        meta_path = hf_hub_download(
            repo_id=EMBSPATIAL_REPO,
            filename="embspatial_bench.json",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"[scp/embspatial] hf_hub_download failed: {e}", flush=True)
        return start_offset, False
    with open(meta_path) as f:
        items = json.load(f)
    print(f"[scp/embspatial] {len(items)} items, start_offset={start_offset}", flush=True)
    last_idx = start_offset
    for idx, item in enumerate(items):
        if idx < start_offset:
            continue
        last_idx = idx
        image_b64 = item.get("image")
        if not image_b64 or not isinstance(image_b64, str):
            continue
        try:
            img_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(img_bytes))
        except Exception:
            continue
        question = item.get("question") or ""
        options = item.get("answer_options") or []
        answer = item.get("answer")
        if isinstance(answer, int) and 0 <= answer < len(options):
            answer_text = str(options[answer])
        else:
            answer_text = str(answer) if answer is not None else ""
        if not answer_text:
            continue
        option_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
        full_q = f"{question}\nOptions:\n{option_text}" if option_text else question
        conversations = [
            {"from": "human", "value": f"<image>\n{full_q}"},
            {"from": "gpt", "value": answer_text},
        ]
        local_rel = f"{SOURCE_NAME}/embspatial/embspatial_{idx:06d}.jpg"
        local_full = images_dir / local_rel
        if local_full.exists() and local_full.stat().st_size >= MIN_IMAGE_BYTES:
            size = local_full.stat().st_size
        else:
            try:
                size = _save_pil(image, local_full)
            except Exception:
                size = None
            if size is None:
                continue
        meta = {"source": SOURCE_NAME, "repo": EMBSPATIAL_REPO, "subset": "embspatial", "row": idx}
        if not on_record(local_rel, conversations, size, meta, idx):
            return last_idx, False
    return last_idx + 1, True


def _collect_synthdog(images_dir, on_record, start_offset=0, max_items=200000):
    print(f"[scp/synthdog] streaming {SYNTHDOG_REPO} from idx={start_offset}", flush=True)
    try:
        ds = load_dataset(SYNTHDOG_REPO, split="train", streaming=True)
    except Exception as e:
        print(f"[scp/synthdog] load_dataset failed: {e}", flush=True)
        return start_offset, False
    last_idx = start_offset
    for idx, item in enumerate(ds):
        if idx < start_offset:
            continue
        if idx >= max_items:
            return last_idx, True
        last_idx = idx
        image = item.get("image")
        gt = item.get("ground_truth") or item.get("text") or ""
        if image is None:
            continue
        if isinstance(gt, str) and gt.startswith("{"):
            try:
                gt_obj = json.loads(gt)
                gt_text = gt_obj.get("gt_parse", {}).get("text_sequence") or json.dumps(gt_obj, ensure_ascii=False)[:2000]
            except Exception:
                gt_text = gt[:2000]
        else:
            gt_text = str(gt)[:2000]
        if not gt_text:
            continue
        local_rel = f"{SOURCE_NAME}/synthdog/{idx:08d}.jpg"
        local_full = images_dir / local_rel
        if local_full.exists() and local_full.stat().st_size >= MIN_IMAGE_BYTES:
            size = local_full.stat().st_size
        else:
            try:
                size = _save_pil(image, local_full)
            except Exception:
                size = None
            if size is None:
                continue
        conversations = [
            {"from": "human", "value": "<image>\nRead all visible text in the image."},
            {"from": "gpt", "value": gt_text},
        ]
        meta = {"source": SOURCE_NAME, "repo": SYNTHDOG_REPO, "subset": "synthdog_en", "row": idx}
        if not on_record(local_rel, conversations, size, meta, idx):
            return last_idx, False
    return last_idx + 1, True


def _collect_cord(images_dir, on_record, start_offset=0):
    print(f"[scp/cord] streaming {CORD_REPO} from idx={start_offset}", flush=True)
    try:
        ds = load_dataset(CORD_REPO, split="train", streaming=True)
    except Exception as e:
        print(f"[scp/cord] load_dataset failed: {e}", flush=True)
        return start_offset, False
    last_idx = start_offset
    for idx, item in enumerate(ds):
        if idx < start_offset:
            continue
        last_idx = idx
        image = item.get("image")
        gt = item.get("ground_truth") or ""
        if image is None or not gt:
            continue
        try:
            gt_obj = json.loads(gt) if isinstance(gt, str) else gt
            valid_lines = gt_obj.get("valid_line", [])
            text_parts = []
            for line in valid_lines:
                for w in line.get("words", []):
                    text_parts.append(w.get("text", ""))
            text = " ".join(text_parts)[:1500]
        except Exception:
            text = ""
        if not text:
            continue
        local_rel = f"{SOURCE_NAME}/cord/{idx:08d}.jpg"
        local_full = images_dir / local_rel
        if local_full.exists() and local_full.stat().st_size >= MIN_IMAGE_BYTES:
            size = local_full.stat().st_size
        else:
            try:
                size = _save_pil(image, local_full)
            except Exception:
                size = None
            if size is None:
                continue
        conversations = [
            {"from": "human", "value": "<image>\nExtract all text from this receipt."},
            {"from": "gpt", "value": text},
        ]
        meta = {"source": SOURCE_NAME, "repo": CORD_REPO, "subset": "cord_v2", "row": idx}
        if not on_record(local_rel, conversations, size, meta, idx):
            return last_idx, False
    return last_idx + 1, True


PIPELINE = [
    ("uground", _collect_uground),
    ("embspatial", _collect_embspatial),
    ("synthdog_en", _collect_synthdog),
    ("cord_v2", _collect_cord),
]


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
    print(f"[scp] resume rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)

    completed_subsets = set(writer.get_extra("completed_subsets", []))
    subset_offsets = dict(writer.get_extra("subset_offsets", {}))
    subset_bytes_record = dict(writer.get_extra("subset_bytes_record", {}))

    report = CollectorReport(source=SOURCE_NAME)
    report.skipped["ok"] = 0
    report.skipped["subset_failed"] = 0

    images_since_check = [0]
    current_subset_name = [None]

    def coverage_done():
        cfg = _COVERAGE_MIN.get(SOURCE_NAME, {})
        for k, min_b in cfg.items():
            if k.startswith("_"):
                continue
            if subset_bytes_record.get(k, 0) < min_b:
                return False
        return True

    def on_record(local_rel, conversations, size, meta, idx):
        # coverage-aware: keep collecting even past target_bytes if a required subset is missing
        if writer.total_bytes >= target_image_bytes and coverage_done():
            return False
        row = {
            "images_1": {"type": "image", "url": local_rel},
            "conversations": normalize_conversations(conversations),
            "is_robot": False,
        }
        writer.add(row, size, meta, None)
        if current_subset_name[0]:
            subset_bytes_record[current_subset_name[0]] = subset_bytes_record.get(current_subset_name[0], 0) + size
        report.skipped["ok"] += 1
        images_since_check[0] += 1
        if images_since_check[0] >= disk_check_every:
            fuse.check()
            images_since_check[0] = 0
        if writer.should_flush():
            if current_subset_name[0] is not None:
                subset_offsets[current_subset_name[0]] = idx + 1
                writer.update_extra(
                    completed_subsets=list(completed_subsets),
                    subset_offsets=subset_offsets,
                    subset_bytes_record=subset_bytes_record,
                )
            out = writer.flush()
            if out:
                print(f"[scp] flushed {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
        return True

    for name, fn in PIPELINE:
        if writer.total_bytes >= target_image_bytes and coverage_done():
            print(f"[scp] target reached and coverage satisfied, stop", flush=True)
            break
        cfg = _COVERAGE_MIN.get(SOURCE_NAME, {})
        min_b = cfg.get(name, 0)
        actual = subset_bytes_record.get(name, 0)
        progress_v = int(writer.get_extra("progress_version", 0))
        if actual >= min_b and (
            name in completed_subsets
            or progress_v >= 2
            or writer.total_bytes >= target_image_bytes
        ):
            print(f"[scp] skip {name} (bytes={actual/1024**2:.1f}MB >= {min_b/1024**2:.1f}MB)", flush=True)
            completed_subsets.add(name)
            writer.update_extra(
                completed_subsets=list(completed_subsets),
                subset_offsets=subset_offsets,
                subset_bytes_record=subset_bytes_record,
            )
            continue
        if name in completed_subsets and actual < min_b:
            print(f"[scp] FORCE re-collect {name}: bytes={actual/1024**2:.1f}MB < min={min_b/1024**2:.1f}MB", flush=True)
            completed_subsets.discard(name)

        print(f"[scp] === subset {name} === current={writer.total_bytes/1024**3:.2f}GB", flush=True)
        rows_before = writer.total_rows
        bytes_before = writer.total_bytes
        current_subset_name[0] = name
        try:
            last_idx, exhausted = fn(images_dir, on_record, start_offset=int(subset_offsets.get(name, 0)))
        except Exception as e:
            print(f"[scp] subset {name} crashed: {e}", flush=True)
            report.skipped["subset_failed"] += 1
            continue

        subset_collected = writer.total_bytes - bytes_before
        subset_offsets[name] = last_idx
        if subset_collected > 0 and exhausted:
            completed_subsets.add(name)
        elif exhausted and writer.total_rows == rows_before:
            completed_subsets.add(name)
        writer.update_extra(
            completed_subsets=list(completed_subsets),
            subset_offsets=subset_offsets,
            subset_bytes_record=subset_bytes_record,
        )

    if writer._pending_rows:
        out = writer.flush()
        if out:
            print(f"[scp] final flush {out.name} rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB", flush=True)
    writer.final_save(
        completed_subsets=list(completed_subsets),
        subset_offsets=subset_offsets,
        subset_bytes_record=subset_bytes_record,
    )

    report.rows = writer.flushed_rows
    report.image_bytes = writer.flushed_bytes
    report.jsonl_files = writer.jsonl_files
    report.items_meta = writer._pending_meta_history
    print(f"[scp] DONE rows={writer.flushed_rows} bytes={writer.flushed_bytes/1024**3:.2f}GB "
          f"completed={len(completed_subsets)}/{len(PIPELINE)} skipped={dict(report.skipped)}", flush=True)
    return report
