# Entrypoint for the 50GB VL collection. Auto-discovers collectors from
# tools/collectors/ via tools/recipe.COLLECTOR_ORDER. Resumes automatically.
import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import disk_guard
import precompute_index_cache
from recipe import (
    ANNOTATIONS_DIR,
    COLLECTOR_ORDER,
    DATA_SOURCE_DIR,
    DISK_CHECK_EVERY,
    DISK_FLOOR_GB,
    EPISODE_BATCH_SIZE,
    IMAGES_DIR,
    PROGRESS_DIR,
    RAW_DIR,
    RECIPE,
    REGISTER_PREFIX,
    ROOT_DIR,
)


def parse_size(text):
    text = text.strip().upper()
    if text.endswith("G"):
        return int(float(text[:-1]) * 1024**3)
    if text.endswith("M"):
        return int(float(text[:-1]) * 1024**2)
    if text.endswith("K"):
        return int(float(text[:-1]) * 1024)
    return int(text)


def ensure_dirs():
    for p in (ANNOTATIONS_DIR, IMAGES_DIR, RAW_DIR, PROGRESS_DIR, DATA_SOURCE_DIR, ROOT_DIR / "docs"):
        p.mkdir(parents=True, exist_ok=True)
    for src in COLLECTOR_ORDER:
        (ANNOTATIONS_DIR / src).mkdir(parents=True, exist_ok=True)
        (IMAGES_DIR / src).mkdir(parents=True, exist_ok=True)


def write_data_source_register():
    """Generate a local data_source register file. The official one in
    `dexbotic/data/data_source/dm0vl_midtrain_official.py` is preferred for PRs;
    this file is a local fallback when DM0_VL_ROOT differs from upstream layout."""
    out = DATA_SOURCE_DIR / "dm0_vl_midtrain_50gb.py"
    images_root = str(IMAGES_DIR.resolve())
    annotations_root = str(ANNOTATIONS_DIR.resolve())
    lines = [
        "# Auto-generated. Do not edit by hand.",
        "from dexbotic.data.data_source.register import register_dataset",
        "",
        "",
        "DM0_VL_MIDTRAIN_50GB = {",
    ]
    for src in COLLECTOR_ORDER:
        annot_dir = ANNOTATIONS_DIR / src
        if not list(annot_dir.glob("episode_*.jsonl")):
            print(f"[register] skip {src} (no jsonl)")
            continue
        lines.append(
            f'    "{src}": {{"data_path_prefix": r"{images_root}", '
            f'"annotations": r"{annotations_root}/{src}", "frequency": 1}},'
        )
    lines += [
        "}",
        "",
        f'register_dataset(DM0_VL_MIDTRAIN_50GB, meta_data={{}}, prefix="{REGISTER_PREFIX}")',
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def cleanup_raw(source):
    target = RAW_DIR / source
    if target.exists():
        size_before = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target)
        return size_before
    return 0


def run_one(source, target_bytes):
    if source not in RECIPE:
        raise KeyError(f"unknown source: {source}")
    cfg = RECIPE[source]
    if target_bytes is None:
        target_bytes = cfg["target_image_bytes"]
    print(f"[main] source={source} target={target_bytes/1024**3:.2f}GB", flush=True)
    disk_guard.precheck(ROOT_DIR, max(80, int(target_bytes / 1024**3) + 20))

    module = importlib.import_module(f"collectors.{source}")
    progress_path = PROGRESS_DIR / f"{source}.json"
    t0 = time.time()
    report = module.collect_at_scale(
        target_image_bytes=target_bytes,
        annotations_dir=ANNOTATIONS_DIR / source,
        images_dir=IMAGES_DIR,
        raw_dir=RAW_DIR,
        progress_path=progress_path,
        episode_batch_size=EPISODE_BATCH_SIZE,
        disk_check_every=DISK_CHECK_EVERY,
        disk_floor_gb=DISK_FLOOR_GB,
    )
    elapsed = time.time() - t0
    raw_freed = cleanup_raw(source)
    precompute_index_cache.precompute_one(ANNOTATIONS_DIR / source)

    summary = {
        "source": source,
        "rows": report.rows,
        "image_bytes": report.image_bytes,
        "image_gb": report.image_bytes / 1024**3,
        "elapsed_sec": elapsed,
        "skipped": dict(report.skipped),
        "jsonl_files": report.jsonl_files,
        "raw_bytes_freed": raw_freed,
    }
    out = PROGRESS_DIR / f"manifest_{source}.json"
    out.write_text(json.dumps(
        {"summary": summary, "items_count": len(report.items_meta), "items_head": report.items_meta[:50]},
        ensure_ascii=False, indent=2,
    ))
    print(f"[main] {source} done rows={report.rows} bytes={report.image_bytes/1024**3:.2f}GB elapsed={elapsed:.1f}s", flush=True)
    return summary


def write_global_manifest(per_source_summaries):
    total_rows = sum(s["rows"] for s in per_source_summaries)
    total_bytes = sum(s["image_bytes"] for s in per_source_summaries)
    payload = {
        "version": "1",
        "total_rows": total_rows,
        "total_image_bytes": total_bytes,
        "total_image_gb": total_bytes / 1024**3,
        "sources": {s["source"]: {k: v for k, v in s.items() if k != "jsonl_files"} for s in per_source_summaries},
        "register_prefix": REGISTER_PREFIX,
    }
    out = ROOT_DIR / "manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None, help="single source name; default=all")
    p.add_argument("--target-bytes", default=None, help="override target (e.g. 5G)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    if args.dry_run:
        print(f"[dry-run] root={ROOT_DIR}")
        for src in COLLECTOR_ORDER:
            cfg = RECIPE[src]
            print(f"  {src}: target={cfg['target_image_bytes']/1024**3:.1f}GB")
        return 0

    target_bytes = parse_size(args.target_bytes) if args.target_bytes else None
    sources = [args.source] if args.source else COLLECTOR_ORDER

    summaries = []
    for src in sources:
        summaries.append(run_one(src, target_bytes))

    write_data_source_register()
    if args.source is None:
        precompute_index_cache.precompute_all()
        m = write_global_manifest(summaries)
        print(f"[main] manifest -> {m}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
