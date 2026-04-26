# Rebuild progress.json from on-disk jsonl + image stats.
# Run before resuming a partial collection with newer code; lets the
# DurableEpisodeWriter pick up where the disk actually left off.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe import ANNOTATIONS_DIR, COLLECTOR_ORDER, IMAGES_DIR, PROGRESS_DIR

# Sources whose `subset_bytes_record` should be rebuilt from disk. Their jsonl
# image URLs use a 3-segment path (source/subset/file), so we can recover
# subset-level counts from disk facts alone.
RESET_EXTRA_SOURCES = {"llava_onevision", "self_collected_proxy"}
RESET_EXTRA_KEYS = {"completed_subsets", "subset_offsets", "subset_bytes_record"}

# self_collected_proxy v3 wrote images under "synthdog/" and "cord/" but the
# canonical subset names (matching coverage gate + collector PIPELINE) are
# "synthdog_en" and "cord_v2". Apply the alias on rebuild.
SCP_SUBSET_ALIAS = {"synthdog": "synthdog_en", "cord": "cord_v2"}


def _parse_subset_key(url, source):
    parts = url.split("/", 2)
    if len(parts) < 3:
        return None
    key = parts[1]
    if source == "self_collected_proxy":
        return SCP_SUBSET_ALIAS.get(key, key)
    return key


def reconcile_one(source, dry_run=False, reset_extra=False):
    annotation_dir = Path(ANNOTATIONS_DIR) / source
    progress_path = Path(PROGRESS_DIR) / f"{source}.json"
    if not annotation_dir.exists():
        print(f"[reconcile] {source}: annotation dir missing, skip")
        return None
    files = sorted(annotation_dir.glob("episode_*.jsonl"))
    real_rows = 0
    real_bytes = 0
    seen_keys = set()
    rebuilt_subset_bytes = {}
    bad_lines = 0
    missing_images = 0
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                real_rows += 1
                images_1 = rec.get("images_1") or {}
                url = images_1.get("url")
                if not url:
                    continue
                p = Path(IMAGES_DIR) / url
                if not p.exists():
                    missing_images += 1
                    continue
                try:
                    sz = p.stat().st_size
                except Exception:
                    missing_images += 1
                    continue
                real_bytes += sz
                seen_keys.add(p.name)
                sub = _parse_subset_key(url, source)
                if sub:
                    rebuilt_subset_bytes[sub] = rebuilt_subset_bytes.get(sub, 0) + sz
    next_episode = max(
        len(files),
        (int(files[-1].stem.rsplit("_", 1)[-1]) + 1) if files else 0,
    )

    old_progress = {}
    if progress_path.exists():
        try:
            old_progress = json.loads(progress_path.read_text())
        except Exception:
            old_progress = {}
    new_progress = {
        "rows": real_rows,
        "image_bytes": real_bytes,
        "next_episode": next_episode,
        "seen_image_keys": list(seen_keys),
        "progress_version": 2,
    }
    if source in RESET_EXTRA_SOURCES and rebuilt_subset_bytes:
        new_progress["subset_bytes_record"] = rebuilt_subset_bytes
        print(f"[reconcile] {source}: rebuilt subset_bytes_record="
              f"{ {k: round(v/1024**2, 1) for k, v in rebuilt_subset_bytes.items()} }MB")
    base_keys = set(new_progress.keys()) | {"subset_bytes_record"}
    auto_reset = reset_extra or (source in RESET_EXTRA_SOURCES)
    skipped = []
    for k, v in old_progress.items():
        if k in base_keys:
            continue
        if auto_reset and k in RESET_EXTRA_KEYS:
            skipped.append(k)
            continue
        new_progress[k] = v
    if skipped:
        print(f"[reconcile] {source}: dropped legacy extra {skipped}")

    print(f"[reconcile] {source}: rows={real_rows} bytes={real_bytes/1024**3:.2f}GB "
          f"jsonl_files={len(files)} next_episode={next_episode} "
          f"bad_lines={bad_lines} missing_images={missing_images}")

    if dry_run:
        print(f"[reconcile] {source}: DRY RUN, not writing")
        return new_progress

    if progress_path.exists():
        progress_path.with_suffix(".json.before_reconcile").write_text(progress_path.read_text())
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_progress, ensure_ascii=False, indent=2))
    tmp.replace(progress_path)
    return new_progress


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reset-extra", action="store_true",
                   help="force-reset completed_subsets/subset_offsets/subset_bytes_record for all sources")
    args = p.parse_args()
    sources = [args.source] if args.source else COLLECTOR_ORDER
    for src in sources:
        reconcile_one(src, dry_run=args.dry_run, reset_extra=args.reset_extra)


if __name__ == "__main__":
    raise SystemExit(main())
