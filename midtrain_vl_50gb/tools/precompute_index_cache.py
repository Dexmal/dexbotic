# Pre-compute dexbotic DexDataset index_cache.json so the first-time DexDataset
# constructor doesn't have to scan all jsonl files (saves 30-60s on cold start).
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe import ANNOTATIONS_DIR, COLLECTOR_ORDER


def precompute_one(annotation_dir):
    annotation_dir = Path(annotation_dir)
    files = sorted(annotation_dir.glob("episode_*.jsonl"))
    cache = {
        "meta_data": {"total_samples": 0, "total_jsonl_files": len(files)},
        "data": {},
    }
    for f in files:
        with f.open() as fh:
            n = sum(1 for _ in fh)
        cache["data"][str(f.absolute())] = n
        cache["meta_data"]["total_samples"] += n
    out = annotation_dir / "index_cache.json"
    out.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return cache["meta_data"]["total_samples"]


def precompute_all():
    total = 0
    for src in COLLECTOR_ORDER:
        d = ANNOTATIONS_DIR / src
        if not d.exists():
            continue
        n = precompute_one(d)
        print(f"{src}: {n} samples")
        total += n
    print(f"total: {total} samples")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    args = p.parse_args()
    if args.source:
        n = precompute_one(ANNOTATIONS_DIR / args.source)
        print(f"{args.source}: {n} samples")
    else:
        precompute_all()


if __name__ == "__main__":
    main()
