# Subset-coverage gate. Catches the failure mode where source bytes hit target
# but a critical subset (e.g. embspatial) is empty.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe import COLLECTOR_ORDER, PROGRESS_DIR

# Per-source minimum bytes thresholds. Aligned with paper §3.2 intent for
# subset diversity; tune via PR if downstream tasks need stronger guarantees.
MIN_SUBSET_BYTES = {
    "self_collected_proxy": {
        "uground": 500 * 1024 * 1024,
        "embspatial": 50 * 1024 * 1024,
        "synthdog_en": 100 * 1024 * 1024,
        "cord_v2": 10 * 1024 * 1024,
    },
    "llava_onevision": {
        "_min_subset_count": 5,
        "_min_per_subset_bytes": 100 * 1024 * 1024,
    },
}


def check_source_coverage(source):
    progress_path = Path(PROGRESS_DIR) / f"{source}.json"
    if not progress_path.exists():
        return False, [f"{source}: progress.json missing"]
    try:
        progress = json.loads(progress_path.read_text())
    except Exception as e:
        return False, [f"{source}: progress.json parse error: {e}"]
    failures = []
    cfg = MIN_SUBSET_BYTES.get(source, {})
    record = progress.get("subset_bytes_record", {})
    for subset, min_bytes in cfg.items():
        if subset.startswith("_"):
            continue
        actual = record.get(subset, 0)
        if actual < min_bytes:
            failures.append(f"{source}.{subset}: only {actual/1024**2:.1f}MB < {min_bytes/1024**2:.1f}MB")
    min_count = cfg.get("_min_subset_count")
    min_per = cfg.get("_min_per_subset_bytes")
    if min_count and min_per:
        valid = [s for s, b in record.items() if not s.startswith("_") and b >= min_per]
        if len(valid) < min_count:
            failures.append(
                f"{source}: only {len(valid)} subsets >= {min_per/1024**2:.0f}MB, need {min_count}"
            )
    return len(failures) == 0, failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any source falls below subset coverage thresholds")
    args = p.parse_args()
    sources = [args.source] if args.source else COLLECTOR_ORDER
    all_failures = []
    for src in sources:
        ok, fails = check_source_coverage(src)
        if ok:
            print(f"[coverage] {src}: OK")
        else:
            for f in fails:
                print(f"[coverage] FAIL: {f}")
                all_failures.append(f)
    if all_failures:
        if args.strict:
            print(f"[coverage] STRICT FAIL: {len(all_failures)} subset coverage failures")
            return 1
        print(f"[coverage] WARN: {len(all_failures)} subset coverage failures (use --strict to fail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
