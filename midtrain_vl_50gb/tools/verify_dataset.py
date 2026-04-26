# Random-sample validation of collected jsonl. Verifies image files load with
# PIL and conversations contain `<image>` tokens. Fail-closed by default.
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from recipe import ANNOTATIONS_DIR, COLLECTOR_ORDER, IMAGES_DIR  # noqa: E402


def collect_samples(source, num_samples, rng):
    files = sorted((ANNOTATIONS_DIR / source).glob("episode_*.jsonl"))
    if not files:
        return []
    samples = []
    chosen = rng.sample(files, min(len(files), max(1, num_samples // 100)))
    for f in chosen:
        with f.open() as fh:
            lines = fh.readlines()
        rng.shuffle(lines)
        for line in lines[:num_samples]:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            samples.append((source, rec))
            if len(samples) >= num_samples:
                return samples
    return samples


def verify_one(source, rec):
    images_1 = rec.get("images_1") or {}
    url = images_1.get("url")
    if not url:
        return False, "missing_url"
    p = IMAGES_DIR / url
    if not p.exists():
        return False, f"image_not_found:{p}"
    try:
        Image.open(p).verify()
    except Exception as e:
        return False, f"pil_error:{e}"
    convs = rec.get("conversations") or []
    if not convs:
        return False, "no_conversations"
    if not any("<image>" in (t.get("value") or "") for t in convs if isinstance(t, dict)):
        return False, "no_image_token"
    return True, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--allow-empty", action="store_true",
                   help="don't fail on empty source (default: fail-closed)")
    args = p.parse_args()
    rng = random.Random(20260425)
    sources = [args.source] if args.source else COLLECTOR_ORDER

    total = 0
    failed = 0
    by_status = {}
    empty_sources = []
    for src in sources:
        samples = collect_samples(src, args.num_samples, rng)
        if not samples:
            empty_sources.append(src)
            continue
        for source, rec in samples:
            ok, status = verify_one(source, rec)
            total += 1
            by_status[status] = by_status.get(status, 0) + 1
            if not ok:
                failed += 1

    print(json.dumps(
        {"total": total, "failed": failed, "by_status": by_status, "empty_sources": empty_sources},
        ensure_ascii=False, indent=2,
    ))
    if failed > 0:
        print(f"[verify] FAILED: {failed}/{total} samples failed", file=sys.stderr)
        return 1
    if empty_sources and not args.allow_empty:
        print(f"[verify] FAILED: empty sources {empty_sources} (--allow-empty to bypass)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
