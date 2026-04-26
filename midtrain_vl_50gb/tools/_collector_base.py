# Shared utilities for all source collectors.
# - DurableEpisodeWriter: progress.json mirrors only flushed-to-disk state.
# - stream_jsonl: range-resumable JSONL streaming with proxy-aware reconnect.
# - download_coco_record: the COCO image fetcher reused by Cambrian collectors.
# - normalize_conversations: enforce dexbotic Dexdata conversation format.
import io
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

SSL_CTX = ssl._create_unverified_context()
MIN_IMAGE_BYTES = 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 2.0
HTTP_BACKOFF_MAX = 30.0
DOWNLOAD_TIMEOUT = 60

COCO_BASE = "https://images.cocodataset.org"
COCO_RE = re.compile(r"^coco/(train|val)(201[47])/(?:COCO_(?:train|val)201[47]_)?(\d{12})\.jpg$")


@dataclass
class CollectorReport:
    source: str = ""
    rows: int = 0
    image_bytes: int = 0
    jsonl_files: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    items_meta: list = field(default_factory=list)


def normalize_conversations(conversations):
    """Coerce arbitrary conversation lists to dexbotic Dexdata `[{"from": ..., "value": ...}]`,
    inject `<image>` token into the first human turn if missing."""
    if not isinstance(conversations, list):
        return [
            {"from": "human", "value": "<image>\nDescribe the image."},
            {"from": "gpt", "value": str(conversations)},
        ]
    norm = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "human")
        if role == "user":
            role = "human"
        elif role == "assistant":
            role = "gpt"
        value = turn.get("value") or turn.get("content") or turn.get("text") or ""
        norm.append({"from": role, "value": str(value)})
    if not norm:
        return [
            {"from": "human", "value": "<image>\nDescribe the image."},
            {"from": "gpt", "value": "No answer."},
        ]
    first_human = next((t for t in norm if t["from"] == "human"), norm[0])
    if "<image>" not in first_human["value"]:
        first_human["value"] = "<image>\n" + first_human["value"]
    return norm


def http_fetch(url, timeout=DOWNLOAD_TIMEOUT):
    last_exc = None
    for attempt in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=SSL_CTX) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError) as e:
            last_exc = e
            wait = min(HTTP_BACKOFF_BASE * (2 ** attempt), HTTP_BACKOFF_MAX)
            time.sleep(wait + random.uniform(0, 1))
    raise RuntimeError(f"http_failed: {url}: {last_exc}")


def validate_image(data):
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return False
    return True


def coco_url(image_path):
    """Map a Dexdata image_path like 'coco/train2017/<id>.jpg' to the COCO CDN URL."""
    m = COCO_RE.match(image_path)
    if not m:
        return None
    split, year, image_id = m.groups()
    return f"{COCO_BASE}/{split}{year}/{image_id}.jpg"


def download_coco_record(record, images_dir, source_name, seen_keys):
    """Shared COCO downloader for Cambrian-* collectors.
    Returns ((record, local_rel, size, image_path), status) or (None, status)."""
    image_path = str(record.get("image", ""))
    url = coco_url(image_path)
    if not url:
        return None, "invalid_path"
    image_name = Path(image_path).name
    if image_name in seen_keys:
        return None, "already_present"
    local_rel = f"{source_name}/{image_name}"
    local_path = images_dir / local_rel
    if local_path.exists() and local_path.stat().st_size >= MIN_IMAGE_BYTES:
        return (record, local_rel, local_path.stat().st_size, image_path), "ok_cached"
    try:
        data = http_fetch(url)
    except Exception:
        return None, "http_failed"
    if len(data) > MAX_IMAGE_BYTES:
        return None, "too_large"
    if len(data) < MIN_IMAGE_BYTES:
        return None, "too_small"
    if not validate_image(data):
        return None, "pil_error"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(local_path)
    return (record, local_rel, len(data), image_path), "ok"


class DurableEpisodeWriter:
    """Buffer rows in memory; on flush, fsync + atomic rename a new episode_*.jsonl,
    then update flushed counts and persist progress.json. Crashes only lose unflushed
    in-memory rows; never corrupt on-disk jsonl. Resume reuses progress.json.
    """

    def __init__(self, annotations_dir, progress_path, source_name, episode_batch_size=10000):
        self.annotations_dir = Path(annotations_dir)
        self.progress_path = Path(progress_path)
        self.source = source_name
        self.batch_size = episode_batch_size
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

        self._pending_rows = []
        self._pending_bytes = 0
        self._pending_meta = []
        self._pending_seen_keys = []
        self._reserved_seen_keys = set()
        self._extra_progress = {}
        self._pending_meta_history = []

        self._existing_episodes = sorted(self.annotations_dir.glob("episode_*.jsonl"))
        self._progress = self._load_progress()
        self._next_episode = max(
            int(self._progress.get("next_episode", 0)),
            len(self._existing_episodes),
        )
        self.flushed_rows = int(self._progress.get("rows", 0))
        self.flushed_bytes = int(self._progress.get("image_bytes", 0))
        self.seen_keys = set(self._progress.get("seen_image_keys", []))

    def _load_progress(self):
        if not self.progress_path.exists():
            return {}
        try:
            return json.loads(self.progress_path.read_text())
        except Exception:
            return {}

    def get_extra(self, key, default=None):
        return self._progress.get(key, default)

    def add(self, row, byte_size, meta, seen_key):
        if seen_key is not None:
            if seen_key in self.seen_keys or seen_key in self._reserved_seen_keys:
                return self.flushed_bytes + self._pending_bytes
            self._reserved_seen_keys.add(seen_key)
            self._pending_seen_keys.append(seen_key)
        self._pending_rows.append(row)
        self._pending_bytes += byte_size
        self._pending_meta.append(meta)
        return self.flushed_bytes + self._pending_bytes

    @property
    def total_bytes(self):
        return self.flushed_bytes + self._pending_bytes

    @property
    def total_rows(self):
        return self.flushed_rows + len(self._pending_rows)

    @property
    def jsonl_files(self):
        return [str(p) for p in self._existing_episodes]

    def should_flush(self):
        return len(self._pending_rows) >= self.batch_size

    def flush(self, set_extra=None):
        if not self._pending_rows:
            return None
        if set_extra:
            self._extra_progress.update(set_extra)
        out = self.annotations_dir / f"episode_{self._next_episode:06d}.jsonl"
        while out.exists():
            self._next_episode += 1
            out = self.annotations_dir / f"episode_{self._next_episode:06d}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in self._pending_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(out)
        self._existing_episodes.append(out)
        flushed_count = len(self._pending_rows)
        flushed_size = self._pending_bytes
        flushed_meta = list(self._pending_meta)
        flushed_seen = list(self._pending_seen_keys)
        self._pending_rows.clear()
        self._pending_bytes = 0
        self._pending_meta.clear()
        self._pending_seen_keys.clear()
        self._reserved_seen_keys.clear()
        self.flushed_rows += flushed_count
        self.flushed_bytes += flushed_size
        for k in flushed_seen:
            self.seen_keys.add(k)
        self._next_episode += 1
        self._pending_meta_history.extend(flushed_meta)
        self._save_progress()
        return out

    def update_extra(self, **kwargs):
        self._extra_progress.update(kwargs)

    def _save_progress(self):
        payload = {
            "rows": self.flushed_rows,
            "image_bytes": self.flushed_bytes,
            "next_episode": self._next_episode,
            "seen_image_keys": list(self.seen_keys),
        }
        payload.update(self._extra_progress)
        tmp = self.progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self.progress_path)

    def final_save(self, **extra):
        # Pending rows are intentionally NOT flushed here; they would already be flushed
        # if the run completed normally via flush(). final_save persists extra metadata only.
        if extra:
            self._extra_progress.update(extra)
        self._save_progress()


def stream_jsonl(url, start_offset=0, max_retries=20, line_max_bytes=4 * 1024 * 1024):
    """Stream a remote JSONL with byte-offset Range resume on transient SSL/EOF errors."""
    consumed = start_offset
    for retry in range(max_retries):
        buf = b""
        try:
            headers = {"Range": f"bytes={consumed}-"} if consumed else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180, context=SSL_CTX) as resp:
                got_data = False
                for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                    if chunk:
                        got_data = True
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        consumed += len(line) + 1
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line.decode("utf-8")), consumed
                        except Exception:
                            continue
                if not got_data and retry > 0:
                    return
            try:
                head_req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(head_req, timeout=30, context=SSL_CTX) as hr:
                    total = int(hr.headers.get("X-Linked-Size") or hr.headers.get("Content-Length") or 0)
                if total and consumed >= total - line_max_bytes:
                    return
            except Exception as e:
                print(f"[stream] head check failed: {e}", flush=True)
            print(f"[stream] reconnect at offset={consumed} retry={retry+1}/{max_retries}", flush=True)
            time.sleep(min(5 + retry, 30))
        except Exception as e:
            print(f"[stream] error at offset={consumed}: {e}, retry {retry+1}", flush=True)
            time.sleep(min(5 + retry, 30))
    print(f"[stream] gave up after {max_retries} retries at offset={consumed}", flush=True)
