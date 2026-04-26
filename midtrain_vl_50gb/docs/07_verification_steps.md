# 07 Verification Steps — End-to-End Reproduction Checklist

This document is a self-contained, copy-paste reproducible verification of the 50GB VL mid-train data recipe. Each step is fail-closed: if it returns non-zero, fix before proceeding.

## Prerequisites

```bash
# 1. dexbotic checkout + venv
cd /path/to/dexbotic
source ACTIVATE_DEXBOTIC.sh
pip install "deepspeed==0.14.4"  # version-pinned for torch 2.2 compat

# 2. Standard proxy/env (your network)
source /etc/profile.d/proxy.sh   # or equivalent
export HF_ENDPOINT=https://hf-mirror.com  # optional but often faster

# 3. Pick a data root with ≥60 GB free
export DM0_VL_ROOT=/data/dm0_vl_midtrain_50gb
mkdir -p "$DM0_VL_ROOT"
```

## Step 1 — Dry-run sanity check

```bash
python midtrain_vl_50gb/tools/prepare.py --dry-run
```

Expected: prints 4 sources with their target sizes. Shouldn't touch network.

## Step 2 — Pilot run on a single source (5 GB)

```bash
python midtrain_vl_50gb/tools/prepare.py --source cambrian737k --target-bytes 5G
```

Expected:
- `progress/cambrian737k.json` populated
- `annotations/cambrian737k/episode_*.jsonl` (≥ 1 file, 10000 rows each)
- `images/cambrian737k/*.jpg` ≈ 5 GB
- exit code 0

If it stalls on COCO 429s, that's normal; the collector retries with exponential backoff. Resume with the same command.

## Step 3 — Full collection (all 4 sources)

```bash
python midtrain_vl_50gb/tools/prepare.py
```

Expected runtime: 3-12 hours depending on proxy throughput. Auto-resumes if interrupted; safe to Ctrl-C and re-run. Final state:
- `cambrian737k`: ~12 GB (target=12)
- `cambrian10m_filtered`: ~9-11 GB (source-limited by COCO pool dedupe)
- `llava_onevision`: ~22 GB across 30 subsets
- `self_collected_proxy`: ~8 GB across 4 subsets

`manifest.json` written to `$DM0_VL_ROOT`.

## Step 4 — Hybrid resume rebuild (if you need to update progress.json)

If you change collector code mid-run, use this before continuing:

```bash
python midtrain_vl_50gb/tools/reconcile_progress.py --dry-run
python midtrain_vl_50gb/tools/reconcile_progress.py
```

Rebuilds `rows / image_bytes / seen_image_keys / subset_bytes_record` from on-disk facts. For `self_collected_proxy`, applies `synthdog → synthdog_en` and `cord → cord_v2` aliasing.

## Step 5 — Index cache pre-compute

```bash
python midtrain_vl_50gb/tools/precompute_index_cache.py
```

Speeds up dexbotic `DexDataset` cold start by 30-60s.

## Step 6 — Subset-coverage strict gate

```bash
python midtrain_vl_50gb/tools/subset_coverage_check.py --strict
```

Expected output:
```
[coverage] cambrian737k: OK
[coverage] cambrian10m_filtered: OK
[coverage] llava_onevision: OK
[coverage] self_collected_proxy: OK
```

Exit 0. Specifically validates `self_collected_proxy.embspatial >= 50 MB` (the failure mode where total bytes hit target but a critical subset was empty).

## Step 7 — DexDataset / DataCollator smoke test

```bash
export DEXBOTIC_DATA_PATH=$(realpath dexbotic/data/data_source)
python midtrain_vl_50gb/tools/test_dataloader.py --batch-size 4
```

Expected: dataset_len ≈ 515,000, batch keys `{input_ids, labels, attention_mask, images}`. Exit 0.

## Step 8 — Random-sample verification (8192 PIL.open)

```bash
python midtrain_vl_50gb/tools/verify_dataset.py --num-samples 2048
```

Expected:
```json
{
  "total": 8192,
  "failed": 0,
  "by_status": {"ok": 8192},
  "empty_sources": []
}
```

## Step 9 — Throughput benchmark (optional)

```python
import os, time, torch
from torch.utils.data import DataLoader
from easydict import EasyDict

os.environ["DEXBOTIC_DATA_PATH"] = os.environ["DM0_VL_ROOT"] + "/data_source"
import dexbotic.data.data_source
from dexbotic.data.collator import DataCollatorForSupervisedDataset
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal


class TinyTok:
    pad_token_id = 0; eos_token_id = 1; model_max_length = 32


ds = DexDataset(
    data_args=EasyDict(
        dataset_name="dm0vlmid_cambrian737k+dm0vlmid_cambrian10m_filtered+"
                      "dm0vlmid_llava_onevision+dm0vlmid_self_collected_proxy",
        num_images=1, data_keys=["input_ids", "labels", "image"],
        images_keys=None, depths_keys=None, load_depth=False,
        discrete_state_input=False, aug_policy=None, image_aspect_ratio=None,
    ),
    tokenization_func=DummyTokenization(),
    action_process_func=Pipeline([ToDict(), ToNumpy(), LoadMultiModal(), ToList()]),
    image_process_func=DummyRGBProcessor(),
    depth_process_func=lambda _: torch.zeros(1),
)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=16,
                    collate_fn=DataCollatorForSupervisedDataset(TinyTok()))
it = iter(loader)
for _ in range(5): next(it)  # warm
t0 = time.time(); cnt = 0
for _ in range(30): cnt += next(it)["input_ids"].shape[0]
print(f"{cnt/(time.time()-t0):.1f} samples/s")
```

Expected: ≥ 50 samples/s on a server with NVMe disk and 16 CPU workers (we measured 69.7 samples/s on H200 box).

## Step 10 — Unit tests

```bash
pytest midtrain_vl_50gb/tests/ -v
```

Expected: all tests in `test_collector_base.py`, `test_reconcile.py`, `test_subset_coverage.py` pass.

## Step 11 — DM0 mid-train warmup (optional, requires mixed data)

```bash
export DM0_BASE_CKPT=./checkpoints/DM0-base
mkdir -p checkpoints
huggingface-cli download Dexmal/DM0-base --local-dir checkpoints/DM0-base

# Edit warmup_dm0_vlonly.py: append a robot source to dataset_name, e.g.:
#   "dm0vlmid_*+libero_goal+libero_object+libero_spatial"
# Then download libero data per dexbotic README.

WANDB_DISABLED=true torchrun --nproc_per_node=8 \
  midtrain_vl_50gb/tools/warmup_dm0_vlonly.py
```

See `docs/06_warmup_findings.md` for the design constraint that DM0 dual-expert architecture requires actions tensor and therefore mixed VL+robot data for any forward pass.

## Failure recovery

| Symptom | Action |
|---------|--------|
| `set -u` `${PYTHONPATH}: unbound` | use `${PYTHONPATH:-}` in your shell init |
| `ImportError: deepspeed not available` | `pip install deepspeed==0.14.4` (newer versions break torch 2.2) |
| `module 'torch.library' has no attribute 'custom_op'` | downgrade deepspeed; never above 0.14.x for torch 2.2 |
| `[stream] reconnect at offset=…` repeatedly | proxy is throttling; wait or switch `HF_ENDPOINT=https://hf-mirror.com` |
| `[scp/embspatial] hf_hub_download failed` | network; embspatial metadata file is 250 MB, retry or use the mirror |
| `coverage: FAIL: self_collected_proxy.embspatial …` | run `prepare.py --source self_collected_proxy --target-bytes 9G` to force re-collect |
| `AttributeError: 'NoneType' object has no attribute 'shape'` in DM0 forward | VL-only data alone can't drive DM0; mix in a robot source (see Step 11) |

## Reference: validated state

Validated on 8 × NVIDIA H200 + dexbotic main:
- 50.55 GB images / 515,572 rows / 35 jsonl shards
- 4 sources registered as `dm0vlmid_{cambrian737k,cambrian10m_filtered,llava_onevision,self_collected_proxy}`
- Strict coverage gate passes; verify 8192/8192 OK
- DexDataset constructs in <30 s with pre-computed `index_cache.json`
- Throughput 69.7 samples/s with bs=32 nw=16 dummy processors
