# DM0 mid-train VL 50GB Dataset Recipe

A reproducible pipeline for assembling a 50GB Vision-Language corpus that follows the DM0 paper §3.2 Mid-Training recipe. Output drops directly into the dexbotic `DexDataset` / `DataCollatorForSupervisedDataset` flow.

## What it does

Collects ~515k samples (50.55 GB images) from four open-source sources matching the paper's VL subdivision:

| Source | HF repo | Target | Paper section |
|--------|---------|--------|---------------|
| `dm0vlmid_cambrian737k` | `LanguageBind/Cambrian737k` | 12 GB | §3.2 (1) |
| `dm0vlmid_cambrian10m_filtered` | `nyu-visionx/Cambrian-10M` | 20 GB (source-limited to ~10 GB by COCO pool) | §3.2 (2) |
| `dm0vlmid_llava_onevision` | `lmms-lab/LLaVA-OneVision-Data` (30 subsets, dynamic quota) | 22 GB | §3.2 (3) |
| `dm0vlmid_self_collected_proxy` | `zonghanHZH/UGround-V1-8k`, `Phineas476/EmbSpatial-Bench`, `naver-clova-ix/synthdog-en`, `naver-clova-ix/cord-v2` | 8 GB | §3.2 (4) |

All outputs use the dexbotic Dexdata format (`images_N` + `conversations` + `is_robot=false`).

## Quick start

```bash
# 1. Set the data root (any disk with ~60GB free)
export DM0_VL_ROOT=/path/to/dm0_vl_midtrain_50gb

# 2. Standard proxy / venv (your own setup)
source /etc/profile.d/proxy.sh   # or: export http_proxy=...
source /path/to/dexbotic/.venv-py310/bin/activate
cd /path/to/dexbotic

# 3. Run all four sources (resumes automatically if interrupted)
python midtrain_vl_50gb/tools/prepare.py

# 4. Or run a single source / pilot size
python midtrain_vl_50gb/tools/prepare.py --source cambrian737k --target-bytes 5G

# 5. Build dexbotic index_cache
python midtrain_vl_50gb/tools/precompute_index_cache.py

# 6. Validate dataloader end-to-end
export DEXBOTIC_DATA_PATH=$(realpath dexbotic/data/data_source)
python midtrain_vl_50gb/tools/test_dataloader.py

# 7. Strict subset coverage gate
python midtrain_vl_50gb/tools/subset_coverage_check.py --strict
```

The official register file lives at `dexbotic/data/data_source/dm0vl_midtrain_official.py` and is loaded automatically when `DM0_VL_ROOT` is set.

## Hardware

- Disk: 60+ GB free (final ~52 GB images + 0.5 GB jsonl)
- Memory: 16+ GB (Cambrian737k.json loads to ~5 GB Python objects)
- Network: HuggingFace + COCO reachable (set `HF_ENDPOINT=https://hf-mirror.com` to bypass slow upstream)

## Architecture

```
midtrain_vl_50gb/
├── README.md                     ← this file
├── tools/
│   ├── recipe.py                 ← target bytes + subset lists (single source of truth)
│   ├── _collector_base.py        ← DurableEpisodeWriter, stream_jsonl, COCO helper
│   ├── disk_guard.py             ← precheck / fuse
│   ├── collectors/
│   │   ├── cambrian737k.py
│   │   ├── cambrian10m_filtered.py
│   │   ├── llava_onevision.py
│   │   └── self_collected_proxy.py
│   ├── prepare.py                ← entrypoint (auto-discovers collectors)
│   ├── reconcile_progress.py     ← rebuild progress.json from on-disk jsonl + images
│   ├── precompute_index_cache.py ← speeds up dexbotic dataloader cold start
│   ├── subset_coverage_check.py  ← enforces paper-aligned subset minimums
│   ├── test_dataloader.py        ← 8192-sample PIL.open + DataCollator smoke
│   └── verify_dataset.py         ← random-sample validation (fail-closed)
├── docs/
│   ├── 00_data_index.md          ← dexbotic source code index (path:line)
│   ├── 01_data_processing_log.md ← runtime log template
│   ├── 02_source_card_*.md       ← per-source license + schema
│   ├── 03_recipe.md              ← paper §3.2 alignment
│   ├── 06_warmup_findings.md     ← DM0 mid-train training-loop validation
│   └── 07_verification_steps.md  ← step-by-step reproduction checklist
└── tests/
    ├── test_collector_base.py
    ├── test_reconcile.py
    └── test_subset_coverage.py
```

## Design notes

### Durable resume
`DurableEpisodeWriter` checkpoints **only flushed state** to `progress.json`. A crash mid-batch loses pending in-memory rows but never corrupts written jsonl files. `next_episode = max(progress, len(existing))` prevents overwriting episodes after restart.

### Coverage-aware collection
For `llava_onevision` and `self_collected_proxy`, collectors continue past total-byte target until per-subset minimums are met. `subset_coverage_check.py --strict` is the final gate — it rejects (e.g.) 8GB of `self_collected_proxy` if `embspatial < 50MB`.

### Hybrid resume
`reconcile_progress.py` rebuilds `subset_bytes_record` from on-disk jsonl + image stat (with `synthdog → synthdog_en`, `cord → cord_v2` canonicalisation) before each run. Lets you safely resume a partial run with newer code.

### Source-limited tolerance
`cambrian10m_filtered` is dedupe'd against `cambrian737k` over the COCO image pool. After both sources, the COCO pool is exhausted and `cambrian10m` settles at ~9 GB regardless of target. This is a dataset-level limit, not a bug. Compensated by extending `llava_onevision` subset list (default 30 subsets, ~22 GB).

## Validation results

End-to-end validated on 8 × NVIDIA H200, dexbotic main branch:

| Check | Result |
|-------|--------|
| Total | 50.55 GB / 515,572 rows / 4 sources / 35 jsonl shards |
| `verify_dataset.py --num-samples 2048` | 8192/8192 PIL.open OK, 0 failed |
| `subset_coverage_check.py --strict` | All 4 sources OK (incl. embspatial 267.8 MB) |
| `test_dataloader.py` | DexDataset registers, batch keys `{input_ids, labels, attention_mask, images}` complete |
| Throughput (DummyProcessor) | bs=32 nw=16: **69.7 samples/s** |
| DM0-base load + 8-rank deepspeed init | OK (8 × 6.1 GB GPU mem) |
| DM0 forward | requires mixed VL+robot batch (paper §3.2 design); see `docs/06_warmup_findings.md` |

## Limitations

1. `cambrian10m_filtered` settles at 9-11 GB (COCO pool dedup'd against c737k). Total is `~50 GB` not strictly per the 12/20/10/8 paper recipe; documented as `paper-aligned, source-limited 50GB VL mid-train corpus`.
2. The `self_collected_proxy.embspatial` portion uses the open `Phineas476/EmbSpatial-Bench` (3640 base64-encoded JPEGs) as a proxy for the paper's "caption-reannotated embodied data", which has no public release.
3. DM0 model forward needs `actions` tensor — VL-only data alone is not trainable; use mixed (e.g. `dm0vlmid_* + libero_goal+...`) for actual mid-train.

## License

Data collection scripts: same license as upstream dexbotic.

Each source dataset retains its original license (cf. `docs/02_source_card_*.md`):
- LAION/Cambrian/LLaVA-OneVision: respective HF dataset licenses
- COCO: CC BY-NC-SA 2.0
- UGround / EmbSpatial / SynthDog / CORD: Apache 2.0 / CC-BY 4.0

## Acknowledgements

Built on the dexbotic DexDataset / DataCollatorForSupervisedDataset infrastructure. Data sources are credited in their respective source cards. Closely follows the DM0 paper [arXiv:2602.14974].
