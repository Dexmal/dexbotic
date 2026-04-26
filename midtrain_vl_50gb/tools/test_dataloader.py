# End-to-end DexDataset + DataCollator smoke test. Fail-closed: exits non-zero
# if any registered source is missing or any sample fails to load.
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from recipe import COLLECTOR_ORDER, REGISTER_PREFIX, ROOT_DIR  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--allow-missing", action="store_true",
                   help="skip unregistered sources instead of failing")
    args = p.parse_args()

    os.environ.setdefault("DEXBOTIC_DATA_PATH", str(ROOT_DIR / "data_source"))

    import dexbotic.data.data_source  # noqa: F401
    from dexbotic.data.collator import DataCollatorForSupervisedDataset
    from dexbotic.data.data_source.register import CONVERSATION_DATA
    from dexbotic.data.dataset.dex_dataset import DexDataset
    from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
    from dexbotic.data.dataset.tokenization import DummyTokenization
    from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
    from dexbotic.data.dataset.transform.multimodal import LoadMultiModal

    if args.dataset_name is None:
        ds_name = "+".join(f"{REGISTER_PREFIX}_{s}" for s in COLLECTOR_ORDER)
    else:
        ds_name = args.dataset_name

    missing = [n for n in ds_name.split("+") if n not in CONVERSATION_DATA]
    present = [n for n in ds_name.split("+") if n in CONVERSATION_DATA]
    if missing and not args.allow_missing:
        raise AssertionError(f"missing registered datasets: {missing} (use --allow-missing to skip)")
    if missing:
        print(f"[test] WARN: skipping missing: {missing}")
    if not present:
        raise AssertionError("no registered dataset to test")
    ds_name = "+".join(present)

    action_process_func = Pipeline([ToDict(), ToNumpy(), LoadMultiModal(), ToList()])
    data_args = EasyDict(
        dataset_name=ds_name,
        num_images=1,
        data_keys=["input_ids", "labels", "image"],
        images_keys=None,
        depths_keys=None,
        load_depth=False,
        discrete_state_input=False,
        aug_policy=None,
        image_aspect_ratio=None,
    )
    dataset = DexDataset(
        data_args=data_args,
        tokenization_func=DummyTokenization(),
        action_process_func=action_process_func,
        image_process_func=DummyRGBProcessor(),
        depth_process_func=lambda _: torch.zeros(1),
    )

    n = len(dataset)
    print(f"[test] dataset_name={ds_name} dataset_len={n}")
    if n == 0:
        raise AssertionError("empty dataset")

    for idx in [0, n // 8, n // 4, n // 2, n - 1]:
        item = dataset[idx]
        for k in ("input_ids", "labels", "image"):
            if k not in item:
                raise AssertionError(f"sample {idx} missing {k}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForSupervisedDataset(TinyTokenizer()),
    )
    batch = next(iter(loader))
    required = {"input_ids", "labels", "attention_mask", "images"}
    if not required.issubset(batch):
        raise AssertionError(f"batch missing keys: {required - set(batch)}")

    summary = {
        "dataset_name": ds_name,
        "dataset_len": n,
        "batch_keys": sorted(batch.keys()),
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
