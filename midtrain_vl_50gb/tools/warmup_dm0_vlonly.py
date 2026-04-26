# DM0 mid-train warmup template (100 steps).
#
# Note: DM0 is a dual-expert model and its forward pass requires `actions`
# tensor (`dm0_arch.py` line ~426). VL-only data does not carry actions, so
# this script alone will fail at model.forward — that is expected per paper
# §3.2 design. To run a real warmup, mix this 50GB VL data with a robot source
# (e.g. `libero_goal+libero_object+libero_spatial`) which provides the action
# schema. See `docs/06_warmup_findings.md` for details.
#
# Usage: torchrun --nproc_per_node=8 warmup_dm0_vlonly.py
import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime

# Register 50GB data_source before importing dexbotic.data.data_source
os.environ.setdefault(
    "DEXBOTIC_DATA_PATH",
    os.path.join(os.environ.get("DM0_VL_ROOT", "data/dm0_vl_midtrain_50gb"), "data_source"),
)

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.dm0_exp import DM0Exp as _DM0Exp
from dexbotic.exp.dm0_exp import (
    DM0ActionConfig as _DM0ActionConfig,
    DM0ComputeNormActionConfig as _DM0ComputeNormActionConfig,
    DM0DataConfig as _DM0DataConfig,
    DM0ModelConfig as _DM0ModelConfig,
    DM0OptimizerConfig as _DM0OptimizerConfig,
    DM0TokenizerConfig as _DM0TokenizerConfig,
    DM0TrainerConfig as _DM0TrainerConfig,
)
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM


@dataclass
class WarmupOptimizerConfig(_DM0OptimizerConfig):
    base_lr: float = field(default=1e-5)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=10)
    weight_decay: float = field(default=1e-10)


@dataclass
class WarmupTrainerConfig(_DM0TrainerConfig):
    wandb_project: str = field(default="dm0_midtrain_vl_warmup")
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=100)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    output_dir: str = field(
        default=f"./user_checkpoints/dm0vlmid_warmup/{datetime.now().strftime('%m%d_%H%M')}"
    )
    lr_scheduler_type: str = field(default="constant")
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="none")


class WarmupNormConfig(_DM0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        return Pipeline([
            ToDict(), ToNumpy(),
            PadState(ndim=32, axis=-1),
            PadAction(ndim=32, axis=-1),
            AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
            DeltaAction(enable=True),
            ToList(),
        ])


@dataclass
class WarmupActionConfig(_DM0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = None
        if self.statistic_mapping:
            try:
                statistic_mapping = self._read_norm_stats(self.statistic_mapping)
            except Exception:
                statistic_mapping = None
        steps = [
            ToDict(), ToNumpy(),
            PadState(ndim=32, axis=-1),
            PadAction(ndim=32, axis=-1),
            AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
            DeltaAction(enable=True),
        ]
        if statistic_mapping:
            steps.append(ActionNorm(statistic_mapping=statistic_mapping))
        steps.append(LoadMultiModal(return_masks=True))
        steps.append(ToList())
        return Pipeline(steps)


@dataclass
class WarmupDataConfig(_DM0DataConfig):
    dataset_name: str = field(default=(
        "dm0vlmid_cambrian737k+dm0vlmid_cambrian10m_filtered+"
        "dm0vlmid_llava_onevision+dm0vlmid_self_collected_proxy"
    ))
    num_images: int = field(default=1)
    aug_policy: list = field(default_factory=lambda: ["dm0"])
    data_keys: list = field(default_factory=lambda: ["input_ids", "labels", "image"])
    auto_norm: bool = field(default=False)
    action_config: WarmupActionConfig = field(default_factory=WarmupActionConfig)


@dataclass
class WarmupModelConfig(_DM0ModelConfig):
    model_name_or_path: str = field(
        default=os.environ.get("DM0_BASE_CKPT", "./checkpoints/DM0-base")
    )

    def build_model(self) -> DM0ForCausalLM:
        return DM0ForCausalLM.from_pretrained(self.model_name_or_path)


@dataclass
class WarmupTokenizerConfig(_DM0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class WarmupExp(_DM0Exp):
    model_config: WarmupModelConfig = field(default_factory=WarmupModelConfig)
    optimizer_config: WarmupOptimizerConfig = field(default_factory=WarmupOptimizerConfig)
    trainer_config: WarmupTrainerConfig = field(default_factory=WarmupTrainerConfig)
    data_config: WarmupDataConfig = field(default_factory=WarmupDataConfig)
    tokenizer_config: WarmupTokenizerConfig = field(default_factory=WarmupTokenizerConfig)

    def _auto_compute_norm_stats(self) -> None:
        if self.local_rank == 0:
            print("[warmup] skip auto_compute_norm_stats (VL-only data)", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="train", choices=["train"])
    a, _ = p.parse_known_args()
    return a


if __name__ == "__main__":
    parse_args()
    import dexbotic.data.data_source  # noqa: F401
    WarmupExp().train()
