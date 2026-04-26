# 06 DM0 mid-train warmup 验证结论

> 时间：2026-04-26 17:17-17:30
> 目标：100 steps warmup 确认 DM0 mid-train 训练链路正常
> 结论：**链路前段全部通过；DM0 设计要求 mixed data，VL-only 不可单独训**

---

## 1. 验证维度

| 阶段 | 状态 | 说明 |
|------|------|------|
| dataloader 端到端 | ✅ | 8192/8192 PIL.open 通过；4 配置 throughput benchmark 16-70 samples/s |
| DM0-base 7GB checkpoint 下载 | ✅ | 通过 hf-mirror.com 续下，11.4 MB/s 4m29s 完成 |
| deepspeed 0.14.4 安装 | ✅ | 与 torch 2.2.2 兼容（0.18.9 版本不兼容） |
| torchrun 8 卡分布式启动 | ✅ | 8 ranks 全 init，DeepSpeed comm OK |
| DM0 model 加载（7B params） | ✅ | 8 卡各 6107 MiB（model partitioned via deepspeed zero） |
| Trainer 构建 + dataset build_index | ✅ | 515,572 samples 注册成功 |
| dataloader fetch 第一 batch | ✅ | image_processor / DM0Tokenization 工作（含截断警告 token > 200） |
| **DM0 model.forward** | ❌ | **`AttributeError: 'NoneType' object has no attribute 'shape'`** |

## 2. Forward 失败根因（dexbotic 设计层面）

DM0 是 **dual-expert architecture**：
- VLM (Qwen3-1.7B) — 处理图像 + 文本
- Action Expert (Flow Matching, 300M) — 处理 action trajectory

**`dexbotic/model/dm0/dm0_arch.py:426`**:
```python
batch_size = actions.shape[0]  # ← 必需 actions tensor
```

forward 第一行就要求 `actions` 参数非 None。VL-only 数据（is_robot=false）jsonl 没有 `action` 字段，DexDataset/Collator 不构造该 key → model 收到 actions=None → 崩。

## 3. 论文 §3.2 mid-train 设计意图

论文 §3.2 明确说 mid-train：
> "Cross-embodiment robot data are mixed with vision-language and reasoning data; we also retain a portion of high-quality vision-language data from pretraining to preserve general multimodal capability."

即 **mid-train 是 VL + robot 混合训练**：
- robot 数据：has `action`/`state`/`is_robot=true`，参与 action expert 的 Flow Matching loss
- VL 数据：仅 LM head 的 cross-entropy loss，**仍需 dummy action tensor 占位**（model 实现要求）

DM0 hybrid gradient strategy（§2.2）：
> "for embodied data, gradients from the action expert are not backpropagated to the VLM to preserve generalized representations, while the VLM remains trainable on non-embodied data."

VL data 只更新 VLM，但 **forward 仍需走完 action expert**（哪怕 loss 不回传到 VLM）。

## 4. 实际可行的 warmup 方案

### 方案 A — Mixed dataset warmup（推荐）
混合我们 v6 的 4 源 + dexbotic 内置 robot 数据集（libero/calvin/maniskill2/robotwin/simpler）：

```python
dataset_name = (
    # 50GB VL（无 action）
    "dm0vlmid_cambrian737k+dm0vlmid_cambrian10m_filtered+"
    "dm0vlmid_llava_onevision+dm0vlmid_self_collected_proxy+"
    # 至少 1 个 robot 源（提供 action schema）
    "libero_goal+libero_object+libero_spatial"
)
```

robot 数据需要：
- 下载 libero dataset (`Dexmal/libero` ~10GB)
- 注册到 `dexbotic/data/data_source/libero_official.py`（已有内置）
- 跑 `compute_norm_stats` 算 action 归一化

### 方案 B — Forward-only smoke（极简）
不用 trainer.train()，自己写：

```python
# 1. load DM0 model
# 2. dataloader.next() 拿 1 batch
# 3. 给 batch 加 dummy actions: torch.zeros(bs, 50, 32)
# 4. model.forward(**batch) → 拿 logits / loss
# 5. 报告显存
```

### 方案 C — 包装 DataCollator
subclass `DataCollatorForSupervisedDataset` 在拼 batch 后注入 dummy fields：

```python
class WarmupCollator(DataCollatorForSupervisedDataset):
    def __call__(self, features):
        batch = super().__call__(features)
        bs = batch["input_ids"].shape[0]
        if "actions" not in batch:
            batch["actions"] = torch.zeros(bs, 50, 32)
            batch["states"] = torch.zeros(bs, 32)
            batch["image_masks"] = torch.ones(bs, 1, dtype=torch.bool)
        return batch
```

但 DM0 forward 仍会用这些 dummy 算 Flow Matching loss（含 0 actions 不有意义但不崩）。

## 5. 已就绪资产

| 资产 | 路径 |
|------|------|
| 50GB 数据 | `data/dm0_vl_midtrain_50gb/` (4 源 / 515,572 rows / 50.55GB) |
| data_source register | `data_source/dm0_vl_midtrain_50gb.py` (`dm0vlmid_*`) |
| index_cache 预计算 | `annotations/{source}/index_cache.json` |
| DM0-base checkpoint | `checkpoints/DM0-base/` (7.06GB) |
| deepspeed 0.14.4 | venv-py310 已装 |
| warmup 脚本 | `data/dm0_vl_midtrain_50gb/tools/warmup_dm0_vlonly.py` |
| dataloader benchmark | bs=32 nw=16: 69.7 samples/s ✅ codex round 7 阈值（≥50） |

## 6. 执行命令模板（mixed warmup）

```bash
# 步骤 1: 下 libero 数据
ssh chris-h200-2 'source ~/dexbotic/data/dm0_vl_midtrain_50gb/.progress/_shell_init.sh && \
  cd ~/dexbotic && \
  huggingface-cli download Dexmal/libero --repo-type dataset --local-dir data/libero --max-workers 8'

# 步骤 2: 修改 warmup 脚本 dataset_name 加入 libero
# 编辑 data/dm0_vl_midtrain_50gb/tools/warmup_dm0_vlonly.py:
#   dataset_name = "dm0vlmid_*+libero_goal+libero_object+libero_spatial"

# 步骤 3: 跑（需要 libero 的 norm stats，第一次会自动算）
cd ~/dexbotic
WANDB_DISABLED=true torchrun --nproc_per_node=8 \
  data/dm0_vl_midtrain_50gb/tools/warmup_dm0_vlonly.py 2>&1 | tee /tmp/dm0_mixed_warmup.log
```

## 7. 关键收获 / 下一步

✅ **已验证**：50GB VL 数据本身完整可用，dataloader 链路通过
⚠️ **未验证**：DM0 model forward + backward + optimizer step 完整一轮（受 mixed data 阻塞）
📌 **下一步**（用户决策）：
- 若需严格 100 steps warmup：选方案 A（下 libero ~10GB + 跑 mixed）— 约 1h
- 若仅需 sanity check：选方案 B（forward-only smoke）— 约 10min
- 若直接进入正式 mid-train：用户准备 robot data 后按论文配方混采

## 8. 已知警告（不影响）

- `Sliding Window Attention is enabled but not implemented for sdpa` — Qwen 内部，bf16 + sdpa attention 不支持 SWA，但 fall-back 到 eager 实现，影响速度不影响正确性
- `Token indices > 200` — 部分 conversations 太长被 DM0Tokenization 截断（model_max_length=200）
- `albumentations check_version timeout` — 库初始化时网络检查，不影响功能
