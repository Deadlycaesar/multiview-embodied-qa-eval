# 实验配置说明（v4）

本文与机器可执行配置一一对应。执行源为 `experiments.mcq.json`、`experiments.open.json`、`models/*.json`、`view-plans/mcq-v4.jsonl`、`metrics.json` 和 `judge.json`。实验设计、分发和交付流程见根目录 `README.md`。

## 1. 数据

数据集：`ee6008/multiview-embodied-qa-week9-yang`（私有，`HF_TOKEN` 认证）。

- revision：`41d03c57350430d2a55274308e8488b21f28db9a`
- MCQ subset：`mvqa-local-v1`，280 题
- MCQ questions SHA-256：`e435680010b5d6ea6247e41c4bd5ce6bba890058bca8df52a3a04c2bcdf147fd`
- Open subset：`mvqa-open-v1`，188 题
- Open questions SHA-256：`2162bb32184c97fff586083bc2c17640da2aaba252298233fb4381aee6cdd54f`

样本集：

- S0：全部 MCQ，280 题。
- S1：`dataset == "dexycb"` 且 `type != 4`，190 题。
- S2：S1 中 Y 数小于原图数，166 题。
- H2O：`dataset == "h2o"`，40 题。
- OPEN：开放档全部 188 题。

## 2. 模型注册表与分组

所有 revision 均为不可变 40 位 commit。参考环境不是实机验证声明。

| model key | Hugging Face repo | revision | 参考环境 |
|---|---|---|---|
| `gemma4` | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | `requirements/models/gemma4.txt` |
| `phi4` | `microsoft/Phi-4-multimodal-instruct` | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | `requirements/models/phi4.txt` |
| `molmo2` | `allenai/Molmo2-4B` | `042abfa7a38879a376cec03d949eff0aefaa0600` | `requirements/models/molmo2.txt` |
| `lfm25` | `LiquidAI/LFM2.5-VL-3B` | `35a118d938ce6d123ac2d371649f24a8efb69058` | `requirements/models/lfm25.txt` |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `482adb537c021c86670beed01cd58990d01e72e4` | `requirements/models/smolvlm2.txt` |
| `qwen35` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `requirements/models/qwen35.txt` |
| `qwen3vl` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | `requirements/models/qwen3vl.txt` |
| `internvl35` | `OpenGVLab/InternVL3_5-4B` | `481f6e32467eab4e922ccd7fd6cf420441a62331` | `requirements/models/internvl35.txt` |
| `minicpmv4` | `openbmb/MiniCPM-V-4` | `6f69a8235885be89608f3a71ffd40695b379e1e7` | `requirements/models/minicpmv4.txt` |
| `sailvl2` | `BytedanceDouyinContent/SAIL-VL2-2B` | `3c8b3be75dd51ef77b55e5149e7dbdebd2bbe4c5` | `requirements/models/sailvl2.txt` |

`model_groups.all` 为上述 10 个；`model_groups.view_analysis` 为 `molmo2`、`qwen35`、`qwen3vl`、`minicpmv4`。

默认 `device=cuda`、`dtype=bfloat16`、seed=0。MiniCPM-V-4 必须 `sdpa`；Qwen3.5/Gemma 4 关闭 thinking；Qwen processor 固定 `min_pixels=200704`、`max_pixels=802816`。

## 3. 条件配置

每个条件字段为：

```json
{
  "sample_set": "S1",
  "views": "plan",
  "mode": "mcq",
  "prompt": "mcq",
  "max_new_tokens": 64,
  "model_group": "view_analysis"
}
```

MCQ 为 `views=plan`，Open 为 `views=all`。配置按模型自动筛选条件，同时保留完整 canonical experiments 计算跨模型一致的 `protocol_hash`。

| 条件 | 样本 | 视角 | model_group | 请求数 |
|---|---|---|---|---:|
| `a1_no_image` | S0 | 零图 | all | 280 |
| `a2_remove_evidence` | S2 | 删除全部 Y | view_analysis | 166 |
| `a2_remove_random` | S2 | 随机删除同数量，3 种子 | view_analysis | 498 |
| `b0_all_views` | S1 | 全部 8 视角 | all | 190 |
| `b1_random_views` | S1 | 随机 k=1/2/4，各 3 种子 | view_analysis | 1,710 |
| `b2_evidence_views` | S1 | Y>P>N 证据优先 k=1/2/4 | view_analysis | 570 |
| `g0_all_views` | H2O | 全部 5 视角 | view_analysis | 40 |
| `g1_random_views` | H2O | 随机 k=1/2/4，各 3 种子 | view_analysis | 360 |
| `open_all_views` | OPEN | 全部视角、无选项 | all | 188 |

总量：基础模型 658 次/模型，代表模型 4,002 次/模型，总计 19,956 次。

## 4. 提示与生成

MCQ：

```text
<question>
A. <option A>
B. <option B>
...

Answer with only the label of one of the provided options. Do not explain.
```

- prompt version：`answer-format-v3-variable-options`
- placement：`user_suffix`
- `do_sample=false`、`num_beams=1`、`max_new_tokens=64`
- 每条件预热一次
- Open 只给原始题干，不给选项、答案或格式提示

## 5. 视角计划

`view-plans/mcq-v4.jsonl` 共 3,814 行，字段为 `qid, experiment, condition, k, seed, cameras`。

- cameras 为 0-based 索引，输入按索引升序。
- B1/G1 无放回随机，seed=0/1/2。
- A2 random 与 evidence 在每题、每种子下删除相同数量。
- B2 按 Y>P>N、同级按索引排序，取前 k；仅作 oracle 上限。
- 随机种子由协议字符串、seed、qid、condition、k 的 SHA-256 派生。
- 清单由 `tools/build_view_plans.py` 生成；成员只消费，不自行修改。

## 6. 测量与评分

- `option_probs`：首生成 token 位置，对实际选项单 token logits 做 softmax；版本 `first-token-label-softmax-v1`。
- `visual_tokens`：processor 展开后的视觉嵌入占位 token 数；A1 为 0。
- 同时保存 input/output token、端到端/generate 耗时、CUDA allocated/reserved 峰值、截断和状态。
- MCQ 规则准确率分母为计划请求数，失败/缺失记 0。
- Open Token F1 为 casefold 后 Unicode `\w+` 词频重合。
- Jev 为开放题正式文字 Judge，`model=jev`、`temperature=0`、JSON 输出，密钥只读 `JUDGE_API_KEY`。
- 不同硬件不直接比较耗时或显存排名。

## 7. 运行产物与身份

- `protocol_hash`：数据、完整 canonical 条件、模型分组、视角清单、提示和生成协议，跨模型一致。
- `run_fingerprint`：另含选中模型、代码、环境、请求和图像字节。
- checkpoint 支持原子写入与 `--resume`；输入、代码或环境变化后拒绝混跑。
- 正式提交完整 run 目录；`export` 另外生成 predictions/scores JSONL、summary、环境和模型元数据包。
- `validate` 检查题数、唯一性、概率、A1 零图、内容哈希和完成度。
- 不完整结果不得进入正式榜单；Open 还需 Jev 全覆盖。
