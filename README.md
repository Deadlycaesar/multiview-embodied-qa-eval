# 多视角具身问答：统一实验、分发与交付框架

本仓库是当前唯一的正式实验标准。目标不是简单做一个榜单，而是回答：**模型是否真正依赖图像？减少视角能否降低计算成本？选对视角比随机视角好多少？**

当前协议为 **v4**，不再导入旧预测，也不沿用此前 white/single/dual 条件。机器配置与本文、[`configs/PROTOCOL.md`](configs/PROTOCOL.md) 同步维护。

## 1. 最终实验设计

### 1.1 主实验与附录

主实验使用 **DexYCB S1 的 190 题**，每题统一 8 个视角，保证视角预算严格可比。H2O 的 40 题单独作为附录泛化，不混入主曲线。

- **S0**：全部 280 道 MCQ，只用于 A1 整体无图检查。
- **S1**：DexYCB 且 `type != 4`，190 题，主实验。
- **S2**：S1 中至少有一个非 Y 视角的题，166 题，用于删除 Y 后仍能输入非空视角。
- **H2O**：全部 H2O 题，40 题，每题 5 视角，附录。
- **OPEN**：开放问答档，188 题。

### 1.2 条件、模型覆盖与请求量

| 条件 | 样本与设置 | 运行模型 | 每模型请求数 |
|---|---|---|---:|
| `a1_no_image` | S0；真正零图片 | 全部 10 个 | 280 |
| `b0_all_views` | S1；全部 8 视角 | 全部 10 个 | 190 |
| `a2_remove_evidence` | S2；删除全部 Y | 4 个代表模型 | 166 |
| `a2_remove_random` | S2；随机删除同数量，seed=0/1/2 | 4 个代表模型 | 498 |
| `b1_random_views` | S1；随机 k=1/2/4，各 3 种子 | 4 个代表模型 | 1,710 |
| `b2_evidence_views` | S1；证据优先 k=1/2/4 | 4 个代表模型 | 570 |
| `g0_all_views` | H2O；全部 5 视角 | 4 个代表模型 | 40 |
| `g1_random_views` | H2O；随机 k=1/2/4，各 3 种子 | 4 个代表模型 | 360 |
| `open_all_views` | OPEN；全视角、无选项 | 全部 10 个 | 188 |

四个视角分析代表模型为：

```text
molmo2, qwen35, qwen3vl, minicpmv4
```

分层原因：全部模型形成统一基础榜单，代表模型完成成本较高的视角消融；不为了“所有模型跑满”而缺少关键的 B2 上限对照。

请求总量：

- 6 个基础榜单模型：每个 **470 MCQ + 188 Open = 658**。
- 4 个代表模型：每个 **3,814 MCQ + 188 Open = 4,002**。
- 全部合计：`6 × 658 + 4 × 4,002 = 19,956`，不含预热与冒烟。

### 1.3 每个实验回答什么

- **A1**：不看图能答多少？若接近全视角，说明题目或选项存在泄漏/先验。
- **A2**：删除 Y 是否比随机删除同数量视角更伤？用于验证证据标注。
- **B0**：统一代码下，全视角的准确率与成本基线。
- **B1**：随机少看时的准确率–token 下限曲线。
- **B2**：按 Y > P > N 优先选图时的确定性上限曲线。B2 使用标注，只作 oracle 对照，不是可部署选图方法。
- **G0/G1**：结论在 H2O 上是否出现同样趋势，不与 DexYCB 混合计算。
- **Open**：全视角自由生成能力；Token F1 辅助，Jev 为正式裁判。

主图对四个代表模型绘制：A1 的 S1 零点、B1 随机曲线、B2 证据优先曲线、B0 全视角点。

## 2. 统一配置

### 2.1 数据与模型

数据仓库：`ee6008/multiview-embodied-qa-week9-yang`。

```text
revision: 41d03c57350430d2a55274308e8488b21f28db9a
MCQ subset: mvqa-local-v1
Open subset: mvqa-open-v1
```

10 个模型的 repo、完整 40 位 revision 和参考环境见 [`configs/PROTOCOL.md`](configs/PROTOCOL.md) 与 `configs/models/`。每模型使用独立虚拟环境，不要求所有模型共用 Transformers 版本。

公共默认：`device=cuda`、`dtype=bfloat16`、`seed=0`。MiniCPM-V-4 固定 `sdpa`；模型自己的聊天模板、processor 和必要的 `trust_remote_code` 只在适配器内处理。

### 2.2 提示与生成

MCQ 输入严格为：

```text
<question>
A. <option A>
B. <option B>
...

Answer with only the label of one of the provided options. Do not explain.
```

统一设置：

```json
{
  "generation": {"do_sample": false, "num_beams": 1},
  "runtime": {"warmup_per_condition": 1},
  "max_new_tokens": 64
}
```

- 选项保持源顺序，不 shuffle。
- Open 不显示选项或参考答案。
- 预热使用独立通用请求，不回答 benchmark 题。
- A1 不创建白图、不调用图像 processor，`visual_tokens=0`。

### 2.3 固定视角清单

文件：`configs/view-plans/mcq-v4.jsonl`，共 **3,814 行**。

```json
{"qid":"000001","experiment":"B1","condition":"b1_random_views","k":2,"seed":1,"cameras":[0,5]}
```

- 相机编号为 **0-based**，输入按索引升序。
- B1/G1 无放回随机，seed=0/1/2。
- A2 随机对照与 evidence 删除相同数量。
- B2 按 Y > P > N、同级按原索引排序，取前 k 后再按索引升序输入。
- 随机数由协议版本、qid、condition、k、seed 的 SHA-256 派生。
- 成员不得各自重新随机；只有协议负责人可通过 `tools/build_view_plans.py` 重建清单。

### 2.4 输出与评分

MCQ 每次保存：

- 原始回答与解析字母；
- qid、condition、k、seed、cameras；
- 首生成 token 的 `option_probs`，口径 `first-token-label-softmax-v1`；
- input/visual/output token、总耗时、生成耗时、CUDA 峰值显存、截断和错误状态。

评分口径：

- MCQ 规则准确率为主，失败/缺失计入分母；同时报完成率、格式合规和解析状态。
- Open 的 Token F1 使用 casefold + Unicode 词频重合；Jev 只看文字题干、参考答案和预测。
- Jev 固定于 `configs/judge.json`：`model=jev`、`temperature=0`、JSON 输出，密钥走 `JUDGE_API_KEY`。
- 更换 Jev 配置必须整批重判；未全覆盖时不发布正式开放题准确率或排名。
- 跨机器比较准确率和视觉 token；耗时/显存只在同模型、同硬件、同环境下比较。

## 3. 分发方式

### 3.1 框架负责人分发

通过 Git 分支或源码包分发以下内容：

```text
configs/                 实验、模型、指标、Jev 与固定视角清单
core/ datasets/ protocols/ models/ metrics/ analysis/
requirements/            公共与各模型参考环境
tests/                   离线测试
run.py README.md
```

不分发：数据集图片、模型权重、`.env`、密钥和本地 runs。成员自行获得私有数据权限并下载模型。

### 3.2 成员执行

在仓库根目录，将 `<key>` 替换为分配的模型：

```bash
export HF_TOKEN=hf_...
pip install -r requirements/models/<key>.txt

python run.py prepare --config configs/experiments.mcq.json --model <key>
python run.py preflight --config configs/experiments.mcq.json --model <key> --experiments a1_no_image b0_all_views --limit 1

python run.py run --config configs/experiments.mcq.json --model <key> --run-dir runs/<key>-mcq-v4
python run.py run --config configs/experiments.open.json --model <key> --run-dir runs/<key>-open-v4

# 中断续跑
python run.py run --config configs/experiments.mcq.json --model <key> --run-dir runs/<key>-mcq-v4 --resume

python run.py score --run-dir runs/<key>-mcq-v4
python run.py score --run-dir runs/<key>-open-v4
python run.py validate --run-dir runs/<key>-mcq-v4 --complete
python run.py validate --run-dir runs/<key>-open-v4
```

配置会根据模型自动选择分配条件；非代表模型显式请求 B1/B2 等条件会直接报错，避免误跑。切片冒烟可用 `--limit 1 --offset N`（offset 必须配合 limit）；正式全量运行不得添加这些切片参数。

### 3.3 成员提交

**正式提交以完整 run 目录为主**，因为复核、Jev 和 `analyze` 需要其中的 requests、samples、checkpoints、manifest 和环境信息：

```text
runs/<key>-mcq-v4/
runs/<key>-open-v4/
```

run 目录不包含数据集图片或模型权重。不要只提交自己整理的 CSV，也不要修改原始预测。

另外生成便于阅读和归档的导出包：

```bash
python run.py export --run-dir runs/<key>-mcq-v4 --output-dir runs/<key>-mcq-bundle --complete
python run.py export --run-dir runs/<key>-open-v4 --output-dir runs/<key>-open-bundle
python run.py validate --bundle --run-dir runs/<key>-mcq-bundle --complete
```

导出包包含 manifest、环境、模型元数据、predictions/scores JSONL 和 summary；不完整结果只能标记为开发产物。

### 3.4 汇总者交付

1. 校验每个 run 的数据、协议哈希、条件、题数、概率和内容完整性。
2. 对所有 open run 使用同一份 `configs/judge.json` 和同一 Jev 凭据统一评分。
3. Jev 完成后重新 validate/export；不能混用不同裁判结果。
4. MCQ 与 Open 分开聚合：

```bash
python run.py analyze --run-dirs runs/*-mcq-v4 --output-dir runs/analysis-mcq-v4 --plots
python run.py analyze --run-dirs runs/*-open-v4 --output-dir runs/analysis-open-v4
```

最终交付包括：源码与配置、模型验证状态、各模型 run/导出包、Jev 覆盖情况、`overview.csv`、`a2.csv`、`b1.csv`、`b2.csv`、`g1.csv`、主图和汇总报告。未跑通或未完成的项目必须明确列出，不能从榜单中静默删除。

## 4. 测试与验收

离线检查：

```bash
python -m unittest discover -s tests -v
python -m compileall -q analysis core datasets metrics models protocols tests tools run.py
```

每个模型正式运行前必须完成：A1 零图、多图 MCQ、多图 Open 三条真实冒烟；检查选项概率覆盖、视觉 token、原始输出和导出。仅“代码已适配”不等于实机验收通过。

本次交付验证状态：

- 离线测试 **29/29 通过**，语法编译和 `git diff --check` 通过。
- `qwen3vl`：v4 的 A1/A2/B0/B1/B2/G0/G1/Open 真实冒烟通过；评分、导出、校验和主图通过。
- `qwen35`：v4 的 A1/B0/B1/B2 真实冒烟通过。
- 两个 Qwen 已完成跨模型、跨依赖环境联合汇总，protocol hash 一致。
- 其余 8 个模型已完成配置和静态适配，尚未下载权重实机验证。
- Jev 配置与模拟接口测试通过；真实服务仍需 `JUDGE_API_KEY` 做连通性和全量评分。

尚未执行 19,956 次全量正式实验。完整实验结论只能来自完整且校验通过的 v4 run。
