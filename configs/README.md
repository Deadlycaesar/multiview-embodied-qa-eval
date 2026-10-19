# 配置与实验协议

两档实验配置：`experiments.mcq.json`（选择题，自包含）与 `experiments.open.json`（开放问答，继承前者、替换 dataset 与 experiments 段）。`models/` 保存模型配置；`metrics.json` 保存评分规则。每次运行选一个模型，两个模型共用同一份样本与条件设置。

## 数据集（MVQA）

`datasets/mvqa.py` 读取私有 HF 数据集 `ee6008/multiview-embodied-qa-week9-yang`，用 `HF_TOKEN` 认证。配置字段：

- `revision`：40 位数据集 commit，钉死版本。
- `subset`：`mvqa-local-v1`（280 题选择题）或 `mvqa-open-v1`（188 题开放问答，引用 local 档图片）。
- `questions_sha256`：questions.jsonl 内容哈希，下载后校验。
- `snapshot_dir`：可选，指向本地数据集副本则离线读取（测试用），默认 null 走 `hf_hub_download`。
- `offset` / `count`：题目切片。

选项按列表顺序映射为 A、B、C…，数量不固定；开放档 `choices` 为空，推理只输入图片与题干。`per_view` 标注（Y/P/N）转为 `necessary_view_indices`（Y 的视角），供视角缺失分析使用。图片按列表顺序即视角顺序，逐张校验并记录 SHA-256。

## 条件

| 条件 | 图像 | 回答 | 格式提示 |
|---|---|---|---|
| white_mcq | 一张生成白图 | 选择题 | 有 |
| single_mcq | 第一张 | 选择题 | 有 |
| dual_mcq | 前两张 | 选择题 | 有 |
| multi_mcq | 全部 | 选择题 | 有 |
| multi_open | 全部 | 开放问答 | 无 |

选择题把全部实际选项拼入题目文本；格式要求为 `Answer with only the label of one of the provided options. Do not explain.`，追加到题目末尾。开放问答仅提供题干。

## 模型

- `qwen3vl`：Qwen/Qwen3-VL-4B-Instruct，transformers 4.57.x 原生支持。
- `qwen35`：Qwen/Qwen3.5-4B，需要 transformers git main；`chat_template_kwargs: {"enable_thinking": false}` 关闭思考模式，否则 64 token 预算会被思考内容耗尽。

两者默认 CUDA / bfloat16 / eager，图像像素范围 256×28² 至 1024×28²。每个条件预热一次，CUDA 每题测量前清理未使用缓存并重置峰值。

## 评分口径

| 指标 | 约定 |
|---|---|
| 格式合规 | 去除首尾空白后必须恰好等于该题的一个实际标签，大小写保持一致 |
| 选项准确率 | 单字母、明确的 answer/option/choice 表达、开头带标点的字母，或完整选项文本；多候选保守标为歧义 |
| Token F1 | casefold 后按 Unicode `\w+` 分词、词频重合，多参考取最大值 |
| Judge / BERTScore | 可选；未传 `--judge-config` 不调用服务，未传 `--bertscore` 不加载评分模型，配置见 `judge.example.json` 与 `metrics.json` |

规则指标以计划题数为分母，失败/缺失记 0，并报告完成数。预测与评分分开版本化存储，可直接取预测文本与 ground truth 自行计算后续指标。
