# 多视角具身问答 VLM 评测

数据集 [ee6008/multiview-embodied-qa-week9-yang](https://huggingface.co/datasets/ee6008/multiview-embodied-qa-week9-yang)（私有，DexYCB/H2O 多视角图像）× Qwen3-VL-4B-Instruct / Qwen3.5-4B，先跑通推理拿到 MCQ / 开放问答预测，再与 ground truth 计算指标。

## 数据与配置

两档配置共用同一套协议（贪心解码、64 token 上限、逐条件预热、显存/耗时测量）：

- `configs/experiments.mcq.json`：选择题档 `mvqa-local-v1`，280 题（DexYCB 8 视角 / H2O 5 视角），条件 white_mcq / single_mcq / dual_mcq / multi_mcq / multi_open。
- `configs/experiments.open.json`：开放问答档 `mvqa-open-v1`，188 题，条件 multi_open；不输入选项，参考答案取 `open_reference`。

模型配置在 `configs/models/`：`qwen3vl`（Qwen/Qwen3-VL-4B-Instruct）与 `qwen35`（Qwen/Qwen3.5-4B，已通过 `chat_template_kwargs` 关闭 thinking）。

## 使用

数据集为私有仓库，先设置 token（不要写入任何配置文件）：

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt
```

```bash
python run.py prepare --config configs/experiments.mcq.json --limit 5
python run.py run --config configs/experiments.mcq.json --model qwen3vl --run-dir runs/qwen3vl-mcq
python run.py run --config configs/experiments.open.json --model qwen3vl --run-dir runs/qwen3vl-open
python run.py score --run-dir runs/qwen3vl-mcq
python run.py analyze --run-dirs runs/qwen3vl-mcq runs/qwen35-mcq --output-dir runs/analysis
```

`run` 只生成回答；`score` 默认只算规则指标（选项准确率、格式合规、Token F1）。Judge / BERTScore 为可选项，加 `--judge-config configs/judge.json --bertscore` 启用。中断后用 `run --resume` 续跑。

注意：Qwen3.5-4B 的 `qwen3_5` 架构不在钉死的 transformers 4.57.x 中，运行前需升级到 git main 版本（见 requirements.txt 注释），并与 4.57.x 环境的结果分开报告。

## 目录约定

数据、权重和运行产物进入 `data/`、`weights/`、`runs/`，不提交 Git。所有相对路径以本目录为基准。
