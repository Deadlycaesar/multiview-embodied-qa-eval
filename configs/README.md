# 配置文件导航

实验设计、请求量、分发和交付流程见根目录 `README.md`；精确字段和模型 revision 见 `PROTOCOL.md`。

- `experiments.mcq.json`：schema v4、模型注册表与分组、数据版本、提示/生成、A1/A2/B0/B1/B2/G0/G1。
- `experiments.open.json`：继承公共配置，替换为 188 题开放档和 `open_all_views`。
- `models/<key>.json`：模型 repo、40 位 revision、适配器、device/dtype、模板与 processor。
- `view-plans/mcq-v4.jsonl`：3,814 行固定视角清单，成员不得自行重抽。
- `metrics.json`：MCQ 解析、Token F1 与辅助指标。
- `judge.json`：固定 Jev；`judge.example.json` 是模板。

每个条件包含 `sample_set`、`views`、`mode`、`prompt`、`max_new_tokens`、`model_group`。配置按模型筛选条件，canonical experiments 用于跨模型统一协议身份。

密钥只走 `HF_TOKEN` 和 `JUDGE_API_KEY` 环境变量。`data/`、`weights/`、`runs/`、`.env` 和导出物不进入源码分发。`requirements/models/<key>.txt` 是独立参考环境，不等同于全平台实机验证。
