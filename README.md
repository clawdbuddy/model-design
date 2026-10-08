# Model Design Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20%2B%20CUDA-orange.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)

LLM 驱动的神经架构搜索（NAS）Agent：用大模型提出候选网络结构，用零训练指标自动评分，闭环迭代出满足约束的最优架构。无需训练数据、无需 GPU 即可完成一轮完整的架构搜索。

## 工作原理

每轮搜索按如下闭环推进：

```
┌────────────┐   架构规格(JSON)   ┌────────────┐   nn.Module   ┌────────────┐
│  提案器      │ ───────────────▶ │ 模型构建器   │ ────────────▶ │ NASWOT 评分 │
│ (Mock/API)  │                  │ build_model │               │ (零训练)     │
└────────────┘                  └────────────┘               └────────────┘
      ▲                                                             │
      └───────────────── 最优规格 + 历史反馈 ◀────────────────────────┘
```

1. **提案器** 根据当前最优架构与历史反馈，生成下一批候选架构规格（JSON 层列表）。
2. **模型构建器** 将规格编译为 PyTorch `nn.Sequential`（Conv 层恒为 Conv+BN+ReLU 块）。
3. **NASWOT 评分器** 在随机输入上前向一次，取 ReLU 激活核矩阵的对数行列式作为分数——完全不需要训练。
4. 参数量超预算或构建失败的候选判为无效；最优候选的规格进入下一轮反馈。

提案器两种模式：

- `mock` —— 随机变异（widen / deepen / shrink），离线可用，用于流程验证；
- `api` —— 调用 OpenAI Chat Completions（JSON mode），由 LLM 提出改进变体。

## 快速开始

环境要求：Linux，Conda。Python 环境由 conda 统一管理（python 3.12）：

```bash
# 1. 创建 conda 环境（首次；同时把 .env.example 拷贝为 .env）
make env          # CPU 版 PyTorch（默认，体积小）
make env-cuda     # CUDA 版 PyTorch（GPU 机器；可在 environment-cuda.yml 里选 CUDA 版本）
conda activate model-design

# 2. Mock 模式跑通全流程（无需 API key）
make run-mock
#    等价于：./run.sh --mock

# 3. API 模式（需要 OpenAI key）
vim .env                        # 填入 OPENAI_API_KEY
make run-api                    # 或：./run.sh --api-key sk-...
```

直接调用入口（已激活 conda 环境时）：

```bash
python -m src.main --mode mock --config config/default.yaml \
    --max-rounds 5 --candidates-per-round 3
```

清理实验产物并删除 conda 环境：

```bash
make clean
```

## 测试

项目使用 pytest，测试位于 `tests/`（覆盖模型构建、NASWOT 评分、提案器、工厂装配与完整搜索闭环）：

```bash
make test                                          # 运行全部测试
make test-one T=tests/test_model_builder.py        # 单个测试文件
make test-one T=tests/test_naswot.py::test_score_is_finite  # 单个用例
```

已激活 conda 环境时也可直接：`python -m pytest tests/ -v`。

## 从需求描述生成配置

`generate_config.py` 可以把一句自然语言需求交给 LLM，生成符合本项目 schema 的 YAML 配置（需要 `OPENAI_API_KEY`）：

```bash
python generate_config.py \
    --requirement "设计一个 10 万参数以内的 CIFAR-10 图像分类模型" \
    --output config/my.yaml
```

## 配置说明

配置文件为 YAML，共四段（参见 `config/default.yaml`）：

```yaml
constraints:              # 搜索约束
  max_params: 500000      # 参数量上限，超出判为无效
  input_shape: [3, 32, 32]
  input_channels: 3
  num_classes: 10
  device: "cpu"

scoring:
  method: "naswot"        # 目前仅实现 naswot
  batch_size: 32

proposal:
  style: "nngpt"          # 提示词风格
  temperature: 0.7

search:
  max_rounds: 5
  candidates_per_round: 3
  early_stop_patience: 3
```

`constraints.device` 支持 `"cpu"` 或 `"cuda"`：配置为 `cuda` 时若当前机器没有可用 GPU，会自动回退到 CPU，同一份配置在有/无 GPU 的机器上都能直接跑。

所有组件之间以**架构规格**为通用语言：

```json
{
  "input_channels": 3,
  "layers": [
    {"type": "Conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
    {"type": "Conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
    {"type": "MaxPool2d", "kernel_size": 2},
    {"type": "AvgPool2d"},
    {"type": "Linear", "out_features": 10}
  ]
}
```

支持的层类型：`Conv2d`、`DepthwiseConv2d`、`MaxPool2d`、`AvgPool2d`、`Linear`。

## 运行输出

搜索过程中的产物统一写入 `outputs/`：

| 路径 | 内容 |
|---|---|
| `outputs/candidates/*.json` | 每个候选的规格、得分、参数量、有效性 |
| `outputs/checkpoint.json` | 按轮保存的检查点（当前最优、历史长度） |
| `outputs/reports/final_report.json` | 搜索结束后的汇总报告 |

## 项目结构

```
src/
├── main.py                 # 入口：解析参数、驱动搜索
├── search/agent.py         # SearchAgent：核心搜索循环
├── proposal/               # 提案器（factory 可插拔）
│   ├── mock.py             #   MockProposer：随机变异
│   ├── openai_adapter.py   #   OpenAIProposer：LLM 提案
│   └── nngpt_adapter.py / lmsearcher_adapter.py  # 提示词风格参考
├── scoring/                # 评分器（factory 可插拔）
│   └── naswot.py           #   NASWOT 零训练评分
└── utils/
    ├── model_builder.py    # 规格 JSON → nn.Sequential
    ├── io.py               # 配置/结果读写
    └── logger.py
config/default.yaml         # 默认配置
generate_config.py          # 需求 → YAML 配置的 LLM 工具
run.sh / Makefile           # 运行与环境管理
```

扩展新的提案器或评分器：新增模块 + 在对应 `factory.py` 中加一个分支即可；提案器需实现 `generate(current_spec, num_candidates, feedback) -> [{"architecture_spec": ..., "strategy": ...}]`。

## 第三方参考

`third_party/` 下以 git submodule 引入了三个参考项目（本仓库代码不直接 import，仅供适配器设计参考）：

- [NN-GPT](https://github.com/ABrain-One/NN-GPT) —— LLM 神经网络生成器
- [LM-Searcher](https://github.com/Ashone3/LM-Searcher) —— NCode 统一编码的 LLM NAS
- [nas-without-training](https://github.com/BayesWatch/nas-without-training) —— NASWOT 等零样本架构评分

首次拉取子模块：

```bash
git submodule update --init --recursive
```

## 贡献

欢迎 Issue 和 Pull Request！开发环境搭建、测试要求、扩展新的提案器/评分器等约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

本项目基于 [MIT License](LICENSE) 开源。
