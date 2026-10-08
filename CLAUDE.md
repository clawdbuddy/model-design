# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

LLM 驱动的神经架构搜索（NAS）Agent。每轮搜索流程：**提案器（proposer）** 生成候选架构规格（JSON 层列表）→ `build_model` 将规格构建为 PyTorch 模型 → **NASWOT** 零训练评分器排序（ReLU 激活核矩阵的对数行列式）→ 最优规格反馈给下一轮提案。提案器有两种模式：`mock`（随机变异，无需联网）和 `api`（OpenAI chat completions）。

## 常用命令

Python 环境统一用 conda 管理（python 3.12）：`environment.yml` 是默认的 CPU 版 torch，`environment-cuda.yml` 是 CUDA 版（`make env-cuda`；`pytorch-cuda=12.4`，可按驱动改 11.8/12.1/12.6）。两者环境名同为 `model-design`：

```bash
# 创建环境（首次；之后 conda activate model-design）
make env            # CPU 版：conda env create -f environment.yml（并把 .env.example 拷贝为 .env）
make env-cuda       # CUDA 版：conda env create -f environment-cuda.yml

# 运行搜索
make run-mock                       # ./run.sh --mock
make run-api API_KEY=sk-...         # 或：export OPENAI_API_KEY=... && ./run.sh
./run.sh --mock                     # 直接调用（自动激活 conda 环境 model-design）
python -m src.main --mode mock --config config/default.yaml [--max-rounds 5 --candidates-per-round 3]

# 从自然语言需求生成配置 YAML（需要 OPENAI_API_KEY）
python generate_config.py --requirement "..." --output config/my.yaml

make clean                          # 清空 outputs/ 并删除 conda 环境
```

测试用 pytest，位于 `tests/`（模型构建、NASWOT、提案器、工厂、SearchAgent 闭环）：

```bash
make test                                              # 全部测试
make test-one T=tests/test_naswot.py::test_score_is_finite  # 单文件/单用例
python -m pytest tests/ -v                             # 已激活环境时直接用
```

lint 尚未配置。新增行为（尤其新评分器/提案器）请同步补测试。贡献约定见 `CONTRIBUTING.md`。

## 架构

核心搜索循环在 `src/search/agent.py`（`SearchAgent`）：由 `proposal.factory` + `scoring.factory` 构建组件，候选逐个评分并按轮写入检查点到 `outputs/`（`candidates/*.json`、`checkpoint.json`、`reports/final_report.json`）。

- **规格格式**（所有组件之间的通用语言）：`{"input_channels": int, "layers": [{"type": "Conv2d"|"DepthwiseConv2d"|"MaxPool2d"|"AvgPool2d"|"Linear", ...}]}` — `src/utils/model_builder.py` 将其编译为 `nn.Sequential`（Conv 层恒为 Conv+BN+ReLU 块）。提案器接口约定：`generate(current_spec, num_candidates, feedback) -> [{"architecture_spec": ..., "strategy": ...}]`。
- **可插拔工厂**：`src/proposal/factory.py` 根据 `mode`/`api_key` 选择 `MockProposer`（对最优规格做 exploit/explore 变异）或 `OpenAIProposer`（JSON mode 补全）；`src/scoring/factory.py` 目前只实现了 `naswot`（`src/scoring/naswot.py`）。新增提案器/评分器 = 新模块 + 对应工厂加一个分支。
- **配置**（`config/default.yaml`）：`constraints`（max_params、input_shape、num_classes、device）、`scoring`、`proposal`、`search` 四段。参数量超过 `max_params` 或模型构建失败的候选记为无效（得分 0）。`constraints.device` 经 `src/utils/device.py` 的 `resolve_device` 解析（cuda 不可用时回退 cpu），NASWOT 评分在该 device 上执行。
- **框架适配器**（`src/proposal/nngpt_adapter.py`、`lmsearcher_adapter.py`、`src/scoring/naswot_adapter.py`）：借鉴 NN-GPT 与 LM-Searcher 的提示词模板和 NCode 编码，是*风格适配器*，并不真正 import 那两个项目。只有 `src/run_search.py`（遗留入口，独立的 `ModelDesignWithLibs` 类，自有循环和 `results/` 输出）会用到它们 — 注意它以 `from scoring...`/`from proposal...` 导入，运行时必须把 `src/` 放进 `sys.path`。`naswot_adapter.py` 现状是坏的：它把 `sys.path` 指向 `./nas-without-training`（子模块实际在 `third_party/nas-without-training`），且 `from naswot import compute_naswot_score` 这个符号在该仓库里根本不存在（官方零样本评分 API 是 `scores.py` 中的 `hooklogdet` / `get_score_func`）。真正可用的评分器是 `src/scoring/naswot.py`。新的循环开发请以 `src/search/agent.py` 为准。
- **`generate_config.py`** 是独立的 LLM 工具：需求文本 → 符合上述四段式 schema 的 YAML 配置。

## 第三方依赖

参考用的子模块都在 `third_party/` 下：`NN-GPT`（ABrain-One，LLM 神经网络生成器）、`LM-Searcher`（Ashone3，NCode 统一编码的 LLM NAS）、`nas-without-training`（BayesWatch，NASWOT/synflow 零样本评分）。`src/` 中没有任何代码 import 它们 — 仅作为适配器的参考材料。`.gitmodules` 里还残留三个失效的根路径条目（`LM-Searcher`、`NN-GPT`、`nas-without-training`），磁盘上已不存在；编辑 `.gitmodules` 时可顺手删掉。`examples/` 与 `scripts/` 是空的占位目录。

`src/` 下的注释与 `generate_config.py` 的提示词本身就是中文的 — 编辑这些文件时保持一致风格即可。
