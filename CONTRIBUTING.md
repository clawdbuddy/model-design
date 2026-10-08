# 贡献指南

感谢你对 Model Design Agent 的关注！欢迎提交 Issue 和 Pull Request。

## 开发环境

环境统一用 Conda 管理：

```bash
make env                      # 创建 conda 环境 model-design
conda activate model-design
```

修改 `environment.yml` 后，用下面的命令同步已有环境：

```bash
conda env update -f environment.yml --prune
```

## 运行测试

本项目使用 pytest，测试位于 `tests/`：

```bash
make test                                    # 运行全部测试
make test-one T=tests/test_model_builder.py  # 运行单个测试文件
make test-one T=tests/test_naswot.py::test_score_is_finite  # 运行单个用例
```

提交前请确保 `make test` 全部通过。新增行为（尤其是新的评分器/提案器）请同步补充测试。

## 代码结构与扩展点

核心约定：所有组件之间以**架构规格**（JSON 层列表）为通用语言，提案器接口为
`generate(current_spec, num_candidates, feedback) -> [{"architecture_spec": ..., "strategy": ...}]`。

- **新增提案器**：在 `src/proposal/` 下新建模块实现上述接口，并在 `src/proposal/factory.py` 中接入；
- **新增评分器**：实现 `__call__(model, input_shape, batch_size) -> float`，在 `src/scoring/factory.py` 中接入，并在 `config/default.yaml` 的 `scoring.method` 中支持对应取值；
- **新增层类型**：扩展 `src/utils/model_builder.py` 的 `build_model`，并保证与 NASWOT 的 ReLU hook 兼容。

`src/run_search.py` 及 `src/proposal/*_adapter.py`、`src/scoring/naswot_adapter.py` 属于遗留/参考代码，新功能请落在 `src/search/agent.py` 这条主链路上。

## 代码风格

- 暂未配置 lint 工具，保持与现有代码一致的风格即可（4 空格缩进、类型注解用在公共接口上）；
- 注释可以使用中文，与现有文件保持一致；
- 配置项改动请同步更新 `config/default.yaml` 示例与 README 中的配置说明。

## 提交 Pull Request

1. 从 `main` 拉出分支，命名清晰（如 `feat/xxx`、`fix/xxx`）；
2. 一个 PR 聚焦一件事，附上改动说明和必要的测试；
3. 涉及行为/接口变化时，同步更新 README 与 CLAUDE.md；
4. PR 会经过 review，合并前请保持与 `main` 无冲突。
