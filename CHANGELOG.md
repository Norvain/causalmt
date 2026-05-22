# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-05-22

首个公开版本。

### 新增

- `GACRNet`：多值干预因果效应估计器，扩展 DragonNet 到 K 类干预。
  - GPS 路由注意力（`routing="gps" | "self" | "concat" | "none"`）
  - 多尺度 MMD 表示正则（`use_mmd`、`mmd_gammas`）
  - 目标正则化的双稳健 ATE 估计（`use_tarreg`）
  - MLP / Transformer 两种 backbone，uplift / separate 两种 outcome head
  - sklearn 风格 API：`fit` / `predict_potential_outcomes` / `estimate_ate` /
    `estimate_cate` / `predict_gps` / `save` / `load`
- `IDCIO`：交互效应分解的组合干预推荐管线。
  - 三阶段：原子效应估计 → 交互效应 MLP → QUBO 求解
  - 两种构造方式：`from_estimator`（配套 GACRNet）、`from_effects`（解耦外部效应）
  - QUBO 求解器：穷举（`exhaustive`，K≤12）与贪心（`greedy`）
  - 基于 MC Dropout 的推荐置信度估计
- 数据加载器：`load_ihdp`、`load_multi_attribution`，支持本地路径或 URL 自动下载缓存。
- 评估指标：`pehe`、`ate_error`、`policy_risk`。
- 3 个端到端示例 notebook（IHDP 单干预、多值干预归因、IDCIO 组合推荐）。
- 86 个测试覆盖网络、估计器、QUBO、IDCIO 与数据加载。

[0.1.0]: https://github.com/Norvain/causalmt/releases/tag/v0.1.0
