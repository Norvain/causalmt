# causalmt

面向多值干预决策的**因果表示学习**与**组合干预推荐**工具包。基于毕业论文方法实现，提供 sklearn 风格的简洁 API。

---

## 这个库解决什么问题

传统因果推断方法（如 DragonNet）多聚焦**二元干预**（干预 / 不干预）。但现实决策中常常面对**多种可选干预**，而且多个干预可以**组合施加**、彼此存在交互效应。`causalmt` 提供两个核心组件分别应对这两个场景：

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } __`GACRNet` · 多值干预效应估计__

    ---

    扩展 DragonNet 到 K 类干预，集成 GPS 路由注意力、多尺度 MMD 表示正则、目标正则化，估计每种干预的因果效应。

    [:octicons-arrow-right-24: GACRNet 指南](guide/gacrnet.md)

-   :material-sitemap:{ .lg .middle } __`IDCIO` · 组合干预推荐__

    ---

    交互效应分解三阶段管线：原子效应估计 → 交互效应学习 → QUBO 组合优化求解，给出每个个体的最优干预组合。

    [:octicons-arrow-right-24: IDCIO 指南](guide/idcio.md)

</div>

## 安装

```bash
pip install causalmt
```

依赖 `torch>=2.0`、`numpy`、`pandas`、`scikit-learn`、`tqdm`。支持 Python 3.9–3.12，自动选择 CUDA / MPS / CPU。

## 一瞥

```python
from causalmt import GACRNet, IDCIO

# 多值干预效应估计
est = GACRNet(num_treatments=3, routing="gps", use_mmd=True)
est.fit(x_train, t_train, y_train)
ate = est.estimate_ate(x_test, treatment_a=1, treatment_b=0)

# 组合干预推荐
rec = IDCIO.from_estimator(atomic_estimator=est)
rec.fit_interaction(x_combined, y_combined)
actions = rec.recommend(x_new, method="exhaustive")
```

## 下一步

- 第一次使用 → [快速上手](quickstart.md)
- 估计某种干预的因果效应 → [GACRNet 指南](guide/gacrnet.md)
- 推荐最优干预组合 → [IDCIO 指南](guide/idcio.md)
- 查每个接口的参数 → [API 参考](api/gacrnet.md)

## 引用

```bibtex
@thesis{xu2026causalmt,
  title  = {面向多值干预决策的因果表示学习与交互效应分解优化方法研究},
  author = {xuhaoli},
  year   = {2026},
}
```

本项目以 [MIT License](https://github.com/Norvain/causalmt/blob/main/LICENSE) 开源。
