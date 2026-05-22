# IDCIO · 组合干预推荐

`IDCIO`（Interaction-Decomposed Combined Intervention Optimization）回答这样一个问题：当多个干预可以**同时施加**、彼此存在交互效应时，应该给每个个体上哪几个干预、收益最大？

## 三阶段管线

```
① 原子效应估计    τ̂_k(x)        每个单一干预的效应
        ↓
② 交互效应学习    τ̂_{ij}(x)     干预 i、j 同时施加的额外增量
        ↓
③ QUBO 求解       z* ∈ {0,1}^K  最优组合（二元决策向量）
```

第三阶段把「选哪些干预」建模成一个 QUBO（二次无约束二值优化）问题：目标函数的对角项是原子效应、上三角项是交互效应，求解使总收益最大的 0/1 向量。

## 两种构造方式

### 配套 GACRNet

最常见的用法 —— 用训练好的 `GACRNet` 提供原子效应：

```python
from causalmt import GACRNet, IDCIO

est = GACRNet(num_treatments=3).fit(x, t_atomic, y)
rec = IDCIO.from_estimator(atomic_estimator=est)
```

此时 `IDCIO` 会自动复用 `GACRNet` 的原子效应预测与控制组基准 μ̂₀。

### 解耦使用

如果原子效应来自别的模型（例如 EconML），直接传效应数组：

```python
import numpy as np
rec = IDCIO.from_effects(atomic_effects=tau_array)   # tau_array: (n, K_atomic)
```

## 拟合交互效应

用**接受过组合干预**的样本训练交互项 MLP：

```python
rec.fit_interaction(
    x_combined, y_combined,
    treatment_pairs=[(0, 1)],   # 要拟合的交互对；None 则拟合全部 K(K-1)/2 对
)
```

!!! note "交互项标签如何构造"
    对干预对 `(i, j)`，交互项的训练目标为
    `y_{ij} = y_obs − μ̂_0 − τ̂_i − τ̂_j`，
    即从观测结果里剥掉控制组基准和两个原子效应后的残差。

## 推荐组合干预

```python
# 基本推荐：返回 (n, K) 的 0/1 矩阵
actions = rec.recommend(x_new, method="exhaustive")

# 带成本约束：costs 是各原子干预的单位成本
actions = rec.recommend(x_new, costs=np.array([1.4, 0.6]))
```

| `method` | 适用场景 |
|---|---|
| `"exhaustive"` | 穷举所有 2^K 组合，K ≤ 12 时精确最优 |
| `"greedy"` | 贪心启发式，K 较大时使用；`exhaustive` 在 K>12 时也会自动回退到它 |

## 不确定性估计

开启 MC Dropout 后，可在推荐时同时得到置信度：

```python
rec = IDCIO.from_estimator(atomic_estimator=est, use_mc_dropout=True, mc_samples=100)
rec.fit_interaction(x_combined, y_combined)

actions, confidence = rec.recommend(x_new, return_uncertainty=True)
# confidence: (n,) 每个样本推荐方案的稳定性，越接近 1 越可靠
```

`confidence` 通过多次 MC Dropout 采样、重新求解 QUBO，统计推荐结果与最终方案一致的比例得到。

## 完整示例

[`03_idcio_combined_recommendation.ipynb`](https://github.com/Norvain/causalmt/blob/main/examples/03_idcio_combined_recommendation.ipynb) 演示了从 `GACRNet` 训练到 `IDCIO` 推荐的全流程。

完整参数与方法签名见 [API 参考 · IDCIO](../api/idcio.md)。
