# 快速上手

本页用最短的代码跑通 `causalmt` 的两个核心组件。假设你已经 `pip install causalmt`。

## 数据约定

`causalmt` 全程使用三个数组描述观测数据：

| 变量 | 形状 | 含义 |
|---|---|---|
| `x` | `(n, d)` | 协变量（特征） |
| `t` | `(n,)` | 干预索引，取值 `0..K-1`，其中 `0` 约定为控制组 |
| `y` | `(n,)` | 观测到的结果 |

## GACRNet：估计多值干预效应

```python
import numpy as np
from causalmt import GACRNet

# 准备数据（这里用随机数据演示，实际换成你的数据）
rng = np.random.default_rng(0)
x = rng.normal(size=(500, 8)).astype("float32")
t = rng.integers(0, 3, size=500)           # 3 类干预：0/1/2
y = (x[:, 0] + t * 0.7).astype("float32")

# 训练
est = GACRNet(num_treatments=3, epochs=100, verbose=False)
est.fit(x, t, y)

# 估计因果效应
ate = est.estimate_ate(x, treatment_a=1, treatment_b=0)   # 干预 1 相对 0 的平均效应
cate = est.estimate_cate(x, treatment_a=1)                 # 每个个体的效应 (n,)
po = est.predict_potential_outcomes(x)                     # K 种干预下的潜在结果 (n, 3)

print(f"ATE(1 vs 0) = {ate:.3f}")
```

## IDCIO：推荐最优干预组合

当多个干预可以**同时施加**，要回答「给每个个体上哪几个干预收益最大」时，用 `IDCIO`。

```python
from causalmt import IDCIO

# 复用上面训练好的 GACRNet 作为原子效应来源
rec = IDCIO.from_estimator(atomic_estimator=est)

# 用接受过组合干预的样本拟合交互效应
rec.fit_interaction(x_combined, y_combined, treatment_pairs=[(0, 1)])

# 推荐：返回 (n, K) 的 0/1 矩阵，1 表示推荐该干预
actions = rec.recommend(x_new, method="exhaustive")

# 带成本约束的推荐：costs 是各原子干预的单位成本
actions = rec.recommend(x_new, costs=np.array([1.4, 0.6]))
```

## 评估

如果有反事实真值（如 IHDP 基准数据集），可以评估估计质量：

```python
from causalmt.metrics import pehe, ate_error

print("PEHE:", pehe(cate_pred, cate_true))
print("ATE error:", ate_error(cate_pred, ate_true=true_ate))
```

## 接下来

- [GACRNet 指南](guide/gacrnet.md) —— backbone、routing、正则化等参数怎么调
- [IDCIO 指南](guide/idcio.md) —— 三阶段管线原理、不确定性估计
- [API 参考](api/gacrnet.md) —— 每个方法的完整签名与参数表
