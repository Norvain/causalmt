# GACRNet · 多值干预效应估计

`GACRNet`（GPS-Attention Causal Routing Network）把 DragonNet 从二元干预扩展到 **K 类干预**，估计每种干预相对控制组的因果效应。

## 它做什么

给定观测数据 `(x, t, y)`，`GACRNet` 学习一个能预测**潜在结果**的模型：对任意个体 `x`，预测它在每种干预 `k` 下的结果 `ŷ_k(x)`。有了潜在结果，就能算各种因果量：

- **CATE**（个体处理效应）：`τ(x) = ŷ_a(x) − ŷ_b(x)`
- **ATE**（平均处理效应）：CATE 在样本上的均值
- **GPS**（广义倾向得分）：个体接受各干预的概率

## 基本用法

```python
from causalmt import GACRNet

est = GACRNet(num_treatments=3)
est.fit(x, t, y)

po = est.predict_potential_outcomes(x_test)              # (n, K) 潜在结果
cate = est.estimate_cate(x_test, treatment_a=1, treatment_b=0)
ate = est.estimate_ate(x_test, treatment_a=1, treatment_b=0)
```

## 关键参数

### 网络结构

| 参数 | 取值 | 说明 |
|---|---|---|
| `backbone` | `"mlp"`（默认）/ `"transformer"` | 共享表示骨干网络 |
| `head` | `"uplift"`（默认）/ `"separate"` | `uplift` 用 μ₀+Δ_k 分解，对效应估计更稳；`separate` 每个干预独立 head |
| `routing` | `"gps"`（默认）/ `"self"` / `"concat"` / `"none"` | 干预路由注意力机制 |
| `hidden_dim` | 默认 `200` | 共享表示维度 |

!!! tip "routing 怎么选"
    `"gps"` 用广义倾向得分作为注意力 query，是论文的默认方案；样本量小时可试 `"none"` 关闭路由作对照。

### 正则化

`GACRNet` 内置三种正则，均可独立开关：

| 参数 | 默认 | 作用 |
|---|---|---|
| `use_mmd` / `mmd_weight` / `mmd_gammas` | `True` / `1.0` / `(0.1,1.0,10.0)` | 多尺度 MMD，平衡不同干预组的表示分布 |
| `use_tarreg` / `tarreg_ratio` | `True` / `1.0` | 目标正则化，提供双稳健的 ATE 估计 |
| `gps_weight` | `1.0` | GPS 分类损失权重 |
| `reg_l2` | `0.01` | outcome head 与 routing 的 L2 正则 |

### 训练

```python
est = GACRNet(
    num_treatments=3,
    epochs=300,
    batch_size=64,
    lr=1e-3,
    early_stopping_patience=30,   # 验证损失 30 轮不降则早停
    val_split=0.2,                # 未显式给验证集时，从训练集切 20%
    device="auto",                # auto: cuda > mps > cpu
    random_state=42,
)
```

显式指定验证集：

```python
est.fit(x_train, t_train, y_train, val_data=(x_val, t_val, y_val))
```

## 保存与加载

```python
est.save("gacrnet.pt")
est = GACRNet.load("gacrnet.pt", device="auto")
```

保存的文件同时包含模型权重、全部超参与训练历史，加载后可直接预测。

## 完整示例

仓库 `examples/` 下有两个端到端 notebook：

- [`01_ihdp_single_treatment.ipynb`](https://github.com/Norvain/causalmt/blob/main/examples/01_ihdp_single_treatment.ipynb) —— IHDP 标准基准
- [`02_multi_treatment_attribution.ipynb`](https://github.com/Norvain/causalmt/blob/main/examples/02_multi_treatment_attribution.ipynb) —— 多值干预归因

完整参数与方法签名见 [API 参考 · GACRNet](../api/gacrnet.md)。
