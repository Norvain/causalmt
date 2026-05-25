# causalmt

面向多值干预决策的因果表示学习与组合干预推荐工具包。基于毕业论文方法实现，提供 sklearn 风格的简洁 API。

📖 在线文档：<https://norvain.github.io/causalmt/>

- **`GACRNet`** —— 多值干预效应估计，扩展 DragonNet 到 K 类干预
- **`IDCIO`** —— 交互效应分解的组合干预推荐（原子效应 → 交互效应 → QUBO 求解）

## 安装

```bash
pip install causalmt
```

开发模式安装（含测试与示例依赖）：

```bash
git clone https://github.com/Norvain/causalmt.git
cd causalmt
pip install -e ".[dev,examples]"
```

依赖：`torch>=2.0`、`numpy`、`pandas`、`scikit-learn`、`tqdm`。支持 Python 3.9–3.12，自动选择 CUDA / MPS / CPU。

## 快速上手

### 1. `GACRNet` —— 多值干预效应估计

```python
import numpy as np
from causalmt import GACRNet

# x: (n, d) 协变量   t: (n,) 干预索引 0..K-1   y: (n,) 观测结果
est = GACRNet(num_treatments=3, routing="gps", use_mmd=True, use_tarreg=True)
est.fit(x_train, t_train, y_train)

ate = est.estimate_ate(x_test, treatment_a=1, treatment_b=0)   # 平均处理效应
cate = est.estimate_cate(x_test, treatment_a=1)                # 个体处理效应 (n,)
y_potential = est.predict_potential_outcomes(x_test)           # (n, K) 反事实
gps = est.predict_gps(x_test)                                  # (n, K) 广义倾向得分

est.save("gacrnet.pt")
est = GACRNet.load("gacrnet.pt")
```

### 2. `IDCIO` —— 组合干预推荐

在原子干预效应基础上估计交互项，并通过 QUBO 求解每个样本的最优组合。

```python
import numpy as np
from causalmt import GACRNet, IDCIO

# 路径 A：配套 GACRNet（自动复用其原子效应与控制组基准）
rec = IDCIO.from_estimator(atomic_estimator=est)

# 路径 B：解耦使用，接受任意外部原子效应数组 (n, K_atomic)
rec = IDCIO.from_effects(atomic_effects=tau_array)

# 用接受组合干预的样本拟合交互效应
rec.fit_interaction(x_combined, y_combined, treatment_pairs=[(0, 1)])

# 推荐最优组合：costs 为各原子干预单位成本数组 (K_atomic,)
actions = rec.recommend(x_new, costs=np.array([1.4, 0.6]), method="exhaustive")
# actions: (n, K_atomic) 0/1 矩阵，1 表示推荐该干预
```

带不确定性的推荐（需 `use_mc_dropout=True`）：

```python
rec = IDCIO.from_estimator(atomic_estimator=est, use_mc_dropout=True)
rec.fit_interaction(x_combined, y_combined)
actions, confidence = rec.recommend(x_new, return_uncertainty=True)
# confidence: (n,) 每个样本推荐方案的稳定性置信度
```

## 数据加载与评估

```python
from causalmt.data import load_ihdp, load_multi_attribution
from causalmt.metrics import pehe, ate_error, policy_risk

# IHDP 标准基准（二元干预）；仓库内数据位于 data/ihdp/
train, test = load_ihdp("data/ihdp", slice_idx=0)

# EconML 多值干预归因数据；接受本地路径或 URL，自动缓存到 ~/.causalmt_cache/
ds = load_multi_attribution("multi_attribution_sample.csv", split="atomic")

est = GACRNet(num_treatments=train.num_treatments).fit(train.x, train.t, train.y)
cate_pred = est.estimate_cate(test.x)
print("PEHE:", pehe(cate_pred, test.cate_true))
print("ATE error:", ate_error(cate_pred, ate_true=test.ate_true))
```

仓库内提供 IHDP 示例数据；`load_multi_attribution` 接受本地路径或 URL，下载结果缓存到
`~/.causalmt_cache/`（可用环境变量 `CAUSALMT_CACHE_DIR` 覆盖）。

## API 速查

| 对象 | 用途 |
|---|---|
| `GACRNet` | 多值干预效应估计器 |
| `GACRNet.fit(x, t, y, *, val_data=None)` | 训练 |
| `.estimate_ate(x, *, treatment_a, treatment_b)` | 平均处理效应（标量） |
| `.estimate_cate(x, *, treatment_a, treatment_b)` | 个体处理效应 `(n,)` |
| `.predict_potential_outcomes(x)` | K 种干预下潜在结果 `(n, K)` |
| `.predict_gps(x)` | 广义倾向得分 `(n, K)` |
| `.save(path)` / `GACRNet.load(path)` | 模型持久化 |
| `IDCIO.from_estimator(atomic_estimator, **kw)` | 配套 GACRNet 构造 |
| `IDCIO.from_effects(atomic_effects, **kw)` | 解耦外部效应构造 |
| `.fit_interaction(x, y, *, treatment_pairs=None)` | 训练交互效应 MLP |
| `.recommend(x, *, costs=None, method, return_uncertainty=False)` | 组合干预推荐 |
| `load_ihdp` / `load_multi_attribution` | 数据加载 |
| `pehe` / `ate_error` / `policy_risk` | 评估指标 |

`GACRNet` 关键参数：`num_treatments`、`backbone`（`"mlp"｜"transformer"`）、
`head`（`"uplift"｜"separate"`）、`routing`（`"gps"｜"self"｜"concat"｜"none"`）、
`use_mmd`、`use_tarreg`、`epochs`、`batch_size`、`lr`、`device`、`random_state`。
完整参数见 `help(GACRNet)`。

## 示例

仓库 `examples/` 下提供 2 个端到端 notebook（已嵌入运行输出）：

| Notebook | 内容 |
|---|---|
| `01_ihdp_single_treatment.ipynb` | IHDP 标准基准（二元干预） |
| `02_multi_treatment_attribution.ipynb` | EconML multi_attribution 多值干预与 IDCIO 组合推荐 |

## 引用

```bibtex
@thesis{xu2026causalmt,
  title  = {面向多值干预决策的因果表示学习与交互效应分解优化方法研究},
  author = {xuhaoli},
  year   = {2026},
}
```

## License

[MIT](LICENSE)
