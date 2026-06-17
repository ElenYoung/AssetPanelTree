# P-Tree Algorithm Specification

> 本文档是 P-Tree / P-Forest / P-Boost 三种算法的**规范性说明**：
> 重点回答「算法定义是什么、为什么这样定义、产物如何使用」。
> API 调用细节见 `README.md`；工程优化与里程碑见 `OPTIMIZATION_PLAN.md`。
>
> 推荐阅读顺序：§0 术语表 → §1 问题设定 → §2 单棵 P-Tree → §3 切分准则 → §4 P-Forest → §5 P-Boost → §6 三者对照。

---

## 0. Glossary（术语表）

文档中出现的关键缩写与术语在此一次性约定，正文不再重复定义。

| 术语 | 全称 / 含义 | 备注 |
|------|-------------|------|
| **Panel data** | 面板数据 | 同时具有截面维（资产 $i$）与时间维（时刻 $t$）的二维数据。 |
| **Cross-section** | 截面 | 固定一个时刻 $t$、跨所有资产的一行数据。 |
| **Regime** | 区制 / 状态 | 一段在统计性质（如可预测性强弱）上相对一致的 $(t,i)$ 子集。P-Tree 找的就是「可预测性 regime」。 |
| **Mosaic**（马赛克） | 叶子 × 时间 的热力图 | 把每个叶子在每个时间截面上的指标（如逐期 $R^2$）排列出来，用以观察「模型在何时何处灵 / 失效」。 |
| **Block bootstrap** | 时间块自助 | 把时间轴切成长度为 `block_size` 的连续块，对**块**做有放回抽样。保留收益的时序自相关，是面板数据替代逐样本 bootstrap 的标准做法。 |
| **OOB（Out-Of-Bag）** | 袋外 | 在一次 bootstrap 抽样中**未被抽中**的样本。在 P-Forest 里具体指未被任何 block 选中的时间块所对应的所有 $(t,i)$ 行。用作免费的 hold-out 验证集。 |
| **Honest split** | 诚实切分 | 节点内部把样本随机划成「fit 集」与「eval 集」：fit 集用来拟合叶子模型，eval 集用来评估切分质量。目的是消除「在同一批样本上既挑选又评估」带来的偏差。 |
| **Co-association / Consensus matrix** | 共识 / 协同关联矩阵 | $C_{ij}\in[0,1]$，表示样本 $i,j$ 落入同一叶子的树占比。森林产物之一，对应一种鲁棒的「监督相似度」。 |
| **`regime_membership`** | 区制隶属概率 | 森林产物之一，对每个样本输出一个 $[0,1]$ 的标量，含义是「有多少比例的树把它路由进了该树的高指标叶子集合」。详见 §4.3.b。 |
| **SDF** | Stochastic Discount Factor（随机贴现因子） | 资产定价里的核心对象。本文档只在 `MeanVarianceCriterion` 一节使用其「最大夏普 = 切线组合」的等价形式。 |
| **Adjusted-$R^2$** | 调整 $R^2$ | $R^2_{\text{adj}}=1-(1-R^2)\frac{n-1}{n-p-1}$，把回归维度 $p$ 当作复杂度惩罚。 |
| **`ccp_alpha`** | Cost-Complexity Pruning 系数 | 后剪枝的复杂度惩罚强度，越大剪得越狠。 |
| **Learning rate $\nu$ (shrinkage)** | 收缩率 | P-Boost 每一轮预测的乘子，$\nu\in(0,1]$；越小越正则化、越慢收敛。 |

---

## 1. Problem Setup

### 1.1 数据与记号

面板数据由 $(t,i)$ 单元构成，每个单元一条观测：

$$
\big(\, x_{t,i}\in\mathbb{R}^{p},\ \ y_{t,i}\in\mathbb{R}\,\big),
\qquad t=1,\dots,T,\quad i=1,\dots,N_t.
$$

- $x_{t,i}$：$p$ 维**特征向量**（characteristics / 因子），如动量、估值、波动率、规模；可拼接宏观状态变量。
- $y_{t,i}$：**目标变量**，金融场景下通常是下一期收益率 $r_{t+1,i}$。在分类任务里取 $\{0,1\}$。

总样本量 $n=\sum_t N_t$。一个"模型"就是一个映射 $f:\mathbb{R}^p\to\mathbb{R}$，在每个截面上施加于所有资产。

### 1.2 截面秩标准化（cross-sectional rank standardisation）

P-Tree 的切分阈值需要在不同时刻具有可比的语义。原始特征量纲会随时间漂移（例如波动率水平整体上升），用全局常数阈值会失真。`DataHandler` 在**每个时间截面内**把每个特征转为秩并映射到 $[0,1]$：

$$
\tilde{x}_{t,i,k}\;=\;\frac{\operatorname{rank}_{t}(x_{t,i,k})-1}{N_t-1}\;\in[0,1],
\qquad k=1,\dots,p.
$$

这样阈值 $c=0.5$ 恒等于「该时刻该特征的截面中位数」，跨期语义稳定；下文「固定阈值列表」「自适应分位」「时间块自助」均依赖此性质。

### 1.3 与普通决策树的本质区别

| 维度 | 标准 CART | **P-Tree** |
|------|-----------|------------|
| 切分目标 | 最小化子节点 MSE / Gini | **最大化两子节点的可预测性差异**：如 $\lvert R^2_L-R^2_R\rvert$ |
| 叶子模型 | 常数（叶均值） | **岭回归 / Logit**：每个叶子内拟合一个局部线性因子模型 |
| 目的 | 点预测 | 识别「可预测性集中在何时、何种资产上」的 **regime** |
| 主要产物 | 单一预测 | 预测 + 预测马赛克 + 可解读的分区规则 |

一句话：P-Tree **不是为了把误差降到最低**，而是为了找出「哪一片 $(时间,资产)$ 区域里因子模型特别有效」。这一点决定了它的切分准则、集成方式都与经典树不同。

---

## 2. The Single P-Tree

### 2.1 叶子区域与局部模型

树把全样本递归划分为互不相交的叶子区域 $\{\mathcal R_1,\dots,\mathcal R_M\}$。每个 $\mathcal R_m$ 由一串「特征-阈值」规则定义，例如

$$
\mathcal R_m=\big\{(t,i):\ \tilde x_{\cdot,1}\ge 0.5\ \wedge\ \tilde x_{\cdot,3}<0.7\big\}.
$$

每个叶子内部拟合一个**局部岭回归**（leaf model）：

$$
\hat\beta_m
=\arg\min_{\beta}\ \sum_{(t,i)\in\mathcal R_m} w_{t,i}\,(y_{t,i}-x_{t,i}^\top\beta)^2
\;+\;\alpha\lVert\beta\rVert_2^2
\;=\;\big(X_m^\top W_m X_m+\alpha I\big)^{-1}X_m^\top W_m y_m.
$$

权重 $w_{t,i}$ 可取逆波动率（`VolWeightedRidge`），缓解异方差。**预测时**一个观测沿树下行到所属叶子 $m$，由该叶子的 $\hat\beta_m$ 给出 $\hat y=x^\top\hat\beta_m$。

> **闭式解 + 可加充分统计量。** 岭回归只依赖 $A_m=X_m^\top W_m X_m$（$p\times p$）与 $b_m=X_m^\top W_m y_m$（$p$）。$A,b$ 对样本可加，于是父节点的 $(A,b)$ 等于左右子节点之和。评估候选切分时，只对较小一侧直接做 $O(p^2)$ 矩阵乘得到 $(A_{\text{small}},b_{\text{small}})$，较大一侧用 $A_{\text{large}}=A_{\text{parent}}-A_{\text{small}}$ 纯减法得到，结果与"两侧都重算"逐位一致。

### 2.2 贪心生长算法

P-Tree 用自顶向下贪心生长，结构与 CART 相同，仅切分准则不同：

```text
function GROW(node):
    if 停止条件(node):              # 见下方
        标记为叶子, 拟合 β_m, return
    best_score = -∞
    for 每个特征 k in features(node):           # 可由 max_features 随机子集化
        for 每个候选阈值 c in thresholds(k):     # 三种来源见下方
            L = {样本: x_k <  c};  R = {样本: x_k >= c}
            if min(|L|,|R|) < min_samples: continue
            拟合 β_L, β_R (岭回归闭式解)
            m_L = metrics(L);  m_R = metrics(R)   # 含 R² / precision / ...
            score = criterion(m_L, m_R)            # §3
            if score > best_score: 记录 (k*, c*)
    if best_score < min_impurity_decrease:
        标记为叶子, return
    按 (k*, c*) 切分; 递归 GROW(left), GROW(right)
```

**停止条件**：达到 `max_depth`、节点样本数 `< min_samples`、或最优 score `< min_impurity_decrease`。

**候选阈值的三种来源**（由参数 `splitter` / `split_thresholds` 控制）：

1. **固定列表**（默认 `[0.3, 0.5, 0.7]`）：因秩标准化后特征 $\in[0,1]$，固定分位即有跨期一致的含义。
2. **自适应分位** `split_thresholds="adaptive"`：用本节点该特征的实际分位点（如 25/50/75 百分位），对非均匀分布更稳。
3. **随机阈值** `splitter="random"`：在节点内 $[\min,\max]$ 范围随机取 `n_random_splits` 个点（Extra-Trees 风格），主要服务于集成多样性（见 §4.2）。

### 2.3 预测路径

给定一个新观测 $x$：

1. 从根节点出发，按每个内部节点保存的 $(k^*,c^*)$ 比较 $\tilde x_{k^*}$ 与 $c^*$，向左 / 向右下行；
2. 到达叶子 $m$ 时，调用 $\hat y=x^\top\hat\beta_m$（回归）或 $\hat p=\sigma(x^\top\hat\beta_m)$（分类）。

因此 P-Tree 的预测函数是**逐叶分段线性**的：路径给出一条可读的 if-else 规则，叶子给出一个局部线性模型。

### 2.4 Prediction Mosaic（预测马赛克）

把每个叶子在每个时间截面上的表现指标（如逐期 $R^2$、precision、收益）排成"叶子 × 时间"的热力图，得到**预测马赛克**。它直观回答：

- **何时**模型整体灵 / 失效（横向看一行）；
- **哪类资产**长期可预测（纵向看一列叶子）。

这是 P-Tree 区别于黑箱预测器的可解释产物，也是后文 P-Forest 「软马赛克 / 共识矩阵」的源头。

### 2.5 可选稳健化

#### 2.5.1 Honest Split

**问题。** 标准 P-Tree 在同一批样本上既**枚举挑选**切分点、又**评估**切分质量。"挑出最大 $\lvert R^2_L-R^2_R\rvert$"这一步天然高估了切分的真实优度（selection bias / 数据窥探）。

**Honest split**（Athey & Imbens, 2016）的做法：在每个节点内部，把样本按比例 `honest_frac` 随机（或按时间前后）划成两份：

| 子集 | 用途 |
|------|------|
| **fit 集** | 拟合左右两侧叶子的 $\hat\beta_L,\hat\beta_R$ |
| **eval 集** | 用拟合好的 $\hat\beta$ 计算 $R^2_L,R^2_R$ 与切分分 |

即「选切分的样本 ≠ 评估切分的样本」，从根上消除"两次复用同一批数据"的乐观偏差。`honest_refit_full=True` 可在树长好后用全样本重训末端叶子，以恢复预测精度。

与全局 train/validation 的关系：honest 切分是**每个节点内部**各自再分一次，专门治"树结构选择"层面的过拟合；它与外层 train/val 切分**可叠加**。

#### 2.5.2 Cost-Complexity Pruning（后剪枝）

贪心生长容易长出过深、过拟合的树。`cost_complexity_pruning_path()` 引入复杂度惩罚 `ccp_alpha`，自底向上比较子树的累计 score 增益与叶子数惩罚：

$$
\text{保留子树}\iff\ \text{Gain(子树)}\ >\ \texttt{ccp\_alpha}\cdot\big(|\text{leaves(子树)}|-1\big).
$$

净收益为负的子树被剪掉。函数返回一串 $(\alpha,\#\text{leaves},\text{score})$，配合外层验证集即可选出折中点。

---

## 3. Splitting Criteria

切分准则把左右子节点的评估指标 $(m_L,m_R)$ 映射成一个**切分质量分** $\text{score}=\texttt{criterion}(m_L,m_R)$，越大越好。准则是 P-Tree 的"灵魂"，因为它把"什么叫好分裂"显式写进了算法。

> **统一约定。** 每个准则对象暴露一个 `metric_key()` 方法，返回它用来评估叶子质量的主指标名（回归 → `"r2"`，分类 → `"precision"` / `"f1"` / `"auc"` / `"logloss"`）。后文 §4 / §5 的输出函数会读取这个键，从而**任务无关地**适配回归与分类。

### 3.1 R²Diff

$$
\boxed{\ \text{score}=\lvert R^2_L-R^2_R\rvert\ },\qquad
R^2_{\bullet}=1-\frac{\sum w(y-\hat y)^2}{\sum w(y-\bar y)^2}.
$$

**直觉。** 我们要的不是「两边都拟合得好」，而是「**两边的可预测性显著不同**」——一侧高 $R^2$（强 regime，alpha 集中）、另一侧低 $R^2$（近噪声）。把高/低预测性区域**分开**本身就是有价值的发现。

可选 `weight_by_size=True` 用 $\min(n_L,n_R)/(n_L+n_R)$ 惩罚极不平衡切分。

### 3.2 Weighted R²Diff

小样本叶子的 $R^2$ 容易虚高（过拟合），让贪心被"假 regime"误导。`WeightedR2DiffCriterion` 在保留核心的同时叠加两个稳健项：

$$
\text{score}=\lvert R^2_L-R^2_R\rvert\;
\cdot\;\underbrace{\frac{\min(n_L,n_R)}{n_L+n_R}}_{\text{平衡项}}
\;\cdot\;\underbrace{\frac{n_{\min}-1}{n_{\min}-1+\kappa}}_{\text{样本量收缩}}.
$$

可选使用 adjusted-$R^2$（见 §0）替代 $R^2$，把回归维度 $p$ 纳入复杂度惩罚，进一步抑制"多因子 + 少样本"刷分。

### 3.3 Mean-Variance / SDF

论文《Growing the Efficient Frontier on Panel Trees》的真正目标是**组合层面**的：让切分出的 regime 能张出更靠外的均值-方差有效前沿。`MeanVarianceCriterion` 据此定义：

1. 每个子叶子用预测值在每个截面上构造一个**预测加权、截面去均值**的多空组合，得到一条收益时序 $\{r^L_t\}$、$\{r^R_t\}$；
2. 取两条时序的均值向量 $\mu\in\mathbb{R}^2$ 与协方差矩阵 $\Sigma\in\mathbb{R}^{2\times 2}$；
3. 切分分 = 两组合可达到的**切线组合（最大）夏普**（年化）：

$$
\text{score}=\sqrt{\mu^\top \Sigma^{-1}\mu}\;\cdot\;\sqrt{A},\qquad A=\text{年化因子}.
$$

**直觉。** 分高意味着左右两个 regime 的多空组合**互补**（低相关、各自含 alpha），合在一起把有效前沿推得更远——这正是"组合视角的可预测性差异"，比单看 $R^2$ 更贴近最终交易目标。

使用前提：调用 `fit(..., time_index=...)` 让引擎能按时间分组计算组合收益。

### 3.4 Classification

二分类（如涨跌方向）用 `ClassificationCriterion(metric=...)`，把 §3.1 中的 $R^2$ 替换为分类指标的左右差异：

| `metric` | 切分分定义 | 备注 |
|----------|-----------|------|
| `"precision"` | $\lvert\text{prec}_L-\text{prec}_R\rvert$ | 最常用，等价于「找一边精度集中」 |
| `"f1"` | $\lvert F1_L-F1_R\rvert$ | 兼顾召回 |
| `"auc"` | $\lvert\text{AUC}_L-\text{AUC}_R\rvert$ | 用 Mann–Whitney U 的秩公式 $O(n\log n)$ 实现，避免双循环 |
| `"logloss"` | $\lvert\text{logloss}_L-\text{logloss}_R\rvert$ | 注意 logloss 越小越好，但作"差的绝对值"时仍是越大越好 |

`metric_key()` 返回所选指标名，§4 的 forest 会用它来定义"高指标叶子集合"。

---

## 4. P-Forest（Bagging Ensemble）

> 对应「单棵决策树 ↔ 随机森林」。源码：`ensemble.PanelForest`。

### 4.1 为什么 P-Tree 特别需要 Bagging

P-Tree 的贪心步骤"在 $p\times|\text{thresholds}|$ 个候选里挑 $\lvert R^2_L-R^2_R\rvert$ 最大者"是一个**高方差的离散选择**：数据轻微扰动就可能让胜出的 $(k^*,c^*)$ 翻转，从而得到**完全不同的分区**。这正是 bagging（自助聚合）最能发挥威力的场景——用扰动生成多棵去相关的树，再在**输出层**聚合以降方差。

### 4.2 集成什么、不集成什么

> 关键点：P-Tree 的准则 $\lvert R^2_L-R^2_R\rvert$ **不是一个可被平均的损失**，直接对它做平均没有数学意义。

破解之道是区分单棵 P-Tree 的两类输出，分别用不同方式聚合：

| 输出 | 形态 | 聚合方式 |
|------|------|----------|
| **预测** $\hat y_b(x)$ | 连续数值 | 直接平均（袋装均值） |
| **分区**（样本落到哪个叶子） | 离散结构 | 共识 / 协同（co-association） |

由此得出**P-Forest 的设计准则**：

> 每棵树内部的切分准则保持 $\lvert R^2_L-R^2_R\rvert$（或 §3 其他准则）**不变**；集成只发生在「分区 / 预测 / 组合」这一**输出层**。

### 4.3 两类面板专属扰动

**(a) 样本扰动 = 时间块自助（block bootstrap）。**
不能照搬普通 RF 的逐样本 bootstrap——那会打散截面、制造前视信息。P-Forest 把排好序的唯一时间标签切成长度为 `block_size` 的连续块，对**块**做有放回抽样：

$$
\{1,2,\dots,T\}\to \text{blocks}=\big[\{1..b\},\{b{+}1..2b\},\dots\big],\quad
b^{(1)},\dots,b^{(n_{\text{blk}})}\stackrel{\text{i.i.d.}}{\sim}\text{Unif(blocks)}.
$$

- 被抽中的块（**可重复**，等价于上调权重）拼接成该树的**训练集**；
- **从未被抽中**的块拼接成该树的**袋外 (OOB) 集**。

按块抽样保留了收益的时序自相关结构，这是面板场景的关键。`block_size` 通常应不小于目标的自相关阶。

**(b) 特征扰动 = 节点级随机特征子集。**
每个节点只在随机的 `max_features` 个特征里搜索切分（默认 $\sqrt p$）。这进一步强制不同树发现不同的弱 regime，降低树间相关性。

### 4.4 三类聚合输出

记森林由 $B$ 棵树组成。第 $b$ 棵树把观测 $x$ 路由到的叶子记作 $\ell_b(x)$，该树对 $x$ 的预测记作 $\hat y_b(x)$。下面三种输出对应森林的三种使用姿势。

#### (a) Bagged Prediction —— `predict(X)`

$$
\hat y_{\text{forest}}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat y_b(x).
$$

平均抵消单树贪心切分带来的预测方差。

- **回归任务**：返回连续预测。
- **分类任务**：返回平均的 $\hat P(y=1\mid x)$；同名 `predict_proba(X)` 是一个语义更清晰的别名（回归森林上调用会报错）。

```python
yhat = forest.predict(X)           # 回归或 P(y=1|X)
p1   = forest.predict_proba(X)     # 仅分类
```

#### (b) Regime Membership —— `regime_membership(X)`

**这是 P-Forest 最容易被误解的产物，单独说清楚。**

**一句话直观含义。** 对每个样本 $x$，输出一个 $[0,1]$ 的标量，表示「有多少比例的树把它路由进了'高指标叶子'」。把单树脆弱的 0/1 硬规则升级为森林共识下的**软概率**。

**形式化定义。** 第 $b$ 棵树的"高指标叶子集合"是该树中主指标（即 `criterion.metric_key()`，回归为 $R^2$，分类为 precision / F1 / AUC）高于其叶子中位数的叶子：

$$
\mathcal H_b\;=\;\Big\{\,\ell\ :\ \texttt{metric}_b(\ell)\ \ge\ \operatorname{median}_{\ell'\in\text{leaves}(b)}\ \texttt{metric}_b(\ell')\,\Big\}.
$$

于是 regime 隶属概率为：

$$
\widehat P(\text{high-predictability regime}\mid x)
\;=\;\frac{1}{B}\sum_{b=1}^{B}\mathbf 1\big[\ell_b(x)\in\mathcal H_b\big]\ \in[0,1].
$$

- **值域**：$[0,1]$。
- **与单棵 P-Tree 的对应关系**：单树用一条硬规则路径（如 $x_1>0.7\wedge x_3<0.3$）判定"高可预测"；森林对应地用 `regime_membership(X) > τ`（如 $\tau=0.7$）做**软阈值化**。两者的语义完全平行，只是判定从单一脆弱规则变成 $B$ 棵树的投票共识。
- **与 mosaic 的关系**：可视作单棵 P-Tree 硬马赛克（0/1）的**软化版本**——森林版的"软马赛克"。
- **与 co-association 的区别**：`regime_membership` 是**逐样本**标量（"这个样本是否高可预测"）；`coassociation_matrix` 是**成对**相似度（"哪些样本一致地共属同一 regime"），二者用途不同，详见 (c)。

**两种排序方式（`regime_aggregation` 参数）。** 排"高指标叶子"时，主指标可以在两种样本上计算：

| `regime_aggregation` | 含义 | 取舍 |
|----------------------|------|------|
| `"train"`（默认） | 用每棵树自己 bootstrap-train 集的指标 | 便宜、确定；但训练集指标在噪声面板上偏乐观 |
| `"oof"` | 在每棵树的 OOB 行上重算指标 | 更稳健；当某棵树的 bootstrap 覆盖了全部时间块时退化为 train |

**怎么用。**

```python
forest.fit(X, y, feature_names, time_index="date")
p = forest.regime_membership(X)           # 每个样本一个 [0,1] 概率
X_high = X[p > 0.7]                       # 筛"高可预测样本"
```

若仍需可读规则，可拟合一棵浅**代理树（surrogate tree）**把森林的软标签翻译回 if-else：

```python
label = (forest.regime_membership(X) > 0.7).astype(int)
# 再用一棵浅 PanelTree 或 sklearn DecisionTree 拟合 X -> label,
# 读出近似规则 "x1 > ?, x3 < ?"
```

#### (c) Co-association / Consensus Matrix —— `coassociation_matrix(X)`

$$
C_{ij}\;=\;\frac{1}{B}\sum_{b=1}^{B}\mathbf 1\big[\ell_b(x_i)=\ell_b(x_j)\big]\ \in[0,1].
$$

即"样本 $i,j$ 落入同一叶子的树占比"。

- 这把森林里许多脆弱的硬分区融合成一个**鲁棒的相似度矩阵**，回答"哪些 $(t,i)$ 单元*一致地*共享同一可预测性 regime"。
- 它本质上是**任务无关**的（只依赖分区，不依赖叶子模型），因此回归森林与分类森林产出的 $C$ 同构。
- 可作为预计算 affinity 喂给谱聚类，得到稳定的"元 regime"。
- **注意**：$C$ 是 $O(n^2)$ 内存。请控制 $n$。

### 4.5 OOB Evaluation —— `oob_score_`

「袋外」(Out-Of-Bag, OOB) 指的是单棵树在其自身 bootstrap 中**没抽到**的时间块上的所有 $(t,i)$ 行。OOB 给出**无需额外 hold-out 的泛化估计**：

1. 对每个样本 $i$，找出所有把 $i$ 视为 OOB 的树，仅用这些树对它打分，取平均得到一条 OOB 预测 $\hat y_i^{\text{oob}}$。
2. 在所有有 OOB 预测的样本上计算指标：
   - **回归**：OOB $R^2$；
   - **分类**：`criterion.metric_key()` 指定的指标（precision / F1 / AUC；`logloss` 取反号以保持"越大越好"）。

这就是 `forest.oob_score_` 的含义。它与 train/test split 的直觉等价（每个样本都在某些树眼里属于"测试集"），只是不浪费数据。

### 4.6 与经典 Random Forest 的对照

| 维度 | 随机森林 (RF) | **P-Forest** |
|------|---------------|--------------|
| 基学习器 | CART（MSE / Gini） | P-Tree（R²Diff 等） |
| 样本扰动 | 逐样本 bootstrap | **时间块** block bootstrap |
| 特征扰动 | 节点随机子集 | 节点随机子集（相同思想） |
| 主要产物 | 平均预测 | 平均预测 **+ regime 隶属概率 + 共识矩阵** |
| 降的是 | 预测方差 | 预测方差 **+ 分区方差** |

### 4.7 适用范围与局限

- **回归 / 分类一体化。** 当 `base_params["criterion"]` 是回归准则（`R2DiffCriterion` / `WeightedR2DiffCriterion` / `RankICDiffCriterion` / `MeanVarianceCriterion`），叶子默认使用 `RidgeRegressor`；当是 `ClassificationCriterion(...)`，叶子默认使用 `RidgeLogitClassifier` 并自动启用 `predict_proba`。`regime_membership` / `coassociation_matrix` / `oob_score_` 的行为在两种任务下都成立——`regime_membership` 把"高 $R^2$ 叶子"自动换成"高 metric 叶子"，其它逻辑逐字不变。
- **Mean-Variance / SDF 准则只对回归收益序列有定义**，分类下不适用。
- **何时 Forest 的增益有限。** 若真实可预测性由**单一强特征**主导（例如演示数据里的 `char_1`），所有树都会在它上面分裂 → 高度相关 → 森林增益微弱，甚至可能略低于单树（随机特征子集反而牺牲了最优分裂）。**P-Forest 的收益主要来自多个、弱识别的特征共同驱动可预测性**的场景；此时随机特征子集让不同树发现不同弱 regime，聚合才显著获益。`example/demo_pforest.py` 特意构造了这种场景。

---

## 5. P-Boost（Residual Boosting Ensemble）

> 对应「单棵决策树 ↔ 梯度提升树」。源码：`ensemble.BoostedPanelTree`。

### 5.1 提升什么：目标，而不是准则

最常见的误解是去"提升 $\lvert R^2_L-R^2_R\rvert$ 这个准则值"——它不可加、不是损失，提升它没有意义。

**P-Boost 提升的是目标变量本身（残差）**：每一轮把上一轮已经解释掉的收益成分扣除，让下一棵 P-Tree 在**残差**上重新寻找可预测性 regime。每棵子树**内部的切分准则仍是 R²Diff，保持不变**。

### 5.2 算法：前向分步残差提升

$$
\begin{aligned}
&F_0(x)=0\\
&\textbf{for }m=1,\dots,M:\\
&\qquad r^{(m)}_{t,i}=y_{t,i}-\nu\,F_{m-1}(x_{t,i})
   &&\text{(残差，}\nu=\text{learning\_rate)}\\
&\qquad T_m=\texttt{PanelTreeEngine}_{R^2\text{Diff}}.\text{fit}\big(X,\ r^{(m)}\big)
   &&\text{(在残差上长一棵浅 P-Tree)}\\
&\qquad F_m(x)=F_{m-1}(x)+T_m(x)\\
&\textbf{return }\hat y(x)=\nu\sum_{m=1}^{M}T_m(x)
\end{aligned}
$$

- $\nu\in(0,1]$（`learning_rate`）：**收缩率 (shrinkage)**，越小正则化越强、收敛越慢。
- 通常取浅树（`max_depth=1~2`）作弱学习器。
- 可选 `subsample`：按时间块随机抽样做**随机提升**（stochastic boosting）进一步正则化。
- `residual_norms_`：记录每一轮 $\lVert r^{(m)}\rVert_2$，便于观察收敛行为。

### 5.3 为何这种结构有用：分层揭示弱 regime

单棵贪心 P-Tree 有个结构性缺陷：**一旦在主导特征上分裂，被它掩盖的、更弱但真实的可预测性 regime 可能永远不会浮现**。Boosting 恰好治这个病：

1. **第 1 棵树 $T_1$** 捕获最强 regime（如 `x1>0.7` 处的强 alpha），并从 $y$ 中扣除；
2. **残差 $r$** 已经抹掉了被解释的成分，于是 **$T_2$ 在残差上搜索时，被主导特征压制的次级、更弱 regime 浮现**并被进一步分区；
3. 依此类推，每深一层揭示更弱一档的可预测性。

因此 **P-Boost = 嵌套 / 分层地发现可预测性 regime**，是 P-Tree"找可预测性集中区"初衷在深度方向的自然延伸。

### 5.4 自限性（self-limiting）

在**单特征主导**的数据上，第 1 棵树解释掉主导 regime 后，残差在该 regime 已近纯噪声、在弱 regime 本就弱；第 2 棵起几乎找不到可分裂的可预测性差异 → boosting **自动收敛**。表现为残差范数序列先降后趋平：

```
round  0: ||residual|| = 83.59
round  2: ||residual|| = 77.03
round  4: ||residual|| = 74.56
 ...                       (迅速趋平)
round 20: ||residual|| = 72.36
```

这说明 P-Boost **不会无中生有地过拟合 regime**，可作为正确性演示。

### 5.5 怎么用 P-Boost 筛"高可预测样本"

由于 P-Boost 的产物是一个**分层的规则栈**——每一层 $T_m$ 都是一棵完整的 P-Tree，其规则可直接读出——它给出了两种使用方式：

**方式 a：预测幅度排序（最快）。**

$$
\hat y_{\text{boost}}(x)=\nu\sum_{m=1}^{M}T_m(x).
$$

$\lvert\hat y_{\text{boost}}(x)\rvert$ 越大，说明各层叠加给出的信号越强、越一致 → 该样本可预测性越高：

```python
yhat   = boost.predict(X)
X_high = X[np.abs(yhat) > tau]      # |信号| 大 = 高可预测
```

注意这是「信号强度」代理，量纲是 $y$ 本身（收益），不是 $R^2$；适合做快速排序、选交易标的。

**方式 b：分层 hit 画像（最契合 P-Boost 灵魂）。**

每个样本对应一张"在第几层被解释"的画像：

- **第 1 层就命中**该树高 $R^2$ 叶子的样本 → 强、显性可预测；
- 只在**深层**才命中高 $R^2$ 叶子的样本 → 弱、被主导特征掩盖的隐藏可预测（**单棵 P-Tree 会完全错过这批样本**）。

由于每棵 $T_m$ 都是完整的 `PanelTreeEngine`，**每层的规则都可直接读出**，整套画像在保持稳健的同时具备逐层可读性。

### 5.6 与经典 Gradient Boosting (GBDT) 的对照

| 维度 | GBDT | **P-Boost** |
|------|------|-------------|
| 基学习器 | 浅 CART | 浅 P-Tree（R²Diff） |
| 拟合对象 | 损失负梯度（残差） | 目标残差 |
| 准则 | MSE | **R²Diff（不变）** |
| 降的是 | 预测偏差 | 偏差 **+ 逐层揭示弱 regime** |
| 终止 | 早停 / 固定轮数 | 同上，单特征数据**自限** |

### 5.7 实践注意

- 第 $m\ge 2$ 轮的 $R^2$ 是「解释**残差**方差」的 $R^2$，量纲与首轮不同，**不可直接比较**。
- `learning_rate` 越小、`n_estimators` 越大，正则化越强但越慢。
- **当前实现是回归型残差提升**：对 0/1 标签做残差提升并不等价于 GBDT-classification，因此**分类场景请使用 `PanelForest`**。

---

## 6. Unified View

```
                      ┌──────────────────────────────┐
                      │        单棵 P-Tree            │
                      │  贪心最大化 |R²_L − R²_R|     │
                      │  叶子 = 局部岭回归; 高方差     │
                      └───────────────┬──────────────┘
            并行 / 降方差              │              序贯 / 降偏差
         （扰动→去相关→聚合输出）      │        （残差→逐层揭示弱 regime）
                      ▼               │               ▼
   ┌──────────────────────────┐      │      ┌──────────────────────────┐
   │        P-Forest          │      │      │         P-Boost          │
   │  时间块 bootstrap +       │      │      │  F_m = F_{m-1} + T_m(残差)│
   │  节点随机特征子集         │      │      │  弱学习器 = 浅 P-Tree     │
   │  → 袋装预测                │      │      │  → 逐层剥离可预测性       │
   │  → regime 隶属概率         │      │      │  → 单特征数据自限         │
   │  → 共识矩阵                │      │      │                          │
   │  → OOB 评估                │      │      │                          │
   └──────────────────────────┘      │      └──────────────────────────┘
                                      ▼
              共同点：每棵树内部的切分准则 |R²_L − R²_R| 始终不变；
                     集成只发生在「分区 / 预测 / 组合」输出层。
```

**三者在「如何筛高可预测样本」上的对照**（这是最容易混淆的使用维度）：

| 维度 | 单棵 P-Tree | P-Forest | P-Boost |
|------|------------|----------|---------|
| 集成方式 | 无 | 并行 (bagging) | 序贯 (boosting) |
| 筛样本的判定 | 一条规则路径，如 `x1>0.7 ∧ x3<0.3` | `regime_membership(X) > τ` | `|ŷ_boost(X)| > τ` 或逐层 hit 画像 |
| 揭示的 regime | 最主导那一层 | 主导层（聚合更稳） | 主导 + 逐层更弱的隐藏层 |
| 判定形态 | 硬 0/1 | 软 $[0,1]$ 概率 | 连续预测 + 分层画像 |
| 可解释性 | 强（单一 if-else） | 弱（无单一规则） | 强（一摞按层叠加的规则） |
| 稳定性 | 低（规则随扰动翻转） | 高（$B$ 棵树平均） | 中（弱学习器叠加） |
| 独有价值 | 直观 | 鲁棒共识 + OOB 估计 | 挖出被主导特征掩盖的弱 regime |

**一句话总结**：

- **P-Tree** 找「可预测性集中在哪」，可读但高方差；
- **P-Forest** 用面板专属扰动 + 输出层聚合，降分区 / 预测方差，并给出**软 regime 概率 + 共识矩阵 + OOB 评估**；
- **P-Boost** 用残差序贯，把被主导特征压制的**弱 regime 逐层揭示**，并在单特征数据上**自限**。

三者构成完整谱系——同时严格保持 P-Tree「最大化可预测性差异」的算法灵魂不变。

---

## References

- Cong, Feng, He, He. *Growing the Efficient Frontier on Panel Trees.* （P-Tree 的组合 / SDF 视角与有效前沿目标）
- Athey, Imbens (2016). *Recursive Partitioning for Heterogeneous Causal Effects.* PNAS.（honest tree）
- Breiman (2001). *Random Forests.*（bagging / 随机特征子集）
- Friedman (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*（前向分步 / 残差提升 / shrinkage）
- Geurts, Ernst, Wehenkel (2006). *Extremely Randomized Trees.*（随机阈值 splitter）
