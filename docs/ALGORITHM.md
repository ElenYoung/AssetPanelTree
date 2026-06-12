# P-Tree Algorithm: From a Single Tree to P-Forest and P-Boost

> 本文档聚焦**算法本身**（数学定义、统计直觉、与经典方法的异同），而非 package 的 API 用法。
> API 用法见 `README.md`；优化决策与里程碑见 `OPTIMIZATION_PLAN.md`。
>
> 阅读顺序建议：§1 Panel Data and Notation → §2 The Single P-Tree → §3 Splitting Criteria → §4 Statistical Validity → **§5 P-Forest** → **§6 P-Boost** → §7 Unified View。

---

## 1. Panel Data, Objective and Notation

### 1.1 Panel Data

面板数据同时具有**截面维**（cross-section，资产 $i=1,\dots,N$）与**时间维**（time series，时刻 $t=1,\dots,T$）。一条观测是一个 $(t, i)$ 单元：

$$
\big(\, x_{t,i} \in \mathbb{R}^{p},\ \ y_{t,i} \in \mathbb{R} \,\big),
\qquad t = 1,\dots,T,\quad i = 1,\dots,N_t .
$$

- $x_{t,i}$：资产 $i$ 在 $t$ 时刻的 $p$ 个**特征 / 因子**（characteristics），如动量、估值、波动率、规模等；也可拼接宏观状态变量。
- $y_{t,i}$：被预测量，金融场景中通常是**下一期收益率** $r_{t+1,i}$。

总样本量 $n = \sum_t N_t$。在量化语境下，"模型"指一个从截面特征到截面收益的映射 $f:\mathbb{R}^p\to\mathbb{R}$，逐期施加于所有资产。

### 1.2 Cross-Sectional Rank Standardisation

P-Tree 的切分阈值需要在不同时刻可比。原始特征量纲随时间漂移（如波动率水平整体上升），直接用全局阈值会失真。`DataHandler` 在**每个时间截面内**把特征转成秩并映射到 $[0,1]$：

$$
\tilde{x}_{t,i,k} \;=\; \frac{\operatorname{rank}_{t}\big(x_{t,i,k}\big) - 1}{N_t - 1} \;\in\; [0,1],
\qquad k = 1,\dots,p .
$$

于是阈值 $c=0.5$ 恒等于"该时刻该特征的截面中位数"，跨期语义稳定。这一步对后文的"自适应分位阈值""时间块自助"都至关重要。

### 1.3 Why P-Tree Differs From an Ordinary Decision Tree

| 维度 | 标准 CART | **P-Tree** |
|------|-----------|-----------|
| 切分目标 | 最小化子节点**残差 MSE / Gini** | **最大化子节点间可预测性差异** $\lvert R^2_L - R^2_R\rvert$ |
| 叶子模型 | 常数（均值） | **岭回归 / Logit**（局部线性因子模型） |
| 目的 | 点预测 | **识别可预测性 regime**（何时/何处 alpha 集中） |
| 输出 | 单一预测 | 预测马赛克（prediction mosaic）+ 预测 |

关键一句话：**P-Tree 不是为了把误差降到最低，而是为了找出"哪一片 $(时间,资产)$ 区域里因子模型特别有效"**。这决定了它的准则、集成方式都与经典树不同。

---

## 2. The Single P-Tree

### 2.1 Nodes, Leaves and the Local Model

树把全样本递归地划分为互不相交的叶子区域 $\{\mathcal{R}_1,\dots,\mathcal{R}_M\}$。每个区域由一串"特征-阈值"规则定义，例如

$$
\mathcal{R}_m = \{(t,i): \tilde{x}_{\cdot,1}\ge 0.5 \ \wedge\ \tilde{x}_{\cdot,3} < 0.7\}.
$$

在每个叶子内部拟合一个**局部岭回归**（这就是"叶子模型"）：

$$
\hat\beta_m
= \arg\min_{\beta}\ \sum_{(t,i)\in\mathcal{R}_m} w_{t,i}\,\big(y_{t,i} - x_{t,i}^\top \beta\big)^2
\;+\; \alpha\,\lVert\beta\rVert_2^2
\;=\; \big(X_m^\top W_m X_m + \alpha I\big)^{-1} X_m^\top W_m y_m .
$$

权重 $w_{t,i}$ 可取逆波动率（`VolWeightedRidge`），缓解异方差。预测时一个观测沿树下行到所属叶子，用该叶子的 $\hat\beta_m$ 给出 $\hat y = x^\top\hat\beta_m$。

> **闭式解与增量统计量**：岭回归只依赖充分统计量 $A_m=X_m^\top W_m X_m$（$p\times p$）和 $b_m=X_m^\top W_m y_m$（$p$）。由于 $A,b$ 对样本是**可加**的，父节点的 $(A,b)$ 等于两个子节点之和。因此评估一个候选切分时，只需对**较小一侧**直接做一次矩阵乘法得到 $(A_{\text{small}},b_{\text{small}})$，较大一侧用 $A_{\text{large}}=A_{\text{parent}}-A_{\text{small}}$ 纯减法得到（$O(p^2)$）。这是 §A1 的加速核心，对结果**逐位不变**。

### 2.2 The Greedy Growing Algorithm

P-Tree 用自顶向下贪心生长，与 CART 同构但准则不同：

```
function GROW(node):
    if 停止条件(node): 标记为叶子, 拟合 β, return
    best_score = -∞
    for 每个特征 k in 1..p:                 # 可随机子集化 (max_features)
        for 每个候选阈值 c in thresholds(k):  # 固定列表 / 自适应分位 / 随机
            L = {样本: x_k <  c};  R = {样本: x_k >= c}
            if min(|L|,|R|) < min_samples: continue
            拟合 β_L, β_R (岭回归闭式解)
            评估 m_L = metrics(L),  m_R = metrics(R)   # 含 R²
            score = criterion(m_L, m_R)                # 见 §3
            if score > best_score: 记录 (k, c)
    if best_score < min_impurity_decrease: 标记为叶子, return
    用最优 (k*, c*) 切分, 递归 GROW(左), GROW(右)
```

停止条件：达到 `max_depth`、节点样本数 `< min_samples`、或最优 score `< min_impurity_decrease`。

**候选阈值的三种来源**（`splitter` / `split_thresholds`）：
1. **固定列表**（默认 `[0.3,0.5,0.7]`）：因秩标准化后特征 $\in[0,1]$，固定分位即有意义。
2. **自适应分位** `split_thresholds="adaptive"`：用本节点该特征的实际分位点（如 25/50/75 百分位），对非均匀分布更稳。
3. **随机阈值** `splitter="random"`：在节点内 $[\min,\max]$ 范围随机取 `n_random_splits` 个点——Extra-Trees 风格，为集成注入多样性（见 §5.4）。

### 2.3 The Prediction Mosaic

把叶子在每个时间截面上的表现（如逐期 $R^2$ 或 precision）排成 "叶子 × 时间" 的热力图，就是预测马赛克。它直观回答"**模型在什么时候、对哪一类资产最灵 / 最失效**"，是 P-Tree 区别于黑箱预测器的可解释产物。

---

## 3. Splitting Criteria

准则 $\text{criterion}(m_L, m_R)$ 把左右子节点的评估指标映射为一个"切分质量分"，越大越好。这是 P-Tree 的灵魂。

### 3.1 R2Diff Criterion

$$
\boxed{\ \text{score} = \big\lvert R^2_L - R^2_R \big\rvert\ }
$$

其中子节点 $R^2$ 在该子节点的局部岭模型下计算：

$$
R^2_{\bullet} = 1 - \frac{\sum w(y-\hat y)^2}{\sum w(y-\bar y)^2}.
$$

**直觉**：我们要的不是"两边都拟合得好"，而是"**两边的可预测性显著不同**"——一侧高 $R^2$（强 regime，alpha 集中）、另一侧低 $R^2$（近噪声）。把高低预测性区域分开，本身就是有价值的发现。可选 `weight_by_size` 用 $\min(n_L,n_R)/(n_L+n_R)$ 惩罚极不平衡切分。

### 3.2 Weighted R2Diff Criterion

小样本叶子的 $R^2$ 容易虚高（过拟合），导致贪心被"假 regime"误导。`WeightedR2DiffCriterion` 在保留内核的同时叠加两个稳定项：

$$
\text{score} = \big\lvert R^2_L - R^2_R\big\rvert
\;\cdot\; \underbrace{\frac{\min(n_L,n_R)}{n_L+n_R}}_{\text{平衡项}}
\;\cdot\; \underbrace{\frac{n_{\min}-1}{n_{\min}-1+\kappa}}_{\text{样本量收缩}}
$$

并可选用 **adjusted-$R^2$** 把回归维度 $p$ 纳入惩罚：$R^2_{\text{adj}} = 1-(1-R^2)\frac{n-1}{n-p-1}$，防止"多因子+少样本"刷高 $R^2$。默认行为不变，这是 opt-in 的更严格选项。

### 3.3 Mean-Variance / SDF Criterion

论文《Growing the Efficient Frontier on Panel Trees》的真正目标是**组合层面**的：让切分出的 regime 能张出更靠外的均值-方差有效前沿。`MeanVarianceCriterion` 据此定义：

1. 每个子叶子用预测值构造一个**截面去均值、预测加权的多空组合**，得到一条收益时序 $\{r^L_t\}$、$\{r^R_t\}$；
2. 取两条时序的均值向量 $\mu\in\mathbb{R}^2$ 与协方差 $\Sigma\in\mathbb{R}^{2\times2}$；
3. 切分分 = 两组合可达到的**切线（最大）夏普**（年化）：

$$
\text{score} = \sqrt{\mu^\top \Sigma^{-1}\mu}\;\cdot\;\sqrt{A},\qquad A=\text{年化因子}.
$$

**直觉**：分高意味着左右两个 regime 的多空组合**互补**（低相关、各有 alpha），合在一起把有效前沿推得更远——这正是"组合视角的可预测性差异"，比单看 $R^2$ 更贴近最终交易目标。该准则需要引擎在评估时附带逐期组合收益（`fit(..., time_index=...)`）。

### 3.4 Classification Criterion

二分类（如涨跌方向）用 `ClassificationCriterion`，把上面的 $R^2$ 换成 precision / F1 / AUC / logloss 的左右差异。AUC 用 $O(n\log n)$ 的秩（Mann–Whitney U）实现，避免 $O(n^+n^-)$ 双循环。

---

## 4. Statistical Validity: Honest Splits and Cost-Complexity Pruning

这两项不改变"找可预测性差异"的目标，而是**防止 in-sample 乐观偏差**，让树更可信。

### 4.1 Honest Split

**问题**：标准 P-Tree 在同一批样本上既**枚举挑选**切分点、又**评估**切分质量。"挑最大 $\lvert R^2_L-R^2_R\rvert$"这一步天然高估了切分质量（数据窥探 / selection bias）。

**做法**（Athey & Imbens, 2016 的 honest tree 思想）：进入节点时把样本随机（或按时间前后）分成 fit 集与 eval 集（比例 `honest_frac`）：
1. 在 **fit 集**上拟合左右叶子的 $\hat\beta_L,\hat\beta_R$；
2. 在 **eval 集**上计算 $R^2_L,R^2_R$ 与切分分——**选切分的样本 ≠ 评估切分的样本**；
3. 末端叶子可在全样本上重训（`honest_refit_full=True`）以提升预测精度。

与全局 Train/Validation 的区别：honest 是**每个节点内部**各自再切，针对的是"树结构选择"层面的过拟合，二者可叠加。

### 4.2 Cost-Complexity Pruning

贪心生长容易长出过深、过拟合的树。后剪枝引入复杂度惩罚 $\alpha$（`ccp_alpha`），自底向上比较"保留子树带来的累计 score 增益"与"叶子数惩罚 $\alpha\cdot|\text{leaves}|$"，剪去净收益为负的子树：

$$
\text{保留子树} \iff \text{Gain(子树)} \;>\; \alpha\cdot\big(|\text{leaves(子树)}| - 1\big).
$$

`cost_complexity_pruning_path()` 返回一串 $(\alpha, \#\text{叶子}, \text{score})$ 供选参（配合外层 validation）。

---

## 5. P-Forest: A Bagging Ensemble

> 对应"二叉树 ↔ 随机森林"。源码：`ensemble.PanelForest`。

### 5.1 Why P-Tree Especially Needs a Forest

P-Tree 的贪心步骤"在 $p\times|\text{thresholds}|$ 个候选里挑 $\lvert R^2_L-R^2_R\rvert$ 最大者"是一个**高方差**的离散选择：数据轻微扰动就可能让胜出的 $(k^*,c^*)$ 翻转，从而得到**完全不同的分区**。这正是 bagging（自助聚合）最能发挥威力的场景——用扰动生成多棵去相关的树，再在输出层聚合以降方差。

### 5.2 The Key Insight: What Exactly Is Being Aggregated

经典随机森林建立在"最小化预测误差"之上（平均降方差）。但 P-Tree 的准则 $\lvert R^2_L-R^2_R\rvert$ **不是一个可被平均的损失**——直接对它做平均没有数学意义。

破解之道是区分一棵拟合好的 P-Tree 的**两类输出**：

| 输出 | 形态 | 能否集成 | 聚合方式 |
|------|------|----------|----------|
| (1) **分区 / 聚类**（样本落到哪个叶子） | 离散、结构性 | ✔ 需特殊聚合 | 共识聚类（co-association） |
| (2) **逐叶模型的面板预测 $\hat y$** | 连续数值 | ✔ 可直接平均 | 袋装均值 / 组合平均 |

**结论：集成作用在"分区/预测/组合"这一输出层，而每棵树内部的切分准则 $\lvert R^2_L-R^2_R\rvert$ 始终不变。** 这样 P-Forest 才有自洽定义。

### 5.3 Panel-Specific Perturbation Design

不能照搬普通 RF 的逐样本 bootstrap（会打散截面、制造前视）。P-Forest 用两种扰动：

**(a) 样本扰动 = 时间块自助（block bootstrap）。** 把排序后的唯一时间标签切成长度 `block_size` 的**连续块**，对块**有放回**抽样：

$$
\{1,2,\dots,T\}\ \to\ \text{blocks}=\big[\{1..b\},\{b{+}1..2b\},\dots\big],\quad
\text{抽样}\ b^{(1)},\dots,b^{(n_{\text{blk}})}\stackrel{\text{i.i.d.}}{\sim}\text{Unif(blocks)} .
$$

被抽中的块（可重复，等价于上调权重）构成训练集；**从未被抽中的块**构成该树的**袋外（OOB）集**。按块抽样**保留了收益的时序自相关结构**，这是面板场景的关键。

**(b) 特征扰动 = 节点级随机子集。** 每个节点只在随机的 `max_features`（如 $\sqrt{p}$）个特征里找切分，强制不同树发现不同的弱 regime，进一步去相关。

### 5.4 Three Aggregated Outputs

记森林有 $B$ 棵树，第 $b$ 棵的预测为 $\hat y_b(\cdot)$，把观测 $x$ 路由到的叶子为 $\ell_b(x)$。

**① 袋装预测（降方差）**
$$
\hat y_{\text{forest}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat y_b(x).
$$
平均抵消单树贪心分裂带来的预测方差。

**② Regime 隶属概率（软马赛克）**。定义"高 $R^2$ 叶子"为该树中 $R^2$ 高于其叶子中位数的叶子集合 $\mathcal H_b$，则
$$
\widehat{P}(\text{高可预测 regime}\mid x) = \frac{1}{B}\sum_{b=1}^{B}\mathbf 1\big[\ell_b(x)\in\mathcal H_b\big]\ \in[0,1].
$$
把单树脆弱的 0/1 硬标签升级为平滑、鲁棒的概率，直接改善马赛克可解释性。

**③ 共识 / 协同关联矩阵（consensus / co-association）—— 最契合"聚类"语义。**
$$
C_{ij} = \frac{1}{B}\sum_{b=1}^{B}\mathbf 1\big[\ell_b(x_i)=\ell_b(x_j)\big]\ \in[0,1],
$$
即"观测 $i,j$ 落入同一叶子的树占比"。它把许多脆弱的硬分区融合成一个**鲁棒的相似度矩阵**，回答"哪些 $(时间,资产)$ 单元*一致地*共享同一可预测性 regime"。$C$ 可作为预计算 affinity 喂给谱聚类，得到稳定的"元 regime"。注意 $C$ 是 $O(n^2)$ 内存，需控制 $n$。

### 5.5 Out-of-Bag Evaluation

每棵树用它**未抽中的时间块**做袋外预测，对所有树在各自袋外样本上的预测取平均后与 $y$ 比较，得到 OOB $R^2$。这无需额外 hold-out 即可估计泛化能力。

### 5.6 Honest Limitations

**若真实可预测性由单一强特征主导**（如 demo 里的 `char_1`），所有树都会在它上面分裂 → 高度相关 → 森林增益有限，甚至可能略低于单树（随机特征子集反而牺牲了最优分裂）。**P-Forest 的收益主要来自"可预测性由多个、弱识别的特征共同驱动"的场景**——此时随机特征子集让不同树发现不同弱 regime，聚合才显著获益。`example/demo_pforest.py` 特意构造了多特征驱动数据来体现这一点。

### 5.7 Comparison With the Classic Random Forest

| 维度 | 随机森林 (RF) | **P-Forest** |
|------|---------------|-------------|
| 基学习器 | CART（MSE/Gini） | P-Tree（R²Diff） |
| 样本扰动 | 逐样本 bootstrap | **时间块** block bootstrap |
| 特征扰动 | 节点随机子集 | 节点随机子集（相同思想）|
| 主要产物 | 平均预测 | 平均预测 **+ 共识聚类 + regime 概率** |
| 降的是 | 预测方差 | 预测方差 **+ 分区方差** |

### 5.8 How to Identify High-Predictability Samples in a Forest

这是 P-Forest 相对单棵 P-Tree 最容易引起困惑的地方，单独说清楚。

**单棵 P-Tree 怎么筛高可预测样本。** 你沿树读出**一条人类可读的硬规则路径**，落到某个高 $R^2$ 叶子，例如

$$
\{\,\tilde x_{\cdot,1} > 0.7 \ \wedge\ \tilde x_{\cdot,3} < 0.3\,\}\ \Rightarrow\ \text{高可预测 regime}.
$$

直观、可解释，但**高方差**——数据轻微扰动会让胜出的 $(k^*,c^*)$ 翻转，整条规则随之改变。

**P-Forest 怎么筛高可预测样本。** 森林里**不存在那条唯一规则**（$B$ 棵树各有各的分区）。取而代之的是一个**软分数**：§5.4 的 regime 隶属概率

$$
\widehat{P}(\text{高可预测}\mid x) \;=\; \frac{1}{B}\sum_{b=1}^{B}\mathbf 1\big[\ell_b(x)\in\mathcal H_b\big]\ \in[0,1],
$$

即"有多少比例的树把样本 $x$ 路由进了高 $R^2$ 叶子"（$\mathcal H_b$ = 第 $b$ 棵树中 $R^2$ 高于其叶子中位数的叶子集合）。**筛选方式与单树完全平行，只是把硬规则换成概率阈值**：

```python
forest.fit(X, y, feature_names, time_index=...)
p = forest.regime_membership(X)     # 每个样本一个 [0,1] 概率
X_high = X[p > 0.7]                 # 类比单树里的 “x1>0.7 & x3<0.3”
```

> 一句话：单树写 `x1>0.7 & x3<0.3` 选样本，P-Forest 写 `regime_membership(X) > 0.7` 选样本。后者的判定来自 **$B$ 棵树的投票共识**，不依赖任何单一脆弱阈值，因此更稳健。

**两类产物的本质区别。**

| 维度 | 单棵 P-Tree | P-Forest |
|------|------------|----------|
| 如何识别高可预测样本 | 读出**一条规则路径** $x_1>0.7\wedge x_3<0.3$ | `regime_membership(X)` 给**概率**，阈值化即可 |
| 判定形态 | 硬 0/1 标签 | 软 $[0,1]$ 概率 |
| 可解释性 | **强**（直接是 if-else 规则） | 弱（无单一规则），但更**鲁棒** |
| 稳定性 | 低（规则随扰动翻转） | 高（$B$ 棵树平均） |
| 额外产物 | 单一分区 + 预测马赛克 | 袋装预测 + **共识矩阵** + OOB $R^2$ |

**想兼顾鲁棒与可读规则：代理树（surrogate tree）。** 若既要森林的稳健判定、又要单树式的 if-else 解释，可用一棵浅树去拟合森林给出的 0/1 标签，把"森林共识"翻译回近似规则：

```python
p = forest.regime_membership(X)
label = (p > 0.7).astype(int)       # 高可预测 = 1
# 再用一棵浅 P-Tree / 普通决策树拟合 X -> label,
# 读出近似的 “x1>?, x3<?” 规则
```

**别混淆 `regime_membership` 与 `coassociation_matrix`。** 前者回答"**这个样本是否高可预测**"（逐样本概率，可直接阈值化筛样本）；后者 $C_{ij}$ 回答"**哪些样本一致地同属一个 regime**"（成对相似度，适合喂给谱聚类得到稳定的元 regime），二者用途不同。

---

## 6. P-Boost: A Boosting Ensemble

> 对应"二叉树 ↔ 梯度提升"。源码：`ensemble.BoostedPanelTree`。

### 6.1 The Key Insight: Boosting the Target, Not the Criterion

最易犯的错误是去"提升 $\lvert R^2_L-R^2_R\rvert$ 这个准则值"——它不可加、非损失，提升它没有意义。**P-Boost 提升的是目标变量 / 残差**：每一轮把上一轮已解释掉的收益成分扣除，让下一棵 P-Tree 在**残差**上重新寻找可预测性。每棵树内部的切分准则**仍是 R²Diff，保持不变**。

### 6.2 The Algorithm: Residual Boosting / Forward Stagewise

$$
\begin{aligned}
&F_0(x) = 0\\
&\textbf{for } m = 1,\dots,M:\\
&\qquad r^{(m)}_{t,i} = y_{t,i} - \nu\, F_{m-1}(x_{t,i})
   &&\text{(残差，}\nu=\text{learning\_rate)}\\
&\qquad T_m = \text{PanelTree}_{R^2\text{Diff}}.\text{fit}\big(X,\ r^{(m)}\big)
   &&\text{(在残差上长一棵浅 P-Tree)}\\
&\qquad F_m(x) = F_{m-1}(x) + T_m(x)\\
&\textbf{return } \hat y(x) = \nu\sum_{m=1}^{M} T_m(x)
\end{aligned}
$$

其中 $\nu\in(0,1]$ 是收缩率（shrinkage），浅树（`max_depth=1~2`）作弱学习器，可选 `subsample` 按时间块随机抽样做随机提升（正则化）。

### 6.3 Why This Matters in the Predictability Framework

单棵贪心 P-Tree 有个结构性缺陷：**一旦在主导特征上分裂，被它掩盖的、更弱但真实的可预测性 regime 可能永远不会浮现**。Boosting 恰好治这个病：

- **第 1 棵树**捕获最强 regime（如 `char_1>0` 处的强 alpha）并从 $y$ 中扣除；
- **残差** $r$ 抹掉了已解释成分，于是**第 2 棵树在残差上搜索时，被主导特征压制的次级、更弱可预测性结构得以显现**并被进一步分区；
- 依此类推。

因此 **P-Boost = "嵌套 / 分层地发现可预测性 regime"**，是 P-Tree"寻找可预测性集中区"初衷在深度上的自然延伸。

### 6.4 The Self-Limiting Property

在**单特征主导**的数据上，第 1 棵树解释掉主导 regime 后，残差在该 regime 已近纯噪声、在弱 regime 本就弱；第 2 棵起几乎找不到可分裂的可预测性差异 → boosting **自动收敛**。表现为残差范数序列 $\{\lVert r^{(m)}\rVert_2\}$ 先降后趋平：

```
round  0: ||residual|| = 83.59
round  2: ||residual|| = 77.03
round  4: ||residual|| = 74.56
 ...                       (迅速趋平)
round 20: ||residual|| = 72.36
```

这说明 P-Boost **不会无中生有地过拟合 regime**——可作为正确性演示（见 `residual_norms_`）。

### 6.5 Practical Notes

- 第 $m\ge2$ 轮的 $R^2$ 是"解释**残差**方差"的 $R^2$，量纲与首轮不同，不可直接比较；
- `learning_rate` 越小、$M$ 越大，正则化越强但越慢；
- 进阶方案（B-2，对齐 SDF）：每轮加入"对当前集成夏普边际贡献最大"的叶子组合，并对已有组合做 Gram-Schmidt 正交化，更贴近《Growing the Efficient Frontier》的 boosting 精神，需组合层评估管线（属可选增强）。

### 6.6 Comparison With Classic Gradient Boosting

| 维度 | GBDT | **P-Boost** |
|------|------|------------|
| 基学习器 | 浅 CART | 浅 P-Tree（R²Diff）|
| 拟合对象 | 损失的负梯度（残差）| 收益**残差** |
| 准则 | MSE | **R²Diff（不变）** |
| 降的是 | 预测偏差 | 偏差 + **逐层揭示弱 regime** |
| 终止 | 早停 / 固定轮数 | 同上，且单特征数据**自限** |

### 6.7 How to Identify High-Predictability Samples in a Boosting Sequence

与 §5.8 对照，这里说清楚 P-Boost 下"如何筛高可预测样本"，以及它的产物为何与 P-Tree、P-Forest 都不同。

**先厘清三者的产物形态。** 关键在于"集成是并行还是序贯"：

- **P-Tree** = **一层**分区：一套硬规则，只能揭示**最主导**的 regime；
- **P-Forest** = **并行**多棵树：投票平均成**软概率**，丢失了单一可读规则（§5.8）；
- **P-Boost** = **序贯**多棵树：每棵树长在上一轮的**残差**上，因此得到一个**分层的规则栈**——每一层仍是一套**可读硬规则**，可解释性**没有丢失**，只是从"一条规则"变成"一摞按层叠加的规则"。

这正是 P-Boost 与 P-Forest 最大的不同：**森林为降方差牺牲了可读性，提升为降偏差反而保留了逐层可读性。**

**怎么在 P-Boost 下筛高可预测样本——两种视角。**

**① 预测幅度视角（最快）。** 提升后的预测 $\hat y_{\text{boost}}(x)=\nu\sum_m T_m(x)$ 的**绝对值**越大，说明各层叠加给出的信号越强、越一致 → 该样本可预测性越高：

```python
yhat = boost.predict(X)             # nu * sum_m T_m.predict(X)
X_high = X[np.abs(yhat) > tau]      # |信号| 大 = 高可预测
```

注意这是"**信号强度**"代理，量纲是收益本身，不是 $R^2$；适合快速排序、选交易标的。

**② 分层 regime 视角（最契合 P-Boost 灵魂）。** P-Boost 的本质是"**逐层揭示可预测性**"（§6.3）。因此一个样本的可预测性不是单一标签，而是一张**"在第几层被解释"的画像**：

- **第 1 层 $T_1$** 捕获最强 regime——落入其高 $R^2$ 叶子的样本是"**主导可预测**"样本，对应可读规则如 `x1>0.7`；
- **残差上的 $T_2$** 揭示被主导特征压制的次级 regime——只在这一层才落入高 $R^2$ 叶子的样本，是"**弱/隐藏可预测**"样本，对应规则如 `x4<0.2`（在残差上）；
- 依此类推，每深一层揭示更弱一档的可预测性。

由于每棵 $T_m$ 都是一棵完整的 `PanelTreeEngine`，**每一层的规则都能直接读出来**。于是"高可预测样本"被精细化为一个**逐层命中画像**：

```python
# 逐层读出每棵树把样本路由到的叶子 R^2, 得到“分层可预测画像”
for m, tree in enumerate(boost.trees_):
    leaf_r2_m = ...   # tree 在残差上各叶子的 R^2
    # 样本是否落入该层的高 R^2 叶子 -> 它在第 m 层是否“可预测”
```

- 被**第 1 层**就解释的样本 → 强、显性可预测；
- 只有到**深层**才被解释的样本 → 弱、隐藏可预测（**单棵 P-Tree 会完全错过这批样本**，这正是 P-Boost 的增量价值）。

**三类产物对照。**

| 维度 | 单棵 P-Tree | P-Forest | **P-Boost** |
|------|------------|----------|-------------|
| 集成方式 | 无（单树） | 并行（bagging） | **序贯（boosting）** |
| 如何识别高可预测样本 | 一条规则路径 | `regime_membership` 概率阈值 | $\lvert\hat y_{\text{boost}}\rvert$ 排序 **或** 逐层命中画像 |
| 揭示的 regime | 仅**最主导**一层 | 主导层（聚合更稳） | **主导 + 逐层更弱的隐藏层** |
| 规则可读性 | 强（一套规则） | 弱（无单一规则） | **强（一摞按层叠加的规则）** |
| 形态 | 硬 0/1 | 软 $[0,1]$ | 连续预测 + **分层画像** |
| 独有价值 | 直观 | 鲁棒共识 | **挖出被主导特征掩盖的弱 regime** |

**一句话**：P-Tree 只告诉你"最强那批高可预测样本"；P-Forest 用投票把"是不是高可预测"变成稳健概率；**P-Boost 则告诉你"高可预测样本分几层、每层是哪批、各自服从什么规则"**——把"可预测性"从单一标签升级为可逐层解读的层次结构。

---

## 7. Unified View of the Three Algorithms

```
                      ┌──────────────────────────────┐
                      │        单棵 P-Tree            │
                      │  贪心最大化 |R²_L − R²_R|     │
                      │  叶子=局部岭回归; 高方差       │
                      └───────────────┬──────────────┘
            并行 / 降方差              │              序贯 / 降偏差
         （扰动→去相关→聚合输出）      │        （残差→逐层揭示弱 regime）
                      ▼               │               ▼
   ┌──────────────────────────┐      │      ┌──────────────────────────┐
   │        P-Forest          │      │      │         P-Boost          │
   │  时间块 bootstrap +       │      │      │  F_m = F_{m-1}+T_m(残差) │
   │  节点随机特征子集         │      │      │  弱学习器=浅 P-Tree       │
   │  → 袋装预测                │      │      │  → 逐层剥离可预测性       │
   │  → regime 隶属概率         │      │      │  → 单特征数据自限         │
   │  → 共识/协同关联矩阵       │      │      │                          │
   │  → OOB 评估                │      │      │                          │
   └──────────────────────────┘      │      └──────────────────────────┘
                                      ▼
              共同点：每棵树内部的切分准则 |R²_L − R²_R| 始终不变；
                     集成只发生在"分区 / 预测 / 组合"输出层。
```

**一句话总结**：
- **P-Tree** 找"可预测性集中在哪"，但单树高方差、且会被主导特征掩盖弱 regime；
- **P-Forest** 用面板专属扰动 + 输出层聚合，降低分区/预测方差，并把脆弱硬聚类升级为鲁棒的**共识聚类**与**软 regime 概率**；
- **P-Boost** 用残差序贯，把被主导特征压制的**弱 regime 逐层揭示**出来，并在单特征数据上**自限**。

三者构成一个完整谱系：从单一可解释的树，到方差缩减的森林，再到偏差缩减、层次化揭示可预测性的提升——同时严格保持 P-Tree"最大化可预测性差异"的算法灵魂不变。

---

## References

- Cong, Feng, He, He. *Growing the Efficient Frontier on Panel Trees.* （P-Tree 的组合 / SDF 视角与有效前沿目标）
- Athey, Imbens (2016). *Recursive Partitioning for Heterogeneous Causal Effects.* PNAS.（honest tree）
- Breiman (2001). *Random Forests.*（bagging / 随机特征子集）
- Friedman (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*（前向分步 / 残差提升 / 收缩）
- Geurts, Ernst, Wehenkel (2006). *Extremely Randomized Trees.*（随机阈值 splitter）
