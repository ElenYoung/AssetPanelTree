# aptree Examples

四个独立脚本，演示 `aptree` 的不同侧面。所有运行产物（图片、CSV、文本、DOT 源码）统一写入 `example/outputs/`，该目录已在 `.gitignore` 中忽略。

| 脚本 | 演示内容 | 主要产物 (`outputs/`) |
|------|---------|---------------------|
| [`demo_ptree.py`](demo_ptree.py) | 单棵 P-Tree 的端到端用法：DataHandler、回归 / 波动率加权 / 分类、快速模式、节点报告、马赛克、叶子样本提取 | `demo_ptree_mosaic.png` |
| [`demo_pforest.py`](demo_pforest.py) | PanelForest（时间块自助 + OOB）与 BoostedPanelTree（残差提升），含 Extra-Trees 风格变体与样本外 R² 对比 | （仅控制台输出） |
| [`demo_visualization.py`](demo_visualization.py) | **v0.2 可视化**：`engine.evaluate` 逐节点 OOS；`print_tree` / `plot_tree`（纯 matplotlib）/ `to_graphviz` 加 OOS 覆盖；`tune_ccp_alpha` 剪枝路径；`MosaicVisualizer` 自动配色 + 支持 `metric="mean"` / `"ic"` / `"r2"` 等；`RankICDiffCriterion` 训练 | `tree_overlay.txt`、`tree_rank_ic_overlay.txt`、`tree.dot`、`tree.png`、`node_eval.csv`、`ccp_alpha_sweep.{csv,png}`、`mosaic_leaf_mean.png`、`mosaic_r2.png`、`mosaic_ic.png` |
| [`benchmark.py`](benchmark.py) | 小型性能基准：朴素 vs 快速模式 vs Cython 加速对比 | （仅控制台输出） |

## 运行

```bash
# 推荐用项目 .venv / 或 ~/.venv（参考 .clinerules/python_env.md）
~/.venv/bin/python example/demo_ptree.py
~/.venv/bin/python example/demo_pforest.py
~/.venv/bin/python example/demo_visualization.py
~/.venv/bin/python example/benchmark.py
```

每个脚本运行结束时都会打印产物的绝对路径。

## 关于 Graphviz 渲染

`demo_visualization.py` 总是会把 DOT 源码写到 `outputs/tree.dot`。如果系统装了 Graphviz 二进制（`dot`）和 Python 绑定（`pip install graphviz`），脚本会自动渲染为 `outputs/tree.svg`；否则会打印一条手动命令：

```bash
dot -Tsvg example/outputs/tree.dot -o example/outputs/tree.svg
```
