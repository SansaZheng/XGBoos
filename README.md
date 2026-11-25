# XGBoos · Knob 方法迁移到 XGBoost,并与 Kronos_SupervisedAD 比较

把 Huang、Mao、Jiang、Cambria《Uncovering the Knowledge Boundaries and Mechanistic "Knob" of LLMs》的知识边界 / 旋钮方法迁移到 XGBoost,在 [Kronos_SupervisedAD](https://github.com/SansaZheng/Kronos_SupervisedAD) 的九项公开任务上、用与它相同的划分检验。

**最新报告:[中文 v5](reports/KNOB_XGB_REPORT_CN_v5.md) · [English v5](reports/KNOB_XGB_REPORT_EN_v5.md)** —— 改动 XGBoost 的组成部件:同时去掉正类权重和按先验拟合的起始分数,IBM AML 的 AP 从 0.330 升到 0.501,PaySim 从 0.924 升到 0.970。

此前版本:[中文 v4](reports/KNOB_XGB_REPORT_CN_v4.md) · [English v4](reports/KNOB_XGB_REPORT_EN_v4.md)(超参数旋钮的重点分析);[中文 v3](reports/KNOB_XGB_REPORT_CN_v3.md) · [English v3](reports/KNOB_XGB_REPORT_EN_v3.md)(哪些参数控制异常 / 正常的区别);[中文 v2](reports/KNOB_XGB_REPORT_CN_v2.md) · [English v2](reports/KNOB_XGB_REPORT_EN_v2.md)(九项任务、与 Kronos 比较);[中文 v1](reports/KNOB_XGB_REPORT_CN_v1.md) · [English v1](reports/KNOB_XGB_REPORT_EN_v1.md)(三项行情任务)

- XGBoost 在九项中的六项明显高于已记录的 Kronos-mini,ULB 持平,SD-FMM 股票和 TabFormer 低于它。
- Knob 没有带来可靠的检测提升:测试 AP 变化在 −0.004 到 +0.015 之间,有提升处普通的不确定性加权重训同样有效。
- 在特征较多的交易表数据上旋钮确实存在且特异,但不改变排序。
- v5:有效的结构改动是“删”——去掉正类权重和拟合的起始分数,九项平均 +0.025;我设计的“加法”(平衡装袋 + 新颖度分支)没有站住;焦点损失的提升经对照证明来自起始分数。
- v4:旋钮的作用取决于任务类型。IBM AML 上正类权重越小越好(0.344 对 0.267);PaySim、ULB、TSTR 上基线的正类权重是种子间最不稳定的取值;按验证集选一个旋钮,九项的平均测试 AP 都没有下降,SD-FMM 股票 +0.062、IBM AML +0.014。
- v3:模型内部控制异常 / 正常区别的旋钮通常只是一个特征,调它的增益没有用;真正起作用的是超参数 `scale_pos_weight`,调后 PaySim 的 AP 从 0.924 升到 0.967,SD-FMM 股票从 0.602 升到 0.651(调 `reg_lambda`)。

测试 AP(Kronos 为该仓库记录,未重跑):

| 任务 | Kronos-mini 修复版 | XGBoost | Knob-XGBoost |
| --- | ---: | ---: | ---: |
| Sapienza 15S | 0.912 | 0.965 ± 0.003 | 0.965 ± 0.003 |
| SD-FMM 股票 | 0.630 | 0.602 ± 0.015 | 0.606 ± 0.007 |
| SD-FMM 加密货币 | 0.480 | 0.657 ± 0.014 | 0.653 ± 0.014 |
| TSTR | 0.486 | 0.830 ± 0.023 | 0.836 ± 0.011 |
| ULB | 0.754 ± 0.010 | 0.755 ± 0.031 | 0.770 ± 0.004 |
| Elliptic | 0.435 | 0.537 ± 0.004 | 0.540 ± 0.002 |
| PaySim | 0.452 ± 0.197 | 0.924 ± 0.058 | 0.931 ± 0.062 |
| IBM AML | 0.266 | 0.330 ± 0.002 | 0.332 ± 0.003 |
| TabFormer | 0.0074 ± 0.0008 | 0.0052 ± 0.0007 | 0.0044 ± 0.0011 |

## 目录

- `baselines/knob_xgb.py`:方法实现与实验;`baselines/knob_aggregate.py`:汇总、作图、复核
- `baselines/kronos_data.py`:调用 Kronos_SupervisedAD 的数据准备代码生成相同划分的输入;`baselines/kronos_compare.py`:XGBoost 各特征版本与该仓库的梯度提升对照;`baselines/kronos_table.py`:总对比表
- `baselines/intercept_xgb.py`、`intercept_probe.py`、`intercept_aggregate.py`:去掉正类权重与拟合起始分数的实验、机制检验与汇总,结果在 `results/intercept_xgb/`
- `baselines/dual_xgb.py`、`focal_xgb.py`、`dual_aggregate.py`:种子集成、平衡装袋、焦点损失、新颖度分支,结果在 `results/dual_xgb/`
- `baselines/knob_hyper_focus.py`:四个超参数旋钮的细扫与内部量记录;`baselines/knob_hyper_aggregate.py`:汇总与作图;结果在 `results/knob_hyper/`
- `baselines/knob_params.py`:异常 / 正常边界上的模型内部旋钮与超参数旋钮;`baselines/knob_params_aggregate.py`:汇总与作图;结果在 `results/knob_params/`
- `baselines/market.py`、`baselines/common.py`:三项行情任务的数据读取与指标,来自 [Knob](https://github.com/SansaZheng/Knob)
- `data/`:三项行情任务的原始数据与划分清单;其余六项数据在 Kronos_SupervisedAD 仓库,不在此重复存放
- `results/knob/`:逐次运行记录、测试分数、汇总表与图;`results/kronos_compare/`:对照结果与总对比表;`results/kronos_reference/`:从 Kronos_SupervisedAD 提交 `1f3f0aa` 复制的 Kronos 记录

复现命令见报告末尾。
