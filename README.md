# XGBoos · Knob 方法迁移到 XGBoost,并与 Kronos_SupervisedAD 比较

把 Huang、Mao、Jiang、Cambria《Uncovering the Knowledge Boundaries and Mechanistic "Knob" of LLMs》的知识边界 / 旋钮方法迁移到 XGBoost,在 [Kronos_SupervisedAD](https://github.com/SansaZheng/Kronos_SupervisedAD) 的九项公开任务上、用与它相同的划分检验。

**最新报告:[中文 v2](reports/KNOB_XGB_REPORT_CN_v2.md) · [English v2](reports/KNOB_XGB_REPORT_EN_v2.md)**(上一版:[中文 v1](reports/KNOB_XGB_REPORT_CN_v1.md) · [English v1](reports/KNOB_XGB_REPORT_EN_v1.md))

- XGBoost 在九项中的六项明显高于已记录的 Kronos-mini,ULB 持平,SD-FMM 股票和 TabFormer 低于它。
- Knob 没有带来可靠的检测提升:测试 AP 变化在 −0.004 到 +0.015 之间,有提升处普通的不确定性加权重训同样有效。
- 在特征较多的交易表数据上旋钮确实存在且特异,但不改变排序。

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
- `baselines/market.py`、`baselines/common.py`:三项行情任务的数据读取与指标,来自 [Knob](https://github.com/SansaZheng/Knob)
- `data/`:三项行情任务的原始数据与划分清单;其余六项数据在 Kronos_SupervisedAD 仓库,不在此重复存放
- `results/knob/`:逐次运行记录、测试分数、汇总表与图;`results/kronos_compare/`:对照结果与总对比表;`results/kronos_reference/`:从 Kronos_SupervisedAD 提交 `1f3f0aa` 复制的 Kronos 记录

复现命令见报告末尾。
