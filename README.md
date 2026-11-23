# XGBoos · Knob 方法迁移到 XGBoost 的异常检测实验

把 Huang、Mao、Jiang、Cambria《Uncovering the Knowledge Boundaries and Mechanistic "Knob" of LLMs》的知识边界 / 旋钮方法迁移到 XGBoost,在 Sapienza 15S、SD-FMM 股票、SD-FMM 加密货币三项金融异常检测任务上检验。

**结果:没有可靠提升。** 测试 AP 的变化都在种子波动之内,旋钮法向干预与随机方向没有区别。

| 任务 | XGBoost 测试 AP | Knob-XGBoost 测试 AP | 漏报 FN(基线 → Knob) |
| --- | ---: | ---: | ---: |
| Sapienza 15S | 0.9650 ± 0.0026 | 0.9649 ± 0.0029 | 9.7 → 8.7 |
| SD-FMM 加密货币 | 0.6571 ± 0.0139 | 0.6533 ± 0.0142 | 31.7 → 35.0 |
| SD-FMM 股票(raw) | 0.6019 ± 0.0148 | 0.6058 ± 0.0073 | 39.7 → 41.0 |

完整报告:[中文 v1](reports/KNOB_XGB_REPORT_CN_v1.md) · [English v1](reports/KNOB_XGB_REPORT_EN_v1.md)

## 目录

- `baselines/knob_xgb.py`:方法实现与实验;`baselines/knob_aggregate.py`:汇总、作图、复核
- `baselines/market.py`、`baselines/common.py`:数据读取、划分与指标,来自 [Knob](https://github.com/SansaZheng/Knob)
- `data/`:三项任务的原始发布数据与划分清单(同 Knob 仓库),许可见 `data/notices/`
- `results/knob/`:逐次运行记录(JSON)、测试分数(NPZ)、汇总表与图

复现命令见报告末尾。
