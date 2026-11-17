# 场景训练调参记录（2026-11-16）

> Historical measurements below used the retired box benchmark. The current study page measures the bundled skatepark and reports illuminated model and scene fractions.


实验页：`/check/training-study.html`。查询串可覆盖 `config.optimizer` 的任一字段，另有
`grid`（每边顶点数，默认 33）、`every`（检查点间隔）、`timing`（用 timestamp-query 平均的步数）。
每个检查点输出笔画内能量占比（与 `?stroke` 门槛同一口径）、透镜形状指标
（`src/optimization/lens-metrics.ts`）和 Adam 诊断。

形状指标：`lapRms` 是拉普拉斯 RMS（1/m，和网格无关）；`rippleRms` 是拉普拉斯减去四邻域均值后的 RMS，
专门量顶点尺度的波纹（聚焦需要的大而缓变的曲率不计入）；`slopeOverPct` 是坡度超过
`maxSurfaceSlope` 的三角形占比。

## 真实 GPU 计时（RTX 2060，Chrome WebGPU，ms/步）

| 网格（顶点） | sceneTrace | sceneForward | loss | filter | sceneBackward | gather | update | 合计 |
|---|---|---|---|---|---|---|---|---|
| 33 | 0.03 | 0.24 | 0.02 | 0.42 | 0.39 | 0.01 | 0.02 | 1.13 |
| 129 | 0.03 | 0.90 | 0.01 | 0.20 | 0.19 | 0.01 | 0.02 | 1.36 |
| 513 | 0.27 | 15.7 | 0.01 | 0.17 | 2.65 | 0.08 | 0.07 | 19.0 |
| 1025（含梯度平滑） | 1.03 | 45.8 | 0.01 | 0.18 | 11.0 | 0.34 | 2.22 | 60.6 |

- 生产网格上 `sceneForward` 占 75%。它对每个像素用 CAS 循环做浮点原子加，笔画把光聚到少数像素上，竞争最重。
  后续可以考虑定点 `atomicAdd` 或工作组内先合并（属于光学模块）。
- 原来每次 `simulation.encode` 在训练步之后还会再跑一次 forward 作预览，等于每步多 40% 以上。
  现在训练进行中直接显示该步已经算好的 forward（落后一步），收尾那一步仍重新计算。

## 主要结论

1. **33 网格的回归用例掩盖了生产网格的问题。** 旧默认值（lr 5e-5，3000 步，无平滑）在 1025 网格上
   最终笔画内能量只有 2.9%，`rippleRms` 约 4500；在 129 网格上滤波宽度退到 0 时（第 2400 步）
   全视图占比从 79% 掉到 67%，波纹翻倍。
2. **原因是细网格上每束光小于一个接收像素**，滤波退完后梯度只来自光斑边缘，逐顶点噪声很大，
   Adam 又按顶点归一化，把噪声放大成波纹。曲率正则按顶点间距无量纲化，网格越细惩罚越弱。
3. **按物理宽度平滑光学梯度最有效**（`gradientSmoothingSigma`，单位米，可分离高斯，自动适配分辨率）：
   - 129：σ=3 mm 加末端滤波 1 px，92.6% / 86.7%，`rippleRms` 8（无平滑时 300）。
   - 513：σ=2 到 3 mm，86% / 77%，过门槛（无平滑时发散）。
   - 1025：σ=3 mm 从 3% 提到约 71% / 39%，仍未过门槛，见下面"未解决"。
4. **学习率 1e-4、1500 步**在 33 网格上达到旧 3000 步的质量（约 94% / 83%），训练时间减半；
   2e-4 在 33 上更好，但在 513/1025 上更差，所以默认取 1e-4。
5. 平滑 Adam 步长（`updateSmoothingSigma`）在 33 网格最好（96% / 85%），但在 129 以上明显不如平滑梯度，
   默认关闭。单独加大 `smoothnessWeight` 会降低占比；`curvatureLengthScale` 固定物理尺度能压曲率，
   但 129/513 上收敛变慢，默认保持 0（旧行为）。

## 新默认值

`learningRate 1e-4`、`maxIterations 1500`、`gradientSmoothingSigma 0.003`、`lossBlurFinalSigma 1`。
`?stroke` 门槛仍固定 3000 步，以便和历史基线比较。

## 未解决

1025 网格（应用默认）在 1500 步内只到约 71% / 39%。这里每个像素覆盖约 8×8 个顶点，
单靠梯度平滑不够。下一步建议做由粗到细的训练：先在 65 或 129 网格上收敛，再插值到 1025 继续，
或者直接用粗网格系数参数化高度场。
