# 三个数据集的多样本素材索引

每类 4 组，共 12 组。所有新样本使用固定检查点在 CPU 上重新推理；原版单样本素材仍保留在根目录 assets/。先打开 ASSET_GALLERY.html 选择，再进入 multi_samples/数据集/编号/ 取素材。

|编号|数据集|样本|视角|主帧|参考类型|
|---|---|---|---|---|---|
|U01|unity|Anim_Female_Skier_From_Skiing_To_Straight|capture_L0_A000 / capture_L0_A030|10|Unity avatar GT|
|U02|unity|Anim_Male_Skier_Braking|capture_L0_A000 / capture_L0_A060|15|Unity avatar GT|
|U03|unity|Anim_Male_Skier_Braking_Loop|capture_L0_A000 / capture_L0_A120|9|Unity avatar GT|
|U04|unity|Anim_Male_Skier_From_Losing_To_Idle|capture_L0_A000 / capture_L0_A180|9|Unity avatar GT|
|S01|ski|seq_405|cam 0 / 1|121|Constructed canonical reference|
|S02|ski|seq_405|cam 2 / 3|120|Constructed canonical reference|
|S03|ski|seq_412|cam 0 / 1|118|Constructed canonical reference|
|S04|ski|seq_412|cam 2 / 3|116|Constructed canonical reference|
|A01|author|pro_1|left / right|746|No GT: qualitative only|
|A02|author|pro_2|left / right|1231|No GT: qualitative only|
|A03|author|run_3|left / right|180|No GT: qualitative only|
|A04|author|run_4|left / right|180|No GT: qualitative only|

## 使用说明

- 每组包含双视角原图、输入与候选骨架、对齐、五路特征相关素材、时间示例、门控和真实注意力热图、输出对照、完整 30 帧数据。
- 注意力热图是当前帧、多头平均的真实权重，区别于原版中的示意连线；横轴为右流 key 关节，纵轴为左流 query 关节。
- Unity 为 15 关节；Ski 为 13 个共同关节，不补画缺失的双眼；自采为 15 关节。
- Unity 与 Ski 保留原测试批次锚点（256 / 4），不能把这些结果解释为各样本独立首帧归一化。自采采用单窗口首帧锚点。
- 自采统一使用 Unity 检查点，按片段中间时刻选样；没有按外观或效果选择模型。相同帧号为已有双视角对应，本次未重新估计时间同步。
- 自采数据没有 GT，未导出误差文件；其输出外观不能证明绝对三维精度提高。Ski 的参考必须写 Constructed reference，不能改成独立实测 GT。
- 这些是预选示例，不是独立受试者样本或全数据集统计。Ski 的四组来自同一受试者的两段序列，存在相关性。
- 所有热图采用固定 0–1 色标；输出/参考在同一示例内共用投影与范围。三维骨架展示投影不能直接叠到 RGB 上。
- 原任务书中的固定 sample 0 / frame 10 描述只适用于原版素材；新增样本的帧号、关节数和来源以本索引及每组 provenance.json 为准。

## 本帧对照与选图提示

以下数值是预选主帧的平均关节距离，单位为各数据集 canonical coordinate units；不能跨数据集比较，不能作为总体性能结论。保留所有预选样本，包括效果不佳者。

|编号|原始帧|Canonical average|CanonFuse3D|选图提示|
|---|---|---|---|---|
|U01|10|0.13331|0.04067|本帧误差降低，可作局部对照|
|U02|15|0.57729|1.79997|本帧误差增大，不宜展示改善|
|U03|9|0.71915|0.41989|本帧误差降低，可作局部对照|
|U04|9|0.62751|0.30388|本帧误差降低，可作局部对照|
|S01|121|0.35474|0.17225|本帧误差降低，可作局部对照|
|S02|120|0.47635|0.16461|本帧误差降低，可作局部对照|
|S03|118|0.70937|0.28795|本帧误差降低，可作局部对照|
|S04|116|0.21980|0.11988|本帧误差降低，可作局部对照|
|A01|746|—|—|无 GT，仅定性展示|
|A02|1231|—|—|无 GT，仅定性展示|
|A03|180|—|—|无 GT，仅定性展示|
|A04|180|—|—|无 GT，仅定性展示|

U01 适合说明同一 Unity 示例的输入与输出改善；U02 保留为不利示例，不应用来说明普遍改善。S01–S04 可以展示相对于构造参考的本帧变化；A01–A04 适合展示室内外双视角输入和定性输出。选择其中某组用于论文时，仍应保留其样本范围说明。
