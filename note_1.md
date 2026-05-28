# 周报告：rPPG/非接触生命体征监测临床成文思路整理

## 1. 本周工作概述

主要围绕现有 rPPG / 非接触生命体征监测项目，重新梳理了论文选题方向和临床化表达方式。

目前项目已经具备较好的数据和实验基础，包括：

* 视频信号提取：面部；
* 多模态输入：ICU 场景下同步采集的 RGB 与 IR 视频；
* 多任务方向：心率、血氧、血压、呼吸，以及新生儿呼吸暂停下游任务；
* 多源临床标签：ECG、BP、SpO₂、单点血糖等，其中 ECG/BP/血糖等标签目前尚未充分利用；
* 已有实验结果：使用 RGB 分支，在心率任务上，以 UBFC 训练、ICU 数据测试，基于我复刻的 FacePhys Torch 版本，MAE 最好可达到 2.1 bpm。

这一结果初步证明 ICU 视频数据中确实存在可用的生理信号，但如果论文仅围绕“心率估计 MAE”展开，医学价值和临床表达仍然不够强。因此，本周重点从“模型方法论文”转向“临床问题驱动论文”，希望将工作包装为医院方更容易认可的临床交叉研究。

---

## 2. 思路总结

### 2.1 在 ICU / NICU 等真实临床场景中，利用 RGB+IR 多部位视频，实现非接触、连续、低负担的生命体征监测和临床事件识别。

也就是说，论文应从“估计某个生命体征数值”转向“识别有临床意义的生理恶化事件”。

---

### 2.2 ICU 中更临床的核心思想

参考 NICU 呼吸暂停任务，我认为成人 ICU 场景中可以提出一个更临床化的核心概念：

> 基于多部位 RGB–IR 视频的 ICU 患者呼吸-循环失代偿事件非接触检测。

英文表达可以是：

> Contactless detection of cardiorespiratory deterioration in ICU patients using multisite RGB–infrared video monitoring.

这一方向比单纯心率估计更接近临床问题。ICU 医生真正关心患者是否正在发生或即将发生：

* 呼吸暂停；
* 低通气；
* 呼吸频率异常；
* SpO₂ 下降；
* 心动过缓或心动过速；
* 低血压或 MAP 下降；
* 外周低灌注；
* 呼吸异常合并循环异常。

因此，可以将论文问题定义为：

> RGB–IR 视频能否在 ICU 患者中非接触识别具有临床意义的呼吸-循环失代偿事件？

这与 NICU apnea 的逻辑类似，只是成人 ICU 中的终点可以更综合，不局限于单一呼吸暂停事件。

---

## 3. 现有实验结果的分析

### 3.1 UBFC 训练、ICU 测试 MAE 2.1 bpm 的意义

目前基于 RGB 分支和 FacePhys Torch 复刻版本，在心率任务上达到 ICU 测试 MAE 约 2.1 bpm。这说明：

1. ICU 视频中确实包含可恢复的 rPPG 信号；
2. 当前视频采集质量、同步流程和部分 ROI 提取策略是可用的；
3. 公开数据预训练或训练得到的 clean rPPG prior 能够迁移到 ICU 测试场景；
4. 该结果可以作为后续临床论文中的基础可行性证据。

但该结果不宜作为整篇论文的唯一核心。更好的写法是：

> HR estimation performance demonstrates the physiological validity of the video signal, while the main clinical contribution lies in detecting cardiorespiratory deterioration events.

也就是，心率 MAE 用来证明数据和信号可用，但论文主线应进一步上升到临床事件检测。

---

### 3.2 ICU 数据训练效果不好的可能原因

目前 ICU 训练结果尚不理想，我认为原因可能包括以下几类。

#### 1. ICU 标签本身可能较噪

ICU 中的 ECG、SpO₂、BP 等标签并不一定天然干净：

* ECG HR 可能经过监护仪平滑和滤波；
* SpO₂ 通常存在设备平均窗口和时间延迟；
* 袖带 BP 是离散测量，不适合作为逐秒视频监督；
* 动脉压波形可能受 flush、阻尼、导管状态影响；
* 低灌注、体动、探头脱落、护理操作都会影响 reference label 质量。

因此，ICU train 效果不好不一定说明视频信号无效，也可能说明 label alignment、reference quality 和窗口定义存在问题。

#### 2. ICU 数据分布复杂

ICU 场景相比 UBFC 复杂得多，包括：

* 患者体动；
* 遮挡；
* 氧气面罩、管路、贴片、被子遮盖；
* 夜间低光照；
* 肤色差异；
* 低灌注；
* 血管活性药物；
* 镇静、机械通气、护理操作。

这些因素会导致 ICU 内部训练时模型很容易学到场景噪声，而不是稳定的生理信号。

#### 3. 有效样本量可能低于表面样本量

ICU 长视频虽然可以切出大量窗口，但相邻窗口高度相关。真正独立的信息量取决于：

* 患者数量；
* session 数量；
* 不同生理状态数量；
* 不同事件数量；
* 不同光照和遮挡条件数量。

因此，后续应优先使用 patient-level split，并报告独立患者数和事件数，而不是只报告窗口数。

#### 4. 多任务联合训练可能存在负迁移

心率、血氧、血压、呼吸虽然都属于生命体征，但它们依赖的视觉线索不同：

* HR 主要依赖皮肤微弱颜色变化；
* RR 主要依赖胸腹运动和节律；
* SpO₂ 对波长、光谱和灌注状态更敏感；
* BP 很难通过短视频直接稳定回归；
* 外周灌注更依赖手掌、四肢和 IR 信号。

如果强行共享同一个 backbone 或同一套特征，可能会引入 negative transfer。

---

## 4. 整理文献

重点关注偏医学、临床和 Nature / JCR Q1 方向的相关论文。初步整理如下。

### 4.1 NICU / ICU 非接触生命体征监测核心文献

#### 1. Non-contact physiological monitoring of preterm infants in the Neonatal Intensive Care Unit

* 期刊：npj Digital Medicine
* 年份：2019
* DOI：10.1038/s41746-019-0199-5
* 作用：这是 NICU 场景下非接触生命体征监测的重要临床论文，适合对标新生儿呼吸、心率和临床可行性研究。

#### 2. Non-contact physiological monitoring of post-operative patients in the intensive care unit

* 期刊：npj Digital Medicine
* 年份：2022
* DOI：10.1038/s41746-021-00543-z
* 作用：这是成人 ICU / 术后 ICU 场景下 video-based physiological monitoring 的重要参考，适合作为 ICU 论文主线的直接对标文献。

#### 3. Continuous non-contact vital sign monitoring of neonates in intensive care units using RGB-D cameras

* 期刊：Scientific Reports
* 年份：2025
* DOI：10.1038/s41598-025-00539-9
* 作用：该文使用 RGB-D camera 在 NICU 中估计 HR、SpO₂、RR 和潮气量，非常接近本项目的 RGB+IR、多模态、多任务方向。

#### 4. Video-based physiologic monitoring: promising applications for the ICU and beyond

* 期刊：npj Digital Medicine
* 年份：2022
* DOI：10.1038/s41746-022-00575-z
* 作用：这篇文章适合用来支撑 ICU 视频生命体征监测的临床价值，包括连续监测、减少接触式监护负担、辅助 ICU 工作流等。

#### 5. Challenges and prospects of visual contactless physiological monitoring in clinical study

* 期刊：npj Digital Medicine
* 年份：2023
* DOI：10.1038/s41746-023-00973-x
* 作用：这是视觉非接触生命体征监测的综述型文献，适合放在 Introduction 和 Discussion 中，用于总结临床转化难点和证据缺口。

---

### 4.2 ICU 相关临床事件与下游任务文献

#### 6. High-Throughput, Contact-Free Detection of Atrial Fibrillation From Video With Deep Learning

* 期刊：JAMA Cardiology
* 年份：2020
* DOI：10.1001/jamacardio.2019.4004
* 作用：该文说明视频生理信号不仅可以估计心率，也可以用于临床疾病或事件检测。对本项目利用 ECG label 做心律异常、bradycardia 或事件检测有启发。

#### 7. Contact-Free Screening of Atrial Fibrillation by a Smartphone Using Facial Pulsatile Photoplethysmographic Signals

* 期刊：Journal of the American Heart Association
* 年份：2018
* DOI：10.1161/JAHA.118.008585
* 作用：该文适合支撑“rPPG 可用于临床筛查，而不仅是生命体征回归”的论点。

#### 8. Contactless facial video recording with deep learning models for the detection of atrial fibrillation

* 期刊：Scientific Reports
* 年份：2022
* DOI：10.1038/s41598-021-03453-y
* 作用：Nature Portfolio 中的视频 AF 检测研究，可作为从 rPPG 向临床事件识别转化的参考。

---

### 4.3 呼吸、睡眠和 SpO₂ 相关文献

#### 9. Camera-Based Vital Signs Monitoring During Sleep—A Proof of Concept Study

* 期刊：IEEE Journal of Biomedical and Health Informatics
* 年份：2021
* DOI：10.1109/JBHI.2020.3045859
* 作用：适合支持长时间、弱运动、呼吸节律和非接触监测场景，和 ICU / NICU 呼吸任务有一定共性。

#### 10. Notch RGB-camera based SpO₂ estimation: a clinical trial in a neonatal intensive care unit

* 期刊：Biomedical Optics Express
* 年份：2024
* DOI：10.1364/BOE.510925
* 作用：该文直接对应 NICU SpO₂ 视频估计任务，说明 SpO₂ 任务需要考虑光谱设计、波长选择和 clinical trial validation。

#### 11. Non-Contact Measurement of Blood Oxygen Saturation Using Facial Video Without Reference Values

* 期刊：IEEE Journal of Translational Engineering in Health and Medicine
* 年份：2023
* DOI：10.1109/JTEHM.2023.3318643
* 作用：适合补充 SpO₂ 非接触估计相关方法和临床局限性讨论。

---

### 4.4 临床综述与系统评价

#### 12. Clinical applications of contactless photoplethysmography for monitoring in adults: A systematic review and meta-analysis

* 期刊：Journal of Clinical and Translational Science
* 年份：2023
* DOI：10.1017/cts.2023.547
* 作用：适合支持成人临床 cPPG 应用现状，说明领域仍缺乏大规模、真实场景、临床事件导向的研究。

#### 13. Clinical Applications of Contactless Photoplethysmography for Vital Signs Monitoring in Paediatrics: A Systematic Review and Meta-Analysis

* 期刊：Journal of Clinical and Translational Science
* 年份：2023
* DOI：10.1017/cts.2023.557
* 作用：适合支撑儿科和新生儿方向的研究背景，尤其是 NICU 呼吸暂停和儿科非接触监测需求。

#### 14. Effectiveness of consumer-grade contactless vital signs monitors: a systematic review and meta-analysis

* 期刊：Journal of Clinical Monitoring and Computing
* 年份：2022
* DOI：10.1007/s10877-021-00734-9
* 作用：适合讨论 contactless vital sign monitoring 的验证标准、误差报告、设备级临床评估和监管转化问题。

---

## 5. 后续成文方向

### 5.1 思路一：ICU 呼吸-循环失代偿事件检测


拟定题目：

> Contactless Detection of Cardiorespiratory Deterioration in ICU Patients Using Multisite RGB–Infrared Video Monitoring

中文题目：

> 基于多部位 RGB–IR 视频的 ICU 患者呼吸-循环失代偿事件非接触检测

核心假设：

> RGB–IR 多部位视频不仅能估计基础生命体征，还可以非接触识别 ICU 患者具有临床意义的呼吸-循环失代偿事件。

可定义的事件包括：

* apnea；
* hypopnea；
* abnormal respiratory rhythm；
* desaturation；
* bradycardia；
* tachycardia；
* hypotension；
* apnea-associated desaturation；
* desaturation-associated HR/BP abnormality。

建议将事件分为三级：

| 事件级别    | 定义                    |
| ------- | --------------------- |
| Level 1 | 视频检测到呼吸暂停、低通气或异常 RR   |
| Level 2 | Level 1 合并 SpO₂ 下降    |
| Level 3 | Level 2 合并 HR 或 BP 异常 |

该主线最接近 NICU apnea 的临床逻辑，也最适合与医院方合作。

---

### 5.2思路二：ICU 外周低灌注 / 血流动力学不稳定

第二个值得推进的方向是利用手掌、四肢和 IR 视频分析外周灌注状态。

拟定题目：

> Contactless Assessment of Peripheral Perfusion and Hemodynamic Instability in ICU Patients Using RGB–Infrared Video

核心假设：

> 手掌和四肢 RGB–IR 视频中的脉搏幅度、颜色变化和信号可用性，可以反映外周低灌注和血流动力学不稳定。

相比直接做 BP 回归，这个方向更合理。因为 BP label 可能是离散、不连续或延迟的，而外周灌注本身就是一个更适合视频观察的临床变量。

可用 endpoint 包括：

* MAP < 65 mmHg；
* hypotension episode；
* peripheral pulse amplitude drop；
* video-derived perfusion signal loss；
* low perfusion index surrogate；
* vasopressor escalation，如果后续能拿到用药信息；
* SpO₂ poor perfusion / signal loss；
* 四肢与面部信号差异。

---

### 5.3 思路三：接触式监护失效时的视频冗余监护

第三个方向是把视频监测定位为 ICU 接触式监护的补充层或冗余层。

拟定题目：

> Video-Based Contactless Monitoring as a Redundancy Layer for ICU Vital Sign Surveillance

核心假设：

> 当 ECG 电极、SpO₂ 探头或 BP 监测出现缺失、伪影或不稳定时，视频非接触监测可以提供额外的生命体征趋势信息。

该方向的临床意义较强，尤其适合强调：

* 接触式传感器脱落；
* 低灌注导致 SpO₂ 不可靠；
* 护理操作造成监测中断；
* 线缆影响护理流程；
* 皮肤脆弱患者的接触式监测负担；
* ICU 报警疲劳。

但这个方向需要能够标记 contact sensor failure 或 reference signal artifact，因此对数据标注要求略高。

---

## 6. 不建议作为当前主线的方向

### 6.1 不建议主打单点血糖估计

虽然目前采集了单点血糖 label，但我认为不适合作为当前论文主线。

原因包括：

* 单点血糖与短视频窗口之间缺乏明确同步关系；
* 血糖变化时间尺度较长；
* ICU 血糖受胰岛素、营养、感染、应激、激素等因素影响；
* 直接从视频估计血糖的生理机制不足，容易受到审稿质疑。

后续可考虑作为 exploratory analysis，例如糖尿病/非糖尿病亚组、血糖异常患者的外周灌注差异等。

---

### 6.2 不建议主打直接 BP 数值回归

除非后续有高质量连续 arterial BP waveform，否则不建议将 BP 数值估计作为主任务。

更合理的方向是：

* hypotension event detection；
* MAP 下降趋势预测；
* hemodynamic instability classification；
* BP drop risk stratification。

这样比直接回归 SBP/DBP/MAP 更符合临床逻辑。

---

## 7. 目标期刊初步判断

如果论文强调医学临床结合，同时希望保留 IEEE 体系，可以考虑以下方向。

### 第一优先级

#### IEEE Journal of Biomedical and Health Informatics

虽然不是 Transactions，但它非常适合“医疗 AI + 临床数据 + 生命体征监测 + 下游事件检测”的方向。若论文重点是 ICU/NICU 场景、多模态生命体征、临床事件识别，J-BHI 是最匹配的 IEEE 期刊。

#### IEEE Transactions on Biomedical Engineering

如果希望投稿 IEEE Transactions，TBME 是最合适的目标。该期刊适合 biomedical engineering system、非接触生理测量、多模态传感和临床验证。

### 第二优先级

#### IEEE Transactions on Instrumentation and Measurement

如果后续将论文写成“ICU 环境下 RGB–IR 非接触生命体征测量系统的可靠性评估”，则 TIM 也较合适。它更看重测量系统、误差来源、校准、不确定性和可靠性。

#### IEEE Transactions on Medical Imaging

只有当论文能够强化为 optical functional imaging、perfusion/oxygenation imaging 或 computational physiological imaging 时，才建议考虑 TMI。否则风险较高。

---

## 8. 下一步计划

### 8.1 数据和标签整理

下一步我计划优先完成以下标签重构：

1. 从 ECG / monitor HR 中提取：

   * bradycardia；
   * tachycardia；
   * HR variability；
   * possible arrhythmia flags。

2. 从 SpO₂ 中提取：

   * SpO₂ < 90%；
   * SpO₂ drop ≥ 3% 或 ≥ 4%；
   * desaturation duration；
   * desaturation burden。

3. 从 BP 中提取：

   * MAP < 65 mmHg；
   * hypotension episode；
   * BP drop trend；
   * hemodynamic instability label。

4. 从胸部视频或呼吸 reference 中提取：

   * apnea；
   * hypopnea；
   * abnormal RR；
   * respiratory irregularity。

---

### 8.2 实验设计调整

后续实验不再只围绕 MAE，而是加入事件级评价指标：

* AUROC；
* AUPRC；
* sensitivity；
* specificity；
* F1-score；
* false alarm per patient-hour；
* event onset time error；
* signal availability；
* coverage；
* Bland–Altman analysis；
* subgroup analysis。

同时，我会加强以下分层分析：

* RGB vs IR vs RGB+IR；
* face vs palm vs limb vs chest；
* 白天 vs 夜间；
* 遮挡 vs 非遮挡；
* 机械通气 vs 非机械通气；
* 镇静 vs 非镇静；
* 高质量 reference vs 低质量 reference；
* 不同患者亚组。

---

### 8.3 成文结构初稿

后续文章结构可以按以下方式组织。

#### Introduction

从 ICU 临床需求出发，而不是从 rPPG 算法出发：

1. ICU 患者需要连续生命体征监测；
2. 接触式监护存在皮肤损伤、线缆、探头脱落、护理负担和报警疲劳等问题；
3. 视频非接触监测可作为补充监测方式；
4. 当前研究多数停留在 HR/RR 数值估计，缺乏面向 ICU 临床事件的验证；
5. 本研究提出基于多部位 RGB–IR 视频识别 ICU 呼吸-循环失代偿事件。

#### Methods

重点描述：

1. ICU 数据采集；
2. RGB+IR 同步系统；
3. 多部位 ROI；
4. ECG/BP/SpO₂ reference alignment；
5. 事件定义；
6. patient-level split；
7. signal quality control；
8. 模型和评价指标。

#### Results

建议结果顺序：

1. 数据集和患者基本信息；
2. HR/RR 基础准确性；
3. 呼吸异常事件检测；
4. desaturation 事件检测；
5. HR/BP 联合异常检测；
6. 多模态和多部位消融；
7. signal availability；
8. failure cases；
9. 临床亚组分析。

#### Discussion

重点讨论：

1. ICU 非接触监测的临床可行性；
2. 事件检测相比单纯 MAE 的医学意义；
3. RGB+IR 和多部位采集的价值；
4. 低灌注、遮挡、护理操作等真实 ICU 挑战；
5. 视频监测不能替代 bedside monitor，但可以作为 supplementary / redundancy layer；
6. 后续需要更大样本、多中心、前瞻性验证。

---

## 9. 本周结论

整体判断：

1. 现有 ICU 视频数据是可用的，HR MAE 2.1 bpm 结果可以作为重要可行性证据；
2. 论文不应仅以 rPPG 心率估计为核心，而应转向临床事件识别；
3. 最推荐的成文方向是 ICU 呼吸-循环失代偿事件检测；
4. 第二推荐方向是外周低灌注 / 血流动力学不稳定的非接触评估；
5. 单点血糖和直接 BP 回归暂时不适合作为主线；
6. 后续应优先构造 apnea、desaturation、bradycardia、tachycardia、hypotension 等事件级标签；
7. 投稿方向上，J-BHI 和 TBME 是目前最合适的目标，TIM 可作为测量系统版本的备选。

总体来说，下一阶段的重点是把项目从“rPPG 多任务模型”升级为“ICU 临床事件非接触监测研究”，使论文更加贴近医院需求和临床价值。
