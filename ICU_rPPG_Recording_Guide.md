# ICU rPPG 录制与对齐复现指南

## 目标与适用范围
本指南用于严谨复现 ICU 场景下的 rPPG 数据采集流程，适配两类设备方案：
- RealSense（RGB + IR1 + IR2，多模态）
- MX BRIO（仅 RGB，低部署复杂度）

适用脚本：
- 采集脚本：`/home/gem/czx/Code/ZPH/rec3.py`
- 严格帧标签对齐脚本：`/home/gem/czx/Code/ZPH/strict_frame_label_align.py`

本指南重点解决两个核心问题：
- 如何稳定录制可用于训练/评估的视频与生理信号
- 如何解释并修正 ICU 场景中 `label` 常比视频多 `30` 或 `60` 行的问题

## 采集系统架构与时间基准
### 架构说明
- 视频链路：摄像头持续输出帧（RealSense 可同时输出 RGB/IR；MX BRIO 输出 RGB）。
- 生理链路：迈瑞监护仪通过串口/中间日志输出 HL7 波形包（包含 PLETH、采样率、OBR 时间窗等）。
- 对齐链路：先记录原始日志与视频帧时间，再通过离线严格对齐生成“一帧一标签”。

### 时间基准原则
- 采集阶段使用主机系统时钟和单调时钟做统一锚点，不直接依赖“包到达时刻=真实采样时刻”。
- `rec3.py` 中保存：
  - `timestamp_log.txt`（启动、触发、偏移估计）
  - `video_frame_log.csv`（每帧主机时间和原始帧号）
  - `mindray_parsed_samples.csv`（逐样本展开后的生理数据）
- 离线阶段用 `strict_frame_label_align.py` 统一执行首尾裁剪和重采样，输出严格匹配的帧标签表。

## 部署方法
### A. RealSense（RGB + IR）
- 推荐配置：`RGB 1280x720@30fps`，`IR1/IR2 848x480@30fps`（与 `rec3.py` 一致）。
- 关键依赖：
  - `pyrealsense2`
  - `imageio`（FFV1 写入）
  - `ffprobe`（后处理和校验）
- 实施要点：
  - USB 带宽优先保证摄像头稳定供流，尽量直连主板高速口。
  - 固定曝光/白平衡优先于自动模式，避免光照变化导致脸部强波动。
  - 开始前确认 IR 通道与 RGB 同时可读，否则直接中止并重启设备。

### B. MX BRIO（仅 RGB）
- 适合部署快速、床旁空间受限场景。
- 保留 `video_frame_log.csv` 机制，确保后续与 PLETH 可回放对齐。
- 若脚本由 RealSense 改为 BRIO，至少保持以下不变：
  - 每帧 `HostMonotonicSec` 与 `HostSystemSec` 写入
  - `RGB_FrameNumber` 连续性检查
  - `timestamp_log.txt` 时间锚点写入

## 采集前准备（床旁环境）
### 病床与相机相对位置
- 建议脸部到镜头距离：`0.6m ~ 1.2m`（优先保证面部占画面 1/4~1/2）。
- 镜头高度尽量与面部中轴接近，俯仰角控制在小角度范围，减少鼻梁/额头阴影。
- 机位固定后全程不移动，避免重定位导致 ROI 统计分布漂移。

### 环境光控制
- 避免强背光、频闪光源和床旁设备指示灯直射面部。
- 夜班场景尽量维持稳定低照，不要频繁开关顶灯。
- 如必须变化照明，记录时间点，便于后处理分段筛除异常。

### 对焦与清晰度
- 以额头和双颊区域清晰为优先，不追求背景清晰。
- 自动对焦设备建议锁焦后再录制；若无法锁焦，至少保持机位与病人头位稳定。
- 开录前保存一张预览帧进行人工确认（曝光不过饱和、肤色区域不过暗）。

## 标准录制流程（可复现）
### 1. 启动检查
- 确认监护仪波形流正常输出（可见 HL7 `OBR/OBX` 连续包）。
- 确认摄像头可稳定出帧（帧号连续、无长时间卡顿）。
- 清空或新建独立采集目录，避免历史文件混入。

### 2. 执行录制
- 启动 `rec3.py`，录制期间不要手工中断串口监听线程。
- 录制目标时长按方案设定（脚本默认 `TARGET_SECONDS=1800`）。
- 过程中文件应持续增长：
  - `recorded_mindray.txt`
  - `mindray_parsed_samples.csv`
  - `video_frame_log.csv`
  - `frame_level_sync.csv`

### 3. 结束与完整性核验
- 确认 `timestamp_log.txt` 包含：
  - `Camera open time`
  - `Video write start time`
  - `Recording end time`
  - `Estimated clock offset`
- 用 `ffprobe` 验证视频可读且有连续帧。
- 快速检查 `video_frame_log.csv` 是否存在明显帧号跳变。

## ICU 场景常见问题与针对性方案
### 问题 1：`label` 比视频多 30 或 60 行
- 现象解释：
  - 迈瑞 PLETH 常为 `60Hz`，而训练标签一般按 `30fps`对齐。
  - 若波形在首尾多一个 1 秒窗口，会表现为约 `+30`；多两个窗口则 `+60`。
  - 这属于“窗口边界与视频启动/结束不完全同刻”的系统性偏差，不等于采集失败。
- 代码层根因：
  - `rec3.py` 在收到首个包后触发视频，且停止时会处理尾包，边界可能多吃包。
- 解决方案：
  - 禁止直接用原始 `label.csv` 长度判断失败。
  - 统一执行 `strict_frame_label_align.py`，按真实帧时间做重采样并裁剪，输出严格一帧一标签。

### 问题 2：帧号不连续、局部掉帧
- 现象：
  - `RGB_FrameNumber` 非 `+1` 连续，或 PTS 出现异常大跳变。
- 解决方案：
  - `strict_frame_label_align.py` 已内置“最长连续有效段”选择机制，会剔除断裂段。
  - 采集端优化 USB 带宽和编码写盘性能，优先减少源头掉帧。

### 问题 3：视频正常但对齐后可用段很短
- 原因：
  - 时钟偏移估计不稳、监护仪日志间断、病人姿态变化剧烈。
- 解决方案：
  - 检查 `timestamp_log.txt` 是否写出了 `Estimated clock offset`。
  - 检查 `recorded_mindray.txt` 是否存在长时间无 `Received` 包。
  - 复核床旁机位与光照是否中途变化过大。

### 问题 4：ffprobe 报错或视频损坏
- 现象：
  - `EBML header parsing failed` 或 `Invalid data found when processing input`。
- 解决方案：
  - 采集端保证正常退出写入器，避免强制杀进程。
  - 后处理时将坏文件标记为异常样本，不参与训练集主干。

## 严格对齐与重采样操作
### 执行命令
```bash
python /home/gem/czx/Code/ZPH/strict_frame_label_align.py --base-dir <单次采集目录> --fps 30
```

### 输出文件说明
- `frame_label_aligned_output/rgb_aligned.mkv`：裁剪后视频
- `frame_label_aligned_output/pleth_aligned.csv`：与视频逐帧严格一一对应的标签
- `frame_label_aligned_output/trim_log.txt`：本次裁剪依据、偏移来源和丢弃帧统计

### 结果验收标准
- `pleth_aligned.csv` 行数应等于 `rgb_aligned.mkv` 帧数。
- `trim_log.txt` 中 `selected_run_len` 合理，`total_dropped_frames` 不应异常高。
- 对齐后随机抽查波形趋势与视频中脉搏可见区域变化是否同频。

## 快评估追加分析（写回CSV）
### 目的与方法
- 目的：在总评估表中补充“较好/较差 subject”的快速可解释分析，辅助判断问题来自对齐、运动、遮挡还是光照噪声。
- 方法：使用无监督基线 `GREEN`（ROI 绿色通道均值）或预训练模型进行滑窗分析，与重采样后的标签做相关性对比。
- 脚本：`/home/gem/czx/Code/ZPH/quick_subject_rppg_analysis.py`

### 执行命令
```bash
python /home/gem/czx/Code/ZPH/quick_subject_rppg_analysis.py \
  --assessment-csv /home/gem/czx/Code/ZPH/rppg_data_quality_assessment.csv \
  --output-csv /home/gem/czx/Code/ZPH/rppg_data_quality_assessment.csv \
  --shots-dir /home/gem/czx/Code/ZPH/subject_frame_shots
```

### 追加字段说明
- 该脚本会在 `rppg_data_quality_assessment.csv` 追加以下列：
  - `quick_method`、`quick_corr_mean`、`quick_corr_median`
  - `quick_corr_best`、`quick_corr_worst`
  - `quick_best_range_sec`、`quick_worst_range_sec`
  - `quick_motion_mean`、`quick_illum_std`
  - `quick_reason`、`quick_summary_level`
  - `quick_best_shot`、`quick_worst_shot`
- 其中 `quick_best_range_sec/quick_worst_range_sec` 为效果较好/较差时间段，`quick_best_shot/quick_worst_shot` 为对应截图路径。

### 结果使用规范
- 若 `quick_reason` 指向“对齐边界问题”，优先回查 `label_rows-rgb_frames` 和 `trim_log.txt`。
- 若 `quick_motion_mean` 偏高，优先归因为运动/ROI漂移并回看截图。
- 若 `quick_illum_std` 偏高，优先归因为光照不稳或噪声放大。
- 快评估用于快速筛查，不替代最终模型实验；最终结论仍需结合训练/验证指标确认。

## 质量控制建议
### 采集阶段
- 每次采集保留完整原始日志，不只保留最终 `label.csv`。
- 对每个 subject 固化“机位-光照-距离”记录模板。
- 同一病区尽量使用同款相机和同一参数模板，减少域偏移。

### 后处理阶段
- 先跑严格对齐，再进行数据质检统计（帧数、掉帧、有效率）。
- 将异常样本（坏视频、长中断、偏移不可估）单独分层管理，不混入主训练集。

## 版本与复现记录模板
建议每批次记录以下元数据：
- 采集脚本版本（git commit / md5）
- 相机型号与固件版本
- 监护仪型号与数据出口版本
- Python 版本与关键依赖版本
- 采集日期、病区、机位距离、光照条件
- 知情同意书（每例病人均需获取1份）

---

cite：MPU澳门理工大学 ZHENGXUAN CHEN 2026.4 撰写和首发
