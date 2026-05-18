# FacePhys Torch Demo 指南

## 1. 这是什么

本项目当前提供了一个基于 PyTorch 版 `FacePhys_re.py` 和权重 `UBFC_PURE_FacePhys_Torch_Basic_Epoch9.pth` 的实时 demo 脚本：

- 脚本：`facephys_torch_demo.py`
- 模型定义：`FacePhys_re.py`
- 权重：`UBFC_PURE_FacePhys_Torch_Basic_Epoch9.pth`

当前 demo 目标不是一次性完全照搬官方 Web Demo，而是按 phase 渐进复刻官方的工程妙点，先把最核心、最稳定的实时主链做对。

---

## 2. 当前 demo 已实现的功能

当前版本是在 `Phase 1 + M15` 基础上，重建了一条更稳的轻量后处理链，已经完成：

- 权重加载
  - 严格加载当前 `FacePhys_re.py` 与 `Epoch9` 权重
- 单帧递推推理
  - 使用 `init_state()` 初始化状态
  - 每帧调用 `step()` 做 stateful 推理
- ROI 处理
  - MediaPipe 人脸检测
  - 检测降频，非每帧都做真正检测
  - 检测前缩小到较小分辨率再送入检测器
  - 人脸框平滑
  - 人脸框扩框
  - 丢脸短时缓冲，避免闪脸后立刻重置
  - ROI 裁剪并缩放到 `72 x 72`
- 时间处理
  - 摄像头模式下使用独立采集线程持续抓帧
  - `dt` 使用真实采集时间戳而不是显示循环时间
  - 不直接假定固定 30 FPS
- BVP 主链
  - 每帧输出一个 BVP 值
  - 维护实时 BVP 缓冲区
- 无脸处理
  - 单次漏检不会立即闪脸
  - 会在短时宽限内继续沿用最近一次 ROI
  - 连续漏检超过阈值后才判定 `Face: No`
  - 真正失效时才清空 BVP 主缓冲并重置模型 state
- 当前轻量后处理
  - 基于最近 `300` 点 BVP 的 FFT 心率估计
  - 可靠性分数
  - HR 防跳变与 EMA 平滑
  - 固定量程 + 环形缓存的 BVP 波形显示
- 基础显示
  - 视频画面
  - 人脸框
  - ROI 小窗
  - BVP 波形小窗
  - FPS、dt、BVP、Frame、Face、Track、HR、Rel 状态文字
  - `Cap FPS / Proc FPS / Det / Infer / Render / Drops` 性能指标
- 多速率调度
  - 采集线程持续抓帧
  - 推理主链尽量消费最新帧
  - 显示面板按单独显示帧率节流，减少 UI 对主链的拖累
- CSV 导出
  - 保存每帧时间戳、帧号、dt、bvp、face_found、face_status、bbox、hr、reliability、valid
- 视频离线模式
  - `--video` 时不再走 realtime 单帧 demo
  - 先抽整段 ROI，再按 `CHUNK_LENGTH=160` 的 clip 方式做 `model.forward()`
  - 最后统一导出波形 CSV 和 HR CSV

---

## 3. 当前 demo 的工作原理

当前脚本现在有两条模式：

- `camera`：实时 demo，走单帧 `step()` 状态递推
- `video`：离线评估，走 clip `forward()` + 统一 HR/波形导出

其中实时模式可以理解为一条流：

`视频帧 -> 人脸检测 -> ROI裁剪 -> 72x72 RGB -> 单帧 step 推理 -> 当前帧 BVP -> 缓冲 -> 显示/导出`

更具体一点：

### 3.1 输入

- 摄像头或视频文件提供原始 BGR 帧
- 摄像头模式下，后台采集线程会持续读取最新帧
- 主线程推理时总是取最近一次采集到的帧，尽量贴近真实相机时间轴
- 视频模式下，会顺序读完整段视频并缓存 ROI 序列，之后再统一做离线推理

### 3.2 ROI

- 先做人脸检测
- 检测不是每帧都整图执行
- 中间帧优先复用上一帧平滑后的框
- 取最大人脸
- 做扩框
- 再做指数平滑，减少 ROI 抖动
- 检测前会先缩到较小宽度，降低 MediaPipe 负担
- 裁剪后 resize 到训练时一致的 `72 x 72`

### 3.3 单帧推理

- 模型不是一次输入一段 clip 再统一输出
- 而是维护内部状态 `state`
- 第一次运行时用 `model.init_state(batch_size, height, width, ...)`
- 之后每帧调用 `model.step(x_t, state, dt)`

这正是当前 demo 最关键的“官方风格”部分：

- `stateful`
- `single-frame inference`
- `dt-aware`

### 3.4 dt

- 每帧根据真实采集时间计算原始间隔
- 再做平滑，得到更稳定的 `dt`
- 最后把这个 `dt` 显式送进模型

这和“假设永远 30fps”的写法不同，更接近真实部署环境。

### 3.4b 视频离线模式

- `--video` 时，重点不再是实时显示，而是尽量贴近 `test` 路线
- 先抽完整段人脸 ROI，并保持时间轴
- 将整段 ROI 做 `Standardized` 预处理
- 按 `160` 帧 clip 喂给 `model.forward()`
- 拼接得到整段预测波形
- 再对整段波形做 `detrend + bandpass + FFT HR`
- 输出：
  - `*_wave.csv`
  - `*_hr.csv`

### 3.5 采集与显示解耦

- 摄像头模式下，采集线程和显示线程解耦
- 即使窗口显示只能到 `23 FPS` 左右，模型仍优先消费最新采集帧
- 这不能保证推理一定满 `30 FPS`
- 但能避免“显示卡顿导致输入时间轴也被拖慢”的问题
- 因此对 stateful 模型更合理
- 当前还额外把显示刷新从推理主链中节流出来
- 也就是推理尽量跟着最新采集帧走，而窗口面板允许低于推理频率刷新
- 这和官方 demo 里“推理、绘图、趋势更新分频”的思路更接近

### 3.6 输出

- 每帧得到一个标量 BVP
- BVP 被放进缓冲区
- 波形显示采用固定量程 + 环形缓存
- 波形会在丢脸后插入空洞，避免把两段信号错误连在一起
- 如果当前无脸：
  - 单次漏检不会立刻中断
  - 在短时宽限内仍可沿用最近一次 ROI
  - 超过连续漏检阈值后才真正停止 BVP 累积
  - 此时波形断开，HR/Rel/Valid 进入无效状态
- 如果缓冲长度足够：
  - 从最近 `300` 点 BVP 估计真实采样率
  - 做 FFT 并在生理心率带内找主峰
  - 计算可靠性分数
  - 对跳变过大的 HR 做短时抑制
  - 再做 EMA 平滑后输出 HR
- 最终写入 CSV

---

## 4. 当前配置说明

脚本主要配置来自 `DemoConfig`，重要项如下：

| 配置项 | 作用 | 当前默认值 |
| --- | --- | --- |
| `weights_path` | 预训练权重路径 | `UBFC_PURE_FacePhys_Torch_Basic_Epoch9.pth` |
| `output_dir` | CSV 输出目录 | `facephys_torch_demo_outputs` |
| `camera_id` | 摄像头编号 | `0` |
| `video_path` | 视频文件路径 | `None` |
| `device` | 推理设备 | 自动选 `cuda` / `cpu` |
| `input_size` | 模型输入尺寸 | `72` |
| `camera_backend` | 摄像头后端，Windows 下 `auto` 会优先尝试 `MSMF` | `auto` |
| `target_fps` | 初始目标帧率 | `30.0` |
| `camera_buffer_size` | 摄像头缓存大小，建议尽量小 | `1` |
| `use_async_capture` | 是否使用异步采集线程 | `True` |
| `bbox_smooth` | 人脸框平滑系数 | `0.85` |
| `face_expand` | 人脸框扩框比例 | `1.5` |
| `face_min_size` | 最小人脸尺寸 | `120` |
| `mediapipe_min_confidence` | MediaPipe 检测最小置信度 | `0.5` |
| `face_detect_interval` | 每隔多少处理帧才做一次真正检测 | `3` |
| `face_detect_input_width` | 检测前缩放到的目标宽度 | `320` |
| `face_miss_tolerance_frames` | 允许连续漏检的最大帧数 | `6` |
| `face_miss_tolerance_ms` | 允许连续漏检的最大时间窗口 | `350` |
| `buffer_size` | BVP 缓冲长度 | `450` |
| `wave_points` | 显示波形长度 | `225` |
| `wave_y_min` | 波形固定量程下界 | `-1.5` |
| `wave_y_max` | 波形固定量程上界 | `2.0` |
| `wave_scan_gap` | 环形波形的扫描缺口宽度 | `10` |
| `hr_window_size` | FFT HR 使用的最近样本数 | `300` |
| `hr_update_stride` | 每隔多少帧重估一次 HR | `5` |
| `hr_min_bpm` | 心率下界 | `42` |
| `hr_max_bpm` | 心率上界 | `180` |
| `hr_reliability_threshold` | HR 可靠性阈值 | `0.35` |
| `hr_ema_alpha` | HR EMA 平滑系数 | `0.25` |
| `hr_max_jump_bpm` | 单次 HR 允许直接跳变的上限 | `15.0` |
| `hr_jump_confirm_frames` | 极端跳变需要连续确认的次数 | `3` |
| `save_csv` | 是否导出 CSV | `True` |
| `display_max_fps` | 显示面板的最大刷新率 | `15.0` |
| `video_chunk_length` | 视频离线推理的 clip 长度 | `160` |
| `video_hr_window_seconds` | 视频 HR CSV 的窗口长度 | `10.0` |
| `display_scale` | 显示缩放比例 | `1.0` |
| `no_display` | 是否不弹窗运行 | `False` |

说明：

- 默认配置 `72 x 72` ，区别于官方 Web Demo 的 `36 x 36`
- 当前实现优先保持和 Torch 权重一致

---

## 5. 如何运行

### 5.1 摄像头运行

```bash
python facephys_torch_demo.py
```

### 5.2 指定 CPU

```bash
python facephys_torch_demo.py --device cpu
```

### 5.3 指定视频文件

```bash
python facephys_torch_demo.py --video your_video.mp4
```

说明：

- 这条命令现在会进入离线模式
- 不再按 realtime 单帧 UI 路线跑
- 重点输出整段视频的波形 CSV 和 HR CSV

### 5.4 后台运行，不弹窗

```bash
python facephys_torch_demo.py --no-display
```

### 5.5 不保存 CSV

```bash
python facephys_torch_demo.py --no-save-csv
```

### 5.6 可用控制

- 按 `q` 退出
- 按 `ESC` 退出

### 5.7 采集相关参数

```bash
python facephys_torch_demo.py --target-fps 30 --camera-buffer-size 1
python facephys_torch_demo.py --disable-async-capture
python facephys_torch_demo.py --face-detect-interval 3 --face-detect-input-width 320
python facephys_torch_demo.py --camera-backend auto
```

说明：

- 默认摄像头模式会开启异步采集线程
- Windows 下默认 `camera_backend=auto`，会优先尝试 `MSMF`，再回退 `DSHOW/ANY`
- 如果你怀疑某些驱动和线程读取有兼容问题，可以用 `--disable-async-capture` 回退为同步模式
- 如果 `Det` 明显偏高，可以先把 `--face-detect-interval` 调大，或把 `--face-detect-input-width` 再减小一些
- 如果某台机器在 `DSHOW` 下只能拿到 `15fps`，可显式指定 `--camera-backend msmf`

### 5.8 HR 参数

```bash
python facephys_torch_demo.py --hr-window-size 300 --hr-reliability-threshold 0.35
python facephys_torch_demo.py --hr-max-jump-bpm 15 --hr-jump-confirm-frames 3
```

### 5.8b 视频离线参数

```bash
python facephys_torch_demo.py --video your_video.mp4 --video-chunk-length 160 --video-hr-window-seconds 10
```

说明：

- `video_chunk_length` 默认对齐当前训练/测试配置中的 `CHUNK_LENGTH=160`
- `video_hr_window_seconds` 用于生成 HR CSV 的分窗结果
- 同时还会额外给出一条 `full_video` 的整段 FFT HR

### 5.9 波形参数

```bash
python facephys_torch_demo.py --wave-y-min -1.5 --wave-y-max 2.0 --wave-scan-gap 10
```

### 5.10 显示刷新参数

```bash
python facephys_torch_demo.py --display-max-fps 15
```

说明：

- 这个参数限制的是窗口面板刷新速度，不是模型输入速度
- 当机器较慢时，优先降低显示刷新率，通常比直接降低采集目标帧率更符合当前主线

### 5.11 闪脸保护

```bash
python facephys_torch_demo.py --face-miss-tolerance-frames 8 --face-miss-tolerance-ms 450
```

---

## 6. CSV 导出内容

当前 CSV 每帧保存以下字段：

- `timestamp_ms`
- `frame_idx`
- `dt_sec`
- `bvp`
- `face_found`
- `face_status`
- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`
- `hr_bpm`
- `reliability`
- `hr_valid`

用途：

- 做离线调试
- 检查 ROI 抖动
- 检查 dt 是否异常
- 用于后续 HR / reliability 离线分析
- 检查最终 HR 是否仍有大幅跳变

---

## 7. 当前 demo 与官方 FacePhys Demo 的区别

下面尽量直观对比，不长篇展开。

### 7.1 总体定位

| 项目 | 当前 Torch Demo | 官方 FacePhys Demo |
| --- | --- | --- |
| 技术栈 | Python + OpenCV + PyTorch | Web + LiteRT + Web Workers |
| 主要目标 | 先打通实时主链 | 完整产品化实时展示 |
| 权重来源 | 你的 Torch 预训练 | 官方 LiteRT / ONNX / JAX 发布链 |

### 7.2 推理主链

| 项目 | 当前 Torch Demo | 官方 Demo |
| --- | --- | --- |
| 单帧递推 | 已实现 | 已实现 |
| `dt` 显式输入 | 已实现 | 已实现 |
| 状态缓存 | 已实现 | 已实现 |
| 周期性状态持久化 | 未实现 | 已实现 |

### 7.3 ROI 与输入

| 项目 | 当前 Torch Demo | 官方 Demo |
| --- | --- | --- |
| 检测器 | MediaPipe Face Detection | MediaPipe FaceDetector |
| 输入尺寸 | `72 x 72` | `36 x 36` |
| 检测节奏 | 降频检测 + 中间复用 ROI | 视频模式检测 + 主链复用 ROI |
| 人脸框平滑 | 已实现 | 已实现 |
| 扩框 | 已实现 | 已实现 |

### 7.4 后处理与展示

| 项目 | 当前 Torch Demo | 官方 Demo |
| --- | --- | --- |
| BVP 波形 | 固定量程 + 环形缓存版 | 已实现完整版 |
| PSD | 未实现 | 已实现 |
| HR 数值 | 最近 300 点 FFT + 防跳变 | 已实现 |
| SQI 门控 | 未实现独立模型 | 已实现模型版 |
| HR Trend | 未实现 | 已实现 |
| ROI 小窗 | 已实现 | 已实现 |
| Attention 热力图 | 未实现 | 已实现 |
| Heart State 轨迹 | 未实现 | 已实现 |

### 7.5 工程封装

| 项目 | 当前 Torch Demo | 官方 Demo |
| --- | --- | --- |
| 采集异步化 | 已实现 | 已实现 |
| 推理/显示分频 | 已实现基础版 | 已实现 |
| 多线程/多 worker | 基础版 | 已实现完整版 |
| OffscreenCanvas | 不适用 | 已实现 |
| PWA / 离线安装 | 未实现 | 已实现 |
| ZIP 导出 | 未实现 | 已实现 |
| 浏览器端本地推理 | 不适用 | 已实现 |

一句话总结：

- 当前 Torch Demo 已经对齐了官方最重要的“实时单帧主链”
- 并开始对齐官方“多速率实时调度”的工程思路
- 但还没有对齐官方完整的“展示系统”和“解释性旁路”

---

## 8. 当前 phase 进度

### 已完成阶段

当前处于：

- `Phase 1`
- 外加 `M15 CSV 导出`
- 并已接入轻量 HR 后处理

也就是：

- `M1 模型与权重加载`
- `M2 单帧状态推理`
- `M3 ROI 检测与裁剪`
- `M4 dt 估计`
- `M5 BVP 环形缓冲`
- `M14 Demo 显示最小版`
- `M15 CSV 导出`
- `M6 HR 后处理（300 点 FFT）`
- `M16 生命周期与容错的一部分`
- MediaPipe ROI、闪脸缓冲、固定量程环形波形、HR 防跳变
- ROI 降频检测、小图检测、基础性能统计、显示刷新节流

### 当前还没有做的部分

- `M8 波形展示进一步增强`
- `M11 中间特征导出`
- `M12 注意力/热力图`
- `M13 Heart State 轨迹`
- `M16 生命周期与容错增强（完整）`
- `M17 更完整配置管理`

---

## 9. 后续 phase 计划

### Phase 2：实用版监测

目标：不改模型、不重训，先把 demo 做到“能用、能看、能分析”。

当前已完成的轻量版后处理：

- `M6 HR 后处理`
  - 当前基于最近 300 点 BVP 的 FFT
- `M16 生命周期与容错`
  - 当前已处理短时丢脸缓冲和真正失脸后的重置

后续建议补全的 Phase 2：

- `M8 波形展示增强`
  - 继续向官方示波器风格靠近
- `M16 生命周期与容错`
  - 丢脸后的短时重连策略
  - 视频结束、重置、摄像头异常处理更完整
- Phase2 面板布局与刷新策略继续优化
- CSV / 日志进一步结构化

这一阶段通常：

- 不需要改 `model.py`
- 不需要重训

### Phase 3：解释性与官方展示感增强

目标：靠近官方 demo 的可解释性和展示效果。

建议包含：

- `M11 中间特征导出`
- `M12 注意力/热力图`
- `M13 Heart State 轨迹`
- 更完整 UI 拼接和多面板展示

这一阶段通常：

- 可能需要改模型暴露接口
- 某些功能可能需要补训练
- 尤其是如果想接近官方 `projModel` 风格的解释性旁路

---

## 10. 是否需要改模型并重训

### 对 Phase 2

一般不需要。

因为这一阶段主要是：

- 从已有 BVP 做 HR
- 做可靠性和防跳变
- 做更好的 ROI、波形和显示

这些更偏后处理和工程组织，不强依赖改模型结构。

- 当前 HR 稳定化链条包含：
- MediaPipe ROI
- 闪脸短时缓冲
- 最近 300 点 FFT
- 可靠性分数
- HR EMA 平滑
- 极端跳变需要连续多次确认后才放行

### 对 Phase 3

不一定，但概率明显更高。

如果只想做“近似解释性”：

- 可以先用 hooks
- 或做简化的特征可视化

如果想尽量接近官方 demo：

- 很可能需要额外的解释性投影头
- 甚至需要额外训练

---

## 11. 当前文件关系

以下几个关键文件：

- `facephys_torch_demo.py`
  - 当前 realtime demo 主脚本
- `FacePhys_re.py`
  - 当前与 `Epoch9` 权重对齐的 Torch 模型实现
- `UBFC_PURE_FacePhys_Torch_Basic_Epoch9.pth`
  - 当前 demo 使用的权重
- `FACEPHYS_TORCH_DEMO_GUIDE.md`
  - 本说明文档

---

## 12. 当前最推荐的后续开发顺序

后续待开发，继续沿主线推进，顺序如下：

1. 完善 `Phase 2`
2. 优化波形、PSD、Trend 的显示风格与刷新逻辑
3. 增强无脸重连和容错策略
4. 再进入 `Phase 3`
5. 考虑解释性和官方可视化增强组件
