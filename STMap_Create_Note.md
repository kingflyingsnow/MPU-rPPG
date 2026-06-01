区分：

PhysMLE/HSRD 原版 STMap 风格：更偏“landmark 对齐人脸后再 5×5 分块”。HSRD 的官方流程是 Landmark.py → Landmark_proce.py → Align_Face.py → STMap.py，README 明确写了先用 face_alignment 取关键点，再插值异常关键点、对齐人脸、生成 30 FPS STMap。
Y5F ROI STMap 风格：用 YOLO5Face 检测框裁脸，resize 后直接做 5×5 区域 RGB 均值，得到 [25, T, 3] 的 STMap。这和原版不完全一致，但很实用，尤其适合 ICU 视频里姿态、遮挡、检测失败多的情况。

rPPG-Toolbox 里 BACKEND 支持 HC 和 Y5F，其中 Y5F 是 YOLO5Face；它还支持动态检测、检测频率、中位数 face box、大框系数等参数。 它的 BaseLoader.crop_face_resize() 会先调用 Y5F/HC 做 bbox，再裁剪并 resize 成固定大小帧；Y5F 返回的是 square face box。

建议

训练自己的 ICU 数据：可以用 Y5F ROI-STMap。
复现 PhysMLE/GAP 的公开结果或加载它们预训练模型做公平比较：最好用 landmark-aligned STMap。

原因是：PhysMLE/HSRD 那套 STMap 的 domain 包含“人脸对齐”的先验；Y5F bbox-STMap 的每个 grid 位置不一定稳定对应同一块脸部区域。例如第 1 行第 2 列，在 landmark-align 后可能总是额头/眼周附近；但 bbox crop 在转头、低头、呼吸机遮挡时会漂。对 SpO₂ 这种依赖 RGB 通道比例/低频颜色变化的任务，ROI 稳定性很重要。

不过 ICU 场景里，landmark 经常被口罩、鼻导管、氧管、低光、偏头影响。Y5F bbox 反而更稳，工程上更容易批处理。

推荐实现：Y5F crop + median box + 5×5 STMap

把 STMap 定义成：

video frames
  → Y5F 检脸
  → 用 median face box 稳定整段视频 ROI
  → crop + resize, 例如 160×160
  → 5×5 grid
  → 每个 grid 求 RGB mean
  → [T, 25, 3]
  → 插值到 30 FPS
  → 每个 region/channel min-max 到 0–255
  → 保存为 [25, T, 3] 的 STMap.png

这个和 HSRD STMap.py 的核心形式兼容：HSRD 的 getValue() 也是把图像分成 5×5 区域并取 RGB 均值，之后 cubic spline 到 30 FPS，再归一化到 0–255，并 swapaxes 成 [25, T, 3]。
