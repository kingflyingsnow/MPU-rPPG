#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_physmle_stmap_roi.py

一句话意义：把“原始视频 + CSV 标签”用 ROI/bbox 方式一次性导出为 PhysMLE 可读取的 STMap 数据目录。

输出目录形态：
    <out_root>/<dataset_name>/<subject_id>/STMap/STMap.png     # uint8, [25, T30, 3]
    <out_root>/<dataset_name>/<subject_id>/Label/HR.mat        # key='HR', shape=[T30]
    <out_root>/<dataset_name>/<subject_id>/Label/SPO2.mat      # key='SPO2', shape=[T30]
    <out_root>/<dataset_name>/<subject_id>/Label/BVP.mat       # key='BVP', shape=[T30]
    <out_root>/STMap_Index/<dataset_name>/*.mat                # key='Path','Step_Index'

依赖：
    pip install opencv-python numpy pandas scipy

Y5F 依赖：
    - 如果使用 --detector y5f，需要给 --rppg_toolbox_root 指向 rPPG-Toolbox 根目录。
    - 脚本会 import dataset.data_loader.face_detector.YOLO5Face，并复用 detect_face()。

说明：
    - 该版本不做 landmark 对齐，而是仿照 rPPG-Toolbox 的 Y5F 方框逻辑：检测 bbox -> 变成正方形 -> 可选放大 -> median box 稳定 -> crop/resize -> 5x5 STMap。
    - 输出的 STMap.png 维度仍是 PhysMLE 需要的一张 RGB 图：[25, T30, 3]。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d


# =========================
# 数据结构
# =========================

@dataclass
class SubjectIO:
    """
    一句话意义：保存单个受试者的原始输入路径与导出输出路径。
    输入：subject_dir 为原始受试者目录；video_path/label_path 为实际文件路径。
    输出：out_subject_dir 为 PhysMLE-like 的单受试者输出目录。
    维度：不涉及数组维度。
    """
    subject_id: str
    subject_dir: Path
    video_path: Path
    label_path: Path
    out_subject_dir: Path


# =========================
# 文件发现模块
# =========================

def find_first_file(subject_dir: Path, patterns: Iterable[str]) -> Optional[Path]:
    """
    一句话意义：在受试者目录下按 glob 规则找到第一个匹配文件。
    输入：subject_dir=目录；patterns=如 ['*.mp4','*.avi']。
    输出：Path 或 None。
    维度：不涉及数组维度。
    """
    for pat in patterns:
        hits = sorted(subject_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def discover_subjects(raw_root: Path, out_root: Path, dataset_name: str, video_patterns: List[str], label_name: str) -> List[SubjectIO]:
    """
    一句话意义：从 UBFC-like 原始目录自动发现每个 subject 的视频和标签。
    输入：raw_root 下每个子目录代表一个 subject；每个 subject 内含视频和 label CSV。
    输出：List[SubjectIO]。
    维度：不涉及数组维度。
    """
    subjects: List[SubjectIO] = []
    for subject_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        video_path = find_first_file(subject_dir, video_patterns)
        label_path = subject_dir / label_name
        if video_path is None:
            print(f"[跳过] {subject_dir.name}: 未找到视频，patterns={video_patterns}")
            continue
        if not label_path.exists():
            print(f"[跳过] {subject_dir.name}: 未找到标签 {label_path}")
            continue
        subjects.append(SubjectIO(subject_dir.name, subject_dir, video_path, label_path, out_root / dataset_name / subject_dir.name))
    return subjects


# =========================
# 视频读取模块
# =========================

def read_video_rgb(video_path: Path, max_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    一句话意义：读取视频为 RGB 帧序列，并生成每帧时间戳。
    输入：video_path=视频路径；max_frames=最多读取帧数，None 表示全部读取。
    输出：frames_rgb shape=[T,H,W,3] uint8；frame_times shape=[T] 秒；fps=float。
    维度：T=视频帧数，H/W=原始分辨率，3=RGB。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6 or np.isnan(fps):
        raise RuntimeError(f"视频 FPS 非法: {video_path}, fps={fps}")

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"视频没有读到帧: {video_path}")

    frames_rgb = np.stack(frames, axis=0).astype(np.uint8)
    frame_times = np.arange(frames_rgb.shape[0], dtype=np.float32) / fps
    return frames_rgb, frame_times, fps


# =========================
# 标签读取与同步模块
# =========================

def _pick_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """
    一句话意义：按别名从 CSV 表头里选择一列。
    输入：df=标签表；aliases=候选列名小写。
    输出：实际列名或 None。
    维度：不涉及数组维度。
    """
    lower_to_real = {c.lower().strip(): c for c in df.columns}
    for name in aliases:
        if name.lower() in lower_to_real:
            return lower_to_real[name.lower()]
    return None


def load_label_csv(label_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Optional[np.ndarray]]]:
    """
    一句话意义：读取 ICU/UBFC-like CSV 标签并自动识别 time/hr/spo2/bvp 列。
    输入：label_path=CSV；支持列名 time/timestamp/sec、HR、SPO2、BVP/PPG 等。
    输出：label_times shape=[L] 秒或 None；signals 字典，每个值 shape=[L] 或 None。
    维度：L=标签采样点数。
    """
    df = pd.read_csv(label_path)
    time_col = _pick_column(df, ["time", "timestamp", "sec", "seconds", "t"])
    hr_col = _pick_column(df, ["hr", "heart_rate", "heartrate", "pulse", "pulse_rate"])
    spo2_col = _pick_column(df, ["spo2", "sp_o2", "spo_2", "oxygen", "oxygen_saturation", "sao2"])
    bvp_col = _pick_column(df, ["bvp", "ppg", "pleth", "wave", "pulse_wave"])

    label_times = df[time_col].to_numpy(np.float32) if time_col else None
    signals: Dict[str, Optional[np.ndarray]] = {
        "HR": df[hr_col].to_numpy(np.float32) if hr_col else None,
        "SPO2": df[spo2_col].to_numpy(np.float32) if spo2_col else None,
        "BVP": df[bvp_col].to_numpy(np.float32) if bvp_col else None,
    }
    return label_times, signals


def resample_signal_to_video_time(label_times: Optional[np.ndarray], values: Optional[np.ndarray], target_times: np.ndarray, default_value: float = 0.0) -> np.ndarray:
    """
    一句话意义：把低频/不等频标签线性插值到 STMap 的 30Hz 时间轴。
    输入：label_times shape=[L] 秒或 None；values shape=[L]；target_times shape=[T30]。
    输出：resampled shape=[T30] float32。
    维度：L=标签采样点数，T30=30Hz STMap 时间长度。
    """
    if values is None:
        return np.full(target_times.shape[0], default_value, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.full(target_times.shape[0], default_value, dtype=np.float32)

    if label_times is None:
        old_t = np.linspace(target_times[0], target_times[-1], values.size, dtype=np.float32)
    else:
        old_t = np.asarray(label_times, dtype=np.float32).reshape(-1)

    n = min(old_t.size, values.size)
    old_t, values = old_t[:n], values[:n]
    ok = np.isfinite(old_t) & np.isfinite(values)
    old_t, values = old_t[ok], values[ok]
    if values.size == 0:
        return np.full(target_times.shape[0], default_value, dtype=np.float32)
    if values.size == 1:
        return np.full(target_times.shape[0], float(values[0]), dtype=np.float32)

    order = np.argsort(old_t)
    old_t, values = old_t[order], values[order]
    unique_t, unique_idx = np.unique(old_t, return_index=True)
    values = values[unique_idx]

    f = interp1d(unique_t, values, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(target_times).astype(np.float32)


def save_physmle_labels(label_path: Path, target_times: np.ndarray, out_subject_dir: Path) -> None:
    """
    一句话意义：把 CSV 标签保存成 PhysMLE 读取的 .mat 键名格式。
    输入：label_path=CSV；target_times shape=[T30]；out_subject_dir=<dataset>/<subject>。
    输出：Label/HR.mat、Label/SPO2.mat、Label/BVP.mat。
    维度：每个 .mat 内 shape=[T30]。
    """
    label_times, signals = load_label_csv(label_path)
    label_dir = out_subject_dir / "Label"
    label_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(label_dir / "HR.mat", {"HR": resample_signal_to_video_time(label_times, signals["HR"], target_times, 0.0)})
    sio.savemat(label_dir / "SPO2.mat", {"SPO2": resample_signal_to_video_time(label_times, signals["SPO2"], target_times, 0.0)})
    sio.savemat(label_dir / "BVP.mat", {"BVP": resample_signal_to_video_time(label_times, signals["BVP"], target_times, 0.0)})


# =========================
# ROI/Y5F 检测模块
# =========================

def init_y5f_detector(rppg_toolbox_root: Path, backend: str, device: str):
    """
    一句话意义：从 rPPG-Toolbox 中初始化 YOLO5Face 检测器。
    输入：rppg_toolbox_root=rPPG-Toolbox 根目录；backend 通常为 'Y5F'；device='cpu'/'cuda'。
    输出：YOLO5Face 实例，具备 detect_face(frame) 方法。
    维度：不涉及数组维度。
    """
    sys.path.insert(0, str(rppg_toolbox_root))
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face  # type: ignore
    return YOLO5Face(backend, device)


def y5f_result_to_square_box(res: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int], large_box_coef: float) -> np.ndarray:
    """
    一句话意义：把 Y5F 的 x1,y1,x2,y2 转成 rPPG-Toolbox 风格的正方形 bbox。
    输入：res=(x_min,y_min,x_max,y_max)；frame_shape=(H,W,3)；large_box_coef=放大系数。
    输出：box shape=[4]，格式 [x,y,w,h]。
    维度：4 表示左上角 x/y 与宽高。
    """
    x_min, y_min, x_max, y_max = map(float, res)
    width, height = x_max - x_min, y_max - y_min
    cx, cy = x_min + width / 2.0, y_min + height / 2.0
    size = max(width, height) * large_box_coef
    x = cx - size / 2.0
    y = cy - size / 2.0
    return np.asarray([x, y, size, size], dtype=np.float32)


def detect_face_box_y5f(frame_rgb: np.ndarray, y5f_obj, large_box_coef: float) -> Optional[np.ndarray]:
    """
    一句话意义：用 rPPG-Toolbox YOLO5Face 对单帧检测人脸 ROI。
    输入：frame_rgb shape=[H,W,3] uint8；y5f_obj=YOLO5Face 实例。
    输出：box shape=[4] [x,y,w,h] 或 None。
    维度：H/W=原视频分辨率。
    """
    res = y5f_obj.detect_face(frame_rgb[:, :, :3].astype(np.uint8))
    if res is None:
        return None
    return y5f_result_to_square_box(res, frame_rgb.shape, large_box_coef)


def detect_face_box_hc(frame_rgb: np.ndarray, large_box_coef: float) -> Optional[np.ndarray]:
    """
    一句话意义：用 OpenCV Haar 作为无 rPPG-Toolbox 时的轻量 ROI fallback。
    输入：frame_rgb shape=[H,W,3] uint8。
    输出：box shape=[4] [x,y,w,h] 或 None。
    维度：H/W=原视频分辨率。
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
    x, y, w, h = faces[idx].astype(np.float32)
    cx, cy = x + w / 2.0, y + h / 2.0
    size = max(w, h) * large_box_coef
    return np.asarray([cx - size / 2.0, cy - size / 2.0, size, size], dtype=np.float32)


def detect_dynamic_boxes(
    frames_rgb: np.ndarray,
    detector: str,
    detection_freq: int,
    use_median_box: bool,
    large_box_coef: float,
    rppg_toolbox_root: Optional[Path],
    y5f_backend: str,
    device: str,
) -> Tuple[np.ndarray, float]:
    """
    一句话意义：按固定间隔检测 ROI，并可用 median box 稳定整段视频。
    输入：frames_rgb shape=[T,H,W,3]；detector='y5f'/'hc'/'full'。
    输出：boxes shape=[T,4] [x,y,w,h]；valid_ratio=float。
    维度：T=帧数，4=bbox 参数。
    """
    T, H, W, _ = frames_rgb.shape
    detection_freq = max(1, int(detection_freq))
    det_indices = list(range(0, T, detection_freq))
    if det_indices[-1] != T - 1:
        det_indices.append(T - 1)

    y5f_obj = None
    if detector == "y5f":
        if rppg_toolbox_root is None:
            raise ValueError("detector=y5f 时必须提供 --rppg_toolbox_root")
        y5f_obj = init_y5f_detector(rppg_toolbox_root, y5f_backend, device)

    det_boxes: List[Optional[np.ndarray]] = []
    valid_flags: List[bool] = []
    for idx in det_indices:
        frame = frames_rgb[idx]
        if detector == "y5f":
            box = detect_face_box_y5f(frame, y5f_obj, large_box_coef)
        elif detector == "hc":
            box = detect_face_box_hc(frame, large_box_coef)
        elif detector == "full":
            box = np.asarray([0, 0, W, H], dtype=np.float32)
        else:
            raise ValueError(f"未知 detector: {detector}")
        det_boxes.append(box)
        valid_flags.append(box is not None)

    valid_ratio = float(np.mean(valid_flags)) if valid_flags else 0.0
    if valid_ratio == 0.0:
        print("[警告] 没有任何 ROI 检测成功，退化为整帧 ROI")
        boxes = np.tile(np.asarray([0, 0, W, H], dtype=np.float32), (T, 1))
        return boxes, valid_ratio

    valid_det_indices = np.asarray([i for i, ok in zip(det_indices, valid_flags) if ok], dtype=np.float32)
    valid_boxes = np.stack([b for b in det_boxes if b is not None], axis=0).astype(np.float32)

    if use_median_box:
        # 与 rPPG-Toolbox USE_MEDIAN_FACE_BOX 思路一致：用所有检测框中位数稳定整段视频。
        median_box = np.median(valid_boxes, axis=0).astype(np.float32)
        boxes = np.tile(median_box, (T, 1))
    else:
        # 不用 median 时，对动态检测框做逐坐标线性插值到每帧。
        frame_idx = np.arange(T, dtype=np.float32)
        boxes = np.zeros((T, 4), dtype=np.float32)
        for j in range(4):
            if valid_det_indices.size == 1:
                boxes[:, j] = valid_boxes[0, j]
            else:
                f = interp1d(valid_det_indices, valid_boxes[:, j], kind="linear", bounds_error=False, fill_value=(valid_boxes[0, j], valid_boxes[-1, j]))
                boxes[:, j] = f(frame_idx)
    return boxes, valid_ratio


def crop_resize_by_boxes(frames_rgb: np.ndarray, boxes: np.ndarray, out_size: int = 160) -> np.ndarray:
    """
    一句话意义：按每帧 bbox 裁剪 ROI 并 resize 成固定大小。
    输入：frames_rgb shape=[T,H,W,3]；boxes shape=[T,4] [x,y,w,h]。
    输出：roi_frames shape=[T,out_size,out_size,3] uint8。
    维度：T=帧数，out_size 默认 160。
    """
    T, H, W, _ = frames_rgb.shape
    roi_frames = np.zeros((T, out_size, out_size, 3), dtype=np.uint8)
    for i in range(T):
        x, y, bw, bh = boxes[i]
        x1 = int(round(max(0, x)))
        y1 = int(round(max(0, y)))
        x2 = int(round(min(W, x + bw)))
        y2 = int(round(min(H, y + bh)))
        if x2 <= x1 or y2 <= y1:
            crop = frames_rgb[i]
        else:
            crop = frames_rgb[i, y1:y2, x1:x2, :]
        roi_frames[i] = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return roi_frames


# =========================
# STMap 生成模块
# =========================

def frames_to_grid_rgb_traces(frames_rgb: np.ndarray, grid: int = 5) -> np.ndarray:
    """
    一句话意义：把 ROI 帧压缩为每帧 5x5 区域的 RGB 均值。
    输入：frames_rgb shape=[T,H,W,3]。
    输出：traces shape=[T,25,3] float32。
    维度：25=5x5 ROI，3=RGB。
    """
    T, H, W, C = frames_rgb.shape
    assert C == 3
    h_step, w_step = H // grid, W // grid
    traces = np.zeros((T, grid * grid, 3), dtype=np.float32)
    # 与 HSRD getValue 的循环顺序一致：外层 w_index，内层 h_index。
    for t in range(T):
        k = 0
        for w_idx in range(grid):
            for h_idx in range(grid):
                patch = frames_rgb[t, h_idx * h_step:(h_idx + 1) * h_step, w_idx * w_step:(w_idx + 1) * w_step, :]
                traces[t, k] = np.nanmean(patch, axis=(0, 1))
                k += 1
    return traces


def resample_traces_to_fps(traces: np.ndarray, frame_times: np.ndarray, target_fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    一句话意义：把原视频帧率的 RGB trace 插值到 30Hz。
    输入：traces shape=[T,25,3]；frame_times shape=[T] 秒。
    输出：traces_30 shape=[T30,25,3]；target_times shape=[T30] 秒。
    维度：T30≈视频时长*30。
    """
    target_times = np.arange(float(frame_times[0]), float(frame_times[-1]), 1.0 / target_fps, dtype=np.float32)
    if target_times.size < 2:
        raise RuntimeError("视频太短，无法生成 STMap")
    traces_30 = np.zeros((target_times.size, traces.shape[1], traces.shape[2]), dtype=np.float32)
    kind = "cubic" if frame_times.size >= 4 else "linear"
    for roi in range(traces.shape[1]):
        for ch in range(traces.shape[2]):
            f = interp1d(frame_times, traces[:, roi, ch], kind=kind, bounds_error=False, fill_value="extrapolate")
            traces_30[:, roi, ch] = f(target_times)
    return traces_30, target_times


def normalize_stmap_uint8(traces_30: np.ndarray) -> np.ndarray:
    """
    一句话意义：按每个 ROI/通道独立 min-max 归一化为 uint8 STMap 图像。
    输入：traces_30 shape=[T30,25,3] float32。
    输出：stmap_img shape=[25,T30,3] uint8。
    维度：25=空间 ROI 行，T30=时间列，3=RGB。
    """
    x = traces_30.copy().astype(np.float32)
    for roi in range(x.shape[1]):
        for ch in range(x.shape[2]):
            v = x[:, roi, ch]
            mn, mx = np.nanmin(v), np.nanmax(v)
            x[:, roi, ch] = 255.0 * (v - mn) / (0.001 + mx - mn)
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return np.swapaxes(x, 0, 1)


def save_stmap_png(stmap_img_rgb: np.ndarray, out_subject_dir: Path, filename: str = "STMap.png") -> None:
    """
    一句话意义：把 RGB STMap 保存为 PhysMLE 读取的 PNG 文件。
    输入：stmap_img_rgb shape=[25,T30,3] uint8。
    输出：<subject>/STMap/STMap.png。
    维度：25=ROI，T30=时间，3=RGB。
    """
    stmap_dir = out_subject_dir / "STMap"
    stmap_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stmap_dir / filename), cv2.cvtColor(stmap_img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


# =========================
# PhysMLE index 生成模块
# =========================

def write_physmle_index_files(dataset_root: Path, index_dir: Path, frames_num: int = 300, step: int = 10, stmap_name: str = "STMap.png") -> int:
    """
    一句话意义：生成 PhysMLE getIndex 同款滑窗索引 .mat 文件。
    输入：dataset_root=<out_root>/<dataset_name>；index_dir=<out_root>/STMap_Index/<dataset_name>。
    输出：写入多个 .mat，每个含 Path 和 Step_Index；返回索引数量。
    维度：每个窗口长度 frames_num，滑窗步长 step，索引沿 STMap 宽度 T30 生成。
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for subject_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        stmap_path = subject_dir / "STMap" / stmap_name
        img = cv2.imread(str(stmap_path))
        if img is None:
            print(f"[index 跳过] 不能读取 {stmap_path}")
            continue
        num_frames = img.shape[1]
        res = num_frames - frames_num - 1
        step_num = int(res / step) if res > 0 else 0
        for i in range(step_num):
            step_index = i * step
            index_name = f"{subject_dir.name}_{1000 + i}_.mat"
            sio.savemat(index_dir / index_name, {"Path": str(subject_dir), "Step_Index": step_index})
            count += 1
    return count


# =========================
# 单受试者处理主模块
# =========================

def process_subject(args: argparse.Namespace, item: SubjectIO) -> Dict[str, float]:
    """
    一句话意义：处理一个 subject，从原始视频直出 ROI-STMap.png 和三个标签 mat。
    输入：args=命令行参数；item=SubjectIO。
    输出：QC 字典，包括有效 ROI 检测比例、STMap 长度等。
    维度：STMap 输出 [25,T30,3]。
    """
    print(f"\n[处理] {item.subject_id}")
    print(f"  video: {item.video_path}")
    print(f"  label: {item.label_path}")

    frames_rgb, frame_times, fps = read_video_rgb(item.video_path, max_frames=args.max_frames)
    boxes, valid_ratio = detect_dynamic_boxes(
        frames_rgb=frames_rgb,
        detector=args.detector,
        detection_freq=args.detection_freq,
        use_median_box=args.use_median_box,
        large_box_coef=args.large_box_coef,
        rppg_toolbox_root=args.rppg_toolbox_root,
        y5f_backend=args.y5f_backend,
        device=args.device,
    )
    roi_frames = crop_resize_by_boxes(frames_rgb, boxes, out_size=args.roi_size)

    if args.save_roi:
        debug_dir = item.out_subject_dir / "ROI"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(roi_frames):
            cv2.imwrite(str(debug_dir / f"{10000 + i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    traces = frames_to_grid_rgb_traces(roi_frames, grid=args.grid)
    traces_30, target_times = resample_traces_to_fps(traces, frame_times, target_fps=args.target_fps)
    stmap_img = normalize_stmap_uint8(traces_30)
    save_stmap_png(stmap_img, item.out_subject_dir, filename=args.stmap_name)
    save_physmle_labels(item.label_path, target_times, item.out_subject_dir)

    qc = {
        "subject": item.subject_id,
        "fps": fps,
        "raw_frames": float(frames_rgb.shape[0]),
        "target_frames": float(stmap_img.shape[1]),
        "valid_roi_ratio": float(valid_ratio),
        "stmap_height": float(stmap_img.shape[0]),
    }
    print(f"  saved: {item.out_subject_dir} | STMap={stmap_img.shape} | valid_roi={valid_ratio:.2%}")
    return qc


# =========================
# CLI 主入口
# =========================

def parse_args() -> argparse.Namespace:
    """
    一句话意义：解析命令行参数。
    输入：命令行。
    输出：argparse.Namespace。
    维度：不涉及数组维度。
    """
    parser = argparse.ArgumentParser(description="导出 ROI/Y5F bbox 风格 PhysMLE STMap")
    parser.add_argument("--raw_root", type=Path, required=True, help="原始数据根目录；每个子目录为一个 subject")
    parser.add_argument("--out_root", type=Path, required=True, help="输出根目录")
    parser.add_argument("--dataset_name", type=str, default="ICU", help="输出数据集名；若不改 PhysMLE 可临时用 PURE")
    parser.add_argument("--video_patterns", nargs="+", default=["*.mp4", "*.avi", "*.mov", "*.mkv"], help="视频文件匹配规则")
    parser.add_argument("--label_name", type=str, default="label.csv", help="每个 subject 的标签 CSV 文件名")

    parser.add_argument("--detector", choices=["y5f", "hc", "full"], default="y5f", help="ROI 检测器：y5f=复用 rPPG-Toolbox；hc=OpenCV Haar；full=整帧")
    parser.add_argument("--rppg_toolbox_root", type=Path, default=None, help="rPPG-Toolbox 根目录；detector=y5f 时必填")
    parser.add_argument("--y5f_backend", type=str, default="Y5F", help="传给 rPPG-Toolbox YOLO5Face 的 backend 名称")
    parser.add_argument("--device", type=str, default="cuda", help="Y5F 使用 cpu/cuda")
    parser.add_argument("--detection_freq", type=int, default=30, help="每隔多少帧检测一次 ROI")
    parser.add_argument("--use_median_box", action="store_true", help="用所有检测框中位数作为全视频 ROI，强烈建议 ICU 使用")
    parser.add_argument("--large_box_coef", type=float, default=1.2, help="ROI 放大系数")
    parser.add_argument("--roi_size", type=int, default=160, help="ROI crop 后 resize 大小")

    parser.add_argument("--target_fps", type=float, default=30.0, help="STMap 和标签同步目标 FPS")
    parser.add_argument("--grid", type=int, default=5, help="STMap 空间网格，原版风格为 5x5=25")
    parser.add_argument("--stmap_name", type=str, default="STMap.png", help="STMap 文件名")
    parser.add_argument("--frames_num", type=int, default=300, help="PhysMLE 每个训练窗口长度")
    parser.add_argument("--step", type=int, default=10, help="PhysMLE index 滑窗步长")
    parser.add_argument("--no_index", action="store_true", help="不生成 STMap_Index")
    parser.add_argument("--save_roi", action="store_true", help="保存中间 ROI crop PNG，便于检查")
    parser.add_argument("--max_frames", type=int, default=None, help="调试用：最多读取多少帧")
    return parser.parse_args()


def main() -> None:
    """
    一句话意义：批量导出所有 subject 的 ROI-STMap、标签和 PhysMLE index。
    输入：命令行参数。
    输出：PhysMLE-like 数据目录和 qc_roi.csv。
    维度：每个 subject 输出 STMap [25,T30,3]。
    """
    args = parse_args()
    subjects = discover_subjects(args.raw_root, args.out_root, args.dataset_name, args.video_patterns, args.label_name)
    if not subjects:
        raise RuntimeError("没有发现可处理的 subject")

    qc_rows: List[Dict[str, float]] = []
    for item in subjects:
        try:
            qc_rows.append(process_subject(args, item))
        except Exception as exc:
            print(f"[失败] {item.subject_id}: {exc}")

    qc_path = args.out_root / f"qc_{args.dataset_name}_roi.csv"
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        print(f"\nQC saved: {qc_path}")

    if not args.no_index:
        dataset_root = args.out_root / args.dataset_name
        index_dir = args.out_root / "STMap_Index" / args.dataset_name
        n = write_physmle_index_files(dataset_root, index_dir, frames_num=args.frames_num, step=args.step, stmap_name=args.stmap_name)
        print(f"Index saved: {index_dir}, num={n}")


if __name__ == "__main__":
    main()
