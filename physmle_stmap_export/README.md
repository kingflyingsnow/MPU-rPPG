# PhysMLE STMap 导出脚本包

本包包含两个独立脚本，均可从 `原始视频 + label.csv` 直接导出 PhysMLE 可读的 STMap 目录。

## 1. 原版风格：landmark-aligned STMap

```bash
python export_physmle_stmap_landmark.py \
  --raw_root /path/to/raw_icu \
  --out_root /path/to/physmle_data \
  --dataset_name ICU \
  --label_name label.csv \
  --device cuda \
  --frames_num 300 \
  --step 10
```

意义：把 HSRD 原来的 `Landmark.py + Landmark_proce.py + Align_Face.py + STMap.py` 思路整合到一个脚本里。

输出：

```text
/path/to/physmle_data/ICU/subject_001/STMap/STMap.png    # [25,T30,3]
/path/to/physmle_data/ICU/subject_001/Label/HR.mat       # key HR, [T30]
/path/to/physmle_data/ICU/subject_001/Label/SPO2.mat     # key SPO2, [T30]
/path/to/physmle_data/ICU/subject_001/Label/BVP.mat      # key BVP, [T30]
/path/to/physmle_data/STMap_Index/ICU/*.mat              # key Path, Step_Index
```

依赖：

```bash
pip install opencv-python numpy pandas scipy face-alignment torch
```

## 2. ROI/Y5F 风格：bbox STMap

```bash
python export_physmle_stmap_roi.py \
  --raw_root /path/to/raw_icu \
  --out_root /path/to/physmle_data_roi \
  --dataset_name ICU \
  --label_name label.csv \
  --detector y5f \
  --rppg_toolbox_root /path/to/rPPG-Toolbox \
  --device cuda \
  --use_median_box \
  --detection_freq 30 \
  --large_box_coef 1.2 \
  --frames_num 300 \
  --step 10
```

意义：复用 rPPG-Toolbox 的 Y5F 检脸输出，裁剪 ROI 后生成与 PhysMLE 形状兼容的 `[25,T30,3]` STMap。

无 rPPG-Toolbox 时可先用 Haar 调试：

```bash
python export_physmle_stmap_roi.py \
  --raw_root /path/to/raw_icu \
  --out_root /path/to/physmle_data_roi \
  --dataset_name ICU \
  --detector hc \
  --use_median_box
```

## 原始数据格式

每个 subject 一个文件夹：

```text
raw_icu/
  subject_001/
    vid.mp4        # 或 .avi/.mov/.mkv
    label.csv
  subject_002/
    vid.avi
    label.csv
```

`label.csv` 支持自动识别这些列名：

- 时间列：`time`, `timestamp`, `sec`, `seconds`, `t`
- 心率列：`HR`, `heart_rate`, `pulse`, `pulse_rate`
- 血氧列：`SPO2`, `sp_o2`, `oxygen_saturation`, `sao2`
- BVP/PPG 列：`BVP`, `PPG`, `pleth`, `wave`, `pulse_wave`

没有时间列时，脚本会假设标签等间隔覆盖整段视频。

## 接入 PhysMLE 的注意点

PhysMLE 原代码只在 `dataName in ['PURE','VIPL']` 时返回 `sp` 血氧标签。如果你输出为 `dataset_name=ICU`，需要在 `MyDataset.py` 里加 ICU 分支，或临时把 `--dataset_name PURE` 用作零改代码测试。

推荐正式实验保留 `dataset_name=ICU`，并显式 patch `MyDataset.py`。
