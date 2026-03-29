Using deep learning to automatically segment cardiac magnetic resonance imaging. 

Model weights and independent application files (which can be used directly by clicking ".exe" file) are also availabel at: https://pan.baidu.com/s/1BM9viKgzGoECovzjxbMgtg?pwd=4396

<div align="center">

# 🫀 CorSeg-CineSAX

**Automatic Segmentation of Short-Axis Cine Cardiac MRI**

基于深度学习的心脏短轴位电影磁共振（Cine SAX）自动分割工具

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-%E2%89%A51.3-green?logo=data:image/png;base64,)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GUI: PyQt6](https://img.shields.io/badge/GUI-PyQt6-41cd52)](https://www.riverbankcomputing.com/software/pyqt/)

[English](#overview) · [中文说明](#中文说明)

</div>

---

## Overview

**CorSeg-CineSAX** is a desktop application for fully automatic segmentation of short-axis cine cardiac MRI. It provides an intuitive PyQt6 GUI for model inference, real-time visualization, and anatomical post-processing — no coding required.

### Segmentation Targets

| Label | Color | Structure |
|:-----:|:-----:|-----------|
| 1 | 🔴 Red | Left Ventricle Myocardium (LV Myo) |
| 2 | 🔵 Blue | Left Ventricle Blood Pool (LV Cavity) |
| 3 | 🟢 Green | Right Ventricle Blood Pool (RV) |

### Key Features

- **MedNeXt-L** backbone — modern ConvNeXt-based medical image segmentation network
- **PyQt6 GUI** with real-time overlay visualization and opacity control
- **Batch & single-file** inference modes
- **Three-step anatomical post-processing** pipeline with violation detection
- **DICOM & NIfTI** input support (`.dcm`, `.nii`, `.nii.gz`)
- **GPU (CUDA) / CPU** selectable compute device
- **Bilingual** — English (`_en`) and Chinese (`_ch`) versions included
- **JSON statistics report** for post-processing results

---

## 中文说明

**CorSeg-CineSAX** 是一款基于深度学习的心脏短轴位电影 MRI 全自动分割桌面工具。提供直观的 PyQt6 图形界面，支持模型推理、实时可视化和解剖学后处理，无需编写代码。

### 分割目标

| 标签 | 颜色 | 结构 |
|:----:|:----:|------|
| 1 | 🔴 红色 | 左心室心肌 (LV Myocardium) |
| 2 | 🔵 蓝色 | 左心室血池 (LV Blood Pool) |
| 3 | 🟢 绿色 | 右心室血池 (RV Blood Pool) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RunhaoXu2003/CorSeg.git
cd CorSeg
```

### 2. Create Environment (Recommended)

```bash
conda create -n corseg python=3.10 -y
conda activate corseg
```

### 3. Install Dependencies

```bash
# PyTorch (choose the command matching your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install monai>=1.3 nibabel numpy matplotlib PyQt6

# Optional but recommended
pip install scipy       # Required for anatomical post-processing
pip install pydicom     # Required for DICOM file support
```

### Requirements Summary

| Package | Version | Required | Purpose |
|---------|---------|:--------:|---------|
| Python | ≥ 3.9 | ✅ | Runtime |
| PyTorch | ≥ 2.0 | ✅ | Deep learning backend |
| MONAI | ≥ 1.3 | ✅ | MedNeXt-L architecture |
| nibabel | ≥ 3.0 | ✅ | NIfTI I/O |
| numpy | ≥ 1.21 | ✅ | Array operations |
| matplotlib | ≥ 3.5 | ✅ | Visualization |
| PyQt6 | ≥ 6.4 | ✅ | GUI framework |
| scipy | ≥ 1.7 | ⬜ | Post-processing |
| pydicom | ≥ 2.3 | ⬜ | DICOM support |

---

## Model Weights

Download the pre-trained model weights and place them in a folder (e.g., `weights/`):

```
weights/
└── best_model.pth
```

> **Download Link:** [https://pan.baidu.com/s/1BM9viKgzGoECovzjxbMgtg?pwd=4396] (You can also get it from releases)
> 
> The `.pth` file contains:
> - `model_state_dict` — MedNeXt-L trained weights
> - `config` — training configuration (spatial dims, image size, etc.)

If the checkpoint contains a `config` key, it will be used automatically. Otherwise, default settings are applied:

```python
{
    "spatial_dims": 2,
    "in_channels": 1,
    "num_classes": 4,
    "mednext_variant": "L",
    "mednext_kernel": 5,
    "img_size": (224, 224)
}
```

---

## Usage

### Launch the Application

```bash
# English version
python CorSeg-CineSAX_en.py

# 中文版本
python CorSeg-CineSAX_ch.py
```

### Step-by-Step

1. **Select inference mode** — Folder (batch) or single file
2. **Choose compute device** — GPU (CUDA) or CPU
3. **Set paths:**
   - **Model Folder** — directory containing `.pth` weight file
   - **Input Path** — image file or folder with cardiac MRI data
   - **Output Path** — directory for saving segmentation masks
4. **Configure post-processing** (optional) — enable/disable individual steps
5. **Click "Start Segmentation"** — results display in real time on the right panel
6. **Browse results** — use Previous/Next buttons, adjust overlay opacity, toggle between raw prediction and post-processed views

### Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| NIfTI | `.nii`, `.nii.gz` | Recommended |
| DICOM | `.dcm` | Requires `pydicom` |

### Output Structure

**Without post-processing:**
```
output/
├── ROI-image001.nii.gz
├── ROI-image002.nii.gz
└── ...
```

**With post-processing enabled:**
```
output/
├── Prediction/          # Raw model predictions
│   ├── ROI-image001.nii.gz
│   └── ...
├── Postprocess/         # Post-processed results
│   ├── ROI-image001.nii.gz
│   └── ...
└── postprocess_stats.json   # Statistics report
```

---

## Anatomical Post-Processing Pipeline

The three-step post-processing pipeline enforces anatomical priors on the model's raw predictions:

### Step 1 — Connected Component Constraint

> Remove isolated fragments: keep only the largest connected region per label.

Anatomically, each cardiac structure (LV Myo, LV Cavity, RV) should be a single contiguous region on each slice.

### Step 2 — Containment Constraint

> Ensure the LV blood pool (label 2) is fully enclosed by LV myocardium (label 1).

Exposed LV cavity pixels adjacent to background or RV are iteratively reassigned to myocardium.

### Step 3 — Gap Filling Constraint

> Fill background holes enclosed by cardiac structures.

- **Part A:** Holes fully enclosed by cardiac tissue are filled with the dominant neighboring label.
- **Part B:** Narrow gaps between LV myocardium and RV at the septum are filled.

### Statistics Report

When post-processing is enabled, a `postprocess_stats.json` file is generated:

```json
{
  "total_files": 100,
  "aggregate": {
    "before": {
      "fragment": 12,
      "containment_violation": 8,
      "gap": 5
    },
    "after": {
      "fragment": 0,
      "containment_violation": 0,
      "gap": 0
    }
  },
  "per_file": [
    {
      "file": "image001.nii.gz",
      "pre": { "has_fragment": true, "has_containment_violation": false, "has_gap": false },
      "post": { "has_fragment": false, "has_containment_violation": false, "has_gap": false },
      "pixels_changed": { "step1": 42, "step2": 0, "step3": 0 }
    }
  ]
}
```

---

## Project Structure

```
CorSeg-CineSAX/
├── CorSeg-CineSAX_en.py    # English GUI application
├── CorSeg-CineSAX_ch.py    # Chinese (中文) GUI application
├── weights/                 # Model weights directory
│   └── best_model.pth      # Pre-trained MedNeXt-L weights
├── README.md                # This file
├── LICENSE                  # MIT License
└── assets/                  # (Optional) Screenshots, figures
    └── screenshot.png
```

---

## Technical Details

### Model Architecture

**MedNeXt-L** is a large-kernel ConvNeXt-based architecture designed for medical image segmentation, available in [MONAI](https://docs.monai.io/en/latest/networks.html#mednext). Key characteristics:

- Large receptive field via 5×5 depthwise convolutions
- Efficient modern ConvNeXt design pattern
- 2D variant for slice-wise cardiac MRI segmentation

### Inference Pipeline

```
Input Image (2D)
    │
    ▼
Resize to 224×224 (bilinear)
    │
    ▼
Z-score Normalization (non-zero voxels)
    │
    ▼
MedNeXt-L Forward Pass (FP16 on GPU)
    │
    ▼
Argmax → Predicted Labels
    │
    ▼
Resize to Original Resolution (nearest)
    │
    ▼
[Optional] Anatomical Post-Processing
    │
    ▼
Save as NIfTI (.nii.gz)
```

---

## FAQ

<details>
<summary><b>Q: I get "MedNeXt is not available" error</b></summary>

Ensure MONAI ≥ 1.3 is installed:
```bash
pip install monai>=1.3
```
MedNeXt was added in MONAI 1.3. Older versions do not include it.
</details>

<details>
<summary><b>Q: Post-processing options are greyed out</b></summary>

Install scipy:
```bash
pip install scipy
```
Post-processing requires scipy for connected component analysis and morphological operations.
</details>

<details>
<summary><b>Q: DICOM files are not recognized</b></summary>

Install pydicom:
```bash
pip install pydicom
```
</details>

<details>
<summary><b>Q: Can I use this for 3D volumes?</b></summary>

The current version performs **2D slice-wise** inference. For 3D volumes stored as multi-slice NIfTI files, the tool automatically extracts a representative 2D slice. For full 3D volume segmentation, process each slice individually or modify the inference pipeline.
</details>

<details>
<summary><b>Q: What GPU memory is required?</b></summary>

MedNeXt-L with 224×224 input requires approximately **2–4 GB** GPU memory. Any modern NVIDIA GPU with CUDA support should work.
</details>

---

## Citation

If you use this tool in your research, please consider citing:
MedArxiv:

### Related Works

- **MedNeXt:** Roy, S., et al. "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation." *MICCAI 2023*.
- **MONAI:** The MONAI Consortium. "MONAI: Medical Open Network for AI." [https://monai.io/](https://monai.io/)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Made with ❤️ for the cardiac imaging community**

If you find this useful, please ⭐ star the repository!

</div>
