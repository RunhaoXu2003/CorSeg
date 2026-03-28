#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorSeg-CineSAX_ch.py
────────────────────────────────────────────────────────────────
心脏短轴位磁共振影像 (Cine SAX) 自动分割  ·  PyQt6 推理 GUI
模型：MedNeXt-L
后处理：三步解剖学约束流程
  Step 1: 连通域约束  —  移除孤立碎片
  Step 2: 包含关系约束  —  确保 LV 腔被心肌包围
  Step 3: 间隙填充约束  —  填充被心脏结构包围的背景空洞
────────────────────────────────────────────────────────────────
"""

# ══════════════════════════════════════════════════════════════
# 0.  High-DPI 适配
# ══════════════════════════════════════════════════════════════
import sys, os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

# ══════════════════════════════════════════════════════════════
# 1.  导入
# ══════════════════════════════════════════════════════════════
import json
import traceback
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import nibabel as nib

# --- Matplotlib (Qt 后端) ---
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Source Han Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- PyQt6 ---
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QRadioButton,
    QProgressBar, QTextEdit, QScrollArea, QSlider, QFileDialog,
    QButtonGroup, QSizePolicy, QComboBox, QCheckBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

# --- 可选依赖 ---
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from scipy import ndimage as sp_ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from monai.networks.nets.mednext import create_mednext
    HAS_MEDNEXT = True
except ImportError:
    HAS_MEDNEXT = False


# ══════════════════════════════════════════════════════════════
# 2.  常量 / 默认配置
# ══════════════════════════════════════════════════════════════
LABEL_COLORS_HEX = ["#000000", "#FF4444", "#4488FF", "#44CC44"]
LABEL_NAMES      = ["背景", "LV心肌", "LV血池", "RV血池"]
LABEL_CMAP       = ListedColormap(LABEL_COLORS_HEX)
SUPPORTED_EXT    = (".nii.gz", ".nii", ".dcm")

DEFAULT_CFG = dict(
    spatial_dims=2, in_channels=1, num_classes=4,
    mednext_variant="L", mednext_kernel=5,
    img_size=(224, 224),
)

GUIDE_TEXT = (
    "本工具基于深度学习模型，对心脏短轴位 (Cine SAX) "
    "磁共振影像进行自动分割。\n\n"
    "【模型】\n"
    "  • MedNeXt-L — 基于 ConvNeXt 的高效医学分割网络\n\n"
    "【分割目标】\n"
    "  • 标签 1（红色）：左心室心肌 (LV Myocardium)\n"
    "  • 标签 2（蓝色）：左心室血池 (LV Blood Pool)\n"
    "  • 标签 3（绿色）：右心室血池 (RV Blood Pool)\n\n"
    "【使用步骤】\n"
    "  1. 选择推理模式、计算设备\n"
    "  2. 设置路径（模型、输入、输出）\n"
    "  3. 配置后处理选项（可选）\n"
    "  4. 点击「开始分割」，右侧实时展示结果\n\n"
    "【解剖学后处理】\n"
    "  • Step 1: 连通域约束 — 每个标签仅保留最大连通区域\n"
    "  • Step 2: 包含关系约束 — 确保 LV 腔完全被心肌包围\n"
    "  • Step 3: 间隙填充 — 填充被心脏结构包围的背景空洞\n"
    "  启用后处理时，原始预测保存于 Prediction/ 子目录，\n"
    "  后处理结果保存于 Postprocess/ 子目录，并生成统计报告。\n\n"
    "【命名规则】  输出 = \"ROI-\" + 原文件名\n"
    "【目录结构】  文件夹模式递归遍历子目录，输出保持层级。\n"
)


# ══════════════════════════════════════════════════════════════
# 3.  QSS 全局样式表
# ══════════════════════════════════════════════════════════════
STYLESHEET = """
QMainWindow, QWidget#centralContainer {
    background-color: #f5f5f5;
}
QGroupBox {
    font-size: 12pt; font-weight: bold;
    border: 2px solid #b0bec5; border-radius: 6px;
    margin-top: 14px; padding-top: 18px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 10px; color: #263238;
}
QLabel { font-size: 10pt; color: #333333; }
QLineEdit, QComboBox {
    font-size: 10pt; border: 1px solid #ccc; border-radius: 4px;
    padding: 4px 8px; background-color: #ffffff; min-height: 26px;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #3498db; }
QPushButton {
    font-size: 10pt; background-color: #3498db; color: #ffffff;
    border: none; border-radius: 4px; padding: 6px 14px; min-height: 28px;
}
QPushButton:hover   { background-color: #2980b9; }
QPushButton:pressed { background-color: #21618c; }
QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
QPushButton#segmentBtn {
    font-size: 12pt; font-weight: bold;
    background-color: #e74c3c; min-height: 42px;
}
QPushButton#segmentBtn:hover   { background-color: #c0392b; }
QPushButton#segmentBtn:pressed { background-color: #a93226; }
QRadioButton, QCheckBox { font-size: 10pt; spacing: 6px; }
QProgressBar {
    font-size: 10pt; border: 1px solid #ccc; border-radius: 4px;
    text-align: center; min-height: 22px; background-color: #ecf0f1;
}
QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }
QTextEdit {
    font-size: 10pt; border: 1px solid #ccc; border-radius: 4px;
    background-color: #ffffff;
}
QSlider::groove:horizontal {
    border: 1px solid #ccc; height: 6px;
    background: #ecf0f1; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #3498db; border: 1px solid #2980b9;
    width: 16px; margin: -6px 0; border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: #2980b9; }
QScrollArea { border: none; }
"""


# ══════════════════════════════════════════════════════════════
# 4.  工具函数
# ══════════════════════════════════════════════════════════════

def _ext(filepath: str) -> str:
    fp = filepath.lower()
    if fp.endswith(".nii.gz"):
        return ".nii.gz"
    if fp.endswith(".nii"):
        return ".nii"
    if fp.endswith(".dcm"):
        return ".dcm"
    return ""


def _is_image(filepath: str) -> bool:
    return _ext(filepath) in SUPPORTED_EXT


def collect_files(input_dir: str) -> List[str]:
    result: List[str] = []
    for root, _, fnames in os.walk(input_dir):
        for fn in sorted(fnames):
            if fn.startswith("ROI-"):
                continue
            full = os.path.join(root, fn)
            if _is_image(full):
                result.append(full)
    return result


def _make_output_name(basename: str) -> str:
    low = basename.lower()
    if low.endswith(".nii.gz"):
        return f"ROI-{basename[:-7]}.nii.gz"
    if low.endswith(".nii"):
        return f"ROI-{basename[:-4]}.nii"
    if low.endswith(".dcm"):
        return f"ROI-{basename[:-4]}.nii"
    return f"ROI-{basename}"


def output_path_for(inp_file: str, inp_base: str, out_base: str) -> str:
    rel     = os.path.relpath(inp_file, inp_base)
    rel_dir = os.path.dirname(rel)
    return os.path.join(out_base, rel_dir, _make_output_name(os.path.basename(rel)))


def load_image(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    ext = _ext(filepath)
    if ext in (".nii", ".nii.gz"):
        nib_img = nib.load(filepath)
        data   = nib_img.get_fdata().astype(np.float32)
        affine = nib_img.affine.copy()
    elif ext == ".dcm":
        if not HAS_PYDICOM:
            raise ImportError("处理 DICOM 需安装 pydicom:\n  pip install pydicom")
        dcm  = pydicom.dcmread(filepath)
        data = dcm.pixel_array.astype(np.float32)
        slope  = float(getattr(dcm, "RescaleSlope", 1))
        interc = float(getattr(dcm, "RescaleIntercept", 0))
        data   = data * slope + interc
        if hasattr(dcm, "PixelSpacing"):
            ps = [float(x) for x in dcm.PixelSpacing]
        elif hasattr(dcm, "ImagerPixelSpacing"):
            ps = [float(x) for x in dcm.ImagerPixelSpacing]
        else:
            ps = [1.0, 1.0]
        affine = np.diag([ps[0], ps[1], 1.0, 1.0])
    else:
        raise ValueError(f"不支持的格式: {ext}")

    while data.ndim > 2:
        ax1 = [ax for ax in range(data.ndim) if data.shape[ax] == 1]
        if ax1:
            data = np.squeeze(data, axis=ax1[0])
        else:
            sm = int(np.argmin(data.shape))
            data = np.take(data, data.shape[sm] // 2, axis=sm)
    if data.ndim != 2:
        raise ValueError(f"无法转为 2D（shape={data.shape}）")
    return data, affine, ext


def save_mask(pred: np.ndarray, affine: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), path)


# ══════════════════════════════════════════════════════════════
# 5.  模型构建 + 加载（仅 MedNeXt-L）
# ══════════════════════════════════════════════════════════════

def load_model(
    model_dir: str, device: torch.device,
) -> Tuple[torch.nn.Module, Dict, str, str]:
    if not HAS_MEDNEXT:
        raise ImportError(
            "当前 MONAI 版本不含 MedNeXt，请升级:\n  pip install monai>=1.3"
        )
    mdir = Path(model_dir)
    if not mdir.is_dir():
        raise FileNotFoundError(f"模型文件夹不存在: {mdir}")
    best = mdir / "best_model.pth"
    if best.exists():
        ckpt_path = best
    else:
        pths = sorted(mdir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"未在文件夹中找到 .pth 文件: {mdir}")
        ckpt_path = pths[0]

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    if not cfg:
        cfg = DEFAULT_CFG.copy()

    info_str = f"权重: {ckpt_path.name}"

    model = create_mednext(
        variant=cfg.get("mednext_variant", "L"),
        spatial_dims=cfg.get("spatial_dims", 2),
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("num_classes", 4),
        kernel_size=cfg.get("mednext_kernel", 5),
        deep_supervision=False,
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, ckpt_path.name, info_str


# ══════════════════════════════════════════════════════════════
# 6.  解剖学后处理流程
# ══════════════════════════════════════════════════════════════
#
# 标签定义:  0=背景  1=LV心肌  2=LV腔  3=RV
#
# 解剖学先验:
#   · 每个结构在单张切面上应为单个连续区域
#   · LV Cavity 被 LV Myocardium 完全包围
#   · LV Myo 与 RV 之间不应有被心脏结构包围的背景孔洞

def detect_violations(mask: np.ndarray) -> Dict[str, bool]:
    stats: Dict[str, bool] = {
        "has_fragment": False,
        "has_containment_violation": False,
        "has_gap": False,
    }
    if not HAS_SCIPY:
        return stats

    struct = sp_ndimage.generate_binary_structure(2, 1)

    # ── 碎片 ──
    for lv in (1, 2, 3):
        binary = (mask == lv)
        if not binary.any():
            continue
        _, n_cc = sp_ndimage.label(binary)
        if n_cc > 1:
            stats["has_fragment"] = True
            break

    # ── 包含关系违规 ──
    lv_cav = (mask == 2)
    if lv_cav.any():
        non_lv = (mask == 0) | (mask == 3)
        non_lv_dil = sp_ndimage.binary_dilation(non_lv, structure=struct)
        if (lv_cav & non_lv_dil).any():
            stats["has_containment_violation"] = True

    # ── 间隙 ──
    cardiac = (mask > 0)
    if cardiac.any():
        filled = sp_ndimage.binary_fill_holes(cardiac)
        if (filled & ~cardiac).any():
            stats["has_gap"] = True
        else:
            lvm = (mask == 1)
            rv  = (mask == 3)
            if lvm.any() and rv.any():
                lvm_dil = sp_ndimage.binary_dilation(lvm, structure=struct)
                rv_dil  = sp_ndimage.binary_dilation(rv, structure=struct)
                if (lvm_dil & rv_dil & (mask == 0)).any():
                    stats["has_gap"] = True

    return stats


def pp_step1_largest_component(mask: np.ndarray) -> np.ndarray:
    """Step 1: 连通域约束 — 每个标签仅保留最大连通域。"""
    if not HAS_SCIPY:
        return mask
    result = np.zeros_like(mask)
    for lv in (1, 2, 3):
        binary = (mask == lv)
        if not binary.any():
            continue
        labeled, n_cc = sp_ndimage.label(binary)
        if n_cc <= 1:
            result[binary] = lv
            continue
        sizes = sp_ndimage.sum(binary, labeled, range(1, n_cc + 1))
        largest_id = int(np.argmax(sizes)) + 1
        result[labeled == largest_id] = lv
    return result


def pp_step2_containment(mask: np.ndarray) -> np.ndarray:
    """Step 2: 包含关系约束 — 确保 LV Cavity(2) 被 LV Myocardium(1) 完全包围。"""
    if not HAS_SCIPY:
        return mask
    result = mask.copy()
    struct = sp_ndimage.generate_binary_structure(2, 1)

    original_cav = int(np.sum(result == 2))
    if original_cav == 0:
        return result

    max_iter = 50
    min_remaining_frac = 0.5

    for _ in range(max_iter):
        lv_cav = (result == 2)
        non_lv = (result == 0) | (result == 3)
        exposed = lv_cav & sp_ndimage.binary_dilation(non_lv, structure=struct)
        if not exposed.any():
            break
        result[exposed] = 1
        remaining = int(np.sum(result == 2))
        if remaining < original_cav * min_remaining_frac:
            break

    return result


def pp_step3_fill_gaps(mask: np.ndarray) -> np.ndarray:
    """Step 3: 间隙填充约束 — 填充被心脏结构包围的背景孔洞。"""
    if not HAS_SCIPY:
        return mask
    result = mask.copy()
    struct = sp_ndimage.generate_binary_structure(2, 1)

    # ── Part A: 填充封闭孔洞 ──
    cardiac = (result > 0)
    if cardiac.any():
        filled = sp_ndimage.binary_fill_holes(cardiac)
        holes = filled & ~cardiac
        if holes.any():
            hole_labeled, n_holes = sp_ndimage.label(holes)
            for h_id in range(1, n_holes + 1):
                h_mask = (hole_labeled == h_id)
                border = (sp_ndimage.binary_dilation(h_mask, structure=struct,
                                                      iterations=2)
                          & ~h_mask & (result > 0))
                if border.any():
                    counts = np.bincount(result[border], minlength=4)
                    best = int(np.argmax(counts[1:])) + 1 if counts[1:].sum() > 0 else 1
                    result[h_mask] = best
                else:
                    result[h_mask] = 1

    # ── Part B: 填充 LV Myo-RV 之间的窄缝隙 ──
    bg  = (result == 0)
    lvm = (result == 1)
    rv  = (result == 3)
    if bg.any() and lvm.any() and rv.any():
        lvm_adj = sp_ndimage.binary_dilation(lvm, structure=struct) & bg
        rv_adj  = sp_ndimage.binary_dilation(rv,  structure=struct) & bg
        septum_gap = lvm_adj & rv_adj
        if septum_gap.any():
            result[septum_gap] = 1

    return result


def apply_postprocessing(
    mask: np.ndarray,
    steps: Dict[str, bool],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pre_stats  = detect_violations(mask)
    result     = mask.copy()
    pixels_changed = defaultdict(int)

    if steps.get("step1", False):
        before = result.copy()
        result = pp_step1_largest_component(result)
        pixels_changed["step1"] = int(np.sum(before != result))

    if steps.get("step2", False):
        before = result.copy()
        result = pp_step2_containment(result)
        pixels_changed["step2"] = int(np.sum(before != result))

    if steps.get("step3", False):
        before = result.copy()
        result = pp_step3_fill_gaps(result)
        pixels_changed["step3"] = int(np.sum(before != result))

    post_stats = detect_violations(result)

    return result, {
        "pre": pre_stats,
        "post": post_stats,
        "pixels_changed": dict(pixels_changed),
    }


# ══════════════════════════════════════════════════════════════
# 7.  推理工作线程
# ══════════════════════════════════════════════════════════════

class InferenceWorker(QThread):
    sig_progress = pyqtSignal(int, int)
    sig_file_ok  = pyqtSignal(str, str, str, object)
    sig_error    = pyqtSignal(str)
    sig_done     = pyqtSignal()

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        inp_list: List[str],
        pred_out_list: List[str],
        post_out_list: List[Optional[str]],
        img_size: Tuple[int, int],
        pp_steps: Dict[str, bool],
    ):
        super().__init__()
        self.model         = model
        self.device        = device
        self.inp_list      = inp_list
        self.pred_out_list = pred_out_list
        self.post_out_list = post_out_list
        self.img_size      = img_size
        self.pp_steps      = pp_steps
        self._cancel       = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self.inp_list)
        self.model.eval()
        for idx in range(total):
            if self._cancel:
                break
            inp  = self.inp_list[idx]
            pred = self.pred_out_list[idx]
            post = self.post_out_list[idx]
            try:
                stats = self._infer_one(inp, pred, post)
                self.sig_file_ok.emit(inp, pred, post or "", stats)
            except Exception as exc:
                self.sig_error.emit(
                    f"[{Path(inp).name}] {type(exc).__name__}: {exc}"
                )
            self.sig_progress.emit(idx + 1, total)
        self.sig_done.emit()

    def _infer_one(
        self, inp_path: str, pred_path: str, post_path: Optional[str],
    ) -> Dict[str, Any]:
        data, affine, ext = load_image(inp_path)
        orig_hw = data.shape

        t = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=self.img_size,
                          mode="bilinear", align_corners=False)
        nz = t != 0
        if nz.any():
            mu, sd = t[nz].mean(), t[nz].std()
            if sd > 1e-8:
                t = (t - mu) / sd
            t[~nz] = 0.0
        t = t.to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.model(t)
            else:
                logits = self.model(t)

        pred = logits.argmax(dim=1).squeeze(0).cpu()

        if tuple(pred.shape) != orig_hw:
            pred = F.interpolate(
                pred.float().unsqueeze(0).unsqueeze(0),
                size=orig_hw, mode="nearest",
            ).squeeze().to(torch.uint8)

        pred_np = pred.numpy().astype(np.uint8)
        save_mask(pred_np, affine, pred_path)

        stats: Dict[str, Any] = {"pp_applied": False}
        any_step = any(self.pp_steps.get(k, False) for k in ("step1", "step2", "step3"))

        if post_path is not None and any_step:
            pp_result, pp_stats = apply_postprocessing(pred_np, self.pp_steps)
            save_mask(pp_result, affine, post_path)
            stats = {"pp_applied": True, **pp_stats}
        elif post_path is not None:
            save_mask(pred_np, affine, post_path)
            stats = {"pp_applied": False}

        return stats


# ══════════════════════════════════════════════════════════════
# 8.  Matplotlib 画布
# ══════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=100):
        self.fig = Figure(dpi=dpi, tight_layout=True)
        self.fig.patch.set_facecolor("#f0f0f0")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(400, 320))


# ══════════════════════════════════════════════════════════════
# 9.  主窗口
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "CorSeg CineSAX — 心脏 MRI 自动分割 + 解剖学后处理"
        )
        self.setMinimumSize(1360, 860)

        self.model: Optional[torch.nn.Module] = None
        self.model_cfg: dict   = {}
        self.device: Optional[torch.device] = None
        self.worker: Optional[InferenceWorker] = None

        self.pairs: List[Tuple[str, str, Optional[str]]] = []
        self.cur_idx: int  = 0
        self._running: bool = False

        self._cache_idx:  int = -1
        self._cache_mode: str = ""
        self._cache_img:  Optional[np.ndarray] = None
        self._cache_mask: Optional[np.ndarray] = None

        self._pp_stats: Dict[str, int] = defaultdict(int)
        self._pp_file_stats: List[Dict] = []

        self._build_ui()

    # ═══════════════ UI 构建 ═══════════════════════════════

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setObjectName("centralContainer")
        root = QVBoxLayout(container)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(10)

        # ── 功能说明 ──
        guide_grp = QGroupBox("功能说明")
        gl = QVBoxLayout(guide_grp)
        gl.setContentsMargins(12, 22, 12, 12)
        lbl_guide = QLabel(GUIDE_TEXT)
        lbl_guide.setWordWrap(True)
        lbl_guide.setStyleSheet("font-size: 10pt; line-height: 1.6;")
        gl.addWidget(lbl_guide)
        root.addWidget(guide_grp)

        # ── 左右布局 ──
        h_main = QHBoxLayout()
        h_main.setSpacing(12)
        h_main.addWidget(self._make_left(),  stretch=1)
        h_main.addWidget(self._make_right(), stretch=5)
        root.addLayout(h_main, stretch=1)

        # ── 进度条 ──
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setFormat("%v / %m  (%p%)")
        self.progress.setValue(0)
        root.addWidget(self.progress)

        # ── 日志 ──
        lbl_log = QLabel("日志信息：")
        lbl_log.setStyleSheet("color:#888;")
        root.addWidget(lbl_log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.setPlaceholderText("暂无消息")
        root.addWidget(self.log)

        scroll.setWidget(container)
        self.setCentralWidget(scroll)

    # ─────────── 左侧面板 ───────────
    def _make_left(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(10)

        # ── 模型架构 ──
        g_arch = QGroupBox("模型架构")
        l_arch = QVBoxLayout(g_arch)
        lbl_model = QLabel(
            f"MedNeXt-L  {'✅' if HAS_MEDNEXT else '❌  (pip install monai>=1.3)'}"
        )
        lbl_model.setStyleSheet("font-size: 10pt; font-weight: bold;")
        l_arch.addWidget(lbl_model)
        vb.addWidget(g_arch)

        # ── 推理模式 ──
        g1 = QGroupBox("推理模式")
        l1 = QVBoxLayout(g1)
        self.rb_folder = QRadioButton("推理文件夹（批量）")
        self.rb_file   = QRadioButton("推理单个文件")
        self.rb_folder.setChecked(True)
        self._grp_mode = QButtonGroup(self)
        self._grp_mode.addButton(self.rb_folder)
        self._grp_mode.addButton(self.rb_file)
        l1.addWidget(self.rb_folder)
        l1.addWidget(self.rb_file)
        vb.addWidget(g1)

        # ── 计算设备 ──
        g2 = QGroupBox("计算设备")
        l2 = QVBoxLayout(g2)
        self.rb_gpu = QRadioButton("GPU (CUDA)")
        self.rb_cpu = QRadioButton("CPU")
        self._grp_dev = QButtonGroup(self)
        self._grp_dev.addButton(self.rb_gpu)
        self._grp_dev.addButton(self.rb_cpu)
        if torch.cuda.is_available():
            gname = torch.cuda.get_device_name(0)
            self.rb_gpu.setText(f"GPU ({gname})")
            self.rb_gpu.setChecked(True)
        else:
            self.rb_gpu.setText("GPU（不可用）")
            self.rb_gpu.setEnabled(False)
            self.rb_cpu.setChecked(True)
        l2.addWidget(self.rb_gpu)
        l2.addWidget(self.rb_cpu)
        vb.addWidget(g2)

        # ── 路径 ──
        g3 = QGroupBox("路径设置")
        g3l = QGridLayout(g3)
        g3l.setSpacing(8)
        g3l.addWidget(QLabel("模型文件夹:"), 0, 0)
        self.ed_model = QLineEdit()
        self.ed_model.setPlaceholderText("包含 .pth 权重的文件夹")
        g3l.addWidget(self.ed_model, 0, 1)
        b0 = QPushButton("浏览"); b0.setMaximumWidth(64)
        b0.clicked.connect(self._brw_model)
        g3l.addWidget(b0, 0, 2)

        g3l.addWidget(QLabel("输入路径:"), 1, 0)
        self.ed_inp = QLineEdit()
        self.ed_inp.setPlaceholderText("影像文件或文件夹")
        g3l.addWidget(self.ed_inp, 1, 1)
        b1 = QPushButton("浏览"); b1.setMaximumWidth(64)
        b1.clicked.connect(self._brw_input)
        g3l.addWidget(b1, 1, 2)

        g3l.addWidget(QLabel("输出路径:"), 2, 0)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("掩膜保存文件夹")
        g3l.addWidget(self.ed_out, 2, 1)
        b2 = QPushButton("浏览"); b2.setMaximumWidth(64)
        b2.clicked.connect(self._brw_output)
        g3l.addWidget(b2, 2, 2)
        vb.addWidget(g3)

        # ── 后处理设置 ──
        g_pp = QGroupBox("解剖学后处理")
        l_pp = QVBoxLayout(g_pp)
        l_pp.setSpacing(6)

        self.chk_pp_enable = QCheckBox("启用解剖学后处理")
        self.chk_pp_enable.setChecked(True)
        self.chk_pp_enable.setStyleSheet("font-weight: bold;")
        self.chk_pp_enable.toggled.connect(self._on_pp_toggle)
        l_pp.addWidget(self.chk_pp_enable)

        indent = QWidget()
        indent_l = QVBoxLayout(indent)
        indent_l.setContentsMargins(20, 0, 0, 0)
        indent_l.setSpacing(4)

        self.chk_pp_step1 = QCheckBox("Step 1: 连通域约束 — 移除孤立碎片")
        self.chk_pp_step1.setChecked(True)
        indent_l.addWidget(self.chk_pp_step1)

        self.chk_pp_step2 = QCheckBox("Step 2: 包含关系约束 — LV腔须被心肌包围")
        self.chk_pp_step2.setChecked(True)
        indent_l.addWidget(self.chk_pp_step2)

        self.chk_pp_step3 = QCheckBox("Step 3: 间隙填充约束 — 填充心脏间背景孔洞")
        self.chk_pp_step3.setChecked(True)
        indent_l.addWidget(self.chk_pp_step3)

        l_pp.addWidget(indent)

        if HAS_SCIPY:
            lbl_sp = QLabel("scipy ✅  后处理可用")
            lbl_sp.setStyleSheet("font-size: 9pt; color: #27ae60;")
        else:
            lbl_sp = QLabel("scipy ❌  请安装: pip install scipy")
            lbl_sp.setStyleSheet("font-size: 9pt; color: #e74c3c;")
            self.chk_pp_enable.setChecked(False)
            self.chk_pp_enable.setEnabled(False)
        l_pp.addWidget(lbl_sp)

        vb.addWidget(g_pp)

        # ── 模型加载状态 ──
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet(
            "font-size: 9pt; color: #2980b9; padding: 4px;")
        vb.addWidget(self.lbl_model_info)

        # ── 分割按钮 ──
        self.btn_seg = QPushButton("🫀 开始分割")
        self.btn_seg.setObjectName("segmentBtn")
        self.btn_seg.clicked.connect(self._on_segment_click)
        vb.addWidget(self.btn_seg)

        vb.addStretch(1)
        return w

    # ─────────── 右侧面板 ───────────
    def _make_right(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(8)

        self.canvas = MplCanvas(self)
        vb.addWidget(self.canvas, stretch=1)
        self._draw_placeholder()

        hb = QHBoxLayout()
        hb.setSpacing(10)

        hb.addWidget(QLabel("显示:"))
        self.cb_display_mode = QComboBox()
        self.cb_display_mode.addItems(["后处理结果", "原始预测"])
        self.cb_display_mode.setMinimumWidth(120)
        self.cb_display_mode.setEnabled(False)
        self.cb_display_mode.currentIndexChanged.connect(
            lambda _: self._refresh_display()
        )
        hb.addWidget(self.cb_display_mode)

        hb.addWidget(QLabel("透明度:"))
        self.sl_alpha = QSlider(Qt.Orientation.Horizontal)
        self.sl_alpha.setRange(0, 100)
        self.sl_alpha.setValue(50)
        self.sl_alpha.setMinimumWidth(120)
        self.sl_alpha.valueChanged.connect(self._on_alpha)
        hb.addWidget(self.sl_alpha)
        self.lbl_alpha = QLabel("50 %")
        self.lbl_alpha.setFixedWidth(44)
        hb.addWidget(self.lbl_alpha)

        hb.addStretch()

        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setMaximumWidth(96)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev)
        hb.addWidget(self.btn_prev)

        self.lbl_idx = QLabel("0 / 0")
        self.lbl_idx.setStyleSheet(
            "font-size: 10pt; font-weight: bold; min-width:60px;"
            "qproperty-alignment:'AlignCenter';")
        hb.addWidget(self.lbl_idx)

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setMaximumWidth(96)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next)
        hb.addWidget(self.btn_next)

        vb.addLayout(hb)
        return w

    # ═══════════════ 后处理控件逻辑 ═══════════════════════

    def _on_pp_toggle(self, checked: bool):
        for chk in (self.chk_pp_step1, self.chk_pp_step2, self.chk_pp_step3):
            chk.setEnabled(checked)

    def _get_pp_steps(self) -> Dict[str, bool]:
        if not self.chk_pp_enable.isChecked():
            return {"step1": False, "step2": False, "step3": False}
        return {
            "step1": self.chk_pp_step1.isChecked(),
            "step2": self.chk_pp_step2.isChecked(),
            "step3": self.chk_pp_step3.isChecked(),
        }

    def _is_pp_enabled(self) -> bool:
        steps = self._get_pp_steps()
        return any(steps.values())

    # ═══════════════ 浏览对话框 ═══════════════════════════

    def _brw_model(self):
        p = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if p:
            self.ed_model.setText(p)

    def _brw_input(self):
        if self.rb_folder.isChecked():
            p = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "选择影像文件", "",
                "影像文件 (*.nii *.nii.gz *.dcm);;所有文件 (*.*)")
        if p:
            self.ed_inp.setText(p)

    def _brw_output(self):
        p = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if p:
            self.ed_out.setText(p)

    # ═══════════════ 分割逻辑 ═══════════════════════════════

    def _on_segment_click(self):
        if self._running:
            if self.worker:
                self.worker.cancel()
            self.btn_seg.setText("⏳ 正在取消…")
            self.btn_seg.setEnabled(False)
            return

        model_dir = self.ed_model.text().strip()
        inp_path  = self.ed_inp.text().strip()
        out_path  = self.ed_out.text().strip()

        if not model_dir:
            self._err("请先设置模型文件夹路径"); return
        if not inp_path:
            self._err("请先设置输入路径"); return
        if not out_path:
            self._err("请先设置输出路径"); return
        if not HAS_MEDNEXT:
            self._err("MedNeXt 不可用，请安装 MONAI>=1.3"); return

        device = (torch.device("cuda")
                  if self.rb_gpu.isChecked() and torch.cuda.is_available()
                  else torch.device("cpu"))

        # ── 加载模型 ──
        try:
            self.model, self.model_cfg, ckname, info_str = load_model(
                model_dir, device)
            self.device = device
            self.lbl_model_info.setText(f"✅ 已加载: MedNeXt-L\n{info_str}")
        except Exception as e:
            self._err(f"模型加载失败: {e}")
            self.lbl_model_info.setText("❌ 加载失败")
            return

        # ── 后处理配置 ──
        pp_steps   = self._get_pp_steps()
        pp_enabled = any(pp_steps.values())

        # ── 收集文件 ──
        folder_mode = self.rb_folder.isChecked()
        try:
            if folder_mode:
                if not os.path.isdir(inp_path):
                    self._err(f"输入路径不是有效文件夹: {inp_path}"); return
                inp_files = collect_files(inp_path)
                if not inp_files:
                    self._err("输入文件夹中未找到支持的影像文件"); return
                if pp_enabled:
                    pred_base = os.path.join(out_path, "Prediction")
                    post_base = os.path.join(out_path, "Postprocess")
                    pred_files = [output_path_for(f, inp_path, pred_base)
                                  for f in inp_files]
                    post_files = [output_path_for(f, inp_path, post_base)
                                  for f in inp_files]
                else:
                    pred_files = [output_path_for(f, inp_path, out_path)
                                  for f in inp_files]
                    post_files = [None] * len(inp_files)
            else:
                if not os.path.isfile(inp_path):
                    self._err(f"输入路径不是有效文件: {inp_path}"); return
                if not _is_image(inp_path):
                    self._err(f"不支持的文件格式: {inp_path}"); return
                inp_files = [inp_path]
                out_name  = _make_output_name(os.path.basename(inp_path))
                if pp_enabled:
                    pred_files = [os.path.join(out_path, "Prediction", out_name)]
                    post_files = [os.path.join(out_path, "Postprocess", out_name)]
                else:
                    pred_files = [os.path.join(out_path, out_name)]
                    post_files = [None]
        except Exception as e:
            self._err(f"文件收集失败: {e}"); return

        # ── 状态重置 ──
        self.pairs = []
        self.cur_idx = 0
        self._cache_idx  = -1
        self._cache_mode = ""
        self._cache_img  = None
        self._cache_mask = None
        self._pp_stats   = defaultdict(int)
        self._pp_file_stats = []
        self.progress.setMaximum(len(inp_files))
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m  (%p%)")

        # ── 更新显示模式 ──
        self.cb_display_mode.blockSignals(True)
        self.cb_display_mode.clear()
        if pp_enabled:
            self.cb_display_mode.addItems(["后处理结果", "原始预测"])
            self.cb_display_mode.setCurrentIndex(0)
            self.cb_display_mode.setEnabled(True)
        else:
            self.cb_display_mode.addItems(["原始预测"])
            self.cb_display_mode.setCurrentIndex(0)
            self.cb_display_mode.setEnabled(False)
        self.cb_display_mode.blockSignals(False)

        self._set_running(True)

        img_size = tuple(self.model_cfg.get("img_size", (224, 224)))
        self.worker = InferenceWorker(
            self.model, self.device,
            inp_files, pred_files, post_files,
            img_size, pp_steps,
        )
        self.worker.sig_progress.connect(self._w_progress)
        self.worker.sig_file_ok.connect(self._w_file_ok)
        self.worker.sig_error.connect(self._err)
        self.worker.sig_done.connect(self._w_done)
        self.worker.start()

    # ─────────── worker 回调 ───────────
    def _w_progress(self, done: int, total: int):
        self.progress.setValue(done)

    def _w_file_ok(self, inp: str, pred: str, post: str, stats: object):
        post_path = post if post else None
        self.pairs.append((inp, pred, post_path))
        self.cur_idx = len(self.pairs) - 1
        self._refresh_display()
        self._refresh_nav()

        if isinstance(stats, dict) and stats.get("pp_applied"):
            self._pp_stats["total"] += 1
            pre    = stats.get("pre", {})
            post_s = stats.get("post", {})
            if pre.get("has_fragment"):
                self._pp_stats["pre_fragment"] += 1
            if pre.get("has_containment_violation"):
                self._pp_stats["pre_containment"] += 1
            if pre.get("has_gap"):
                self._pp_stats["pre_gap"] += 1
            if post_s.get("has_fragment"):
                self._pp_stats["post_fragment"] += 1
            if post_s.get("has_containment_violation"):
                self._pp_stats["post_containment"] += 1
            if post_s.get("has_gap"):
                self._pp_stats["post_gap"] += 1
            self._pp_file_stats.append({
                "file": Path(inp).name,
                "pre": pre,
                "post": post_s,
                "pixels_changed": stats.get("pixels_changed", {}),
            })

    def _w_done(self):
        self._set_running(False)
        n = len(self.pairs)

        summary = f"完成！共处理 {n} 个文件"
        total_pp = self._pp_stats.get("total", 0)
        if total_pp > 0:
            pf = self._pp_stats
            lines = [
                f"碎片:     {pf['pre_fragment']}/{total_pp} → {pf['post_fragment']}/{total_pp}",
                f"包含违规: {pf['pre_containment']}/{total_pp} → {pf['post_containment']}/{total_pp}",
                f"间隙:     {pf['pre_gap']}/{total_pp} → {pf['post_gap']}/{total_pp}",
            ]
            self._log_info("═══ 后处理统计 ═══")
            for ln in lines:
                self._log_info(f"  {ln}")
            summary += (f" | PP: 碎片 {pf['pre_fragment']}→{pf['post_fragment']}"
                        f"  包含 {pf['pre_containment']}→{pf['post_containment']}"
                        f"  间隙 {pf['pre_gap']}→{pf['post_gap']}")
            self._save_pp_stats_json()

        self.progress.setFormat(summary)

    def _save_pp_stats_json(self):
        out_dir = self.ed_out.text().strip()
        if not out_dir:
            return
        try:
            stats_path = os.path.join(out_dir, "postprocess_stats.json")
            total_pp = self._pp_stats.get("total", 0)
            report = {
                "total_files": total_pp,
                "aggregate": {
                    "before": {
                        "fragment": self._pp_stats["pre_fragment"],
                        "containment_violation": self._pp_stats["pre_containment"],
                        "gap": self._pp_stats["pre_gap"],
                    },
                    "after": {
                        "fragment": self._pp_stats["post_fragment"],
                        "containment_violation": self._pp_stats["post_containment"],
                        "gap": self._pp_stats["post_gap"],
                    },
                },
                "per_file": self._pp_file_stats,
            }
            os.makedirs(out_dir, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self._log_info(f"统计报告已保存: {stats_path}")
        except Exception as e:
            self._err(f"保存统计 JSON 失败: {e}")

    def _set_running(self, running: bool):
        self._running = running
        self.btn_seg.setEnabled(True)
        if running:
            self.btn_seg.setText("⏹ 取消分割")
        else:
            self.btn_seg.setText("🫀 开始分割")
        lockable = [
            self.ed_model, self.ed_inp, self.ed_out,
            self.rb_folder, self.rb_file, self.rb_cpu,
            self.chk_pp_enable, self.chk_pp_step1,
            self.chk_pp_step2, self.chk_pp_step3,
        ]
        for ctrl in lockable:
            ctrl.setEnabled(not running)
        self.rb_gpu.setEnabled(not running and torch.cuda.is_available())
        if not running and HAS_SCIPY:
            self._on_pp_toggle(self.chk_pp_enable.isChecked())

    # ═══════════════ 显示逻辑 ═══════════════════════════════

    def _draw_placeholder(self):
        ax = self.canvas.ax
        ax.clear()
        ax.text(0.5, 0.5, "请先执行分割以查看结果",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#999999")
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    def _get_display_mode(self) -> str:
        txt = self.cb_display_mode.currentText()
        return "post" if "后处理" in txt else "pred"

    def _refresh_display(self):
        if not self.pairs:
            self._draw_placeholder()
            return

        idx = self.cur_idx
        inp, pred_path, post_path = self.pairs[idx]
        mode  = self._get_display_mode()
        alpha = self.sl_alpha.value() / 100.0
        ax    = self.canvas.ax
        ax.clear()

        try:
            if idx != self._cache_idx:
                self._cache_img, _, _ = load_image(inp)
                self._cache_idx  = idx
                self._cache_mode = ""

            if mode != self._cache_mode:
                if mode == "post" and post_path and os.path.isfile(post_path):
                    mask_file = post_path
                else:
                    mask_file = pred_path
                mn = nib.load(mask_file)
                m  = np.squeeze(mn.get_fdata()).astype(int)
                while m.ndim > 2:
                    ax1 = [a for a in range(m.ndim) if m.shape[a] == 1]
                    if ax1:
                        m = np.squeeze(m, axis=ax1[0])
                    else:
                        sm = int(np.argmin(m.shape))
                        m = np.take(m, m.shape[sm] // 2, axis=sm)
                self._cache_mask = m
                self._cache_mode = mode

            ax.imshow(self._cache_img, cmap="gray", aspect="equal")
            if alpha > 0.01 and self._cache_mask is not None:
                masked = np.ma.masked_where(
                    self._cache_mask == 0, self._cache_mask)
                ax.imshow(masked, cmap=LABEL_CMAP, vmin=0, vmax=3,
                          alpha=alpha, interpolation="nearest")

            tag   = "后处理结果" if mode == "post" else "原始预测"
            fname = Path(inp).name
            if len(fname) > 45:
                fname = "…" + fname[-42:]
            ax.set_title(f"[MedNeXt-L] [{tag}]  {fname}", fontsize=9, pad=4)
            ax.axis("off")

            patches = [Patch(facecolor=LABEL_COLORS_HEX[i],
                             label=LABEL_NAMES[i]) for i in range(1, 4)]
            ax.legend(handles=patches, loc="lower right",
                      fontsize=7, framealpha=0.85)
        except Exception as e:
            ax.text(0.5, 0.5, f"显示失败:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red")
            ax.set_facecolor("#f0f0f0")

        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def _refresh_nav(self):
        n = len(self.pairs)
        self.lbl_idx.setText(f"{self.cur_idx + 1} / {n}" if n else "0 / 0")
        self.btn_prev.setEnabled(self.cur_idx > 0)
        self.btn_next.setEnabled(self.cur_idx < n - 1)

    def _prev(self):
        if self.cur_idx > 0:
            self.cur_idx -= 1
            self._refresh_display()
            self._refresh_nav()

    def _next(self):
        if self.cur_idx < len(self.pairs) - 1:
            self.cur_idx += 1
            self._refresh_display()
            self._refresh_nav()

    def _on_alpha(self, v: int):
        self.lbl_alpha.setText(f"{v} %")
        if self.pairs:
            self._refresh_display()

    # ═══════════════ 日志 ════════════════════════════════════

    def _err(self, msg: str):
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#E00000"))
        cur.insertText(f"⚠ {msg}\n", fmt)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def _log_info(self, msg: str):
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#2980b9"))
        cur.insertText(f"ℹ {msg}\n", fmt)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    # ═══════════════ 窗口关闭 ════════════════════════════════

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()


# ══════════════════════════════════════════════════════════════
# 10.  入口
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(["Source Han Sans SC", "Microsoft YaHei", "SimHei"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
