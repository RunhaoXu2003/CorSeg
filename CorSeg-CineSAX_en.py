#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorSeg-CineSAX_en.py
────────────────────────────────────────────────────────────────
Automatic Segmentation of Short-Axis Cine Cardiac MRI · PyQt6 Inference GUI
Model: MedNeXt-L
Post-Processing: Three-step anatomical constraint pipeline
  Step 1: Connected Component Constraint — Remove isolated fragments
  Step 2: Containment Constraint — Ensure LV cavity is enclosed by myocardium
  Step 3: Gap Filling Constraint — Fill background holes enclosed by cardiac structures
────────────────────────────────────────────────────────────────
"""

# ══════════════════════════════════════════════════════════════
# 0.  High-DPI Scaling
# ══════════════════════════════════════════════════════════════
import sys, os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

# ══════════════════════════════════════════════════════════════
# 1.  Imports
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

# --- Matplotlib (Qt backend) ---
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Arial", "Helvetica", "sans-serif",
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

# --- Optional Dependencies ---
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
# 2.  Constants / Default Configuration
# ══════════════════════════════════════════════════════════════
LABEL_COLORS_HEX = ["#000000", "#FF4444", "#4488FF", "#44CC44"]
LABEL_NAMES      = ["Background", "LV Myocardium", "LV Blood Pool", "RV Blood Pool"]
LABEL_CMAP       = ListedColormap(LABEL_COLORS_HEX)
SUPPORTED_EXT    = (".nii.gz", ".nii", ".dcm")

DEFAULT_CFG = dict(
    spatial_dims=2, in_channels=1, num_classes=4,
    mednext_variant="L", mednext_kernel=5,
    img_size=(224, 224),
)

GUIDE_TEXT = (
    "This tool performs automatic segmentation of short-axis cine "
    "cardiac MRI (Cine SAX) using deep learning.\n\n"
    "[Model]\n"
    "  • MedNeXt-L — Modern ConvNeXt-based medical image segmentation network\n\n"
    "[Segmentation Targets]\n"
    "  • Label 1 (Red):   LV Myocardium\n"
    "  • Label 2 (Blue):  LV Blood Pool (Cavity)\n"
    "  • Label 3 (Green): RV Blood Pool\n\n"
    "[Usage]\n"
    "  1. Select inference mode and compute device\n"
    "  2. Set paths (model weights, input, output)\n"
    "  3. Configure post-processing options (optional)\n"
    "  4. Click 'Start Segmentation' and view results in real time\n\n"
    "[Anatomical Post-Processing]\n"
    "  • Step 1: Connected Component — Keep only the largest region per label\n"
    "  • Step 2: Containment — Ensure LV cavity is fully enclosed by myocardium\n"
    "  • Step 3: Gap Filling — Fill background holes within cardiac structures\n"
    "  When enabled, raw predictions are saved in Prediction/ and\n"
    "  post-processed results in Postprocess/, with a JSON statistics report.\n\n"
    "[Naming]    Output = 'ROI-' + original filename\n"
    "[Structure] Folder mode recursively scans all subdirectories.\n"
)


# ══════════════════════════════════════════════════════════════
# 3.  QSS Global Stylesheet
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
# 4.  Utility Functions
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
            raise ImportError("DICOM support requires pydicom:\n  pip install pydicom")
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
        raise ValueError(f"Unsupported format: {ext}")

    while data.ndim > 2:
        ax1 = [ax for ax in range(data.ndim) if data.shape[ax] == 1]
        if ax1:
            data = np.squeeze(data, axis=ax1[0])
        else:
            sm = int(np.argmin(data.shape))
            data = np.take(data, data.shape[sm] // 2, axis=sm)
    if data.ndim != 2:
        raise ValueError(f"Cannot reduce to 2D (shape={data.shape})")
    return data, affine, ext


def save_mask(pred: np.ndarray, affine: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), path)


# ══════════════════════════════════════════════════════════════
# 5.  Model Build + Load (MedNeXt-L only)
# ══════════════════════════════════════════════════════════════

def load_model(
    model_dir: str, device: torch.device,
) -> Tuple[torch.nn.Module, Dict, str, str]:
    if not HAS_MEDNEXT:
        raise ImportError(
            "MedNeXt is not available. Please install/upgrade MONAI:\n"
            "  pip install monai>=1.3"
        )
    mdir = Path(model_dir)
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {mdir}")
    best = mdir / "best_model.pth"
    if best.exists():
        ckpt_path = best
    else:
        pths = sorted(mdir.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"No .pth file found in: {mdir}")
        ckpt_path = pths[0]

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    if not cfg:
        cfg = DEFAULT_CFG.copy()

    info_str = f"Weights: {ckpt_path.name}"

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
# 6.  Anatomical Post-Processing Pipeline
# ══════════════════════════════════════════════════════════════
#
# Label definition:  0=Background  1=LV Myocardium  2=LV Cavity  3=RV
#
# Anatomical priors:
#   · Each structure should form a single connected region per slice
#   · LV Cavity is fully enclosed by LV Myocardium
#   · No background holes enclosed by cardiac structures

def detect_violations(mask: np.ndarray) -> Dict[str, bool]:
    stats: Dict[str, bool] = {
        "has_fragment": False,
        "has_containment_violation": False,
        "has_gap": False,
    }
    if not HAS_SCIPY:
        return stats

    struct = sp_ndimage.generate_binary_structure(2, 1)

    # -- Fragments: any label with >1 connected component --
    for lv in (1, 2, 3):
        binary = (mask == lv)
        if not binary.any():
            continue
        _, n_cc = sp_ndimage.label(binary)
        if n_cc > 1:
            stats["has_fragment"] = True
            break

    # -- Containment violation: LV Cavity touches Background or RV --
    lv_cav = (mask == 2)
    if lv_cav.any():
        non_lv = (mask == 0) | (mask == 3)
        non_lv_dil = sp_ndimage.binary_dilation(non_lv, structure=struct)
        if (lv_cav & non_lv_dil).any():
            stats["has_containment_violation"] = True

    # -- Gap: background holes enclosed by cardiac structures --
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
    """Step 1: Keep only the largest connected component per label."""
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
    """Step 2: Ensure LV Cavity (2) is fully enclosed by LV Myocardium (1)."""
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
    """Step 3: Fill background holes enclosed by cardiac structures."""
    if not HAS_SCIPY:
        return mask
    result = mask.copy()
    struct = sp_ndimage.generate_binary_structure(2, 1)

    # -- Part A: Fill enclosed holes --
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

    # -- Part B: Fill narrow gaps between LV Myo and RV --
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
# 7.  Inference Worker Thread
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
# 8.  Matplotlib Canvas
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
# 9.  Main Window
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "CorSeg CineSAX — Automatic Cardiac MRI Segmentation"
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

    # ═══════════════ UI Build ═══════════════════════════════

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setObjectName("centralContainer")
        root = QVBoxLayout(container)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(10)

        # -- Guide --
        guide_grp = QGroupBox("About")
        gl = QVBoxLayout(guide_grp)
        gl.setContentsMargins(12, 22, 12, 12)
        lbl_guide = QLabel(GUIDE_TEXT)
        lbl_guide.setWordWrap(True)
        lbl_guide.setStyleSheet("font-size: 10pt; line-height: 1.6;")
        gl.addWidget(lbl_guide)
        root.addWidget(guide_grp)

        # -- Main horizontal layout --
        h_main = QHBoxLayout()
        h_main.setSpacing(12)
        h_main.addWidget(self._make_left(),  stretch=1)
        h_main.addWidget(self._make_right(), stretch=5)
        root.addLayout(h_main, stretch=1)

        # -- Progress bar --
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setFormat("%v / %m  (%p%)")
        self.progress.setValue(0)
        root.addWidget(self.progress)

        # -- Log --
        lbl_log = QLabel("Log:")
        lbl_log.setStyleSheet("color:#888;")
        root.addWidget(lbl_log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.setPlaceholderText("No messages yet")
        root.addWidget(self.log)

        scroll.setWidget(container)
        self.setCentralWidget(scroll)

    # ----------- Left Panel -----------
    def _make_left(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(10)

        # -- Model Architecture --
        g_arch = QGroupBox("Model Architecture")
        l_arch = QVBoxLayout(g_arch)
        lbl_model = QLabel(f"MedNeXt-L  {'✅' if HAS_MEDNEXT else '❌  (pip install monai>=1.3)'}")
        lbl_model.setStyleSheet("font-size: 10pt; font-weight: bold;")
        l_arch.addWidget(lbl_model)
        vb.addWidget(g_arch)

        # -- Inference Mode --
        g1 = QGroupBox("Inference Mode")
        l1 = QVBoxLayout(g1)
        self.rb_folder = QRadioButton("Folder (Batch)")
        self.rb_file   = QRadioButton("Single File")
        self.rb_folder.setChecked(True)
        self._grp_mode = QButtonGroup(self)
        self._grp_mode.addButton(self.rb_folder)
        self._grp_mode.addButton(self.rb_file)
        l1.addWidget(self.rb_folder)
        l1.addWidget(self.rb_file)
        vb.addWidget(g1)

        # -- Compute Device --
        g2 = QGroupBox("Compute Device")
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
            self.rb_gpu.setText("GPU (not available)")
            self.rb_gpu.setEnabled(False)
            self.rb_cpu.setChecked(True)
        l2.addWidget(self.rb_gpu)
        l2.addWidget(self.rb_cpu)
        vb.addWidget(g2)

        # -- Paths --
        g3 = QGroupBox("Path Settings")
        g3l = QGridLayout(g3)
        g3l.setSpacing(8)
        g3l.addWidget(QLabel("Model Folder:"), 0, 0)
        self.ed_model = QLineEdit()
        self.ed_model.setPlaceholderText("Folder containing .pth weights")
        g3l.addWidget(self.ed_model, 0, 1)
        b0 = QPushButton("Browse"); b0.setMaximumWidth(64)
        b0.clicked.connect(self._brw_model)
        g3l.addWidget(b0, 0, 2)

        g3l.addWidget(QLabel("Input Path:"), 1, 0)
        self.ed_inp = QLineEdit()
        self.ed_inp.setPlaceholderText("Image file or folder")
        g3l.addWidget(self.ed_inp, 1, 1)
        b1 = QPushButton("Browse"); b1.setMaximumWidth(64)
        b1.clicked.connect(self._brw_input)
        g3l.addWidget(b1, 1, 2)

        g3l.addWidget(QLabel("Output Path:"), 2, 0)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Folder to save segmentation masks")
        g3l.addWidget(self.ed_out, 2, 1)
        b2 = QPushButton("Browse"); b2.setMaximumWidth(64)
        b2.clicked.connect(self._brw_output)
        g3l.addWidget(b2, 2, 2)
        vb.addWidget(g3)

        # -- Post-Processing --
        g_pp = QGroupBox("Anatomical Post-Processing")
        l_pp = QVBoxLayout(g_pp)
        l_pp.setSpacing(6)

        self.chk_pp_enable = QCheckBox("Enable Anatomical Post-Processing")
        self.chk_pp_enable.setChecked(True)
        self.chk_pp_enable.setStyleSheet("font-weight: bold;")
        self.chk_pp_enable.toggled.connect(self._on_pp_toggle)
        l_pp.addWidget(self.chk_pp_enable)

        indent = QWidget()
        indent_l = QVBoxLayout(indent)
        indent_l.setContentsMargins(20, 0, 0, 0)
        indent_l.setSpacing(4)

        self.chk_pp_step1 = QCheckBox("Step 1: Connected Component — Remove fragments")
        self.chk_pp_step1.setChecked(True)
        indent_l.addWidget(self.chk_pp_step1)

        self.chk_pp_step2 = QCheckBox("Step 2: Containment — LV cavity must be enclosed")
        self.chk_pp_step2.setChecked(True)
        indent_l.addWidget(self.chk_pp_step2)

        self.chk_pp_step3 = QCheckBox("Step 3: Gap Filling — Fill enclosed background holes")
        self.chk_pp_step3.setChecked(True)
        indent_l.addWidget(self.chk_pp_step3)

        l_pp.addWidget(indent)

        if HAS_SCIPY:
            lbl_sp = QLabel("scipy ✅  Post-processing available")
            lbl_sp.setStyleSheet("font-size: 9pt; color: #27ae60;")
        else:
            lbl_sp = QLabel("scipy ❌  Please install: pip install scipy")
            lbl_sp.setStyleSheet("font-size: 9pt; color: #e74c3c;")
            self.chk_pp_enable.setChecked(False)
            self.chk_pp_enable.setEnabled(False)
        l_pp.addWidget(lbl_sp)

        vb.addWidget(g_pp)

        # -- Model Info --
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet(
            "font-size: 9pt; color: #2980b9; padding: 4px;")
        vb.addWidget(self.lbl_model_info)

        # -- Segment Button --
        self.btn_seg = QPushButton("🫀 Start Segmentation")
        self.btn_seg.setObjectName("segmentBtn")
        self.btn_seg.clicked.connect(self._on_segment_click)
        vb.addWidget(self.btn_seg)

        vb.addStretch(1)
        return w

    # ----------- Right Panel -----------
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

        hb.addWidget(QLabel("Display:"))
        self.cb_display_mode = QComboBox()
        self.cb_display_mode.addItems(["Post-Processed", "Raw Prediction"])
        self.cb_display_mode.setMinimumWidth(130)
        self.cb_display_mode.setEnabled(False)
        self.cb_display_mode.currentIndexChanged.connect(
            lambda _: self._refresh_display()
        )
        hb.addWidget(self.cb_display_mode)

        hb.addWidget(QLabel("Opacity:"))
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

        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.setMaximumWidth(96)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev)
        hb.addWidget(self.btn_prev)

        self.lbl_idx = QLabel("0 / 0")
        self.lbl_idx.setStyleSheet(
            "font-size: 10pt; font-weight: bold; min-width:60px;"
            "qproperty-alignment:'AlignCenter';")
        hb.addWidget(self.lbl_idx)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setMaximumWidth(96)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next)
        hb.addWidget(self.btn_next)

        vb.addLayout(hb)
        return w

    # ═══════════════ Post-Processing Controls ═════════════════

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

    # ═══════════════ Browse Dialogs ══════════════════════════

    def _brw_model(self):
        p = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if p:
            self.ed_model.setText(p)

    def _brw_input(self):
        if self.rb_folder.isChecked():
            p = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select Image File", "",
                "Image Files (*.nii *.nii.gz *.dcm);;All Files (*.*)")
        if p:
            self.ed_inp.setText(p)

    def _brw_output(self):
        p = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if p:
            self.ed_out.setText(p)

    # ═══════════════ Segmentation Logic ══════════════════════

    def _on_segment_click(self):
        if self._running:
            if self.worker:
                self.worker.cancel()
            self.btn_seg.setText("⏳ Cancelling...")
            self.btn_seg.setEnabled(False)
            return

        model_dir = self.ed_model.text().strip()
        inp_path  = self.ed_inp.text().strip()
        out_path  = self.ed_out.text().strip()

        if not model_dir:
            self._err("Please set the model folder path first"); return
        if not inp_path:
            self._err("Please set the input path first"); return
        if not out_path:
            self._err("Please set the output path first"); return
        if not HAS_MEDNEXT:
            self._err("MedNeXt is not available. Install MONAI>=1.3"); return

        device = (torch.device("cuda")
                  if self.rb_gpu.isChecked() and torch.cuda.is_available()
                  else torch.device("cpu"))

        # -- Load model --
        try:
            self.model, self.model_cfg, ckname, info_str = load_model(
                model_dir, device)
            self.device = device
            self.lbl_model_info.setText(f"✅ Loaded: MedNeXt-L\n{info_str}")
        except Exception as e:
            self._err(f"Model loading failed: {e}")
            self.lbl_model_info.setText("❌ Loading failed")
            return

        # -- Post-processing config --
        pp_steps   = self._get_pp_steps()
        pp_enabled = any(pp_steps.values())

        # -- Collect files & build output paths --
        folder_mode = self.rb_folder.isChecked()
        try:
            if folder_mode:
                if not os.path.isdir(inp_path):
                    self._err(f"Input path is not a valid folder: {inp_path}")
                    return
                inp_files = collect_files(inp_path)
                if not inp_files:
                    self._err("No supported image files found in input folder")
                    return
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
                    self._err(f"Input path is not a valid file: {inp_path}")
                    return
                if not _is_image(inp_path):
                    self._err(f"Unsupported file format: {inp_path}"); return
                inp_files = [inp_path]
                out_name  = _make_output_name(os.path.basename(inp_path))
                if pp_enabled:
                    pred_files = [os.path.join(out_path, "Prediction", out_name)]
                    post_files = [os.path.join(out_path, "Postprocess", out_name)]
                else:
                    pred_files = [os.path.join(out_path, out_name)]
                    post_files = [None]
        except Exception as e:
            self._err(f"File collection failed: {e}"); return

        # -- Reset state --
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

        # -- Update display mode --
        self.cb_display_mode.blockSignals(True)
        self.cb_display_mode.clear()
        if pp_enabled:
            self.cb_display_mode.addItems(["Post-Processed", "Raw Prediction"])
            self.cb_display_mode.setCurrentIndex(0)
            self.cb_display_mode.setEnabled(True)
        else:
            self.cb_display_mode.addItems(["Raw Prediction"])
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

    # ----------- Worker Callbacks -----------
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

        summary = f"Done! Processed {n} file(s)"
        total_pp = self._pp_stats.get("total", 0)
        if total_pp > 0:
            pf = self._pp_stats
            lines = [
                f"Fragments:   {pf['pre_fragment']}/{total_pp} -> {pf['post_fragment']}/{total_pp}",
                f"Containment: {pf['pre_containment']}/{total_pp} -> {pf['post_containment']}/{total_pp}",
                f"Gaps:        {pf['pre_gap']}/{total_pp} -> {pf['post_gap']}/{total_pp}",
            ]
            self._log_info("=== Post-Processing Summary ===")
            for ln in lines:
                self._log_info(f"  {ln}")
            summary += (f" | PP: Frag {pf['pre_fragment']}->{pf['post_fragment']}"
                        f"  Cont {pf['pre_containment']}->{pf['post_containment']}"
                        f"  Gap {pf['pre_gap']}->{pf['post_gap']}")
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
            self._log_info(f"Statistics saved: {stats_path}")
        except Exception as e:
            self._err(f"Failed to save statistics: {e}")

    def _set_running(self, running: bool):
        self._running = running
        self.btn_seg.setEnabled(True)
        if running:
            self.btn_seg.setText("⏹ Cancel")
        else:
            self.btn_seg.setText("🫀 Start Segmentation")
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

    # ═══════════════ Display ════════════════════════════════

    def _draw_placeholder(self):
        ax = self.canvas.ax
        ax.clear()
        ax.text(0.5, 0.5, "Run segmentation to view results",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#999999")
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    def _get_display_mode(self) -> str:
        txt = self.cb_display_mode.currentText()
        return "post" if "Post" in txt else "pred"

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

            tag   = "Post-Processed" if mode == "post" else "Prediction"
            fname = Path(inp).name
            if len(fname) > 45:
                fname = "..." + fname[-42:]
            ax.set_title(f"[MedNeXt-L] [{tag}]  {fname}", fontsize=9, pad=4)
            ax.axis("off")

            patches = [Patch(facecolor=LABEL_COLORS_HEX[i],
                             label=LABEL_NAMES[i]) for i in range(1, 4)]
            ax.legend(handles=patches, loc="lower right",
                      fontsize=7, framealpha=0.85)
        except Exception as e:
            ax.text(0.5, 0.5, f"Display error:\n{e}",
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

    # ═══════════════ Logging ════════════════════════════════

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

    # ═══════════════ Close ══════════════════════════════════

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()


# ══════════════════════════════════════════════════════════════
# 10.  Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(["Segoe UI", "Arial", "DejaVu Sans"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
