"""
Author: Sreenivas Bhattiprolu
YouTube channel: DigitalSreeni

LLM-Assisted Scientific Image Annotation Tool
==============================================
Combines Grounding DINO (text-prompted detection) and SAM 2 (segmentation)
with a PyQt5 interface for manual correction via point clicks.

Workflow:
  1. Select models and click Load Selected Models
  2. Load image
  3. Add class names (e.g. "mitochondria") — add descriptive phrases per class,
     set per-class thresholds
  4. Run LLM Detection  (one DINO pass per class using all its phrases,
     then cross-class NMS)
  5. Switch to Add/Delete mode; pick Active Class once, click freely
  6. Save masks to folder

Output per image (e.g. cell_001.jpg):
  cell_001_mitochondria_001.png   (uint8: 1=object, 0=background)
  cell_001_mitochondria_002.png
  cell_001_nucleus_001.png
  cell_001_classes.json

Changes vs. v3
--------------
  * Browse button for Grounding DINO model path — point at any local
    checkpoint including fine-tuned models from the Fine-Tuning Tool.
  * Model status line shows which model is currently loaded.
  * Menu bar: View (font size), Tools (Merge COCO JSONs), Help (About).
  * Merge COCO tool built in — accessible without leaving the annotation UI.
  * Scroll panel min-width fix so right panel content is never clipped.

  * Per-class detection phrases : each class can carry multiple DINO text
    prompts (e.g. "mitochondria", "elongated oval organelle", "rod-shaped
    structure").  All phrases for a class are concatenated into a single DINO
    pass so detection recall improves without losing class assignment.
  * Phrase editor panel (Option B) : selecting a class row in the table
    reveals a live phrase list below with Add / Remove buttons.  The class name
    itself is always the first phrase and cannot be removed.
  * Label matching updated to search the full phrase vocabulary, not just
    class names, then maps any matched phrase back to its canonical class.
  * Ambiguity guard : if a phrase is claimed by two classes the conflict is
    flagged at detection time and the box is marked "unknown".

Changes vs. v2
--------------
  * Per-class thresholds  : each class row has its own box / text / NMS spinboxes.
  * Per-class DINO passes : detection runs once per class so thresholds are applied
    independently; a final cross-class NMS removes inter-class duplicates.
  * Label normalisation   : DINO's returned token is fuzzy-matched back to the
    canonical class name (case-insensitive, plural-strip, edit-distance fallback).
    Unmatched boxes are kept but labelled "unknown".
  * Active-class selector : a persistent QComboBox in the Manual Correction panel.
    All SAM clicks go to the selected class — no per-click popup.

Models are cached after first download:
  Windows : C:\\Users\\<you>\\.cache\\huggingface\\hub
  Linux   : ~/.cache/huggingface/hub

Requirements:
  pip install PyQt5 torch transformers accelerate pillow numpy torchvision
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Processor,
    Sam2Model,
)
from torchvision.ops import nms

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QInputDialog, QMessageBox, QStatusBar, QSplitter, QGroupBox,
    QButtonGroup, QRadioButton, QSizePolicy, QComboBox,
    QTextEdit, QDoubleSpinBox, QFormLayout, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame, QAction, QMenu, QSpinBox
)
from PyQt5.QtCore import Qt, QPoint, QRect, QThread, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QPen, QImage, QCursor, QBrush, QFont
)

# ---------------------------------------------------------------------------
# Available models  —  switch between HF hub or local paths as needed
# ---------------------------------------------------------------------------

MODEL_BASE = r"C:\hf_models"

GDINO_MODELS = {
    "grounding-dino-base (recommended)": f"{MODEL_BASE}/grounding-dino-base",
    "grounding-dino-tiny (faster)":      f"{MODEL_BASE}/grounding-dino-tiny",
    "Custom / fine-tuned (browse below)": "",   # path set via Browse button
}

SAM2_MODELS = {
    "sam2.1-hiera-small (recommended)":        f"{MODEL_BASE}/sam2-hiera-small",
    "sam2.1-hiera-base-plus (better quality)": f"{MODEL_BASE}/sam2-hiera-base-plus",
    "sam2.1-hiera-tiny (fastest)":             f"{MODEL_BASE}/sam2-hiera-tiny",
}

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE    = "cpu"
MAX_AREA_FRAC = 0.70   # boxes covering >70 % of the image are discarded

# 20 distinct RGBA colours for mask overlays
MASK_COLORS = [
    (52,152,219,140),(46,204,113,140),(231,76,60,140),(155,89,182,140),
    (241,196,15,140),(230,126,34,140),(26,188,156,140),(236,112,99,140),
    (52,73,94,140),  (39,174,96,140), (192,57,43,140), (142,68,173,140),
    (243,156,18,140),(211,84,0,140),  (22,160,133,140),(127,179,213,140),
    (130,224,170,140),(245,183,177,140),(195,155,211,140),(250,215,160,140),
]

# Default per-class threshold values
DEFAULT_BOX_THR = 0.25
DEFAULT_TXT_THR = 0.25
DEFAULT_NMS_THR = 0.50

# IoU threshold for the *cross-class* NMS pass after merging all class results
CROSS_CLASS_NMS_THR = 0.50


# ---------------------------------------------------------------------------
# Label normalisation helper
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """
    Lower-case, strip whitespace and common plural suffixes.
    Handles regular plurals ('s', 'es', 'ies') and irregular Latin plurals
    ('i', e.g. glomeruli → glomeru, close enough for edit-distance matching).
    """
    t = text.lower().strip().rstrip(".")
    for suffix in ("ies", "es", "i", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) > 2:
            t = t[:-len(suffix)]
            break
    return t


def _edit_dist(a: str, b: str) -> int:
    """Standard Levenshtein edit distance (no external deps)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = temp
    return dp[lb]


def match_label_to_class(raw_label: str, class_names: list[str]) -> str:
    """
    Fuzzy-match DINO's returned token against canonical class names only.
    Used as a fallback; prefer match_label_to_class_from_configs when
    class_configs (with phrases) are available.

    Strategy (in order of strictness):
      1. Exact match after normalisation (case-insensitive, plural-stripped).
      2. Substring: canonical name is contained in the token or vice-versa
         with >= 60 % coverage.
      3. Edit distance <= 2 on normalised forms (handles irregular plurals,
         tokenisation artefacts, minor OCR drift).
      4. Return "unknown" if nothing matches.
    """
    raw_norm      = _normalise(raw_label)
    best_sub      = None
    best_sub_score = 0.0

    for cn in class_names:
        cn_norm = _normalise(cn)
        if raw_norm == cn_norm:
            return cn
        score = 0.0
        if cn_norm in raw_norm:
            score = len(cn_norm) / max(len(raw_norm), 1)
        elif raw_norm in cn_norm:
            score = len(raw_norm) / max(len(cn_norm), 1)
        if score > best_sub_score:
            best_sub_score = score
            best_sub = cn

    if best_sub is not None and best_sub_score >= 0.60:
        return best_sub

    best_ed      = None
    best_ed_dist = 999
    for cn in class_names:
        cn_norm = _normalise(cn)
        d       = _edit_dist(raw_norm, cn_norm)
        max_len = max(len(raw_norm), len(cn_norm))
        if d <= 2 and max_len >= 4 and d < best_ed_dist:
            best_ed_dist = d
            best_ed = cn

    return best_ed if best_ed is not None else "unknown"


def match_label_to_class_from_configs(
        raw_label: str,
        class_configs: list[dict]) -> str:
    """
    Fuzzy-match DINO's returned token against the full phrase vocabulary
    across all classes, then return the canonical class name.

    class_configs is a list of dicts with at least:
        {"name": str, "phrases": [str, ...], ...}

    The class name itself is always included as the first phrase by convention.

    Matching order per phrase:
      1. Exact normalised match  ->  return canonical name immediately.
      2. Substring coverage >= 60 %.
      3. Edit distance <= 2 on normalised forms.

    Ambiguity guard: if the best-matching phrase belongs to more than one
    class (same score), the result is "unknown".
    """
    raw_norm = _normalise(raw_label)

    # Build flat vocabulary: normalised_phrase -> set of canonical class names
    # (a phrase might accidentally appear in two classes)
    vocab: dict[str, set] = {}
    for cfg in class_configs:
        cn = cfg["name"]
        all_phrases = cfg.get("phrases", [cn])
        if cn not in all_phrases:
            all_phrases = [cn] + list(all_phrases)
        for phrase in all_phrases:
            key = _normalise(phrase)
            vocab.setdefault(key, set()).add(cn)

    # --- Pass 1: exact normalised match
    if raw_norm in vocab:
        owners = vocab[raw_norm]
        if len(owners) == 1:
            return next(iter(owners))
        # Phrase claimed by multiple classes: ambiguous
        return "unknown"

    # --- Pass 2: substring — either direction, including raw being a partial
    #     match of a longer phrase (e.g. "oval organelle" inside
    #     "elongated oval organelle")
    best_phrase_norm  = None
    best_score        = 0.0
    for phrase_norm in vocab:
        score = 0.0
        if phrase_norm in raw_norm:        # full phrase token inside raw
            score = len(phrase_norm) / max(len(raw_norm), 1)
        elif raw_norm in phrase_norm:      # raw token is a subset of phrase
            score = len(raw_norm) / max(len(phrase_norm), 1)
        if score > best_score:
            best_score       = score
            best_phrase_norm = phrase_norm

    if best_phrase_norm is not None and best_score >= 0.50:
        owners = vocab[best_phrase_norm]
        if len(owners) == 1:
            return next(iter(owners))
        return "unknown"

    # --- Pass 3: edit distance <= 2
    best_ed_phrase = None
    best_ed_dist   = 999
    for phrase_norm in vocab:
        d       = _edit_dist(raw_norm, phrase_norm)
        max_len = max(len(raw_norm), len(phrase_norm))
        if d <= 2 and max_len >= 4 and d < best_ed_dist:
            best_ed_dist   = d
            best_ed_phrase = phrase_norm

    if best_ed_phrase is not None:
        owners = vocab[best_ed_phrase]
        if len(owners) == 1:
            return next(iter(owners))
        return "unknown"

    return "unknown"


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class ModelLoader(QThread):
    finished = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, gdino_id, sam2_id):
        super().__init__()
        self.gdino_id = gdino_id
        self.sam2_id  = sam2_id

    def run(self):
        try:
            self.progress.emit("Loading Grounding DINO...")
            gdino_proc  = AutoProcessor.from_pretrained(self.gdino_id)
            gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gdino_id)
            gdino_model.eval()

            self.progress.emit("Loading SAM 2...")
            sam2_proc  = Sam2Processor.from_pretrained(self.sam2_id)
            sam2_model = Sam2Model.from_pretrained(self.sam2_id)
            sam2_model.eval()

            self.finished.emit(gdino_proc, gdino_model, sam2_proc, sam2_model)
        except Exception as e:
            self.error.emit(str(e))


class DetectionWorker(QThread):
    """
    Runs one DINO pass per class (with that class's own thresholds and phrase
    list), collects all boxes, applies a final cross-class NMS, then segments
    with SAM 2.

    `class_configs` is a list of dicts:
        {
          "name":    str,
          "phrases": [str, ...],   # class name is always first
          "box_thr": float,
          "txt_thr": float,
          "nms_thr": float
        }
    """
    finished = pyqtSignal(list)
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, image_pil, class_configs,
                 gdino_proc, gdino_model,
                 sam2_proc, sam2_model):
        super().__init__()
        self.image          = image_pil
        self.class_configs  = class_configs
        self.gdino_proc     = gdino_proc
        self.gdino_model    = gdino_model
        self.sam2_proc      = sam2_proc
        self.sam2_model     = sam2_model

    # ------------------------------------------------------------------
    def _run_dino_for_class(self, cfg):
        """
        Single DINO inference for one class using all its phrases.
        Returns (boxes, scores, canonical_labels).

        The prompt is built by joining all phrases with ' . ' as DINO expects.
        Because this is a single-class pass, every non-unknown match is
        overridden to the canonical class name, removing any ambiguity from
        phrase-level tokenisation.
        """
        # Build prompt: "mitochondria . elongated oval organelle . rod-shaped structure ."
        phrases = cfg.get("phrases", [cfg["name"]])
        if cfg["name"] not in phrases:
            phrases = [cfg["name"]] + list(phrases)
        # Normalise each phrase: strip trailing dots/spaces
        clean_phrases = [p.strip().rstrip(".") for p in phrases if p.strip()]
        prompt = " . ".join(clean_phrases) + " ."

        self.progress.emit(
            f'  DINO: "{cfg["name"]}"  '
            f'({len(clean_phrases)} phrase(s))  '
            f'[box={cfg["box_thr"]:.2f}  '
            f'txt={cfg["txt_thr"]:.2f}  '
            f'nms={cfg["nms_thr"]:.2f}]')

        inputs = self.gdino_proc(
            images=self.image,
            text=prompt,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        det = self.gdino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=cfg["box_thr"],
            text_threshold=cfg["txt_thr"],
            target_sizes=[self.image.size[::-1]]
        )[0]

        boxes      = det["boxes"].cpu()
        scores     = det["scores"].cpu()
        raw_labels = det.get("text_labels", det.get("labels", []))

        if len(boxes) == 0:
            return boxes, scores, []

        # Area filter
        iw, ih = self.image.size
        area   = iw * ih
        keep   = [i for i, b in enumerate(boxes)
                  if ((b[2]-b[0])*(b[3]-b[1])).item() / area < MAX_AREA_FRAC]
        if not keep:
            return torch.zeros((0, 4)), torch.zeros(0), []

        boxes      = boxes[keep]
        scores     = scores[keep]
        raw_labels = [raw_labels[i] for i in keep]

        # Per-class NMS
        keep2      = nms(boxes, scores, cfg["nms_thr"]).tolist()
        boxes      = boxes[keep2]
        scores     = scores[keep2]
        raw_labels = [raw_labels[i] for i in keep2]

        # Single-class pass: every box that survived area filter + NMS belongs
        # to this class by definition.  We do NOT run fuzzy matching here —
        # that was the source of spurious "unknown" labels.  The box_thr is the
        # confidence gate; if a box passed it, it is assigned to cfg["name"].
        # Fuzzy matching is only meaningful when a single prompt contains
        # multiple classes and we need to disambiguate, which is not the case
        # in this per-class architecture.
        norm_labels = [cfg["name"]] * len(raw_labels)

        return boxes, scores, norm_labels

    # ------------------------------------------------------------------
    def run(self):
        try:
            # ---- DINO: one pass per class --------------------------------
            self.gdino_model.to(DEVICE)
            all_boxes, all_scores, all_labels = [], [], []

            for cfg in self.class_configs:
                boxes, scores, labels = self._run_dino_for_class(cfg)
                if len(boxes):
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.extend(labels)

            self.gdino_model.to(CPU_DEVICE)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if not all_boxes:
                self.progress.emit("No detections found across any class.")
                self.finished.emit([])
                return

            all_boxes  = torch.cat(all_boxes,  dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            # ---- Cross-class NMS: remove heavy overlaps between classes --
            cross_keep = nms(all_boxes, all_scores, CROSS_CLASS_NMS_THR).tolist()
            all_boxes  = all_boxes[cross_keep]
            all_scores = all_scores[cross_keep]
            all_labels = [all_labels[i] for i in cross_keep]

            n_det = len(all_boxes)
            self.progress.emit(
                f"{n_det} detection(s) after cross-class NMS. Running SAM 2...")

            # ---- SAM 2: segment each box ---------------------------------
            self.sam2_model.to(DEVICE)
            results = []

            for i in range(n_det):
                box = all_boxes[i].numpy().tolist()

                sam_in = self.sam2_proc(
                    images=self.image,
                    input_boxes=[[box]],
                    return_tensors="pt"
                ).to(DEVICE)

                with torch.no_grad():
                    sam_out = self.sam2_model(**sam_in, multimask_output=False)

                masks_t = self.sam2_proc.post_process_masks(
                    sam_out.pred_masks.cpu(),
                    sam_in["original_sizes"]
                )[0]

                mask = masks_t[0, 0].numpy()
                iou  = sam_out.iou_scores.cpu().squeeze().item()

                results.append({
                    "class_name": all_labels[i],
                    "mask":       mask,
                    "score":      float(all_scores[i].item()),
                    "iou":        float(iou),
                    "source":     "llm"
                })

            self.sam2_model.to(CPU_DEVICE)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# SAM 2 point segmentation (synchronous, called from main thread)
# ---------------------------------------------------------------------------

def segment_point(image_pil, x, y, sam2_proc, sam2_model):
    sam2_model.to(DEVICE)

    inputs = sam2_proc(
        images=image_pil,
        input_points=[[[[x, y]]]],
        input_labels=[[[1]]],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam2_model(**inputs, multimask_output=False)

    masks_t = sam2_proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    mask = masks_t[0, 0].numpy()
    iou  = outputs.iou_scores.cpu().squeeze().item()

    sam2_model.to(CPU_DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return mask, float(iou)


# ---------------------------------------------------------------------------
# Class threshold table widget
# ---------------------------------------------------------------------------

# Column indices
_COL_NAME = 0
_COL_BOX  = 1
_COL_TXT  = 2
_COL_NMS  = 3


class ClassThresholdTable(QTableWidget):
    """
    A QTableWidget where each row = one class.
    Columns: Class Name | Box thr | Text thr | NMS thr
    The spinboxes are embedded directly in the cells.
    """

    def __init__(self):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(
            ["Class", "Box thr", "Txt thr", "NMS thr"])
        self.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.Stretch)
        for col in (_COL_BOX, _COL_TXT, _COL_NMS):
            self.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setMaximumHeight(160)
        self.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { font-size: 11px; font-weight: bold; "
            "  background: #e0e0e0; padding: 2px; }"
        )

    # ------------------------------------------------------------------
    def _make_spin(self, value=0.25):
        sp = QDoubleSpinBox()
        sp.setRange(0.01, 0.99)
        sp.setSingleStep(0.05)
        sp.setDecimals(2)
        sp.setValue(value)
        sp.setFrame(False)
        sp.setStyleSheet("font-size: 11px;")
        return sp

    def add_class(self, name: str):
        """Append a new class row with default thresholds."""
        # Reject duplicates
        for r in range(self.rowCount()):
            if self.item(r, _COL_NAME).text() == name:
                return False

        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, _COL_NAME, QTableWidgetItem(name))
        self.setCellWidget(row, _COL_BOX, self._make_spin(DEFAULT_BOX_THR))
        self.setCellWidget(row, _COL_TXT, self._make_spin(DEFAULT_TXT_THR))
        self.setCellWidget(row, _COL_NMS, self._make_spin(DEFAULT_NMS_THR))
        self.setRowHeight(row, 26)
        return True

    def remove_selected(self):
        row = self.currentRow()
        if row >= 0:
            name = self.item(row, _COL_NAME).text()
            self.removeRow(row)
            return name
        return None

    def get_class_configs(self) -> list[dict]:
        """Return list of {name, box_thr, txt_thr, nms_thr}."""
        configs = []
        for r in range(self.rowCount()):
            configs.append({
                "name":    self.item(r, _COL_NAME).text(),
                "box_thr": self.cellWidget(r, _COL_BOX).value(),
                "txt_thr": self.cellWidget(r, _COL_TXT).value(),
                "nms_thr": self.cellWidget(r, _COL_NMS).value(),
            })
        return configs

    def get_class_names(self) -> list[str]:
        return [self.item(r, _COL_NAME).text()
                for r in range(self.rowCount())]


# ---------------------------------------------------------------------------
# Phrase editor panel  (Option B: live panel below the class table)
# ---------------------------------------------------------------------------

class PhraseEditorPanel(QWidget):
    """
    Shows the phrase list for whichever class row is currently selected in
    ClassThresholdTable.  The class name itself is always the first phrase
    and is locked (cannot be removed).

    Phrases are stored in a dict keyed by class name:
        self._phrases = {"mitochondria": ["mitochondria",
                                          "elongated oval organelle",
                                          "rod-shaped structure"], ...}

    The panel is hidden when no class is selected.
    """

    def __init__(self):
        super().__init__()
        self._phrases: dict[str, list[str]] = {}
        self._active_class: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(3)

        self.lbl_title = QLabel("Phrases for: —")
        self.lbl_title.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #333;")
        layout.addWidget(self.lbl_title)

        hint = QLabel(
            "DINO uses all phrases below for this class.\n"
            "First phrase (class name) cannot be removed.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #777; font-style: italic;")
        layout.addWidget(hint)

        self.phrase_list = QListWidget()
        self.phrase_list.setMaximumHeight(90)
        self.phrase_list.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.phrase_list)

        btn_row = QHBoxLayout()
        self.btn_add_phrase = QPushButton("Add Phrase")
        self.btn_add_phrase.setStyleSheet(
            "QPushButton{font-size:11px;padding:3px 6px;}")
        self.btn_add_phrase.clicked.connect(self._add_phrase)

        self.btn_rem_phrase = QPushButton("Remove Selected")
        self.btn_rem_phrase.setStyleSheet(
            "QPushButton{font-size:11px;padding:3px 6px;}")
        self.btn_rem_phrase.clicked.connect(self._remove_phrase)

        btn_row.addWidget(self.btn_add_phrase)
        btn_row.addWidget(self.btn_rem_phrase)
        layout.addLayout(btn_row)

        self.setVisible(False)

    # ------------------------------------------------------------------
    def set_active_class(self, class_name: str | None):
        """Called when the user selects a different class row."""
        self._active_class = class_name
        if class_name is None:
            self.setVisible(False)
            return
        # Ensure the class has at least its own name as a phrase
        if class_name not in self._phrases:
            self._phrases[class_name] = [class_name]
        self.lbl_title.setText(f"Phrases for:  {class_name}")
        self._refresh_list()
        self.setVisible(True)

    def _refresh_list(self):
        self.phrase_list.clear()
        if self._active_class is None:
            return
        for i, phrase in enumerate(self._phrases[self._active_class]):
            item = QListWidgetItem(phrase)
            if i == 0:
                # Class name row: visually locked
                item.setForeground(QColor("#2E75B6"))
                item.setToolTip("Class name — cannot be removed")
            self.phrase_list.addItem(item)

    def _add_phrase(self):
        if self._active_class is None:
            return
        text, ok = QInputDialog.getText(
            self, "Add Phrase",
            f'New detection phrase for "{self._active_class}":\n'
            f'(e.g. "elongated oval organelle", "rod-shaped structure")')
        if not (ok and text.strip()):
            return
        phrase = text.strip().rstrip(".")
        existing = self._phrases[self._active_class]
        if phrase.lower() in [p.lower() for p in existing]:
            QMessageBox.information(self, "Duplicate",
                                    "That phrase already exists for this class.")
            return
        self._phrases[self._active_class].append(phrase)
        self._refresh_list()

    def _remove_phrase(self):
        if self._active_class is None:
            return
        row = self.phrase_list.currentRow()
        if row <= 0:
            if row == 0:
                QMessageBox.information(
                    self, "Cannot Remove",
                    "The class name phrase cannot be removed.")
            return
        self._phrases[self._active_class].pop(row)
        self._refresh_list()

    # ------------------------------------------------------------------
    # Called by AnnotationTool when a class is added or removed
    # ------------------------------------------------------------------

    def on_class_added(self, class_name: str):
        if class_name not in self._phrases:
            self._phrases[class_name] = [class_name]

    def on_class_removed(self, class_name: str):
        self._phrases.pop(class_name, None)
        if self._active_class == class_name:
            self.set_active_class(None)

    # ------------------------------------------------------------------
    # Data access for DetectionWorker
    # ------------------------------------------------------------------

    def get_phrases_for(self, class_name: str) -> list[str]:
        """Return the phrase list for a class (always includes class name)."""
        phrases = self._phrases.get(class_name, [class_name])
        if class_name not in phrases:
            phrases = [class_name] + phrases
        return phrases

    def get_all_phrases(self) -> dict[str, list[str]]:
        return dict(self._phrases)


# ---------------------------------------------------------------------------
# Image canvas
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.original_pixmap = None
        self.annotations     = []
        self.scale           = 1.0
        self.offset          = QPoint(0, 0)
        self.show_masks      = True   # toggled by M key or button
        self.setMinimumSize(600, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(QCursor(Qt.CrossCursor))

    def load_image(self, pil_img):
        arr  = np.array(pil_img.convert("RGB"))
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.annotations     = []
        self.update()

    def set_annotations(self, anns):
        self.annotations = anns
        self.update()

    def paintEvent(self, event):
        if self.original_pixmap is None:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(40, 40, 40))
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignCenter, "Load an image to begin")
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        ww, wh = self.width(), self.height()
        iw = self.original_pixmap.width()
        ih = self.original_pixmap.height()

        self.scale = min(ww/iw, wh/ih)
        dw = int(iw * self.scale)
        dh = int(ih * self.scale)
        ox = (ww - dw) // 2
        oy = (wh - dh) // 2
        self.offset = QPoint(ox, oy)

        scaled = self.original_pixmap.scaled(
            dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap(ox, oy, scaled)

        if self.show_masks:
            for i, ann in enumerate(self.annotations):
                color = MASK_COLORS[i % len(MASK_COLORS)]
                is_unknown = ann["class_name"] == "unknown"
                mask  = ann["mask"]
                mh, mw = mask.shape

                ov = np.zeros((mh, mw, 4), dtype=np.uint8)
                draw_color = (220, 50, 50, 140) if is_unknown else color
                ov[mask] = draw_color
                qi = QImage(ov.data, mw, mh, 4*mw, QImage.Format_RGBA8888)
                pm = QPixmap.fromImage(qi).scaled(
                    dw, dh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                p.drawPixmap(ox, oy, pm)

                ys, xs = np.where(mask)
                if len(xs):
                    cx = int(np.mean(xs) * self.scale) + ox
                    cy = int(np.mean(ys) * self.scale) + oy
                    badge_color = QColor(200, 50, 50) if is_unknown else QColor(*color[:3])
                    p.setPen(QPen(Qt.white, 1))
                    p.setBrush(QBrush(badge_color))
                    p.drawEllipse(cx-13, cy-13, 26, 26)
                    p.setPen(Qt.white)
                    p.drawText(QRect(cx-13, cy-13, 26, 26),
                               Qt.AlignCenter, str(i+1))

    def mousePressEvent(self, event):
        if self.original_pixmap is None or event.button() != Qt.LeftButton:
            return
        wx = event.x() - self.offset.x()
        wy = event.y() - self.offset.y()
        ix = int(wx / self.scale)
        iy = int(wy / self.scale)
        if 0 <= ix < self.original_pixmap.width() and \
           0 <= iy < self.original_pixmap.height():
            self.clicked.emit(ix, iy)



# ---------------------------------------------------------------------------
# Merge COCO dialog  (launched from Tools menu)
# ---------------------------------------------------------------------------

class _MergeCOCODialog(object.__class__.__mro__[0]):
    pass


from PyQt5.QtWidgets import QDialog

class _MergeCOCODialog(QDialog):
    """
    Self-contained dialog that merges per-image COCO JSONs into
    train.json and val.json.  Embedded in the annotation tool so the
    full pipeline (annotate, merge, train) can be run without switching
    between scripts.
    """

    # Phrases used when the category name matches a known structure.
    # Add entries here for new object types.
    KNOWN_PHRASES = {
        "glomerulus": [
            "glomerulus", "glomeruli", "renal glomerulus",
            "small circular structure", "round cellular cluster",
            "spherical capillary tuft",
        ],
        "mitochondria": [
            "mitochondria", "mitochondrion",
            "elongated oval organelle", "rod-shaped structure",
        ],
        "nucleus": [
            "nucleus", "nuclei", "cell nucleus",
            "round dark structure",
        ],
        "cell": [
            "cell", "cells", "individual cell",
        ],
    }

    def __init__(self, parent=None, extra_phrases=None):
        super().__init__(parent)
        self.setWindowTitle("Merge COCO Annotations  |  DigitalSreeni")
        self.setMinimumWidth(540)
        self.setMinimumHeight(520)
        if extra_phrases:
            self.KNOWN_PHRASES.update(extra_phrases)
        self._build_ui()

    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
            QPushButton, QFileDialog, QDoubleSpinBox, QTextEdit,
            QMessageBox,
        )
        layout = QVBoxLayout(self)

        info = QLabel(
            "Merges the *_coco.json files produced by this tool into a single "
            "dataset and splits it into train.json and val.json for fine-tuning.\n\n"
            "Masks folder: the output directory where COCO JSONs were saved.\n"
            "Images folder: the directory containing the original images "
            "(required because only the filename is stored in each JSON).\n"
            "Output folder: where train.json and val.json will be written."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px;color:#444;padding:4px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        def browse_row(placeholder, pick_dir=True):
            row  = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            btn  = QPushButton("Browse")
            btn.setFixedWidth(70)
            if pick_dir:
                btn.clicked.connect(
                    lambda: edit.setText(
                        QFileDialog.getExistingDirectory(self, placeholder)
                        or edit.text()))
            row.addWidget(edit, 1)
            row.addWidget(btn)
            return row, edit

        masks_row,  self._masks_edit  = browse_row("Folder containing *_coco.json files")
        images_row, self._images_edit = browse_row("Folder containing original images")
        output_row, self._output_edit = browse_row("Folder to write train.json / val.json")

        form.addRow("Masks folder:", masks_row)
        form.addRow("Images folder:", images_row)
        form.addRow("Output folder:", output_row)

        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.05, 0.50)
        self._val_spin.setSingleStep(0.05)
        self._val_spin.setDecimals(2)
        self._val_spin.setValue(0.20)
        self._val_spin.setToolTip(
            "Fraction of images per stain type held out for validation.")
        form.addRow("Val fraction:", self._val_spin)
        layout.addLayout(form)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(160)
        self._log.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;"
            "font-family:Consolas,monospace;font-size:11px;}")
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Merge and Split")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#2E75B6;color:white;font-weight:bold;"
            "padding:7px;border-radius:4px;}"
            "QPushButton:hover{background:#1a5490;}")
        self._btn_run.clicked.connect(self._run)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _log_msg(self, msg):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _run(self):
        import json as _j, math as _m, random as _r
        from collections import defaultdict
        from pathlib import Path

        masks_dir  = self._masks_edit.text().strip()
        images_dir = self._images_edit.text().strip()
        output_dir = self._output_edit.text().strip()

        errors = []
        if not masks_dir or not Path(masks_dir).exists():
            errors.append("Masks folder not found.")
        if not output_dir:
            errors.append("Output folder not set.")
        if errors:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing inputs", "\n".join(errors))
            return

        self._btn_run.setEnabled(False)
        self._log.clear()
        try:
            masks_path  = Path(masks_dir)
            images_path = Path(images_dir) if images_dir else None
            out_path    = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            coco_files = sorted(masks_path.glob("*_coco.json"))
            if not coco_files:
                self._log_msg("No *_coco.json files found in masks folder.")
                return
            self._log_msg(f"Found {len(coco_files)} COCO JSON file(s).")

            # Load and validate
            records = []
            for path in coco_files:
                with open(path) as f:
                    data = _j.load(f)
                if not data.get("images") or not data.get("annotations"):
                    self._log_msg(f"  [skip] {path.name}: empty.")
                    continue
                records.append({"source": path.name, "data": data})
            self._log_msg(f"Valid records: {len(records)}")

            # Category map
            name_to_gid = {}
            gid = 1
            for rec in records:
                for cat in rec["data"].get("categories", []):
                    if cat["name"] not in name_to_gid:
                        name_to_gid[cat["name"]] = gid
                        gid += 1

            global_cats = []
            for name, g in name_to_gid.items():
                phrases = self.KNOWN_PHRASES.get(name, [name])
                global_cats.append({
                    "id": g, "name": name,
                    "supercategory": "object", "phrases": phrases})
            global_cats.sort(key=lambda c: c["id"])

            local_to_global = {}
            for rec in records:
                src = rec["source"]
                for cat in rec["data"].get("categories", []):
                    local_to_global[(src, cat["id"])] = name_to_gid[cat["name"]]

            self._log_msg(
                f"Categories: {[c['name'] for c in global_cats]}")
            for cat in global_cats:
                self._log_msg(
                    f"  '{cat['name']}': phrases: "
                    f"{', '.join(cat['phrases'])}")

            # Flatten records with global IDs
            flat = []
            img_ctr = ann_ctr = 1
            for rec in records:
                src  = rec["source"]
                data = rec["data"]
                img_info = data["images"][0]

                # Resolve file_name
                file_name = img_info["file_name"]
                p = Path(file_name)
                if p.parent == Path(".") and images_path:
                    file_name = str(images_path / p.name)

                local_anns = [
                    a for a in data["annotations"]
                    if a["image_id"] == img_info["id"]]

                new_anns = []
                for ann in local_anns:
                    gcat = local_to_global.get((src, ann["category_id"]))
                    if gcat is None:
                        continue
                    bbox = ann["bbox"]
                    new_anns.append({
                        "id": ann_ctr, "image_id": img_ctr,
                        "category_id": gcat,
                        "bbox": bbox,
                        "area": ann.get("area", bbox[2]*bbox[3]),
                        "iscrowd": 0,
                    })
                    ann_ctr += 1

                flat.append({
                    "image_id": img_ctr,
                    "file_name": file_name,
                    "height": img_info["height"],
                    "width":  img_info["width"],
                    "annotations": new_anns,
                })
                img_ctr += 1

            # Stratified split by stain type (IHC vs everything else)
            def stain(fn):
                return "IHC" if "IHC" in Path(fn).stem.upper() else "HE"

            by_stain = defaultdict(list)
            for r in flat:
                by_stain[stain(r["file_name"])].append(r)

            val_frac = self._val_spin.value()
            rng = _r.Random(42)
            train_imgs, val_imgs = [], []
            for group in by_stain.values():
                rng.shuffle(group)
                n_val = max(1, _m.floor(len(group) * val_frac))
                val_imgs.extend(group[:n_val])
                train_imgs.extend(group[n_val:])

            def to_coco(imgs):
                return {
                    "images": [{"id": r["image_id"], "file_name": r["file_name"],
                                "height": r["height"], "width": r["width"]}
                               for r in imgs],
                    "annotations": [a for r in imgs for a in r["annotations"]],
                    "categories": global_cats,
                }

            with open(out_path / "train.json", "w") as f:
                _j.dump(to_coco(train_imgs), f, indent=2)
            with open(out_path / "val.json", "w") as f:
                _j.dump(to_coco(val_imgs), f, indent=2)

            train_anns = sum(len(r["annotations"]) for r in train_imgs)
            val_anns   = sum(len(r["annotations"]) for r in val_imgs)
            self._log_msg(
                f"\ntrain.json: {len(train_imgs)} images, {train_anns} annotations")
            self._log_msg(
                f"val.json:   {len(val_imgs)} images, {val_anns} annotations")
            self._log_msg(f"\nSaved to: {output_dir}")

            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Done",
                f"Merge complete.\n\n"
                f"train.json: {len(train_imgs)} images\n"
                f"val.json:   {len(val_imgs)} images\n\n"
                f"Output folder:\n{output_dir}")

        except Exception:
            import traceback
            self._log_msg(f"ERROR:\n{traceback.format_exc()}")
        finally:
            self._btn_run.setEnabled(True)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class AnnotationTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "LLM-Assisted Annotation Tool  |  DigitalSreeni  |  v4")
        self.resize(1380, 860)

        self.image_pil    = None
        self.image_path   = None
        self.annotations  = []
        self.gdino_proc   = None
        self.gdino_model  = None
        self.sam2_proc    = None
        self.sam2_model   = None
        self.models_ready = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._on_canvas_click)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)

        right_widget.setMinimumWidth(320)
        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # ── 1. Models ──────────────────────────────────────────────────
        mdl_group  = QGroupBox("1. Models")
        mdl_layout = QVBoxLayout(mdl_group)

        mdl_layout.addWidget(QLabel("Grounding DINO:"))
        self.combo_gdino = QComboBox()
        for k in GDINO_MODELS:
            self.combo_gdino.addItem(k)
        self.combo_gdino.currentTextChanged.connect(self._on_gdino_combo_changed)
        mdl_layout.addWidget(self.combo_gdino)

        # Browse row — shown only when "Custom / fine-tuned" is selected
        self._gdino_browse_row = QWidget()
        browse_row_layout = QHBoxLayout(self._gdino_browse_row)
        browse_row_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_custom_gdino = QLabel("No path set")
        self.lbl_custom_gdino.setWordWrap(True)
        self.lbl_custom_gdino.setStyleSheet("font-size:10px;color:#555;")
        btn_browse_gdino = QPushButton("Browse")
        btn_browse_gdino.setFixedWidth(60)
        btn_browse_gdino.clicked.connect(self._browse_gdino_model)
        browse_row_layout.addWidget(self.lbl_custom_gdino, 1)
        browse_row_layout.addWidget(btn_browse_gdino)
        self._gdino_browse_row.setVisible(False)
        mdl_layout.addWidget(self._gdino_browse_row)

        browse_note = QLabel(
            "To use a fine-tuned model: select 'Custom / fine-tuned', "
            "browse to best_checkpoint/ or final_checkpoint/ from the "
            "Fine-Tuning Tool.")
        browse_note.setWordWrap(True)
        browse_note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        mdl_layout.addWidget(browse_note)

        mdl_layout.addWidget(QLabel("SAM 2:"))
        self.combo_sam2 = QComboBox()
        for k in SAM2_MODELS:
            self.combo_sam2.addItem(k)
        mdl_layout.addWidget(self.combo_sam2)

        self.btn_load_models = QPushButton("Load Selected Models")
        self.btn_load_models.clicked.connect(self._load_models)
        self.btn_load_models.setStyleSheet(
            "QPushButton{background:#555;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:hover{background:#333;}")
        mdl_layout.addWidget(self.btn_load_models)

        # Model status: shows which model is loaded, updates after load
        self.lbl_model_status = QLabel("No model loaded")
        self.lbl_model_status.setWordWrap(True)
        self.lbl_model_status.setStyleSheet(
            "color:#888;font-size:11px;"
            "background:#f5f5f5;padding:4px;border-radius:3px;")
        mdl_layout.addWidget(self.lbl_model_status)

        # ── 2. Image ───────────────────────────────────────────────────
        img_group  = QGroupBox("2. Image")
        img_layout = QVBoxLayout(img_group)
        self.btn_load_img = QPushButton("Load Image")
        self.btn_load_img.clicked.connect(self._load_image)
        self.lbl_image = QLabel("No image loaded")
        self.lbl_image.setWordWrap(True)
        self.lbl_image.setStyleSheet("font-size:11px;color:#555;")
        img_layout.addWidget(self.btn_load_img)
        img_layout.addWidget(self.lbl_image)

        # ── 3. Classes + per-class thresholds ─────────────────────────
        cls_group  = QGroupBox("3. Classes  (set per-class thresholds below)")
        cls_layout = QVBoxLayout(cls_group)

        # Threshold column legend
        legend = QLabel(
            "Box thr: DINO box confidence   "
            "Txt thr: text-alignment score   "
            "NMS thr: duplicate suppression")
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size: 10px; color: #777;")
        cls_layout.addWidget(legend)

        self.class_table = ClassThresholdTable()
        self.class_table.itemSelectionChanged.connect(self._on_class_row_changed)
        cls_layout.addWidget(self.class_table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Class")
        btn_add.clicked.connect(self._add_class)
        btn_rem = QPushButton("Remove Selected")
        btn_rem.clicked.connect(self._remove_class)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_rem)
        cls_layout.addLayout(btn_row)

        # Phrase editor panel — revealed when a class row is selected
        sep_phrases = QFrame()
        sep_phrases.setFrameShape(QFrame.HLine)
        sep_phrases.setStyleSheet("color: #ccc; margin-top: 2px;")
        cls_layout.addWidget(sep_phrases)

        self.phrase_panel = PhraseEditorPanel()
        cls_layout.addWidget(self.phrase_panel)

        # ── 4. Detection ───────────────────────────────────────────────
        det_group  = QGroupBox("4. Detection")
        det_layout = QVBoxLayout(det_group)

        note = QLabel(
            "DINO runs once per class using that class's own thresholds.\n"
            "A cross-class NMS pass (IoU ≥ {:.2f}) then removes "
            "duplicates across classes.".format(CROSS_CLASS_NMS_THR))
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: #666;")
        det_layout.addWidget(note)

        self.btn_detect = QPushButton("Run LLM Detection")
        self.btn_detect.clicked.connect(self._run_detection)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet(
            "QPushButton{background:#2E75B6;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1a5490;}")
        det_layout.addWidget(self.btn_detect)

        # ── 5. Manual Correction ───────────────────────────────────────
        man_group  = QGroupBox("5. Manual Correction")
        man_layout = QVBoxLayout(man_group)

        # --- Active class selector (persistent, no popup per click) ---
        ac_row = QHBoxLayout()
        ac_row.addWidget(QLabel("Active class:"))
        self.combo_active_class = QComboBox()
        self.combo_active_class.setToolTip(
            "All SAM clicks in Add mode are assigned to this class.\n"
            "Update the class list (step 3) to add options here.")
        self.combo_active_class.setMinimumWidth(120)
        ac_row.addWidget(self.combo_active_class, 1)
        man_layout.addLayout(ac_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        man_layout.addWidget(sep)

        man_layout.addWidget(QLabel("Click mode:"))
        self.mode_group = QButtonGroup()
        self.rb_view   = QRadioButton("View only")
        self.rb_add    = QRadioButton("Add mask  (click missed object)")
        self.rb_delete = QRadioButton("Delete mask  (click on mask)")
        self.rb_view.setChecked(True)
        self.mode_group.addButton(self.rb_view,   0)
        self.mode_group.addButton(self.rb_add,    1)
        self.mode_group.addButton(self.rb_delete, 2)
        man_layout.addWidget(self.rb_view)
        man_layout.addWidget(self.rb_add)
        man_layout.addWidget(self.rb_delete)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #ddd;")
        man_layout.addWidget(sep2)

        self.btn_toggle_masks = QPushButton("Hide Masks  [M]")
        self.btn_toggle_masks.setCheckable(True)
        self.btn_toggle_masks.setChecked(False)
        self.btn_toggle_masks.setToolTip(
            "Toggle mask overlay on/off to inspect the underlying image.\n"
            "Keyboard shortcut: M")
        self.btn_toggle_masks.setStyleSheet(
            "QPushButton{background:#555;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:checked{background:#E67E22;color:white;}"
            "QPushButton:hover{background:#333;}")
        self.btn_toggle_masks.clicked.connect(self._toggle_masks)
        man_layout.addWidget(self.btn_toggle_masks)

        hint = QLabel(
            "Tip: set Active class above, then click objects freely.\n"
            "No popup will appear. Press M to toggle masks.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #777; font-style: italic;")
        man_layout.addWidget(hint)

        # ── 6. Annotations ─────────────────────────────────────────────
        ann_group  = QGroupBox("6. Annotations")
        ann_layout = QVBoxLayout(ann_group)
        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(150)
        btn_row2 = QHBoxLayout()
        btn_del  = QPushButton("Delete Selected")
        btn_del.clicked.connect(self._delete_selected)
        btn_clr  = QPushButton("Clear All")
        btn_clr.clicked.connect(self._clear_all)
        btn_row2.addWidget(btn_del)
        btn_row2.addWidget(btn_clr)
        ann_layout.addWidget(self.ann_list)
        ann_layout.addLayout(btn_row2)

        # ── 7. Export ──────────────────────────────────────────────────
        save_group  = QGroupBox("7. Export")
        save_layout = QVBoxLayout(save_group)
        self.btn_save = QPushButton("Save Masks to Folder")
        self.btn_save.clicked.connect(self._save_masks)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1e8449;}")
        self.lbl_save = QLabel("")
        self.lbl_save.setWordWrap(True)
        self.lbl_save.setStyleSheet("font-size:11px;color:#555;")
        save_layout.addWidget(self.btn_save)
        save_layout.addWidget(self.lbl_save)

        # Assemble right panel
        for w in [mdl_group, img_group, cls_group, det_group,
                  man_group, ann_group, save_group]:
            right_layout.addWidget(w)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

        self._build_menu()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            f"Device: {DEVICE.upper()}  |  "
            "Select models and click Load Selected Models to begin.")

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        # View: font size
        view_menu = menubar.addMenu("View")
        font_menu = QMenu("Font Size", self)
        for label, size in [("Small", 9), ("Medium", 11), ("Large", 14)]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked, s=size: self._set_font_size(s))
            font_menu.addAction(act)
        view_menu.addMenu(font_menu)

        # Tools: Merge COCO
        tools_menu = menubar.addMenu("Tools")
        merge_act = QAction("Merge COCO Annotations...", self)
        merge_act.setToolTip(
            "Combine per-image COCO JSONs into train.json and val.json "
            "for fine-tuning.")
        merge_act.triggered.connect(self._launch_merge_tool)
        tools_menu.addAction(merge_act)

        # Help: About
        help_menu = menubar.addMenu("Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _set_font_size(self, size: int):
        font = QApplication.instance().font()
        font.setPointSize(size)
        QApplication.instance().setFont(font)
        for widget in QApplication.instance().allWidgets():
            widget.setFont(font)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About — LLM-Assisted Annotation Tool",
            "<b>LLM-Assisted Scientific Image Annotation Tool</b> v4<br>"
            "Part of the <i>Applied LLMs for Scientists</i> series<br><br>"
            "<b>Author:</b> Sreenivas Bhattiprolu (DigitalSreeni)<br>"
            "<b>YouTube:</b> "
            "<a href='https://www.youtube.com/@DigitalSreeni'>"
            "youtube.com/@DigitalSreeni</a><br>"
            "<b>GitHub:</b> "
            "<a href='https://github.com/bnsreenu'>"
            "github.com/bnsreenu</a><br><br>"
            "Combines Grounding DINO text-prompted detection with SAM 2 "
            "segmentation for LLM-assisted annotation of scientific images.<br><br>"
            "Use <b>Tools > Merge COCO Annotations</b> to prepare training data, "
            "then use the Fine-Tuning Tool to adapt the model to your dataset."
        )

    # ------------------------------------------------------------------
    # GDINO model browse (for custom / fine-tuned models)
    # ------------------------------------------------------------------

    def _on_gdino_combo_changed(self, text):
        is_custom = text.startswith("Custom")
        self._gdino_browse_row.setVisible(is_custom)
        self._mark_reload_needed()

    def _mark_reload_needed(self):
        """Turn the Load button orange to signal the model selection changed."""
        self.btn_load_models.setText("Load Selected Models  ← click to apply")
        self.btn_load_models.setStyleSheet(
            "QPushButton{background:#E67E22;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:hover{background:#CA6F1E;}")

    def _mark_model_loaded(self):
        """Restore Load button to neutral state after successful load."""
        self.btn_load_models.setText("Load Selected Models")
        self.btn_load_models.setStyleSheet(
            "QPushButton{background:#555;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:hover{background:#333;}")

    def _browse_gdino_model(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Grounding DINO model folder", MODEL_BASE)
        if not d:
            return
        self.lbl_custom_gdino.setText(d)
        GDINO_MODELS["Custom / fine-tuned (browse below)"] = d
        self._mark_reload_needed()

    # ------------------------------------------------------------------
    # Merge COCO tool (launched from Tools menu)
    # ------------------------------------------------------------------

    def _launch_merge_tool(self):
        """
        Inline Merge COCO dialog — same logic as merge_coco.py's run_gui(),
        embedded here so the user does not need a separate script.
        """
        import json as _json
        import math as _math
        import random as _random
        from collections import defaultdict as _defaultdict
        from pathlib import Path as _Path

        CATEGORY_PHRASES = {
            "glomerulus": [
                "glomerulus", "glomeruli", "renal glomerulus",
                "small circular structure", "round cellular cluster",
                "spherical capillary tuft",
            ],
        }

        dlg = _MergeCOCODialog(self, CATEGORY_PHRASES)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Mask toggle
    # ------------------------------------------------------------------

    def _toggle_masks(self):
        visible = not self.canvas.show_masks
        self.canvas.show_masks = visible
        self.canvas.update()
        if visible:
            self.btn_toggle_masks.setText('Hide Masks  [M]')
            self.btn_toggle_masks.setChecked(False)
            self.status.showMessage('Masks visible.')
        else:
            self.btn_toggle_masks.setText('Show Masks  [M]')
            self.btn_toggle_masks.setChecked(True)
            self.status.showMessage('Masks hidden — press M or click to restore.')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_M:
            self._toggle_masks()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        gdino_id = GDINO_MODELS[self.combo_gdino.currentText()]
        if not gdino_id:
            QMessageBox.warning(
                self, "No model path",
                "Browse to a Grounding DINO model folder before loading.")
            return
        sam2_id  = SAM2_MODELS[self.combo_sam2.currentText()]
        self.btn_load_models.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.models_ready = False
        self.lbl_model_status.setText("Downloading / loading...")
        self.status.showMessage(
            "Loading models... first run downloads from Hugging Face cache.")

        self.loader = ModelLoader(gdino_id, sam2_id)
        self.loader.progress.connect(self._on_mdl_progress)
        self.loader.finished.connect(self._on_mdl_ready)
        self.loader.error.connect(self._on_mdl_error)
        self.loader.start()

    def _on_mdl_progress(self, msg):
        self.lbl_model_status.setText(msg)
        self.status.showMessage(msg)

    def _on_mdl_ready(self, gp, gm, sp, sm):
        self.gdino_proc   = gp
        self.gdino_model  = gm
        self.sam2_proc    = sp
        self.sam2_model   = sm
        self.models_ready = True
        self.btn_load_models.setEnabled(True)
        self._mark_model_loaded()
        gdino_name = self.combo_gdino.currentText()
        if gdino_name.startswith("Custom"):
            gdino_display = self.lbl_custom_gdino.text()
        else:
            gdino_display = gdino_name.split("(")[0].strip()
        self.lbl_model_status.setText(
            f"Loaded on {DEVICE.upper()}:\n"
            f"DINO: {gdino_display}\n"
            f"SAM2: {self.combo_sam2.currentText().split('(')[0].strip()}")
        self.lbl_model_status.setStyleSheet(
            "color:#27AE60;font-size:11px;font-weight:bold;"
            "background:#eafaf1;padding:4px;border-radius:3px;")
        if self.image_pil is not None:
            self.btn_detect.setEnabled(True)
        self.status.showMessage(
            f"Models ready on {DEVICE.upper()}. "
            "Load an image or run detection.")

    def _on_mdl_error(self, msg):
        self.btn_load_models.setEnabled(True)
        self.lbl_model_status.setText("Error loading models.")
        QMessageBox.critical(self, "Model Load Error", msg)

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        self.image_path  = path
        self.image_pil   = Image.open(path).convert("RGB")
        self.annotations = []
        self.canvas.load_image(self.image_pil)
        self.ann_list.clear()
        self.btn_save.setEnabled(False)
        self.lbl_save.setText("")
        self.lbl_image.setText(
            f"{Path(path).name}\n"
            f"{self.image_pil.width} x {self.image_pil.height} px")
        if self.models_ready:
            self.btn_detect.setEnabled(True)
        self.status.showMessage(
            f"Loaded: {Path(path).name}  "
            f"({self.image_pil.width}x{self.image_pil.height})")

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _add_class(self):
        name, ok = QInputDialog.getText(
            self, "Add Class", "Class name (e.g. mitochondria):")
        if not (ok and name.strip()):
            return
        name = name.strip().lower()
        added = self.class_table.add_class(name)
        if added:
            self.phrase_panel.on_class_added(name)
            self._sync_active_class_combo()
        else:
            QMessageBox.information(
                self, "Duplicate", f'Class "{name}" already exists.')

    def _remove_class(self):
        removed = self.class_table.remove_selected()
        if removed:
            self.phrase_panel.on_class_removed(removed)
            self._sync_active_class_combo()

    def _on_class_row_changed(self):
        """Update the phrase editor panel when the user selects a class row."""
        row = self.class_table.currentRow()
        if row >= 0:
            name = self.class_table.item(row, _COL_NAME).text()
            self.phrase_panel.set_active_class(name)
        else:
            self.phrase_panel.set_active_class(None)

    def _sync_active_class_combo(self):
        """Keep the Active Class dropdown in step 5 in sync with the class table."""
        current = self.combo_active_class.currentText()
        names   = self.class_table.get_class_names()
        self.combo_active_class.clear()
        self.combo_active_class.addItems(names)
        # Restore previous selection if it still exists
        if current in names:
            self.combo_active_class.setCurrentText(current)

    def _get_classes(self) -> list[str]:
        return self.class_table.get_class_names()

    # ------------------------------------------------------------------
    # LLM Detection
    # ------------------------------------------------------------------

    def _run_detection(self):
        if self.image_pil is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        if not self.models_ready:
            QMessageBox.warning(self, "Models Not Ready",
                                "Load models first.")
            return

        class_configs = self.class_table.get_class_configs()
        if not class_configs:
            QMessageBox.warning(
                self, "No Classes",
                "Add at least one class in step 3.")
            return

        # Inject phrase lists from the phrase panel into each config dict
        for cfg in class_configs:
            cfg["phrases"] = self.phrase_panel.get_phrases_for(cfg["name"])

        self.btn_detect.setEnabled(False)
        names_str = ", ".join(c["name"] for c in class_configs)
        self.status.showMessage(
            f'Running detection for: {names_str}')

        self.det_worker = DetectionWorker(
            self.image_pil, class_configs,
            self.gdino_proc, self.gdino_model,
            self.sam2_proc,  self.sam2_model
        )
        self.det_worker.progress.connect(self.status.showMessage)
        self.det_worker.finished.connect(self._on_det_done)
        self.det_worker.error.connect(self._on_det_error)
        self.det_worker.start()

    def _on_det_done(self, results):
        n_unknown = sum(1 for r in results if r["class_name"] == "unknown")
        for ann in results:
            self.annotations.append(ann)
        self._refresh()
        self.btn_detect.setEnabled(True)
        self.btn_save.setEnabled(bool(self.annotations))

        msg = (f"Detection done: {len(results)} new mask(s).  "
               f"Total: {len(self.annotations)}.")
        if n_unknown:
            msg += (f"  ⚠ {n_unknown} mask(s) labelled 'unknown' "
                    f"(DINO token did not match any class). "
                    f"Re-assign via Delete + manual Add, or check class names.")
        msg += "  Use Add/Delete mode to correct."
        self.status.showMessage(msg)

    def _on_det_error(self, msg):
        self.btn_detect.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", msg)
        self.status.showMessage("Detection failed.")

    # ------------------------------------------------------------------
    # Canvas click
    # ------------------------------------------------------------------

    def _on_canvas_click(self, ix, iy):
        mode = self.mode_group.checkedId()
        if mode == 1:
            self._add_mask_at(ix, iy)
        elif mode == 2:
            self._delete_mask_at(ix, iy)

    def _add_mask_at(self, ix, iy):
        if not self.models_ready or self.image_pil is None:
            return

        # Use the persistent active-class selector — no popup
        chosen = self.combo_active_class.currentText()
        if not chosen:
            QMessageBox.warning(
                self, "No Active Class",
                "Add at least one class in step 3, "
                "then select the active class in step 5.")
            return

        self.status.showMessage(
            f"Segmenting ({ix}, {iy}) → '{chosen}'...")
        QApplication.processEvents()

        try:
            mask, iou = segment_point(
                self.image_pil, ix, iy,
                self.sam2_proc, self.sam2_model)
            self.annotations.append({
                "class_name": chosen,
                "mask":       mask,
                "score":      1.0,
                "iou":        iou,
                "source":     "manual"
            })
            self._refresh()
            self.btn_save.setEnabled(True)
            self.status.showMessage(
                f"Added manual mask for '{chosen}'  "
                f"SAM IoU={iou:.2f}.  "
                f"Total: {len(self.annotations)}")
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", str(e))

    def _delete_mask_at(self, ix, iy):
        for i in range(len(self.annotations)-1, -1, -1):
            mask = self.annotations[i]["mask"]
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix]:
                    cn = self.annotations[i]["class_name"]
                    self.annotations.pop(i)
                    self._refresh()
                    self.btn_save.setEnabled(bool(self.annotations))
                    self.status.showMessage(
                        f"Deleted annotation {i+1} ('{cn}').  "
                        f"Total: {len(self.annotations)}")
                    return
        self.status.showMessage("No mask at that location.")

    def _delete_selected(self):
        row = self.ann_list.currentRow()
        if 0 <= row < len(self.annotations):
            cn = self.annotations[row]["class_name"]
            self.annotations.pop(row)
            self._refresh()
            self.btn_save.setEnabled(bool(self.annotations))
            self.status.showMessage(
                f"Deleted annotation {row+1} ('{cn}').  "
                f"Total: {len(self.annotations)}")

    def _clear_all(self):
        if not self.annotations:
            return
        if QMessageBox.question(
                self, "Clear All", "Delete all annotations?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.annotations = []
            self._refresh()
            self.btn_save.setEnabled(False)
            self.status.showMessage("All annotations cleared.")

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh(self):
        self.canvas.set_annotations(self.annotations)
        self.ann_list.clear()
        for i, ann in enumerate(self.annotations):
            color   = MASK_COLORS[i % len(MASK_COLORS)]
            pixels  = int(ann["mask"].sum())
            is_unk  = ann["class_name"] == "unknown"
            prefix  = "⚠ " if is_unk else ""
            item = QListWidgetItem(
                f"{i+1}. {prefix}{ann['class_name']}  [{ann['source']}]  "
                f"det={ann['score']:.2f}  iou={ann['iou']:.2f}  "
                f"({pixels}px)")
            if is_unk:
                item.setForeground(QColor(200, 50, 50))
            else:
                item.setForeground(QColor(*color[:3]))
            self.ann_list.addItem(item)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_masks(self):
        if not self.annotations:
            QMessageBox.warning(self, "Nothing to Save", "No annotations to save.")
            return
    
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out_dir:
            return
    
        stem     = Path(self.image_path).stem
        out_path = Path(out_dir)
        counts   = {}
        saved    = []
    
        # ── Per-instance binary mask PNGs (existing behaviour) ────────────────
        for ann in self.annotations:
            cn = ann["class_name"]
            counts[cn] = counts.get(cn, 0) + 1
            fname = f"{stem}_{cn}_{counts[cn]:03d}.png"
            labeled = ann["mask"].astype(np.uint8)
            Image.fromarray(labeled, mode="L").save(str(out_path / fname))
            saved.append({
                "file":   fname,
                "class":  cn,
                "source": ann["source"],
                "score":  ann["score"],
                "iou":    ann["iou"]
            })
    
        # ── Tool summary JSON (existing behaviour) ─────────────────────────────
        summary = {
            "image":         self.image_path,
            "image_size":    list(self.image_pil.size),
            "device":        DEVICE,
            "annotations":   saved
        }
        with open(str(out_path / f"{stem}_classes.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
        # ── COCO-format JSON (new) ─────────────────────────────────────────────
        self._save_coco_json(out_path, stem, saved)
    
        self.lbl_save.setText(f"{len(saved)} masks saved to\n{Path(out_dir).name}/")
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(saved)} mask(s) to:\n{out_dir}\n\n"
            f"Files include individual masks, summary JSON,\n"
            f"and COCO-format annotations for training.")
        self.status.showMessage(f"Saved {len(saved)} masks to {out_dir}")
    
    
    def _save_coco_json(self, out_path, stem, saved):
        """
        Export annotations in COCO format for fine-tuning object detectors.
    
        Output file: {stem}_coco.json
    
        Format:
          images      : one entry per source image
          annotations : one entry per object instance with bounding box
          categories  : one entry per class name
    
        Bounding boxes are in COCO format: [x_min, y_min, width, height]
        computed from each binary mask.
        """
        img_w, img_h = self.image_pil.size
    
        # Build category list from unique class names in this image
        class_names = sorted(set(ann["class_name"] for ann in self.annotations))
        category_id_map = {cn: i + 1 for i, cn in enumerate(class_names)}
    
        categories = [
            {"id": category_id_map[cn], "name": cn, "supercategory": "object"}
            for cn in class_names
        ]
    
        images = [{
            "id":        1,
            "file_name": Path(self.image_path).name,
            "width":     img_w,
            "height":    img_h
        }]
    
        annotations = []
        for ann_idx, ann in enumerate(self.annotations):
            mask = ann["mask"]   # bool (H, W)
    
            # Compute bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
    
            if not rows.any():
                # Empty mask -- skip
                continue
    
            y_min = int(np.argmax(rows))
            y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
            x_min = int(np.argmax(cols))
            x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            area   = int(mask.sum())
    
            annotations.append({
                "id":          ann_idx + 1,
                "image_id":    1,
                "category_id": category_id_map[ann["class_name"]],
                "bbox":        [x_min, y_min, bbox_w, bbox_h],
                "area":        area,
                "iscrowd":     0,
                "source":      ann["source"],   # "llm" or "manual"
                "score":       ann["score"],
                "iou":         ann["iou"]
            })
    
        coco = {
            "info": {
                "description": "Generated by LLM-Assisted Annotation Tool",
                "url":         "https://github.com/bnsreenu/LLM-Assisted-Scientific-Image-Annotation-Tool",
                "version":     "1.0",
                "contributor": "DigitalSreeni"
            },
            "images":      images,
            "annotations": annotations,
            "categories":  categories
        }
    
        coco_path = out_path / f"{stem}_coco.json"
        with open(str(coco_path), "w") as f:
            json.dump(coco, f, indent=2)
    
        print(f"COCO JSON saved: {coco_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
