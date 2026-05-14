"""
Author: Sreenivas Bhattiprolu
YouTube channel: DigitalSreeni

Grounding DINO Fine-Tuning Tool
================================
PyQt5 GUI for fine-tuning Grounding DINO on scientific images using
COCO-format annotations produced by the LLM-Assisted Annotation Tool.

Workflow:
  1. Point to base model folder and training data (train.json + val.json)
  2. Configure training hyperparameters
  3. Run Training — logs and loss curve update live from a worker thread
  4. Load the saved checkpoint and run a before/after comparison on a test image

Requirements:
  pip install PyQt5 torch transformers accelerate pillow numpy torchvision
  pip install matplotlib

Training data must be produced by merge_coco.py, which outputs:
  train.json  — COCO JSON with a "phrases" list per category
  val.json    — same format, held-out images

The text prompt for each training image is built automatically from the
phrases in train.json:
  "glomerulus . glomeruli . renal glomerulus . small circular structure ."
"""

import sys
import json
import math
import copy
import numpy as np
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, nms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QGroupBox, QSizePolicy, QTextEdit, QSpinBox, QDoubleSpinBox,
    QFormLayout, QScrollArea, QSplitter, QFrame, QProgressBar,
    QTabWidget, QAction, QMenu,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor

# ---------------------------------------------------------------------------
# Constants — change MODEL_BASE to match your local model cache
# ---------------------------------------------------------------------------

MODEL_BASE    = r"C:\hf_models"
DEFAULT_MODEL = str(Path(MODEL_BASE) / "grounding-dino-base")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE    = "cpu"

# Inference threshold used for the before/after comparison display
COMPARE_BOX_THR  = 0.30
COMPARE_TEXT_THR = 0.25


# ---------------------------------------------------------------------------
# COCO dataset for Grounding DINO fine-tuning
# ---------------------------------------------------------------------------

class GDINODataset(Dataset):
    """
    Loads images and annotations from a COCO JSON produced by merge_coco.py.

    Each item returns:
        image      : PIL Image (RGB)
        text       : str  — phrases joined with " . ", e.g.
                     "glomerulus . glomeruli . renal glomerulus ."
        boxes      : Tensor (N, 4) in [x_min, y_min, x_max, y_max] absolute pixels
        class_ids  : Tensor (N,) category IDs (1-indexed)
        image_path : str — for logging / display
    """

    def __init__(self, coco_json_path: str):
        with open(coco_json_path) as f:
            coco = json.load(f)

        # Build category lookup
        self.categories = {c["id"]: c for c in coco["categories"]}

        # Build per-image text prompt from category phrases
        # All categories present anywhere in the dataset contribute to the prompt.
        # This matches inference-time behaviour where we pass a fixed prompt.
        all_phrases = []
        seen = set()
        for cat in coco["categories"]:
            for ph in cat.get("phrases", [cat["name"]]):
                if ph.lower() not in seen:
                    all_phrases.append(ph.strip().rstrip("."))
                    seen.add(ph.lower())
        self.text_prompt = " . ".join(all_phrases) + " ."

        # Index annotations by image_id
        ann_by_image = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_info in coco["images"]:
            iid  = img_info["id"]
            anns = ann_by_image.get(iid, [])
            if not anns:
                continue
            # Convert COCO bbox [x,y,w,h] -> [x1,y1,x2,y2]
            boxes = []
            class_ids = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                class_ids.append(ann["category_id"])
            self.samples.append({
                "image_path": img_info["file_name"],
                "boxes":      boxes,
                "class_ids":  class_ids,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        return {
            "image":      image,
            "text":       self.text_prompt,
            "boxes":      torch.tensor(s["boxes"],     dtype=torch.float32),
            "class_ids":  torch.tensor(s["class_ids"], dtype=torch.long),
            "image_path": s["image_path"],
        }


def collate_fn(batch):
    """Keep each sample as a dict; do not stack images or boxes."""
    return batch


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_gdino_loss(model, processor, batch, device):
    """
    Compute Grounding DINO loss for one batch.

    Grounding DINO's HF implementation accepts `labels` as a list of dicts:
        [{"boxes": Tensor(N,4) normalised [cx,cy,w,h], "class_labels": Tensor(N,)}]

    Boxes must be normalised to [0,1] in [cx, cy, w, h] format.
    The model internally computes the bipartite matching loss (Hungarian) over
    all decoder queries, giving us a combined classification + L1 + GIoU loss.
    """
    images = [s["image"] for s in batch]
    texts  = [s["text"]  for s in batch]

    # Use the same text prompt for all images in the batch
    # (they all share the same category set in our single-class use case)
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Build normalised label dicts
    label_list = []
    for s in batch:
        img_w, img_h = s["image"].size
        boxes_abs = s["boxes"].to(device)   # (N,4) x1y1x2y2

        # Convert to normalised cx,cy,w,h
        cx = (boxes_abs[:, 0] + boxes_abs[:, 2]) / 2.0 / img_w
        cy = (boxes_abs[:, 1] + boxes_abs[:, 3]) / 2.0 / img_h
        bw = (boxes_abs[:, 2] - boxes_abs[:, 0]) / img_w
        bh = (boxes_abs[:, 3] - boxes_abs[:, 1]) / img_h
        boxes_norm = torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)

        label_list.append({
            "boxes":        boxes_norm,
            "class_labels": s["class_ids"].to(device) - 1,  # 0-indexed for model
        })

    outputs = model(**inputs, labels=label_list)
    # outputs.loss is a sum over all 900 decoder queries; normalise by the
    # total number of target boxes in the batch so the displayed value is
    # in a human-readable range (typically 1-20) and the effective LR
    # is not sensitive to batch size.
    total_boxes = sum(len(s["boxes"]) for s in batch)
    normalised_loss = outputs.loss / max(total_boxes, 1)
    return normalised_loss


# ---------------------------------------------------------------------------
# Validation metric: mean average precision at IoU 0.5
# ---------------------------------------------------------------------------

def evaluate_map50(model, processor, dataset, device, box_thr=0.20, text_thr=0.20):
    """
    Simple mAP@50 over all images in dataset.
    Returns a float in [0, 1].
    """
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0

    with torch.no_grad():
        for s in dataset.samples:
            image = Image.open(s["image_path"]).convert("RGB")
            inputs = processor(
                images=image,
                text=dataset.text_prompt,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            det = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=[image.size[::-1]],
            )[0]

            pred_boxes = det["boxes"].cpu()  # (M, 4) x1y1x2y2
            gt_boxes   = torch.tensor(s["boxes"], dtype=torch.float32)  # already x1y1x2y2

            if len(gt_boxes) == 0:
                all_fp += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue

            iou_mat = box_iou(pred_boxes, gt_boxes)  # (M, N)
            matched_gt = set()
            tp = 0
            for m in range(len(pred_boxes)):
                best_iou, best_n = iou_mat[m].max(0)
                best_n = best_n.item()
                if best_iou.item() >= 0.50 and best_n not in matched_gt:
                    tp += 1
                    matched_gt.add(best_n)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes)   - tp
            all_tp += tp
            all_fp += fp
            all_fn += fn

    precision = all_tp / max(all_tp + all_fp, 1)
    recall    = all_tp / max(all_tp + all_fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1 as proxy for mAP@50


# ---------------------------------------------------------------------------
# Training worker thread
# ---------------------------------------------------------------------------

class TrainingWorker(QThread):
    progress    = pyqtSignal(str)        # log messages
    epoch_done  = pyqtSignal(int, float, float)  # epoch, train_loss, val_metric
    finished    = pyqtSignal(str)        # path to saved checkpoint
    error       = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop  = False

    def stop(self):
        self._stop = True

    def run(self):
        cfg = self.config
        try:
            # ---- Load model and processor --------------------------------
            self.progress.emit(f"Loading base model from: {cfg['model_path']}")
            processor = AutoProcessor.from_pretrained(cfg["model_path"])
            model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                cfg["model_path"])
            model.train()
            model.to(DEVICE)
            self.progress.emit(f"Model loaded on {DEVICE}.")

            # ---- Datasets ------------------------------------------------
            self.progress.emit(f"Loading training data: {cfg['train_json']}")
            train_ds = GDINODataset(cfg["train_json"])
            self.progress.emit(
                f"  {len(train_ds)} training images  |  "
                f"text prompt: \"{train_ds.text_prompt}\"")

            val_ds = None
            if cfg["val_json"] and Path(cfg["val_json"]).exists():
                val_ds = GDINODataset(cfg["val_json"])
                self.progress.emit(f"  {len(val_ds)} validation images")

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )

            # ---- Optimiser -----------------------------------------------
            # Fine-tune only the transformer encoder + decoder heads.
            # Freeze the backbone (Swin) to preserve pretrained features
            # and reduce memory — critical with 20 GB and small datasets.
            for name, param in model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False

            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            self.progress.emit(
                f"Trainable params: {trainable:,} / {total:,}  "
                f"(backbone frozen)")

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg["lr"],
                weight_decay=1e-4,
            )

            # Linear warmup + cosine decay
            total_steps  = cfg["epochs"] * len(train_loader)
            warmup_steps = max(1, total_steps // 10)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            # ---- Training loop -------------------------------------------
            train_losses, val_metrics = [], []
            best_val   = -1.0
            best_epoch = 0
            output_dir = Path(cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(1, cfg["epochs"] + 1):
                if self._stop:
                    self.progress.emit("Training stopped by user.")
                    break

                model.train()
                epoch_loss = 0.0
                for step, batch in enumerate(train_loader, 1):
                    if self._stop:
                        break
                    optimizer.zero_grad()
                    loss = compute_gdino_loss(model, processor, batch, DEVICE)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=0.1)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                    self.progress.emit(
                        f"  Epoch {epoch}/{cfg['epochs']}  "
                        f"step {step}/{len(train_loader)}  "
                        f"loss={loss.item():.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}")

                avg_loss = epoch_loss / max(len(train_loader), 1)
                train_losses.append(avg_loss)

                # Validation
                val_metric = 0.0
                if val_ds is not None:
                    self.progress.emit(f"  Running validation (epoch {epoch})...")
                    val_metric = evaluate_map50(
                        model, processor, val_ds, DEVICE)
                    self.progress.emit(
                        f"  Epoch {epoch} val F1@50={val_metric:.4f}")
                val_metrics.append(val_metric)

                self.epoch_done.emit(epoch, avg_loss, val_metric)

                # Save best checkpoint
                if val_metric >= best_val:
                    best_val   = val_metric
                    best_epoch = epoch
                    best_path  = output_dir / "best_checkpoint"
                    model.save_pretrained(str(best_path))
                    processor.save_pretrained(str(best_path))
                    self.progress.emit(
                        f"  Saved best checkpoint (epoch {epoch}, "
                        f"val={best_val:.4f}) to {best_path}")

            # Always save final checkpoint
            final_path = output_dir / "final_checkpoint"
            model.save_pretrained(str(final_path))
            processor.save_pretrained(str(final_path))
            self.progress.emit(
                f"Saved final checkpoint to {final_path}")
            self.progress.emit(
                f"Training complete. Best epoch: {best_epoch}  "
                f"val F1@50={best_val:.4f}")

            # Save loss curve
            self._save_loss_plot(
                train_losses, val_metrics, output_dir / "training_curves.png")
            self.progress.emit(
                f"Training curves saved to {output_dir / 'training_curves.png'}")

            self.finished.emit(str(best_path))

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

    def _save_loss_plot(self, losses, val_metrics, path):
        epochs = list(range(1, len(losses) + 1))
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs, losses, "b-o", markersize=4, label="Train loss")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", color="blue", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="blue")
        if any(v > 0 for v in val_metrics):
            ax2 = ax1.twinx()
            ax2.plot(epochs, val_metrics, "r-s", markersize=4,
                     label="Val F1@50")
            ax2.set_ylabel("Val F1@50", color="red", fontsize=11)
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0, 1)
        ax1.spines["top"].set_visible(False)
        ax1.set_title("Training curves", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Post-processing helpers (ported from annotation tool)
# ---------------------------------------------------------------------------

def filter_large_boxes(det: dict, image_pil, max_area_fraction: float = 0.5) -> dict:
    """Remove detections whose box covers more than max_area_fraction of the image."""
    img_w, img_h = image_pil.size
    img_area = img_w * img_h
    boxes  = det["boxes"]   # numpy (N,4)
    scores = det["scores"]
    labels = det.get("labels", det.get("text_labels", []))

    keep = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        if ((x1 - x0) * (y1 - y0)) / img_area < max_area_fraction:
            keep.append(i)

    return {
        "boxes":  boxes[keep],
        "scores": scores[keep],
        "labels": [labels[i] for i in keep],
    }


def apply_nms(det: dict, iou_threshold: float = 0.5) -> dict:
    """Remove duplicate overlapping boxes with Non-Maximum Suppression."""
    boxes  = det["boxes"]
    scores = det["scores"]
    labels = det.get("labels", det.get("text_labels", []))

    if len(boxes) == 0:
        return det

    boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep     = nms(boxes_t, scores_t, iou_threshold).tolist()

    return {
        "boxes":  boxes[keep],
        "scores": scores[keep],
        "labels": [labels[i] for i in keep],
    }


# ---------------------------------------------------------------------------
# Inference worker — for before/after comparison
# ---------------------------------------------------------------------------

class InferenceWorker(QThread):
    finished = pyqtSignal(object, object, object, object)
    # base_result, ft_result, base_image_pil, ft_image_pil
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, image_path, base_model_path, ft_model_path,
                 text_prompt, box_thr, text_thr,
                 max_area_fraction=0.5, nms_iou=0.5):
        super().__init__()
        self.image_path         = image_path
        self.base_model_path    = base_model_path
        self.ft_model_path      = ft_model_path
        self.text_prompt        = text_prompt
        self.box_thr            = box_thr
        self.text_thr           = text_thr
        self.max_area_fraction  = max_area_fraction
        self.nms_iou            = nms_iou

    def _run_inference(self, model_path, image_pil, label):
        self.progress.emit(f"Running {label} inference...")
        processor = AutoProcessor.from_pretrained(model_path)
        model     = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path).to(DEVICE)
        model.eval()

        inputs = processor(
            images=image_pil,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        det = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        model.to(CPU_DEVICE)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        result = {
            "boxes":  det["boxes"].cpu().numpy(),
            "scores": det["scores"].cpu().numpy(),
            "labels": det.get("text_labels", det.get("labels", [])),
        }
        # Apply large-box filter then NMS, same order as annotation tool
        result = filter_large_boxes(result, image_pil,
                                    max_area_fraction=self.max_area_fraction)
        result = apply_nms(result, iou_threshold=self.nms_iou)
        return result

    def run(self):
        try:
            image_pil = Image.open(self.image_path).convert("RGB")
            base_det  = self._run_inference(
                self.base_model_path, image_pil, "base model")
            ft_det    = self._run_inference(
                self.ft_model_path, image_pil, "fine-tuned model")
            self.finished.emit(base_det, ft_det, image_pil, image_pil)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Comparison image rendering
# ---------------------------------------------------------------------------

def render_detections(image_pil: Image.Image, det: dict,
                      title: str, color=(52, 152, 219)) -> QPixmap:
    """
    Draw bounding boxes and confidence scores on a copy of image_pil.
    Returns a QPixmap suitable for display in a QLabel.
    """
    img = image_pil.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    boxes  = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    r, g, b = color
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)
        label_str = f"{labels[i] if i < len(labels) else ''} {score:.2f}"
        tw = max(len(label_str) * 6, 40)
        draw.rectangle([x1, y1 - 16, x1 + tw, y1], fill=(r, g, b, 200))
        draw.text((x1 + 2, y1 - 14), label_str, fill=(255, 255, 255, 255))

    # Title bar
    bar_h = 28
    bar   = Image.new("RGBA", (img.width, bar_h), (40, 40, 40, 220))
    bdraw = ImageDraw.Draw(bar)
    n     = len(boxes)
    bdraw.text((6, 6), f"{title}  |  {n} detection(s)", fill=(255, 255, 255))
    out   = Image.new("RGBA", (img.width, img.height + bar_h), (0, 0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar_h))
    out   = out.convert("RGB")

    arr  = np.array(out)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Scrollable image label
# ---------------------------------------------------------------------------

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#1e1e1e;")
        self.setText("No image")
        self.setStyleSheet(
            "background:#2a2a2a; color:#888; font-size:13px;")
        self.setAlignment(Qt.AlignCenter)
        self._pixmap_orig = None

    def set_pixmap(self, pm: QPixmap):
        self._pixmap_orig = pm
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._pixmap_orig is None:
            return
        self.setPixmap(self._pixmap_orig.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ---------------------------------------------------------------------------
# Live loss plot widget (renders into a QLabel using matplotlib)
# ---------------------------------------------------------------------------

class LossPlotLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background:#1e1e1e;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Loss curve will appear here after the first epoch.")
        self._train_losses  = []
        self._val_metrics   = []

    def update_data(self, train_losses, val_metrics):
        self._train_losses = train_losses
        self._val_metrics  = val_metrics
        self._replot()

    def _replot(self):          # noqa: F811  — override above with PIL version
        if not self._train_losses:
            return
        epochs = list(range(1, len(self._train_losses) + 1))
        fig, ax1 = plt.subplots(figsize=(5, 2.2), dpi=100)
        ax1.plot(epochs, self._train_losses, "b-o", markersize=3)
        ax1.set_xlabel("Epoch", fontsize=9)
        ax1.set_ylabel("Train loss", color="blue", fontsize=9)
        ax1.tick_params(axis="y", labelcolor="blue", labelsize=8)
        ax1.tick_params(axis="x", labelsize=8)
        if any(v > 0 for v in self._val_metrics):
            ax2 = ax1.twinx()
            ax2.plot(epochs, self._val_metrics, "r-s", markersize=3)
            ax2.set_ylabel("Val F1@50", color="red", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="red", labelsize=8)
            ax2.set_ylim(0, 1)
        ax1.spines["top"].set_visible(False)
        fig.tight_layout(pad=0.5)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
        arr = np.array(pil_img)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        pm   = QPixmap.fromImage(qimg)
        self.setPixmap(pm.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._replot()
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class FineTuneTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Grounding DINO Fine-Tuning Tool  |  DigitalSreeni")
        self.resize(1300, 860)

        self._worker         = None
        self._infer_worker   = None
        self._train_losses   = []
        self._val_metrics    = []
        self._ft_model_path  = None   # set after training or via Load button
        self._base_model_path_for_compare = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: tabs (Training | Comparison)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_training_tab(),   "Training")
        self.tabs.addTab(self._build_comparison_tab(), "Before / After Comparison")

        # Right: control panel
        right_widget = QWidget()
        right_widget.setMinimumWidth(320)   # prevents content from being clipped
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._build_right_panel(right_layout)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

        self._build_menu()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            f"Device: {DEVICE.upper()}   |   "
            "Configure paths and hyperparameters, then click Start Training.")

    # -- Menu bar -------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        # View menu: font size
        view_menu = menubar.addMenu("View")
        font_menu = QMenu("Font Size", self)

        for label, size in [("Small", 9), ("Medium", 11), ("Large", 14)]:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, s=size: self._set_font_size(s))
            font_menu.addAction(action)

        view_menu.addMenu(font_menu)

        # Help menu: About
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _set_font_size(self, size: int):
        font = QApplication.instance().font()
        font.setPointSize(size)
        QApplication.instance().setFont(font)
        # Force all child widgets to pick up the new font
        for widget in QApplication.instance().allWidgets():
            widget.setFont(font)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About — Grounding DINO Fine-Tuning Tool",
            "<b>Grounding DINO Fine-Tuning Tool</b><br>"
            "Part of the <i>Applied LLMs for Scientists</i> series<br><br>"
            "<b>Author:</b> Sreenivas Bhattiprolu (DigitalSreeni)<br>"
            "<b>YouTube:</b> "
            "<a href='https://www.youtube.com/@DigitalSreeni'>"
            "youtube.com/@DigitalSreeni</a><br>"
            "<b>GitHub:</b> "
            "<a href='https://github.com/bnsreenu'>"
            "github.com/bnsreenu</a><br><br>"
            "Fine-tune Grounding DINO on your own annotated scientific images "
            "and compare base vs. fine-tuned model performance side by side.<br><br>"
            "Pair with the LLM-Assisted Annotation Tool and merge_coco.py "
            "to build a complete annotation and training pipeline."
        )

    # -- Training tab ------------------------------------------------------

    def _build_training_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(4)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate until training starts
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        # Live loss plot
        self.loss_plot = LossPlotLabel()
        layout.addWidget(self.loss_plot)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#444;")
        layout.addWidget(sep)

        # Log panel
        log_lbl = QLabel("Training log:")
        log_lbl.setStyleSheet("font-size:11px; font-weight:bold; color:#444;")
        layout.addWidget(log_lbl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;"
            "font-family:Consolas,monospace;font-size:11px;"
            "border:1px solid #444;}")
        layout.addWidget(self.log)
        return w

    # -- Comparison tab ----------------------------------------------------

    def _build_comparison_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel("Text prompt for comparison:"))
        self.compare_prompt = QTextEdit()
        self.compare_prompt.setFixedHeight(48)
        self.compare_prompt.setPlaceholderText(
            "glomerulus . glomeruli . renal glomerulus . small circular structure .")
        top.addWidget(self.compare_prompt, 1)
        layout.addLayout(top)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Box thr:"))
        self.spin_cmp_box = QDoubleSpinBox()
        self.spin_cmp_box.setRange(0.01, 0.99)
        self.spin_cmp_box.setSingleStep(0.05)
        self.spin_cmp_box.setValue(COMPARE_BOX_THR)
        self.spin_cmp_box.setToolTip("Minimum confidence score for a detection to be kept.")
        thresh_row.addWidget(self.spin_cmp_box)
        thresh_row.addWidget(QLabel("  Text thr:"))
        self.spin_cmp_txt = QDoubleSpinBox()
        self.spin_cmp_txt.setRange(0.01, 0.99)
        self.spin_cmp_txt.setSingleStep(0.05)
        self.spin_cmp_txt.setValue(COMPARE_TEXT_THR)
        self.spin_cmp_txt.setToolTip("Minimum text-alignment score for a detection to be kept.")
        thresh_row.addWidget(self.spin_cmp_txt)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("NMS IoU:"))
        self.spin_nms_iou = QDoubleSpinBox()
        self.spin_nms_iou.setRange(0.10, 0.95)
        self.spin_nms_iou.setSingleStep(0.05)
        self.spin_nms_iou.setValue(0.50)
        self.spin_nms_iou.setToolTip(
            "IoU threshold for Non-Maximum Suppression. Lower = more aggressive "
            "duplicate removal. 0.5 is a good starting point.")
        filter_row.addWidget(self.spin_nms_iou)
        filter_row.addWidget(QLabel("  Max box area:"))
        self.spin_max_area = QDoubleSpinBox()
        self.spin_max_area.setRange(0.05, 0.99)
        self.spin_max_area.setSingleStep(0.05)
        self.spin_max_area.setValue(0.50)
        self.spin_max_area.setToolTip(
            "Discard boxes covering more than this fraction of the image area. "
            "Removes whole-image false positives. 0.5 = remove boxes larger "
            "than 50% of the image.")
        filter_row.addWidget(self.spin_max_area)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Side-by-side image panels
        img_row = QHBoxLayout()
        img_row.setSpacing(4)

        base_col = QVBoxLayout()
        base_col.addWidget(QLabel("Base model"))
        self.lbl_base = ImageLabel()
        base_col.addWidget(self.lbl_base)

        ft_col = QVBoxLayout()
        ft_col.addWidget(QLabel("Fine-tuned model"))
        self.lbl_ft = ImageLabel()
        ft_col.addWidget(self.lbl_ft)

        img_row.addLayout(base_col)
        img_row.addLayout(ft_col)
        layout.addLayout(img_row, 1)

        # Comparison log
        self.compare_log = QTextEdit()
        self.compare_log.setReadOnly(True)
        self.compare_log.setFixedHeight(80)
        self.compare_log.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;"
            "font-family:Consolas,monospace;font-size:11px;"
            "border:1px solid #444;}")
        layout.addWidget(self.compare_log)
        return w

    # -- Right control panel -----------------------------------------------

    def _build_right_panel(self, layout):

        # 0. About — dataset-agnostic explainer
        about_group  = QGroupBox("About this tool")
        about_layout = QVBoxLayout(about_group)
        about_text = QLabel(
            "This tool fine-tunes Grounding DINO on any object class, "
            "not just glomeruli. It works with any dataset annotated by "
            "the LLM-Assisted Annotation Tool and merged with merge_coco.py.\n\n"
            "Categories, class names, and text phrases are read directly "
            "from train.json — nothing is hardcoded here. To train on "
            "mitochondria, nuclei, cells, or any other structure, simply "
            "point the tool at the correct train.json and val.json.\n\n"
            "The text prompt used during training and comparison is built "
            "automatically from the phrases field in train.json."
        )
        about_text.setWordWrap(True)
        about_text.setStyleSheet(
            "font-size:10px; color:#444; font-style:italic;")
        about_layout.addWidget(about_text)

        # 1. Model path
        mdl_group  = QGroupBox("1. Base Model (starting point for fine-tuning)")
        mdl_layout = QVBoxLayout(mdl_group)
        mdl_layout.addWidget(QLabel("Model folder:"))
        path_row = QHBoxLayout()
        self.lbl_model_path = QLabel(DEFAULT_MODEL)
        self.lbl_model_path.setWordWrap(True)
        self.lbl_model_path.setStyleSheet("font-size:10px;color:#555;")
        self.btn_browse_model = QPushButton("Browse")
        self.btn_browse_model.setFixedWidth(60)
        self.btn_browse_model.clicked.connect(self._browse_model)
        path_row.addWidget(self.lbl_model_path, 1)
        path_row.addWidget(self.btn_browse_model)
        mdl_layout.addLayout(path_row)
        mdl_note = QLabel(
            "Fine-tuning starts from this model and adapts it to your "
            "annotated dataset. The original model files are never changed. "
            "This same model is used as the 'before' baseline in the "
            "Before / After Comparison.")
        mdl_note.setWordWrap(True)
        mdl_note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        mdl_layout.addWidget(mdl_note)

        # 2. Data paths
        data_group  = QGroupBox("2. Training Data")
        data_layout = QFormLayout(data_group)
        data_layout.setLabelAlignment(Qt.AlignRight)

        self.lbl_train_json = QLabel("Not set")
        self.lbl_train_json.setStyleSheet("font-size:10px;color:#555;")
        self.lbl_train_json.setWordWrap(True)
        btn_train = QPushButton("Browse")
        btn_train.setFixedWidth(60)
        btn_train.clicked.connect(lambda: self._browse_json("train"))
        train_row = QHBoxLayout()
        train_row.addWidget(self.lbl_train_json, 1)
        train_row.addWidget(btn_train)
        data_layout.addRow("train.json:", train_row)

        self.lbl_val_json = QLabel("Not set")
        self.lbl_val_json.setStyleSheet("font-size:10px;color:#555;")
        self.lbl_val_json.setWordWrap(True)
        btn_val = QPushButton("Browse")
        btn_val.setFixedWidth(60)
        btn_val.clicked.connect(lambda: self._browse_json("val"))
        val_row = QHBoxLayout()
        val_row.addWidget(self.lbl_val_json, 1)
        val_row.addWidget(btn_val)
        data_layout.addRow("val.json:", val_row)

        self.lbl_output_dir = QLabel("Not set")
        self.lbl_output_dir.setStyleSheet("font-size:10px;color:#555;")
        self.lbl_output_dir.setWordWrap(True)
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(60)
        btn_out.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.lbl_output_dir, 1)
        out_row.addWidget(btn_out)
        data_layout.addRow("Checkpoints folder:", out_row)
        out_note = QLabel(
            "Training saves best_checkpoint/ and final_checkpoint/ "
            "here, plus a loss curve PNG. The base model folder is "
            "never modified.")
        out_note.setWordWrap(True)
        out_note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        data_layout.addRow(out_note)

        # Dataset summary, populated when train.json is loaded
        self.lbl_dataset_summary = QLabel("")
        self.lbl_dataset_summary.setWordWrap(True)
        self.lbl_dataset_summary.setStyleSheet(
            "font-size:10px; color:#2E75B6; "
            "background:#eef4fb; padding:4px; border-radius:3px;")
        self.lbl_dataset_summary.setVisible(False)
        data_layout.addRow(self.lbl_dataset_summary)

        # 3. Hyperparameters
        hp_group  = QGroupBox("3. Hyperparameters")
        hp_layout = QFormLayout(hp_group)
        hp_layout.setLabelAlignment(Qt.AlignRight)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(20)
        hp_layout.addRow("Epochs:", self.spin_epochs)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-7, 1e-2)
        self.spin_lr.setSingleStep(1e-6)
        self.spin_lr.setValue(1e-5)
        hp_layout.addRow("Learning rate:", self.spin_lr)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8)
        self.spin_batch.setValue(1)
        hp_layout.addRow("Batch size:", self.spin_batch)

        note = QLabel(
            "Batch size 1 is recommended for 20 GB VRAM with "
            "grounding-dino-base. Backbone is frozen during fine-tuning.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        hp_layout.addRow(note)

        # 4. Training controls
        ctrl_group  = QGroupBox("4. Training Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)

        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self._start_training)
        self.btn_start.setStyleSheet(
            "QPushButton{background:#2E75B6;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1a5490;}")
        ctrl_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#C0392B;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#922b21;}")
        ctrl_layout.addWidget(self.btn_stop)

        self.lbl_epoch_status = QLabel("")
        self.lbl_epoch_status.setWordWrap(True)
        self.lbl_epoch_status.setStyleSheet("font-size:11px;color:#444;")
        ctrl_layout.addWidget(self.lbl_epoch_status)

        # 5. Comparison
        cmp_group  = QGroupBox("5. Before / After Comparison")
        cmp_layout = QVBoxLayout(cmp_group)

        session_note = QLabel(
            "If you just finished training: skip this button. The best "
            "checkpoint is already loaded automatically.\n\n"
            "If you are returning in a new session: use this button to "
            "point at an existing best_checkpoint/ or final_checkpoint/ "
            "folder from a previous run.")
        session_note.setWordWrap(True)
        session_note.setStyleSheet("font-size:10px;color:#555;font-style:italic;")
        cmp_layout.addWidget(session_note)

        self.btn_load_ft = QPushButton("Load Fine-tuned Model (new session only)")
        self.btn_load_ft.clicked.connect(self._load_finetuned)
        self.btn_load_ft.setStyleSheet(
            "QPushButton{background:#555;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:hover{background:#333;}")
        cmp_layout.addWidget(self.btn_load_ft)

        self.lbl_ft_path = QLabel("No fine-tuned model loaded")
        self.lbl_ft_path.setWordWrap(True)
        self.lbl_ft_path.setStyleSheet("font-size:10px;color:#555;")
        cmp_layout.addWidget(self.lbl_ft_path)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#ddd;")
        cmp_layout.addWidget(sep)

        self.btn_load_test = QPushButton("Load Test Image")
        self.btn_load_test.clicked.connect(self._load_test_image)
        self.btn_load_test.setEnabled(False)
        self.btn_load_test.setStyleSheet(
            "QPushButton{background:#555;color:white;font-weight:bold;"
            "padding:5px;border-radius:3px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#333;}")
        cmp_layout.addWidget(self.btn_load_test)

        self.btn_run_compare = QPushButton("Run Comparison")
        self.btn_run_compare.clicked.connect(self._run_comparison)
        self.btn_run_compare.setEnabled(False)
        self.btn_run_compare.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1e8449;}")
        cmp_layout.addWidget(self.btn_run_compare)

        self.lbl_test_image = QLabel("No test image")
        self.lbl_test_image.setWordWrap(True)
        self.lbl_test_image.setStyleSheet("font-size:10px;color:#555;")
        cmp_layout.addWidget(self.lbl_test_image)

        for w in [about_group, mdl_group, data_group, hp_group, ctrl_group, cmp_group]:
            layout.addWidget(w)

    # ------------------------------------------------------------------
    # Browse handlers
    # ------------------------------------------------------------------

    def _browse_model(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Grounding DINO model folder",
            self.lbl_model_path.text())
        if d:
            self.lbl_model_path.setText(d)

    def _browse_json(self, which):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {which}.json", "", "JSON files (*.json)")
        if not path:
            return
        if which == "train":
            self.lbl_train_json.setText(path)
            self._update_dataset_summary(path)
        else:
            self.lbl_val_json.setText(path)

    def _update_dataset_summary(self, train_json_path: str):
        """
        Parse train.json and display a human-readable summary of what the
        dataset contains: categories, phrases, image count, annotation count.
        Also pre-fills the comparison prompt field.
        """
        try:
            with open(train_json_path) as f:
                coco = json.load(f)
        except Exception as e:
            self.lbl_dataset_summary.setText(f"Could not read JSON: {e}")
            self.lbl_dataset_summary.setVisible(True)
            return

        n_images = len(coco.get("images", []))
        n_anns   = len(coco.get("annotations", []))
        cats     = coco.get("categories", [])

        lines = [f"Dataset: {n_images} images, {n_anns} annotations"]
        phrase_parts = []
        for cat in cats:
            phrases = cat.get("phrases", [cat["name"]])
            lines.append(
                f"  Class: \"{cat['name']}\"  "
                f"({len(phrases)} phrase(s))")
            for ph in phrases:
                lines.append(f"    - {ph}")
            for ph in phrases:
                clean = ph.strip().rstrip(".")
                if clean.lower() not in [p.lower() for p in phrase_parts]:
                    phrase_parts.append(clean)

        # Build and pre-fill the comparison prompt
        prompt = " . ".join(phrase_parts) + " ."
        self.compare_prompt.setPlainText(prompt)

        summary = "\n".join(lines)
        self.lbl_dataset_summary.setText(summary)
        self.lbl_dataset_summary.setVisible(True)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.lbl_output_dir.setText(d)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _start_training(self):
        model_path = self.lbl_model_path.text().strip()
        train_json = self.lbl_train_json.text().strip()
        val_json   = self.lbl_val_json.text().strip()
        output_dir = self.lbl_output_dir.text().strip()

        errors = []
        if not Path(model_path).exists():
            errors.append(f"Model folder not found: {model_path}")
        if not Path(train_json).exists():
            errors.append(f"train.json not found: {train_json}")
        if not output_dir or output_dir == "Not set":
            errors.append("Output folder not set.")
        if errors:
            QMessageBox.warning(self, "Missing inputs", "\n".join(errors))
            return

        self._train_losses  = []
        self._val_metrics   = []
        self.log.clear()
        self._log(f"Starting training on {DEVICE.upper()}...")

        config = {
            "model_path": model_path,
            "train_json": train_json,
            "val_json":   val_json if Path(val_json).exists() else "",
            "output_dir": output_dir,
            "epochs":     self.spin_epochs.value(),
            "lr":         self.spin_lr.value(),
            "batch_size": self.spin_batch.value(),
        }

        self._worker = TrainingWorker(config)
        self._worker.progress.connect(self._log)
        self._worker.epoch_done.connect(self._on_epoch_done)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.error.connect(self._on_training_error)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status.showMessage("Training in progress...")

    def _stop_training(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)
        self.status.showMessage("Stop requested — will finish current step.")

    def _on_epoch_done(self, epoch, train_loss, val_metric):
        self._train_losses.append(train_loss)
        self._val_metrics.append(val_metric)
        self.loss_plot.update_data(self._train_losses, self._val_metrics)
        self.lbl_epoch_status.setText(
            f"Epoch {epoch}  |  loss={train_loss:.4f}  "
            f"|  val={val_metric:.4f}")

    def _on_training_finished(self, checkpoint_path):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._ft_model_path = checkpoint_path
        self._base_model_path_for_compare = self.lbl_model_path.text().strip()
        self.lbl_ft_path.setText(f"Best checkpoint:\n{checkpoint_path}")
        self.btn_load_test.setEnabled(True)
        self.status.showMessage("Training complete. Load a test image to compare.")
        QMessageBox.information(
            self, "Training complete",
            f"Best checkpoint saved to:\n{checkpoint_path}\n\n"
            "Switch to the 'Before / After Comparison' tab, "
            "load a test image, and click Run Comparison.")

    def _on_training_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._log(f"ERROR:\n{msg}")
        self.status.showMessage("Training failed — see log.")
        QMessageBox.critical(self, "Training error", msg[:500])

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _load_finetuned(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select fine-tuned model checkpoint folder")
        if not d:
            return
        self._ft_model_path = d
        # Infer base model: use the path currently in the model field
        self._base_model_path_for_compare = self.lbl_model_path.text().strip()
        self.lbl_ft_path.setText(f"Fine-tuned:\n{d}")
        self.btn_load_test.setEnabled(True)
        self.status.showMessage(
            "Fine-tuned model loaded. Load a test image to compare.")

    def _load_test_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select test image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        self._test_image_path = path
        self.lbl_test_image.setText(Path(path).name)
        self.btn_run_compare.setEnabled(True)
        self.tabs.setCurrentIndex(1)  # switch to comparison tab

    def _run_comparison(self):
        if not self._ft_model_path:
            QMessageBox.warning(
                self, "No model",
                "No fine-tuned model loaded. Train a model or use "
                "'Load Fine-tuned Model'.")
            return

        prompt = self.compare_prompt.toPlainText().strip()
        if not prompt:
            # Try to read prompt from train.json automatically
            train_json = self.lbl_train_json.text().strip()
            if Path(train_json).exists():
                try:
                    with open(train_json) as f:
                        coco = json.load(f)
                    phrases = []
                    seen = set()
                    for cat in coco["categories"]:
                        for ph in cat.get("phrases", [cat["name"]]):
                            if ph.lower() not in seen:
                                phrases.append(ph.strip().rstrip("."))
                                seen.add(ph.lower())
                    prompt = " . ".join(phrases) + " ."
                    self.compare_prompt.setPlainText(prompt)
                    self._compare_log(
                        f"Auto-filled prompt from train.json: {prompt}")
                except Exception:
                    pass
        if not prompt:
            QMessageBox.warning(
                self, "No prompt",
                "Enter a text prompt or point to train.json so it can be "
                "read automatically.")
            return

        self.btn_run_compare.setEnabled(False)
        self._compare_log("Running inference on base model and fine-tuned model...")

        self._infer_worker = InferenceWorker(
            image_path         = self._test_image_path,
            base_model_path    = self._base_model_path_for_compare,
            ft_model_path      = self._ft_model_path,
            text_prompt        = prompt,
            box_thr            = self.spin_cmp_box.value(),
            text_thr           = self.spin_cmp_txt.value(),
            max_area_fraction  = self.spin_max_area.value(),
            nms_iou            = self.spin_nms_iou.value(),
        )
        self._infer_worker.progress.connect(self._compare_log)
        self._infer_worker.finished.connect(self._on_comparison_done)
        self._infer_worker.error.connect(self._on_comparison_error)
        self._infer_worker.start()
        self.status.showMessage("Running comparison inference...")

    def _on_comparison_done(self, base_det, ft_det, base_img, ft_img):
        pm_base = render_detections(
            base_img, base_det,
            title="Base model",
            color=(52, 152, 219))
        pm_ft   = render_detections(
            ft_img, ft_det,
            title="Fine-tuned model",
            color=(39, 174, 96))
        self.lbl_base.set_pixmap(pm_base)
        self.lbl_ft.set_pixmap(pm_ft)

        n_base = len(base_det["boxes"])
        n_ft   = len(ft_det["boxes"])
        base_scores = base_det["scores"]
        ft_scores   = ft_det["scores"]
        msg = (
            f"Base model: {n_base} detection(s)  "
            f"avg conf={float(base_scores.mean()):.3f}" if n_base else
            f"Base model: 0 detections")
        msg += "   |   "
        msg += (
            f"Fine-tuned: {n_ft} detection(s)  "
            f"avg conf={float(ft_scores.mean()):.3f}" if n_ft else
            f"Fine-tuned: 0 detections")
        self._compare_log(msg)
        self.btn_run_compare.setEnabled(True)
        self.status.showMessage("Comparison complete.")

    def _on_comparison_error(self, msg):
        self._compare_log(f"ERROR:\n{msg}")
        self.btn_run_compare.setEnabled(True)
        self.status.showMessage("Comparison failed — see log.")
        QMessageBox.critical(self, "Inference error", msg[:500])

    def _compare_log(self, msg):
        self.compare_log.append(msg)
        self.compare_log.verticalScrollBar().setValue(
            self.compare_log.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FineTuneTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
