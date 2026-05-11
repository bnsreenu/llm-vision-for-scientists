# LLM-Assisted Scientific Image Annotation Tool

A PyQt5 desktop application for generating per-object segmentation masks in scientific images. It combines **Grounding DINO** (open-vocabulary object detection) with **SAM 2** (segment anything) to let you annotate images by typing plain-text descriptions rather than drawing boxes by hand. Missed objects can be added with a single click using SAM 2's point-prompt segmentation.

The repository now also includes a **fine-tuning pipeline** that adapts Grounding DINO to your specific imaging domain, so the model detects structures like glomeruli reliably at higher confidence thresholds after training on as few as 20 annotated images.

[https://www.youtube.com/playlist?list=PLZsOBAyNTZwYhwXhL8rqruLK_3mbf-CTX](video tutorials)

![Demo](product_demo.gif)

---

## Background

### Grounding DINO

Grounding DINO (Liu et al., 2023) is an open-set object detector that accepts free-form text as input rather than a fixed category list. It fuses a transformer-based visual backbone (Swin Transformer) with a BERT-style text encoder, allowing it to detect any object described in natural language. A prompt like `"glomerulus . renal glomerulus . small circular structure ."` is enough for the model to attempt detection without retraining.

> **Reference:** Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., ... & Zhang, L. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499. [https://arxiv.org/abs/2303.05499](https://arxiv.org/abs/2303.05499)

Models used in this project are hosted on Hugging Face:
- `IDEA-Research/grounding-dino-base`
- `IDEA-Research/grounding-dino-tiny`

### SAM 2

Segment Anything Model 2 (Ravi et al., 2024) is Meta AI's second-generation segmentation model. Given a bounding box or a point click anywhere on an image, SAM 2 produces a high-quality binary mask for the object at that location. In this tool SAM 2 is used in two modes: box-prompted (after Grounding DINO detection) and point-prompted (for manual correction clicks).

> **Reference:** Ravi, N., Gabeur, V., Hu, Y. T., Hu, R., Ryali, C., Ma, T., ... & Feichtenhofer, C. (2024). *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714. [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

Models used in this project are hosted on Hugging Face:
- `facebook/sam2.1-hiera-small`
- `facebook/sam2.1-hiera-base-plus`
- `facebook/sam2.1-hiera-tiny`

---

## Repository Contents

| File | Description |
|------|-------------|
| `01_grounding_dino_bboxes.ipynb` | Notebook 1: Grounding DINO detection from scratch — loads the model, runs text-prompted detection, and visualises bounding boxes with confidence scores. |
| `02_dino_plus_sam2_masks.ipynb` | Notebook 2: Extends Notebook 1 by feeding each detected bounding box into SAM 2 to produce a per-object segmentation mask. |
| `annotation_tool_v4.py` | Main annotation application: full PyQt5 GUI combining both models with interactive manual correction, per-class thresholds, multi-phrase prompts, mask export, and built-in COCO merge tool. |
| `finetune_gdino.py` | Fine-tuning application: PyQt5 GUI for adapting Grounding DINO to your annotated dataset, with live training curves and before/after comparison. |
| `download_models.py` | Utility script to download all model weights to a local folder before running offline. |
| `README.md` | This file. |

---

## Full Pipeline

The tools in this repository form a complete annotation and training pipeline. Each step feeds into the next.

```
Annotate images          Merge annotations        Fine-tune model
annotation_tool_v4.py →  Tools menu (built-in) →  finetune_gdino.py
        ↓                        ↓                        ↓
 per-image COCO JSONs      train.json                best_checkpoint/
 binary mask PNGs          val.json                  final_checkpoint/
                                                          ↓
                                              Load back into annotation_tool_v4.py
                                              via Browse button in section 1
```

### Step 1: Annotate

Use `annotation_tool_v4.py` to annotate your images. Add class names and optional detection phrases, set per-class thresholds, run detection, and correct any missed or false-positive objects manually. Save masks to a flat output folder — the tool writes one COCO JSON per image alongside the binary mask PNGs.

### Step 2: Merge

Open the annotation tool and go to **Tools > Merge COCO Annotations**. Point the dialog at your masks folder and your images folder (the tool stores only filenames in each COCO JSON, so the images folder is required to resolve full paths). Set a val fraction (default 0.20) and click Merge. This produces `train.json` and `val.json` in your chosen output folder, stratified by stain type.

### Step 3: Fine-tune

Open `finetune_gdino.py`. Point section 1 at your local `grounding-dino-base` folder. Point section 2 at `train.json`, `val.json`, and a checkpoints output folder. Set hyperparameters (20 epochs, learning rate 1e-5, batch size 1 are good defaults for 20 images on a 20 GB GPU). Click **Start Training**. The training log and live loss curve update after every epoch. The best checkpoint (by val F1@IoU50) is saved automatically.

### Step 4: Compare

After training, switch to the **Before / After Comparison** tab in the fine-tuning tool. Load a held-out test image and click **Run Comparison** to see base model vs. fine-tuned model detections side by side at the same thresholds.

### Step 5: Deploy

Back in `annotation_tool_v4.py`, select **Custom / fine-tuned (browse below)** from the Grounding DINO dropdown and browse to your `best_checkpoint/` folder. Click **Load Selected Models** — the button turns orange if you forget this step. The fine-tuned model now runs inside the annotation tool with the same workflow as before.

---

## What the Annotation Tool Does

1. **Load models** — choose a Grounding DINO variant and a SAM 2 variant from drop-down menus. Select Custom to load a fine-tuned checkpoint. Models load in a background thread.

2. **Load an image** — supports PNG, JPG, TIFF, and BMP.

3. **Define classes and phrases** — add class names and multiple plain-text detection phrases per class. Grounding DINO uses all phrases in a single detection pass, improving recall on visually ambiguous structures.

4. **Set per-class thresholds** — each class has its own box confidence, text alignment, and NMS threshold. Scientific images typically need lower box thresholds (0.05–0.15) than natural images.

5. **Run detection** — one Grounding DINO pass per class, merged with a cross-class NMS pass, followed by SAM 2 segmentation on each surviving box.

6. **Toggle masks** — press **M** or click **Hide Masks** in section 5 to toggle the mask overlay on and off. Useful for inspecting which detections are correct before deleting false positives.

7. **Manual correction** — Add mode: click a missed object, SAM 2 segments it and assigns it to the active class. Delete mode: click on a mask to remove it. No dialog appears per click.

8. **Merge COCO annotations** — available under **Tools > Merge COCO Annotations** without leaving the application.

9. **Export** — binary mask PNGs (uint8, foreground=1), a summary JSON, and a COCO-format JSON per image.

### Output format

For an image named `kidney_001.jpg` with class `glomerulus`:

```
kidney_001_glomerulus_001.png
kidney_001_glomerulus_002.png
kidney_001_classes.json
kidney_001_coco.json
```

Each PNG is a single-channel uint8 image: pixel value `1` = object, `0` = background.

---

## What the Fine-Tuning Tool Does

- Loads any Grounding DINO checkpoint (base, tiny, or a previously fine-tuned model) as the starting point.
- Reads category names and text phrases directly from `train.json` — nothing is hardcoded. Works with any object class, not just glomeruli.
- Freezes the Swin Transformer backbone and fine-tunes only the detection head, which makes training stable on small datasets (20–100 images).
- Applies linear warmup followed by cosine learning rate decay.
- Evaluates val F1@IoU50 after every epoch and saves the best checkpoint automatically.
- Displays a live loss curve (train loss + val F1) inside the GUI, updated after each epoch.
- Saves `best_checkpoint/` and `final_checkpoint/` to the folder you specify. The original base model folder is never modified.
- Includes a **Before / After Comparison** tab: load a test image and run both the base model and the fine-tuned model at identical thresholds, with NMS and large-box filtering applied to both.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bnsreenu/LLM-Assisted-Scientific-Image-Annotation-Tool
cd LLM-Assisted-Scientific-Image-Annotation-Tool
```

### 2. Install Python dependencies

Python 3.10 or later is recommended.

```bash
pip install PyQt5 torch torchvision transformers accelerate pillow numpy matplotlib
```

For GPU acceleration (strongly recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Download the models

By default the application expects models in:

```
C:\hf_models\
```

Change this by editing `MODEL_BASE` at the top of `annotation_tool_v4.py` and `finetune_gdino.py`:

```python
MODEL_BASE = r"C:\hf_models"
```

Run the download script to fetch all weights:

```bash
python download_models.py
```

This will populate:

```
C:\hf_models\
    grounding-dino-base\
    grounding-dino-tiny\
    sam2-hiera-small\
    sam2-hiera-base-plus\
    sam2-hiera-tiny\
```

---

## Running the Tools

```bash
# Annotation tool
python annotation_tool_v4.py

# Fine-tuning tool
python finetune_gdino.py
```

---

## Workflow Tips

**Low detection confidence on scientific images:** drop the box threshold to 0.05–0.10. Fine-tuning will raise effective confidence for your target class.

**Objects with no standard name:** add descriptive phrases. A class called `vesicle` might use phrases like `"small round membrane structure"` or `"circular lipid body"`.

**Fine-tuning on a small dataset:** 20 images is enough to see improvement. Keep the backbone frozen (default), use batch size 1, and run 20–30 epochs. Watch the val F1 curve — if it plateaus before 20 epochs, training has converged.

**Loading a fine-tuned model in the annotation tool:** select Custom from the Grounding DINO dropdown, browse to `best_checkpoint/`, then click **Load Selected Models**. The button turns orange as a reminder if you browse but forget to reload.

**Mask toggle during correction:** press **M** to hide and show the mask overlay while inspecting detections. Annotations are not affected.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (CPU fallback available) | 8–20 GB |
| Storage | 5 GB (all models) | — |

Both tools move models to CPU between inference passes to minimise VRAM usage.

---

## Acknowledgements

- Grounding DINO: IDEA Research — [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- SAM 2: Meta AI — [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Hugging Face Transformers for model hosting and the unified inference API

---

## License

Released for educational and research use. Model weights are subject to their respective licenses (Apache 2.0 for Grounding DINO; Apache 2.0 for SAM 2).

---

*Created by [DigitalSreeni](https://www.youtube.com/@DigitalSreeni)*
