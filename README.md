# LLM-Assisted Scientific Image Annotation Tool

A PyQt5 desktop application for generating per-object segmentation masks in scientific images. It combines **Grounding DINO** (open-vocabulary object detection) with **SAM 2** (segment anything) to let you annotate images by typing plain-text descriptions rather than drawing boxes by hand. Missed objects can be added with a single click using SAM 2's point-prompt segmentation.

![Demo](product_demo.gif)

---

## Background

### Grounding DINO

Grounding DINO (Liu et al., 2023) is an open-set object detector that accepts free-form text as input rather than a fixed category list. It fuses a transformer-based visual backbone (Swin Transformer) with a BERT-style text encoder, allowing it to detect any object described in natural language. You do not need to retrain or fine-tune it for new categories: a prompt like `"elongated oval organelle . rod-shaped structure ."` is enough for the model to attempt detection.

> **Reference:** Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., ... & Zhang, L. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499. [https://arxiv.org/abs/2303.05499](https://arxiv.org/abs/2303.05499)

Models used in this project are hosted on Hugging Face:
- `IDEA-Research/grounding-dino-base`
- `IDEA-Research/grounding-dino-tiny`

### SAM 2

Segment Anything Model 2 (Ravi et al., 2024) is Meta AI's second-generation segmentation model. Given a bounding box or a point click anywhere on an image, SAM 2 produces a high-quality binary mask for the object at that location. It requires no category labels and generalises well to scientific and microscopy images. In this tool SAM 2 is used in two modes: box-prompted (after Grounding DINO detection) and point-prompted (for manual correction clicks).

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
| `annotation_tool.py` | Notebook 3 (Application): Full PyQt5 GUI that combines both models with interactive manual correction, per-class thresholds, multi-phrase prompts, and mask export. |
| `download_models.py` | Utility script to download all Grounding DINO and SAM 2 model weights to a local folder before running the application offline. |
| `README.md` | This file. |

---

## What the Application Does

The application provides a complete annotation workflow without writing any code:

1. **Load models** — choose a Grounding DINO variant and a SAM 2 variant from drop-down menus. Models load in a background thread so the UI stays responsive.

2. **Load an image** — supports PNG, JPG, TIFF, and BMP.

3. **Define classes and phrases** — add one or more class names (e.g. `glomerulus`, `mitochondria`). For each class you can add multiple plain-text detection phrases that describe the object in different ways (e.g. `"elongated oval organelle"`, `"rod-shaped structure"`). Grounding DINO uses all phrases for that class in a single detection pass, improving recall on objects that are visually ambiguous or lack a standard name.

4. **Set per-class thresholds** — each class has its own box confidence threshold, text alignment threshold, and NMS threshold. Scientific images often require lower box thresholds (0.05–0.15) than natural images; per-class control avoids having to compromise across all categories.

5. **Run detection** — one Grounding DINO pass is run per class. Results are merged and a final cross-class NMS pass removes duplicate boxes at class boundaries. Each surviving box is fed to SAM 2 to produce a binary mask.

6. **Manual correction** — switch to Add or Delete mode. Select the active class once from the drop-down; all subsequent point clicks segment and assign to that class automatically using SAM 2. No dialog appears per click. Incorrect masks are removed by clicking on them in Delete mode, or by selecting them in the annotation list.

7. **Export** — masks are saved as individual PNG files (uint8, foreground=1, background=0), one file per object. A JSON summary records the image path, model names, per-class thresholds and phrases, and every annotation's class, source, confidence score, and SAM IoU score.

### Output format

For an image named `kidney_001.jpg` with classes `glomerulus` and `tubule`:

```
kidney_001_glomerulus_001.png
kidney_001_glomerulus_002.png
kidney_001_tubule_001.png
kidney_001_classes.json
```

Each PNG is a single-channel uint8 image: pixel value `1` = object, `0` = background. This format is directly compatible with most scientific image analysis pipelines (ImageJ, Python, MATLAB).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/llm-annotation-tool.git
cd llm-annotation-tool
```

### 2. Install Python dependencies

Python 3.10 or later is recommended.

```bash
pip install PyQt5 torch torchvision transformers accelerate pillow numpy
```

For GPU acceleration (strongly recommended for SAM 2):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Download the models

The application expects models to be pre-downloaded to a local folder. By default this is:

```
C:\hf_models\
```

You can change this path by editing the `MODEL_BASE` variable at the top of `annotation_tool.py`:

```python
MODEL_BASE = r"C:\hf_models"   # change this to any path you prefer
```

Run the provided download script to fetch all weights:

```bash
python download_models.py
```

This will download the following into `MODEL_BASE`:

```
C:\hf_models\
    grounding-dino-base\
    grounding-dino-tiny\
    sam2-hiera-small\
    sam2-hiera-base-plus\
    sam2-hiera-tiny\
```

Alternatively, you can switch back to automatic Hugging Face Hub downloads by using the hub model IDs directly in `annotation_tool.py` (commented-out lines are included at the top of the file for reference).

---

## Running the Application

```bash
python annotation_tool.py
```

The window title shows the device in use (CPU or CUDA). GPU is used for inference and models are moved back to CPU between passes to minimise VRAM usage, allowing large SAM 2 variants to run on GPUs with 6–8 GB VRAM.

---

## Workflow Tips

**For scientific images with low detection confidence:**
Drop the box threshold to 0.05–0.10. Every box that passes this threshold is automatically assigned to its class: the tool does not use the label returned by Grounding DINO for assignment (since the model is run once per class, all returned boxes definitively belong to that class regardless of the label text).

**For objects with no standard name:**
Add descriptive phrases. For example, a class called `vesicle` might benefit from additional phrases like `"small round membrane structure"` or `"circular lipid body"`. DINO will attempt detection on each phrase and all hits are assigned to `vesicle`.

**For overlapping classes:**
Increase the NMS threshold to allow more overlap, or decrease it to enforce stricter separation. The cross-class NMS threshold (`CROSS_CLASS_NMS_THR = 0.50` in the source) controls how aggressively duplicate boxes across different classes are removed.

**Unknown masks:**
Any mask labelled `unknown` in the annotation list means the detection was retained but could not be confidently assigned. This should not occur in normal use after the v3 fix (all boxes from a single-class pass are now directly assigned). If you see unknown masks, check that the class was properly added before running detection.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (CPU fallback available) | 8 GB |
| Storage | 5 GB (all models) | — |

The application runs on CPU if no CUDA GPU is detected, but SAM 2 segmentation will be slow (several seconds per object).

---

## Acknowledgements

- Grounding DINO: IDEA Research — [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- SAM 2: Meta AI — [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Hugging Face Transformers for model hosting and the unified inference API

---

## License

This project is released for educational and research use. Model weights are subject to their respective licenses (Apache 2.0 for Grounding DINO; Apache 2.0 for SAM 2).

---

*Created by [DigitalSreeni](https://www.youtube.com/@DigitalSreeni)*
