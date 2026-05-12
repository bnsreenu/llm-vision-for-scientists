# LLM Vision for Scientists

A growing collection of tools and tutorials showing how large language models and vision-language models can be applied to real scientific image analysis workflows. Built alongside the *Applied LLMs for Scientists* video series on [DigitalSreeni](https://www.youtube.com/@DigitalSreeni).

## Video Playlist

[Applied LLMs for Scientists — YouTube Playlist](https://www.youtube.com/playlist?list=PLZsOBAyNTZwYhwXhL8rqruLK_3mbf-CTX)

---

## Series Overview

| Video | Topic | Code |
|-------|-------|------|
| 1 | Conceptual overview of LLM-assisted image annotation (slides only) | — |
| 2 | Text-prompted object detection with Grounding DINO | `01_grounding_dino_bboxes.ipynb` |
| 3 | Grounding DINO + SAM 2 segmentation pipeline | `02_dino_plus_sam2_masks.ipynb` |
| 4 | Interactive annotation GUI (Grounding DINO + SAM 2) | `annotation_tool_v4.py` |
| 5 | Fine-tuning Grounding DINO on scientific images | `finetune_gdino.py` |
| 6 | Literature-informed object detection using RAG | `rag_literature_tool.py` |
| 7 | SAM3 vs Grounding DINO + SAM 2: a three-way comparison | `annotation_tool_v5.py` |

---

## Repository Contents

| File | Description |
|------|-------------|
| `01_grounding_dino_bboxes.ipynb` | Grounding DINO text-prompted detection: loads the model, runs inference, visualises bounding boxes with confidence scores. |
| `02_dino_plus_sam2_masks.ipynb` | Extends notebook 1 by feeding detected boxes into SAM 2 to produce per-object segmentation masks. |
| `annotation_tool_v4.py` | Annotation GUI (DINO + SAM 2): per-class thresholds, multi-phrase prompts, manual correction by point click, mask export, built-in COCO merge tool. |
| `annotation_tool_v5.py` | Extended annotation GUI adding SAM3 as a second detection backend. Switch between DINO+SAM2 and SAM3 from a dropdown. Includes a prominent model status banner showing which backend is active. |
| `finetune_gdino.py` | Fine-tuning GUI: adapts Grounding DINO to your domain using annotated COCO data, with live loss curves and a before/after comparison tab. |
| `rag_literature_tool.py` | Fully local RAG pipeline: upload scientific PDFs, ask questions, and use retrieved context to guide object detection. No API keys or internet required after setup. |
| `download_models.py` | Downloads all model weights to a local folder before running offline. |

---

## Background

### Grounding DINO

Grounding DINO (Liu et al., 2023) is an open-set object detector that accepts free-form text rather than a fixed category list. It fuses a Swin Transformer visual backbone with a BERT-style text encoder, detecting any object described in natural language. A prompt like `"glomerulus . renal glomerulus . small circular structure ."` is enough to attempt detection without retraining.

> Liu, S. et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499.

Models: `IDEA-Research/grounding-dino-base`, `IDEA-Research/grounding-dino-tiny`

### SAM 2

SAM 2 (Ravi et al., 2024) is Meta AI's second-generation segmentation model. Given a bounding box or a point click, it produces a high-quality binary mask. Used here in two modes: box-prompted (after Grounding DINO detection) and point-prompted (for manual correction clicks).

> Ravi, N. et al. (2024). *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714.

Models: `facebook/sam2.1-hiera-small`, `facebook/sam2.1-hiera-base-plus`, `facebook/sam2.1-hiera-tiny`

### SAM3

SAM3 (Meta AI, 2025) is a unified vision-language segmentation model that accepts text prompts and produces segmentation masks directly, combining detection and segmentation in a single pass. It was evaluated here against the DINO+SAM2 pipeline on kidney histology images (H&E and IHC) under three conditions: SAM3 zero-shot, DINO+SAM2 zero-shot, and fine-tuned DINO+SAM2.

Model: `facebook/sam3.1` — weights at `sam3.1_multiplex.pt` (~3.5 GB), runs on 16 GB VRAM via the Ultralytics API.

### RAG-based Literature-Informed Detection

The RAG tool (video 6) adds a local knowledge base of scientific PDFs to the annotation workflow. PDFs are parsed with PyMuPDF, chunked, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. At query time, relevant passages are retrieved and passed to Llama 3.2 (via Ollama) to synthesise detection guidance. The resulting text prompts are fed into Grounding DINO, grounding detection in domain literature rather than generic descriptions.

---

## Tool Highlights

### Annotation Tool (v5)

Two detection backends selectable from a single dropdown:

**DINO + SAM2:** Grounding DINO for text-prompted bounding-box detection, SAM 2 for mask segmentation. Supports per-class thresholds, multi-phrase prompts, and fine-tuned DINO checkpoints loaded via Browse. Manual Add clicks use SAM 2 point-prompt segmentation.

**SAM3:** Meta's SAM 3.1 unified model. One model handles text-prompted detection and segmentation in a single pass. No DINO or SAM 2 needed.

Both backends share the same class table, phrase editor, Add/Delete correction mode, mask overlay toggle (M key), and COCO export. A prominent model status banner always shows which backend and model variant is active — important when switching between backends during annotation sessions.

### Fine-Tuning Tool

Freezes the Swin Transformer backbone and fine-tunes only the Grounding DINO detection head, which keeps training stable on small datasets. 20 annotated images on an RTX 4000 Ada trains in under 5 minutes and reaches Val F1@IoU50 = 0.70 on kidney histology. Reads class names and phrases directly from `train.json` — nothing is hardcoded.

### RAG Literature Tool

Fully local — no API keys, no cloud calls after initial model download. Streaming architecture processes one PDF at a time with a 200K character / 500 chunk cap per file. Retrieved passages are displayed alongside the generated detection prompt so you can see exactly which literature guided the result.

---

## Full Annotation Pipeline

```
Annotate images            Merge annotations       Fine-tune model
annotation_tool_v5.py  →   Tools > Merge COCO  →   finetune_gdino.py
        |                          |                        |
 per-image COCO JSONs        train.json              best_checkpoint/
 binary mask PNGs            val.json                final_checkpoint/
                                                           |
                                           Load back into annotation_tool_v5.py
                                           via Browse button (DINO+SAM2 backend)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bnsreenu/llm-vision-for-scientists
cd llm-vision-for-scientists
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

For SAM3 support (video 7 onwards):

```bash
pip install ultralytics timm
```

For the RAG literature tool (video 6):

```bash
pip install pymupdf sentence-transformers chromadb
# Also install Ollama and pull llama3.2: https://ollama.com
```

### 3. Download the models

By default the application expects models in `C:\hf_models\`. Edit `MODEL_BASE` at the top of any script to change this.

```bash
python download_models.py
```

This populates:

```
C:\hf_models\
    grounding-dino-base\
    grounding-dino-tiny\
    sam2-hiera-small\
    sam2-hiera-base-plus\
    sam2-hiera-tiny\
    sam3.1\               # for video 7 — download separately from Meta
```

---

## Running the Tools

```bash
# Annotation tool (DINO+SAM2 and SAM3 backends)
python annotation_tool_v5.py

# Fine-tuning tool
python finetune_gdino.py

# RAG literature tool
python rag_literature_tool.py
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (CPU fallback) | 16–20 GB (SAM3 needs ~16 GB) |
| Storage | 10 GB (all models) | — |

DINO+SAM2 mode moves models to CPU between passes to minimise VRAM use. SAM3 keeps the unified model on GPU throughout.

---

## Workflow Tips

**Low detection confidence on scientific images:** drop the box threshold to 0.05–0.10 in DINO+SAM2 mode. Fine-tuning raises effective confidence for your target class after training on as few as 20 images.

**SAM3 on histology images:** SAM3 was trained on natural images and struggles with domain-specific structures like glomeruli under zero-shot conditions. The DINO+SAM2 pipeline with fine-tuning outperforms SAM3 on specialised scientific datasets — this is demonstrated in video 7.

**Multi-phrase prompts:** a class called `glomerulus` might use additional phrases like `"renal glomerulus"` or `"small circular structure in kidney cortex"`. All phrases run in a single detection pass, improving recall on visually ambiguous structures.

**Fine-tuning on a small dataset:** 20 images is enough to see improvement. Keep the backbone frozen (default), use batch size 1, and run 20 to 30 epochs. Watch the val F1 curve — if it plateaus before 20 epochs, training has converged.

**RAG prompts:** upload papers that describe the structure you want to detect. The tool retrieves the most relevant passages and generates detection phrases grounded in the literature rather than generic descriptions.

---

## Acknowledgements

- Grounding DINO: IDEA Research — [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- SAM 2: Meta AI — [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- SAM3: Meta AI — [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Hugging Face Transformers for model hosting and the unified inference API
- Ultralytics for the SAM3 inference API

---

## License

Released for educational and research use. Model weights are subject to their respective licenses (Apache 2.0 for Grounding DINO; Apache 2.0 for SAM 2; see Meta's license for SAM3).

---

*Created by [DigitalSreeni](https://www.youtube.com/@DigitalSreeni) — teaching Python and AI to scientists*
