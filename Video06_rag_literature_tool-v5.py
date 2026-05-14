"""
Author: Sreenivas Bhattiprolu (DigitalSreeni)
YouTube: https://www.youtube.com/@DigitalSreeni

Literature-Informed Object Detection — RAG Tool
================================================
PyQt5 desktop application that builds a searchable index from a folder of
scientific PDFs and answers questions about them using a local Llama model
via Ollama. Runs entirely offline — no API keys, no internet connection
required after initial setup.

Designed as part of the Applied LLMs for Scientists series. Works alongside
the annotation tool and fine-tuning tool to inform detection phrase selection,
threshold choices, and experimental design from published literature.

Workflow:
  1. Point the tool at a folder of PDFs and an output folder for the index.
  2. Click Build Index — PDFs are parsed, chunked, embedded locally using
     all-MiniLM-L6-v2, and stored in a ChromaDB vector database on disk.
  3. Ask questions in plain English. The tool retrieves the most relevant
     passages and passes them to Llama running locally via Ollama, which
     synthesises a grounded answer with citations back to the source papers.

Setup — Ollama and Llama:
  1. Download and install Ollama from https://ollama.com
     (Windows/Mac/Linux installers available)
  2. Open a terminal and pull the Llama model:
         ollama pull llama3.2
     This downloads a ~2 GB model. Larger alternatives if you have VRAM:
         ollama pull llama3.1:8b     (better quality, ~5 GB)
         ollama pull mistral         (good alternative, ~4 GB)
  3. Ollama runs as a local server on http://localhost:11434
     It starts automatically after installation.
  4. Change OLLAMA_MODEL below if you pulled a different model.

Python requirements:
  pip install PyQt5 chromadb sentence-transformers pymupdf python-dotenv
"""

import os
import sys
import json
import textwrap
from pathlib import Path
import urllib.request, json as _json  # for Ollama local API
import chromadb
from chromadb.utils import embedding_functions
import fitz  # pymupdf

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QGroupBox, QTextEdit, QSplitter, QProgressBar, QScrollArea,
    QAction, QMenu, QListWidget, QListWidgetItem, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL    = "all-MiniLM-L6-v2"   # local, no API needed
OLLAMA_MODEL   = "llama3.2"   # change to any model you have pulled
COLLECTION     = "literature"
CHUNK_SIZE     = 800    # characters per chunk
CHUNK_OVERLAP  = 100    # overlap between consecutive chunks
TOP_K          = 6      # number of chunks retrieved per query
MAX_TOKENS     = 1500   # Claude response budget

# Suggested queries shown on first launch
EXAMPLE_QUERIES = [
    "What morphological phrases are used to describe glomeruli?",
    "What dataset sizes have been used for glomerulus detection?",
    "What staining protocols are used in glomerular segmentation studies?",
    "What confidence thresholds have worked for histology object detection?",
    "What deep learning architectures are used for kidney image analysis?",
    "What are common challenges in automated glomerulus detection?",
]

# ---------------------------------------------------------------------------
# PDF ingestion helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path, max_chars: int = 200_000) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Much faster and more memory-efficient than pypdf for scientific PDFs.
    Caps output at max_chars to guard against malformed files.
    Returns __GARBAGE__ if the PDF appears to be image-only.
    """
    try:
        doc = fitz.open(str(pdf_path))
        parts = []
        total = 0
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text.strip())
                total += len(text)
                if total >= max_chars:
                    break
        doc.close()
        full_text = "\n".join(parts)
        if not full_text.strip():
            return ""
        # Detect image-only PDFs: less than 50 chars per page on average
        if len(full_text) / max(len(parts), 1) < 50:
            return "__GARBAGE__"
        return full_text
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP,
               max_chunks: int = 500) -> list[str]:
    """
    Split text into overlapping chunks.
    Caps at max_chunks to prevent runaway memory on malformed input.
    """
    # Hard cap on input length regardless of what caller passed
    if len(text) > 200_000:
        text = text[:200_000]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + chunk_size - 150, end)
            if boundary != -1:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Index build worker
# ---------------------------------------------------------------------------

class IndexWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(int, int)   # n_papers, n_chunks
    error    = pyqtSignal(str)

    def __init__(self, pdf_dir: str, index_dir: str, embed_model):
        super().__init__()
        self.pdf_dir    = Path(pdf_dir)
        self.index_dir  = Path(index_dir)
        self.embed_model = embed_model

    def run(self):
        try:
            pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
            if not pdf_files:
                self.error.emit(f"No PDF files found in {self.pdf_dir}")
                return

            self.progress.emit(
                f"Found {len(pdf_files)} PDF(s). Starting indexing...")
            embed_model = self.embed_model

            # Create fresh ChromaDB collection
            client = chromadb.PersistentClient(path=str(self.index_dir))
            try:
                client.delete_collection(COLLECTION)
            except Exception:
                pass
            collection = client.create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

            # ── Stream: one PDF at a time, embed and write immediately ────
            # Nothing accumulates globally. Memory stays bounded.
            EMBED_BATCH = 32
            total_chunks = 0
            n_papers     = 0
            skipped      = 0
            chunk_counter = 0  # global chunk counter for unique IDs

            for i, pdf_path in enumerate(pdf_files, 1):
                self.progress.emit(
                    f"[{i}/{len(pdf_files)}] {pdf_path.name}")
                text = extract_text_from_pdf(pdf_path)

                if text == "__GARBAGE__":
                    self.progress.emit(
                        f"  [skip] image-only or scanned PDF.")
                    skipped += 1
                    continue
                if not text.strip():
                    self.progress.emit(
                        f"  [skip] no extractable text.")
                    skipped += 1
                    continue

                chunks = chunk_text(text)
                del text  # free immediately

                if not chunks:
                    skipped += 1
                    continue

                # Embed and write this PDF's chunks in small batches
                for b in range(0, len(chunks), EMBED_BATCH):
                    batch_chunks = chunks[b:b + EMBED_BATCH]
                    # convert_to_numpy=True, do NOT call .tolist()
                    # ChromaDB accepts numpy arrays directly
                    embeddings = embed_model.encode(
                        batch_chunks,
                        batch_size=EMBED_BATCH,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    ids   = [f"{pdf_path.stem}__{chunk_counter + k}"
                             for k in range(len(batch_chunks))]
                    metas = [{"source": pdf_path.name,
                              "chunk": chunk_counter + k}
                             for k in range(len(batch_chunks))]
                    collection.add(
                        documents=batch_chunks,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metas,
                    )
                    chunk_counter += len(batch_chunks)

                del chunks  # free immediately
                n_papers += 1
                self.progress.emit(
                    f"  {collection.count()} chunks in index so far.")

            total_chunks = collection.count()
            self.progress.emit(
                f"\nDone: {n_papers} papers indexed, "
                f"{skipped} skipped, "
                f"{total_chunks} total chunks.")
            self.finished.emit(n_papers, total_chunks)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Query worker
# ---------------------------------------------------------------------------

class QueryWorker(QThread):
    finished = pyqtSignal(str, list)   # answer, sources
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, query: str, index_dir: str, embed_model):
        super().__init__()
        self.query       = query
        self.index_dir   = index_dir
        self.embed_model = embed_model

    def run(self):
        try:
            self.progress.emit("Searching index...")
            embed_model = self.embed_model
            query_embedding = embed_model.encode(
                [self.query], convert_to_numpy=True)
            client     = chromadb.PersistentClient(path=self.index_dir)
            collection = client.get_collection(name=COLLECTION)

            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=TOP_K,
            )

            docs   = results["documents"][0]
            metas  = results["metadatas"][0]
            sources = list({m["source"] for m in metas})

            if not docs:
                self.finished.emit(
                    "No relevant passages found in the indexed papers.", [])
                return

            # Build context block for Claude
            context_parts = []
            for doc, meta in zip(docs, metas):
                context_parts.append(
                    f"[Source: {meta['source']}]\n{doc}")
            context = "\n\n---\n\n".join(context_parts)

            self.progress.emit(
                f"Retrieved {len(docs)} passages from "
                f"{len(sources)} paper(s). Asking local model...")

            prompt = (
                f"You are a scientific literature assistant helping a researcher "
                f"who works on automated detection of structures in microscopy images, "
                f"specifically glomeruli in kidney histology.\n\n"
                f"Answer the following question using ONLY the provided passages from "
                f"the scientific literature. Be specific and practical. "
                f"Where relevant, highlight phrases, numbers, or methods that could "
                f"directly inform image annotation or model training decisions. "
                f"Cite the source paper name for each key claim.\n\n"
                f"If the passages do not contain enough information to answer "
                f"the question, say so clearly rather than guessing.\n\n"
                f"Question: {self.query}\n\n"
                f"Passages from literature:\n\n{context}"
            )

            # Call Ollama local API — no internet connection required
            payload = _json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read().decode("utf-8"))
            answer = result.get("response", "No response from model.")
            self.finished.emit(answer, sources)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class LiteratureRAGTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Literature-Informed Object Detection  |  DigitalSreeni")
        self.resize(1200, 800)

        self._index_dir   = None
        self._indexed     = False
        self._worker      = None
        self._embed_model = None

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: query + results
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        # Query input
        query_group  = QGroupBox("Ask a question about your literature")
        query_layout = QVBoxLayout(query_group)

        self.query_edit = QTextEdit()
        self.query_edit.setFixedHeight(72)
        self.query_edit.setPlaceholderText(
            "e.g. What morphological phrases are used to describe glomeruli?")
        self.query_edit.setStyleSheet(
            "QTextEdit{font-size:13px;padding:4px;border:1px solid #ccc;"
            "border-radius:3px;}")
        query_layout.addWidget(self.query_edit)

        btn_row = QHBoxLayout()
        self.btn_ask = QPushButton("Ask")
        self.btn_ask.setEnabled(False)
        self.btn_ask.setStyleSheet(
            "QPushButton{background:#2E75B6;color:white;font-weight:bold;"
            "padding:7px 20px;border-radius:4px;font-size:13px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1a5490;}")
        self.btn_ask.clicked.connect(self._ask)
        btn_row.addWidget(self.btn_ask)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setStyleSheet(
            "QPushButton{background:#777;color:white;font-weight:bold;"
            "padding:7px 14px;border-radius:4px;}"
            "QPushButton:hover{background:#555;}")
        self.btn_clear.clicked.connect(self._clear_results)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        query_layout.addLayout(btn_row)
        left_layout.addWidget(query_group)

        # Answer panel
        ans_group  = QGroupBox("Answer")
        ans_layout = QVBoxLayout(ans_group)
        self.answer_edit = QTextEdit()
        self.answer_edit.setReadOnly(True)
        self.answer_edit.setStyleSheet(
            "QTextEdit{background:#fafafa;font-size:12px;"
            "border:1px solid #ddd;border-radius:3px;padding:6px;}")
        ans_layout.addWidget(self.answer_edit)
        left_layout.addWidget(ans_group, 1)

        # Sources
        src_group  = QGroupBox("Source papers")
        src_layout = QVBoxLayout(src_group)
        self.sources_list = QListWidget()
        self.sources_list.setFixedHeight(90)
        self.sources_list.setStyleSheet("font-size:11px;")
        src_layout.addWidget(self.sources_list)
        left_layout.addWidget(src_group)

        # Right: control panel
        right_widget = QWidget()
        right_widget.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidget(right_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._build_right_panel(right_layout)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Build an index from your PDFs, then ask questions.")

    def _build_right_panel(self, layout):

        # About
        about_group  = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        about_text   = QLabel(
            "Ask questions about your scientific literature. "
            "The tool retrieves relevant passages locally and uses "
            "a local Ollama model to synthesise a grounded answer with citations.\n\n"
            "Build the index once. Ask as many questions as you like "
            "without rebuilding.")
        about_text.setWordWrap(True)
        about_text.setStyleSheet("font-size:10px;color:#444;font-style:italic;")
        about_layout.addWidget(about_text)

        # 1. Index
        idx_group  = QGroupBox("1. Build Index")
        idx_layout = QVBoxLayout(idx_group)

        idx_layout.addWidget(QLabel("PDF folder:"))
        pdf_row = QHBoxLayout()
        self.lbl_pdf_dir = QLabel("Not set")
        self.lbl_pdf_dir.setWordWrap(True)
        self.lbl_pdf_dir.setStyleSheet("font-size:10px;color:#555;")
        btn_pdf = QPushButton("Browse")
        btn_pdf.setFixedWidth(60)
        btn_pdf.clicked.connect(self._browse_pdf_dir)
        pdf_row.addWidget(self.lbl_pdf_dir, 1)
        pdf_row.addWidget(btn_pdf)
        idx_layout.addLayout(pdf_row)

        idx_layout.addWidget(QLabel("Index output folder:"))
        idx_row = QHBoxLayout()
        self.lbl_idx_dir = QLabel("Not set")
        self.lbl_idx_dir.setWordWrap(True)
        self.lbl_idx_dir.setStyleSheet("font-size:10px;color:#555;")
        btn_idx = QPushButton("Browse")
        btn_idx.setFixedWidth(60)
        btn_idx.clicked.connect(self._browse_idx_dir)
        idx_row.addWidget(self.lbl_idx_dir, 1)
        idx_row.addWidget(btn_idx)
        idx_layout.addLayout(idx_row)

        self.btn_build = QPushButton("Build Index")
        self.btn_build.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
            "QPushButton:hover:!disabled{background:#1e8449;}")
        self.btn_build.clicked.connect(self._build_index)
        idx_layout.addWidget(self.btn_build)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(10)
        idx_layout.addWidget(self.progress_bar)

        self.lbl_index_status = QLabel("No index built yet.")
        self.lbl_index_status.setWordWrap(True)
        self.lbl_index_status.setStyleSheet("font-size:10px;color:#888;")
        idx_layout.addWidget(self.lbl_index_status)

        # 2. Load existing index
        load_group  = QGroupBox("2. Load Existing Index")
        load_layout = QVBoxLayout(load_group)
        load_note   = QLabel(
            "If you already built an index in a previous session, "
            "point here to reload it without rebuilding.")
        load_note.setWordWrap(True)
        load_note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        load_layout.addWidget(load_note)

        load_row = QHBoxLayout()
        self.lbl_load_idx = QLabel("Not set")
        self.lbl_load_idx.setWordWrap(True)
        self.lbl_load_idx.setStyleSheet("font-size:10px;color:#555;")
        btn_load = QPushButton("Browse")
        btn_load.setFixedWidth(60)
        btn_load.clicked.connect(self._load_existing_index)
        load_row.addWidget(self.lbl_load_idx, 1)
        load_row.addWidget(btn_load)
        load_layout.addLayout(load_row)

        # 3. Build log
        log_group  = QGroupBox("Index log")
        log_layout = QVBoxLayout(log_group)
        self.log   = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(160)
        self.log.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;"
            "font-family:Consolas,monospace;font-size:10px;"
            "border:1px solid #444;}")
        log_layout.addWidget(self.log)

        # 4. Example queries
        ex_group  = QGroupBox("Example queries")
        ex_layout = QVBoxLayout(ex_group)
        ex_note   = QLabel("Click any query to load it into the question box.")
        ex_note.setWordWrap(True)
        ex_note.setStyleSheet("font-size:10px;color:#777;font-style:italic;")
        ex_layout.addWidget(ex_note)
        self.ex_list = QListWidget()
        self.ex_list.setStyleSheet("font-size:11px;")
        for q in EXAMPLE_QUERIES:
            self.ex_list.addItem(QListWidgetItem(q))
        self.ex_list.itemClicked.connect(
            lambda item: self.query_edit.setPlainText(item.text()))
        ex_layout.addWidget(self.ex_list)

        for w in [about_group, idx_group, load_group, log_group, ex_group]:
            layout.addWidget(w)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        view_menu = menubar.addMenu("View")
        font_menu = QMenu("Font Size", self)
        for label, size in [("Small", 9), ("Medium", 11), ("Large", 14)]:
            act = QAction(label, self)
            act.triggered.connect(
                lambda checked, s=size: self._set_font_size(s))
            font_menu.addAction(act)
        view_menu.addMenu(font_menu)

        help_menu  = menubar.addMenu("Help")
        about_act  = QAction("About", self)
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
            "About — Literature-Informed Object Detection",
            "<b>Literature-Informed Object Detection</b><br>"
            "Part of the <i>Applied LLMs for Scientists</i> series<br><br>"
            "<b>Author:</b> Sreenivas Bhattiprolu (DigitalSreeni)<br>"
            "<b>YouTube:</b> "
            "<a href='https://www.youtube.com/@DigitalSreeni'>"
            "youtube.com/@DigitalSreeni</a><br>"
            "<b>GitHub:</b> "
            "<a href='https://github.com/bnsreenu'>"
            "github.com/bnsreenu</a><br><br>"
            "Build a searchable index from scientific PDFs and ask "
            "questions about them. Retrieved passages are synthesised "
            "by Llama running locally via Ollama into grounded answers "
            "with source citations. No internet connection required.<br><br>"
            "Requires Ollama installed with a pulled model: "
            "<code>ollama pull llama3.2</code><br><br>"
            "Pair with the annotation tool to inform detection phrase "
            "selection and threshold decisions directly from the literature."
        )

    # ------------------------------------------------------------------
    # Browse handlers
    # ------------------------------------------------------------------

    def _browse_pdf_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select PDF folder")
        if d:
            self.lbl_pdf_dir.setText(d)

    def _browse_idx_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select folder for index output")
        if d:
            self.lbl_idx_dir.setText(d)
            self._index_dir = d

    def _load_existing_index(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select existing index folder")
        if not d:
            return
        try:
            client = chromadb.PersistentClient(path=d)
            col    = client.get_collection(name=COLLECTION)
            count  = col.count()
            self._index_dir = d
            self._indexed   = True
            self.lbl_load_idx.setText(d)
            self.lbl_index_status.setText(
                f"Loaded: {count} chunks.")
            self.lbl_index_status.setStyleSheet(
                "font-size:10px;color:#27AE60;font-weight:bold;")
            self.btn_ask.setEnabled(True)
            self.status.showMessage(
                f"Index loaded ({count} chunks). Ready to answer questions.")
            self._log(f"Loaded existing index from: {d} ({count} chunks)")
        except Exception as e:
            QMessageBox.warning(
                self, "Load failed",
                f"Could not load index from that folder:\n{e}\n\n"
                "Make sure you select the folder that was used as the "
                "index output when building.")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        pdf_dir = self.lbl_pdf_dir.text().strip()
        idx_dir = self.lbl_idx_dir.text().strip()

        errors = []
        if not pdf_dir or not Path(pdf_dir).exists():
            errors.append("PDF folder not found.")
        if not idx_dir:
            errors.append("Index output folder not set.")
        if errors:
            QMessageBox.warning(self, "Missing inputs", "\n".join(errors))
            return

        self.log.clear()
        self.btn_build.setEnabled(False)
        self.btn_ask.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._indexed = False

        import torch as _torch
        from sentence_transformers import SentenceTransformer as _ST
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._log(f"Loading embedding model on {_device.upper()}...")
        QApplication.processEvents()
        self._embed_model = _ST(EMBED_MODEL, device=_device)
        self._log("Embedding model ready.")
        self._worker = IndexWorker(pdf_dir, idx_dir, self._embed_model)
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_index_done)
        self._worker.error.connect(self._on_index_error)
        self._worker.start()
        self.status.showMessage("Building index...")

    def _on_index_done(self, n_papers, n_chunks):
        self.btn_build.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._indexed   = True
        self._index_dir = self.lbl_idx_dir.text().strip()
        self.lbl_index_status.setText(
            f"Ready: {n_papers} papers, {n_chunks} chunks.")
        self.lbl_index_status.setStyleSheet(
            "font-size:10px;color:#27AE60;font-weight:bold;")
        self.btn_ask.setEnabled(True)
        self.status.showMessage(
            f"Index built ({n_papers} papers, {n_chunks} chunks). "
            "Ready to answer questions.")

    def _on_index_error(self, msg):
        self.btn_build.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log(f"ERROR:\n{msg}")
        self.status.showMessage("Index build failed — see log.")
        QMessageBox.critical(self, "Index error", msg[:500])

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _ask(self):
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        if not self._indexed or not self._index_dir:
            QMessageBox.warning(
                self, "No index",
                "Build or load an index first.")
            return
        self.answer_edit.setPlainText(
            "Searching literature and asking Llama...")
        self.sources_list.clear()
        self.btn_ask.setEnabled(False)
        self.status.showMessage("Searching index and asking Llama locally...")

        if not hasattr(self, "_embed_model") or self._embed_model is None:
            import torch as _torch
            from sentence_transformers import SentenceTransformer as _ST
            _device = "cuda" if _torch.cuda.is_available() else "cpu"
            self._embed_model = _ST(EMBED_MODEL, device=_device)
        self._worker = QueryWorker(
            query=query,
            index_dir=self._index_dir,
            embed_model=self._embed_model,
        )
        self._worker.progress.connect(
            lambda msg: self.status.showMessage(msg))
        self._worker.finished.connect(self._on_query_done)
        self._worker.error.connect(self._on_query_error)
        self._worker.start()

    def _on_query_done(self, answer: str, sources: list):
        self.answer_edit.setPlainText(answer)
        self.sources_list.clear()
        for src in sorted(sources):
            self.sources_list.addItem(QListWidgetItem(src))
        self.btn_ask.setEnabled(True)
        self.status.showMessage(
            f"Answer generated from {len(sources)} source paper(s).")

    def _on_query_error(self, msg: str):
        self.answer_edit.setPlainText(f"Error:\n{msg}")
        self.btn_ask.setEnabled(True)
        self.status.showMessage("Query failed — see answer panel.")

    def _clear_results(self):
        self.answer_edit.clear()
        self.sources_list.clear()
        self.query_edit.clear()
        self.status.showMessage("Ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LiteratureRAGTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
