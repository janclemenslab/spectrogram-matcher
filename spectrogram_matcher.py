# spectrogram_matcher.py
# Requirements: PySide6, numpy, h5py, matplotlib
import argparse
import json, os, sys, time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import h5py

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------- Utilities ---------------------------------------

def l2_normalize(x: np.ndarray, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def cosine_dist_matrix(emb: np.ndarray) -> np.ndarray:
    # cosine distance = 1 - cosine similarity
    E = l2_normalize(emb, axis=1)
    S = E @ E.T
    S = np.clip(S, -1.0, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

def quantile_indices_from_sorted(sorted_indices: np.ndarray, count: int) -> List[int]:
    """Evenly spaced ranks from a sorted ascending index list (self removed upstream)."""
    n = len(sorted_indices)
    if n == 0:
        return []
    picks = []
    for k in range(1, count + 1):
        pos = k / (count + 1)
        idx = int(round(pos * (n - 1)))
        picks.append(int(sorted_indices[idx]))
    # deduplicate preserving order
    seen, out = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:count]

# ---------------------------- Matplotlib widget -------------------------------

class MplImage(QtWidgets.QFrame):
    def __init__(self, parent=None, height_px: int = 90):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.fig = Figure(figsize=(8, height_px/100), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        # Tight: no padding
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_position([0, 0, 1, 1])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas)
        self.setMinimumHeight(height_px)
        self.setMaximumHeight(height_px)

    def show_spectrogram(self, S: np.ndarray):
        self.ax.clear()
        self.ax.axis("off")
        # accept (H,W) or (H,W,C); select first channel if 3D
        if S.ndim == 2:
            img = S
        elif S.ndim == 3:
            img = S[..., 0]
        else:
            img = np.squeeze(S)

        # transpose, then drop first 35 rows of height if available
        img = img.T
        if img.shape[0] > 35:
            img = img[35:, :]

        # robust scaling + turbo colormap
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(img)), float(np.max(img))

        self.ax.imshow(img, aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        self.ax.invert_yaxis()
        self.canvas.draw_idle()

# ---------------------------- Proposal row ------------------------------------

class ProposalRow(QtWidgets.QFrame):
    labeledChanged = QtCore.Signal()

    def __init__(self, idx: int, spectrogram: np.ndarray, distance: float, bucket: str,
                 img_height: int, sidepanel_width: int, parent=None):
        super().__init__(parent)
        self.idx = int(idx)
        self.bucket = bucket  # "nn" or "quantile"
        self.distance = float(distance)
        # Default: unchecked -> no match
        self.label: Optional[bool] = False

        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setObjectName("proposalRow")
        self.setStyleSheet("QCheckBox { font-size: 12px; }")

        img = MplImage(self, height_px=img_height)
        img.show_spectrogram(spectrogram)

        # Side panel with only the Match checkbox (metadata removed)
        side = QtWidgets.QWidget()
        side.setFixedWidth(sidepanel_width)
        sideLay = QtWidgets.QVBoxLayout(side)
        sideLay.setContentsMargins(0,0,0,0)
        sideLay.setSpacing(4)

        self.chkMatch = QtWidgets.QCheckBox("Match")
        self.chkMatch.setChecked(False)
        self.chkMatch.toggled.connect(self._on_toggle)
        sideLay.addWidget(self.chkMatch, 0, Qt.AlignTop)
        sideLay.addStretch(1)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2,2,2,2)
        row.setSpacing(6)
        row.addWidget(img, 1)
        row.addWidget(side, 0)

    def _on_toggle(self, checked: bool):
        self.label = bool(checked)
        self.labeledChanged.emit()

# ---------------------------- Main Window -------------------------------------

@dataclass
class LabeledProposal:
    index: int
    distance: float
    bucket: str
    match: bool

@dataclass
class QueryResult:
    query_index: int
    timestamp: float
    annotator: str
    proposals: List[LabeledProposal]  # order: 20 nn then 20 quantiles

class SpectrogramMatcher(QtWidgets.QMainWindow):
    def __init__(self, h5_path="res/embeddings.h5", annotator_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Matcher")
        self.resize(1000, 860)

        self.h5_path = h5_path
        self.spectrograms, self.embeddings = self._load_h5(self.h5_path)
        self.N = self.embeddings.shape[0]
        self.dist = cosine_dist_matrix(self.embeddings)

        self.results: List[QueryResult] = []
        self.results_path = "results.json"

        # ----- Layout constants -----
        IMG_HEIGHT = 90                    # half the previous height
        SIDEPANEL_WIDTH = 120              # fixed width for checkbox column

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8,8,8,8)
        root.setSpacing(6)

        # Top bar (compact)  —— now with Annotator box
        topBar = QtWidgets.QHBoxLayout()
        topBar.setContentsMargins(0,0,0,0)
        topBar.setSpacing(6)

        topBar.addWidget(QtWidgets.QLabel("Annotator:"))
        self.annotatorEdit = QtWidgets.QLineEdit()
        self.annotatorEdit.setPlaceholderText("Name")
        if annotator_name:
            self.annotatorEdit.setText(annotator_name)
        self.annotatorEdit.setMinimumWidth(160)
        topBar.addWidget(self.annotatorEdit)

        topBar.addSpacing(10)
        topBar.addWidget(QtWidgets.QLabel("Query:"))
        self.queryCombo = QtWidgets.QComboBox()
        self.queryCombo.addItems([f"{i}" for i in range(self.N)])
        self.queryCombo.currentIndexChanged.connect(self.on_query_changed)

        self.btnRandom = QtWidgets.QPushButton("Random")
        self.btnRandom.clicked.connect(self.pick_random_query)

        self.btnSave = QtWidgets.QPushButton("Save 💾")
        self.btnSave.clicked.connect(self.save_results)

        self.btnNext = QtWidgets.QPushButton("Next ▶")
        self.btnNext.clicked.connect(self.next_query)

        topBar.addWidget(self.queryCombo, 1)
        topBar.addWidget(self.btnRandom)
        topBar.addStretch(1)
        topBar.addWidget(self.btnSave)
        topBar.addWidget(self.btnNext)

        # Query row: image + empty sidepanel placeholder to match proposal width
        queryRowW = QtWidgets.QWidget()
        queryRow = QtWidgets.QHBoxLayout(queryRowW)
        queryRow.setContentsMargins(2,2,2,2)
        queryRow.setSpacing(6)

        self.queryImg = MplImage(height_px=IMG_HEIGHT)
        queryRow.addWidget(self.queryImg, 1)

        querySidePlaceholder = QtWidgets.QWidget()
        querySidePlaceholder.setFixedWidth(SIDEPANEL_WIDTH)
        queryRow.addWidget(querySidePlaceholder)

        # Scrollable proposals area
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollInner = QtWidgets.QWidget()
        self.scroll.setWidget(self.scrollInner)
        self.proposalsLayout = QtWidgets.QVBoxLayout(self.scrollInner)
        self.proposalsLayout.setContentsMargins(0,0,0,0)
        self.proposalsLayout.setSpacing(6)

        # assemble
        root.addLayout(topBar)
        root.addWidget(queryRowW)      # query
        root.addWidget(self.scroll, 1) # proposals

        self.setCentralWidget(central)

        # Menu + shortcuts
        fileMenu = self.menuBar().addMenu("&File")
        actOpen = fileMenu.addAction("Open HDF5…")
        actOpen.triggered.connect(self.open_h5_dialog)
        actSave = fileMenu.addAction("Save Results")
        actSave.triggered.connect(self.save_results)
        fileMenu.addSeparator()
        actQuit = fileMenu.addAction("Quit")
        actQuit.triggered.connect(self.close)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save_results)
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=self.next_query)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.pick_random_query)

        # initialize
        self.current_query: Optional[int] = None
        self.IMG_HEIGHT = IMG_HEIGHT
        self.SIDEPANEL_WIDTH = SIDEPANEL_WIDTH
        self.set_query(0)

    # -------------------- Data loading --------------------

    def _load_h5(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(self, "File not found", f"Cannot find HDF5 at: {path}")
            sys.exit(1)
        with h5py.File(path, "r") as f:
            if "spectrogram" not in f or "embedding" not in f:
                QtWidgets.QMessageBox.critical(self, "Invalid file",
                    "HDF5 must contain datasets 'spectrogram' and 'embedding'.")
                sys.exit(1)
            S = f["spectrogram"][:]
            E = f["embedding"][:]
        if len(S) != len(E):
            QtWidgets.QMessageBox.critical(self, "Length mismatch",
                f"'spectrogram' (len={len(S)}) and 'embedding' (len={len(E)}) differ.")
            sys.exit(1)
        return S, E

    def open_h5_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open embeddings.h5", "", "HDF5 files (*.h5 *.hdf5)")
        if fn:
            self.h5_path = fn
            self.spectrograms, self.embeddings = self._load_h5(self.h5_path)
            self.N = self.embeddings.shape[0]
            self.dist = cosine_dist_matrix(self.embeddings)
            self.queryCombo.clear()
            self.queryCombo.addItems([f"{i}" for i in range(self.N)])
            self.set_query(0)

    # -------------------- Query / proposals --------------------

    def annotator_name(self) -> str:
        return self.annotatorEdit.text().strip()

    def set_query(self, q: int):
        self.current_query = int(q)
        self.queryCombo.blockSignals(True)
        self.queryCombo.setCurrentIndex(self.current_query)
        self.queryCombo.blockSignals(False)

        self.queryImg.show_spectrogram(self.spectrograms[self.current_query])

        # compute proposals
        d = self.dist[self.current_query].copy()
        order = np.argsort(d)
        order = order[order != self.current_query]  # drop self

        # 20 nearest neighbours
        nn_idx = [int(i) for i in order[:20]]

        # 20 quantiles (spread)
        quant_idx = quantile_indices_from_sorted(order, 20)

        # ensure uniqueness between buckets
        nn_set = set(nn_idx)
        q_unique = [int(i) for i in quant_idx if i not in nn_set]
        if len(q_unique) < 20:
            for i in order:
                ii = int(i)
                if ii not in nn_set and ii not in q_unique:
                    q_unique.append(ii)
                    if len(q_unique) == 20:
                        break

        # populate proposals (single column, no headers)
        self._populate_proposals(nn_idx, q_unique, d)

    def _populate_proposals(self, nn_idx: List[int], q_idx: List[int], dvec: np.ndarray):
        # clear
        while self.proposalsLayout.count():
            item = self.proposalsLayout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # add rows (20 nn then 20 quantiles)
        for i in nn_idx + q_idx:
            row = ProposalRow(idx=i,
                              spectrogram=self.spectrograms[i],
                              distance=float(dvec[i]),
                              bucket=("nn" if i in nn_idx else "quantile"),
                              img_height=self.IMG_HEIGHT,
                              sidepanel_width=self.SIDEPANEL_WIDTH)
            row.labeledChanged.connect(self._noop_update_next)  # Next stays enabled
            self.proposalsLayout.addWidget(row)

        self.proposalsLayout.addStretch(1)

    def _noop_update_next(self):
        self.btnNext.setEnabled(True)

    def on_query_changed(self, idx):
        self.set_query(idx)

    def pick_random_query(self):
        q = int(np.random.randint(0, self.N))
        self.set_query(q)

    # -------------------- Flow / persistence --------------------

    def _collect_labels(self) -> List[LabeledProposal]:
        labels: List[LabeledProposal] = []
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if isinstance(w, ProposalRow):
                labels.append(LabeledProposal(
                    index=w.idx,
                    distance=w.distance,
                    bucket=w.bucket,
                    match=bool(w.label)
                ))
        return labels

    def next_query(self):
        rec = QueryResult(
            query_index=int(self.current_query),
            timestamp=time.time(),
            annotator=self.annotator_name(),
            proposals=self._collect_labels()
        )
        self.results.append(rec)
        # autosave after each query
        self._write_results(self.results_path)
        nxt = (self.current_query + 1) % self.N
        self.set_query(nxt)

    def save_results(self):
        self.results_path = self._make_results_filename()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save results JSON", self.results_path, "JSON (*.json)")
        if path:
            self.results_path = path
            self._write_results(self.results_path)

    def _make_results_filename(self) -> str:
        # sanitize annotator name
        import re
        from datetime import datetime
        annot = self.annotator_name() or "anon"
        annot = re.sub(r"[^A-Za-z0-9_-]", "_", annot)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"results_{annot}_{ts}.json"

    def _write_results(self, path: str):
        payload = {
            "h5_path": os.path.abspath(self.h5_path),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_items": int(self.N),
            "annotator": self.annotator_name(),
            "results": [
                {
                    "query_index": r.query_index,
                    "timestamp": r.timestamp,
                    "annotator": r.annotator,
                    "proposals": [asdict(lp) for lp in r.proposals],
                } for r in self.results
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------------------------- main --------------------------------------------

def main():
    # Parse known args first so Qt doesn't choke on them
    parser = argparse.ArgumentParser(description="Spectrogram Matcher GUI")
    parser.add_argument("--annotator", dest="annotator", type=str, default="",
                        help="Annotator name (prefills the GUI and is saved to JSON)")
    parser.add_argument("--h5", dest="h5_path", type=str, default="res/embeddings.h5",
                        help="Path to embeddings HDF5 (default: res/embeddings.h5)")
    args, unknown = parser.parse_known_args()
    # Keep unknown args for Qt
    sys.argv = [sys.argv[0]] + unknown

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SpectrogramMatcher(h5_path=args.h5_path, annotator_name=args.annotator)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()