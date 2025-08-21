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

    def set_match(self, match: bool):
        # Update UI checkbox and internal label without emitting change twice
        self.chkMatch.blockSignals(True)
        self.chkMatch.setChecked(bool(match))
        self.chkMatch.blockSignals(False)
        self.label = bool(match)

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
    def __init__(self, h5_path="embeddings.h5", annotator_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Matcher")
        self.resize(1000, 860)

        self.h5_path = h5_path
        self.spectrograms, self.embeddings = self._load_h5(self.h5_path)
        self.N = self.embeddings.shape[0]
        self.dist = cosine_dist_matrix(self.embeddings)

        self.results: List[QueryResult] = []
        self.results_path = "results.json"

        # In-memory saved labels: query_index -> (proposal_index -> (timestamp, match))
        self._saved_match_state: dict[int, dict[int, Tuple[float, bool]]] = {}

        self.num_nn_idx = 20
        self.num_quantile_idx = 20

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

        # --- Make view more compact ---
        view = QtWidgets.QListView(self.queryCombo)
        view.setUniformItemSizes(True)
        self.queryCombo.setView(view)

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
        # Load any existing annotations from JSON files in CWD
        try:
            self._load_annotations_from_dir(os.getcwd())
        except Exception as e:
            # Non-fatal: continue without preloaded annotations
            print(f"Warning: failed to load annotations: {e}")

        self.set_query(0)
        # Reflect annotations in query dropdown labels
        self._refresh_all_query_item_labels()

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
            self._refresh_all_query_item_labels()

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
        nn_idx = [int(i) for i in order[:self.num_nn_idx]]

        # 20 quantiles (spread)
        quant_idx = quantile_indices_from_sorted(order, self.num_quantile_idx)

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
        # Apply any saved matches for this query
        self._apply_saved_matches_for_query(self.current_query)

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
            # Update Next button enabled and refresh in-memory state + labels
            row.labeledChanged.connect(self._on_any_row_changed)
            self.proposalsLayout.addWidget(row)

        self.proposalsLayout.addStretch(1)

    def _apply_saved_matches_for_query(self, q: int):
        state = self._saved_match_state.get(int(q), {})
        if not state:
            return
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if isinstance(w, ProposalRow):
                if w.idx in state:
                    _ts, match = state[w.idx]
                    w.set_match(match)

    def _noop_update_next(self):
        self.btnNext.setEnabled(True)

    def _on_any_row_changed(self):
        # Enable Next and cache current labels to state (with timestamp)
        self._noop_update_next()
        self._cache_current_query_labels()
        # Update current query label in dropdown
        if self.current_query is not None:
            self._update_query_item_label(int(self.current_query))

    def on_query_changed(self, idx):
        # Cache current labels to in-memory state before switching
        if self.current_query is not None:
            self._cache_current_query_labels()
        self.set_query(idx)

    def pick_random_query(self):
        # Cache current labels before switching
        if self.current_query is not None:
            self._cache_current_query_labels()
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
                    match=bool(w.chkMatch.isChecked())
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
        # Update in-memory saved state with recency
        self._ingest_query_result_into_state(rec)
        # Update dropdown mark for this query
        self._update_query_item_label(rec.query_index)
        # autosave after each query
        self._write_results(self.results_path)
        nxt = (self.current_query + 1) % self.N
        self.set_query(nxt)

    def save_results(self):
        # Ensure any pending UI events are processed, then cache current state
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        self._cache_current_query_labels()
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
        # Serialize from authoritative in-memory state so unsaved UI edits are included
        results_list = self._state_to_results_list()
        payload = {
            "h5_path": os.path.abspath(self.h5_path),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_items": int(self.N),
            "annotator": self.annotator_name(),
            "results": results_list,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _state_to_results_list(self) -> List[dict]:
        out: List[dict] = []
        for q in sorted(self._saved_match_state.keys()):
            state = self._saved_match_state.get(q, {})
            if not state:
                continue
            # Determine a representative timestamp: max over proposals
            ts_q = max((ts for ts, _ in state.values()), default=time.time())
            # Build proposals list with distances if available
            props: List[LabeledProposal] = []
            # Sort by index for stable output
            for idx in sorted(state.keys()):
                ts, match = state[idx]
                dist = 0.0
                try:
                    dist = float(self.dist[q, idx])
                except Exception:
                    pass
                props.append(LabeledProposal(index=idx, distance=dist, bucket="", match=bool(match)))
            out.append({
                "query_index": int(q),
                "timestamp": float(ts_q),
                "annotator": self.annotator_name(),
                "proposals": [asdict(p) for p in props],
            })
        return out

    # -------------------- Annotation loading/merging --------------------

    def _ingest_query_result_into_state(self, rec: QueryResult):
        q = int(rec.query_index)
        ts = float(rec.timestamp)
        if q not in self._saved_match_state:
            self._saved_match_state[q] = {}
        for p in rec.proposals:
            idx = int(p.index)
            match = bool(p.match)
            prev = self._saved_match_state[q].get(idx)
            if prev is None or ts >= prev[0]:
                self._saved_match_state[q][idx] = (ts, match)

    def _load_annotations_from_dir(self, directory: str):
        # Load all *.json files that look like results produced by this tool
        try:
            files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
        except Exception as e:
            print(f"Warning: cannot list directory {directory}: {e}")
            return

        for fname in files:
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue  # skip non-JSON or unreadable

            if not isinstance(data, dict) or 'results' not in data:
                continue

            file_mtime = 0.0
            try:
                file_mtime = os.path.getmtime(fpath)
            except Exception:
                pass

            results = data.get('results') or []
            if not isinstance(results, list):
                continue

            for rec in results:
                try:
                    q = int(rec.get('query_index'))
                except Exception:
                    continue
                ts = rec.get('timestamp')
                try:
                    ts = float(ts) if ts is not None else float(file_mtime)
                except Exception:
                    ts = float(file_mtime)
                annot = rec.get('proposals') or []
                if not isinstance(annot, list):
                    continue
                # Build temporary QueryResult-like object to reuse ingest logic
                props: List[LabeledProposal] = []
                for p in annot:
                    try:
                        idx = int(p.get('index'))
                    except Exception:
                        continue
                    match = bool(p.get('match', False))
                    bucket = str(p.get('bucket', ''))
                    dist = float(p.get('distance', 0.0)) if 'distance' in p else 0.0
                    props.append(LabeledProposal(index=idx, distance=dist, bucket=bucket, match=match))
                qr = QueryResult(query_index=q, timestamp=ts, annotator=str(rec.get('annotator', '')), proposals=props)
                # Guard: only ingest indices within current dataset bounds
                if 0 <= q < self.N:
                    self._ingest_query_result_into_state(qr)
        # After bulk load, refresh dropdown labels
        self._refresh_all_query_item_labels()

    def _cache_current_query_labels(self):
        # Take current UI labels and store in in-memory state with current timestamp
        try:
            curr = int(self.current_query)
        except Exception:
            return
        ts = time.time()
        # Only ingest changes: new True labels, or toggles relative to saved state
        changed: List[LabeledProposal] = []
        current_state = self._saved_match_state.get(curr, {})
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if not isinstance(w, ProposalRow):
                continue
            prev = current_state.get(w.idx)
            prev_match = prev[1] if prev is not None else None
            curr_match = bool(w.chkMatch.isChecked())
            if prev_match is None:
                # New: only record True to avoid overwriting with False by default
                if curr_match:
                    changed.append(LabeledProposal(index=w.idx, distance=w.distance, bucket=w.bucket, match=True))
            else:
                if curr_match != bool(prev_match):
                    changed.append(LabeledProposal(index=w.idx, distance=w.distance, bucket=w.bucket, match=curr_match))

        if changed:
            rec = QueryResult(query_index=curr, timestamp=ts, annotator=self.annotator_name(), proposals=changed)
            self._ingest_query_result_into_state(rec)
        # Ensure dropdown reflects current query state
        self._update_query_item_label(curr)

    # -------------------- Query label helpers --------------------

    def _query_has_any_match(self, q: int) -> bool:
        state = self._saved_match_state.get(int(q), {})
        for _idx, (_ts, match) in state.items():
            if match:
                return True
        return False

    def _format_query_item_text(self, q: int) -> str:
        mark = "✓ " if self._query_has_any_match(q) else ""
        return f"{mark}{q}"

    def _update_query_item_label(self, q: int):
        if 0 <= q < self.queryCombo.count():
            self.queryCombo.setItemText(q, self._format_query_item_text(q))

    def _refresh_all_query_item_labels(self):
        for q in range(self.queryCombo.count()):
            self._update_query_item_label(q)

# ---------------------------- main --------------------------------------------

def main():
    # Parse known args first so Qt doesn't choke on them
    parser = argparse.ArgumentParser(description="Spectrogram Matcher GUI")
    parser.add_argument("--annotator", dest="annotator", type=str, default="",
                        help="Annotator name (prefills the GUI and is saved to JSON)")
    parser.add_argument("--h5", dest="h5_path", type=str, default="embeddings.h5",
                        help="Path to embeddings HDF5 (default: embeddings.h5)")
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
