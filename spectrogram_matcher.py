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


def quantile_indices_from_sorted(
    sorted_indices: np.ndarray, count: int, bias_exponent: float = 0.1
) -> List[int]:
    """
    Return `count` indices sampled from an ascending rank list with a bias
    towards smaller ranks (closer neighbours).

    - `sorted_indices` is expected to be ascending by distance (self removed).
    - `bias_exponent` < 1.0 over-represents close neighbours (log-like spacing);
      `bias_exponent` == 1.0 reduces to even spacing.
    """
    n = len(sorted_indices)
    if n == 0 or count <= 0:
        return []

    # Sample at u_k in (0,1) and compress using u^alpha (alpha<1 concave)
    alpha = float(bias_exponent)
    alpha = 1.0 if not np.isfinite(alpha) or alpha <= 0 else alpha
    picks: List[int] = []
    for k in range(1, count + 1):
        u = k / (count + 1)
        u_biased = u**alpha
        idx = int(round(u_biased * (n - 1)))
        picks.append(int(sorted_indices[idx]))

    # Deduplicate preserving order
    seen, out = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:count]


def _time_profile_from_spectrogram(S: np.ndarray, drop_low_freq_bins: int = 35) -> np.ndarray:
    """
    Generate a 1D time profile from a 2D spectrogram S with shape (time, freq).
    Optionally drop the lowest `drop_low_freq_bins` frequency bins which are often
    dominated by noise. Returns a zero-mean profile.
    """
    if S.ndim == 3:
        S = S[..., 0]
    S2 = np.asarray(S)
    # Assume (time, freq); clamp if not enough bins
    if S2.ndim != 2:
        S2 = np.squeeze(S2)
        if S2.ndim != 2:
            S2 = np.atleast_2d(S2)
    t, f = S2.shape[0], S2.shape[1]
    cut = int(drop_low_freq_bins)
    if cut > 0 and f > cut:
        S2 = S2[:, cut:]
    # Average across frequencies for robustness
    prof = np.mean(S2, axis=1)
    # Zero-mean, unit variance (guarding small variance)
    prof = prof - np.mean(prof)
    std = float(np.std(prof))
    if std > 1e-8:
        prof = prof / std
    return prof.astype(np.float32)


def _best_time_shift(a: np.ndarray, b: np.ndarray, max_shift: Optional[int] = None) -> int:
    """
    Compute the integer time shift s that best aligns b to a, using normalized
    cross-correlation of 1D profiles a,b (same length preferred, but not required).
    Returns s where positive values shift b forward in time (to the right).
    If max_shift is provided, restrict search to |s| <= max_shift.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na, nb = a.shape[0], b.shape[0]
    if na == 0 or nb == 0:
        return 0
    # Use full correlation and pick the best lag
    corr = np.correlate(a, b, mode="full")
    # Optional windowing of allowable shifts
    if max_shift is not None:
        max_shift = int(abs(max_shift))
        center = nb - 1
        start = max(0, center - max_shift)
        end = min(corr.shape[0], center + max_shift + 1)
        window = corr[start:end]
        best_local = int(np.argmax(window))
        best = start + best_local
    else:
        best = int(np.argmax(corr))
    # Convert index in 'full' correlation to shift s for b
    # In numpy's definition, index == nb-1 corresponds to zero shift.
    s = best - (nb - 1)
    return int(s)


def _shift_along_time(S: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift spectrogram S along time axis (assumed axis 0 for shape (time, freq)).
    Positive shift moves content forward (to the right when plotted), padding
    with the minimum value. Keeps the original shape.
    """
    if shift == 0:
        return S
    X = np.asarray(S)
    if X.ndim == 3:
        X = X[..., 0]
    if X.ndim != 2:
        X = np.squeeze(X)
        if X.ndim != 2:
            X = np.atleast_2d(X)
    t, f = X.shape
    out = np.empty_like(X)
    pad = float(np.min(X)) if np.all(np.isfinite(X)) else 0.0
    if shift > 0:
        # pad front, move data later
        if shift >= t:
            out[:] = pad
        else:
            out[:shift, :] = pad
            out[shift:, :] = X[: t - shift, :]
    else:
        s = -int(shift)
        if s >= t:
            out[:] = pad
        else:
            out[: t - s, :] = X[s:, :]
            out[t - s :, :] = pad
    return out


# ---------------------------- Matplotlib widget -------------------------------


class MplImage(QtWidgets.QFrame):
    def __init__(self, parent=None, height_px: int = 90):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.fig = Figure(figsize=(8, height_px / 100), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        # Tight: no padding
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_position([0, 0, 1, 1])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
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

        self.ax.imshow(img, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
        self.ax.invert_yaxis()
        self.canvas.draw_idle()


# ---------------------------- Proposal row ------------------------------------


class ProposalRow(QtWidgets.QFrame):
    labeledChanged = QtCore.Signal()

    def __init__(
        self,
        idx: int,
        spectrogram: np.ndarray,
        distance: float,
        bucket: str,
        img_height: int,
        sidepanel_width: int,
        parent=None,
    ):
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
        # Keep a reference for styling when matched
        self.img = img

        # Side panel with only the Match checkbox (metadata removed)
        side = QtWidgets.QWidget()
        side.setFixedWidth(sidepanel_width)
        sideLay = QtWidgets.QVBoxLayout(side)
        sideLay.setContentsMargins(0, 0, 0, 0)
        sideLay.setSpacing(4)

        self.chkMatch = QtWidgets.QCheckBox("Match")
        self.chkMatch.setChecked(False)
        self.chkMatch.toggled.connect(self._on_toggle)
        sideLay.addWidget(self.chkMatch, 0, Qt.AlignTop)
        sideLay.addStretch(1)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(6)
        row.addWidget(img, 1)
        row.addWidget(side, 0)

    def _on_toggle(self, checked: bool):
        self.label = bool(checked)
        # Update visual emphasis based on match state
        try:
            if bool(checked):
                # Add a salient red frame around the spectrogram
                self.img.setStyleSheet("border: 3px solid red;")
            else:
                # Remove frame when unchecked
                self.img.setStyleSheet("")
        except Exception:
            pass
        self.labeledChanged.emit()

    def set_match(self, match: bool):
        # Update UI checkbox and internal label without emitting change twice
        self.chkMatch.blockSignals(True)
        self.chkMatch.setChecked(bool(match))
        self.chkMatch.blockSignals(False)
        self.label = bool(match)
        # Ensure visual state reflects programmatic changes
        try:
            if bool(match):
                self.img.setStyleSheet("border: 3px solid red;")
            else:
                self.img.setStyleSheet("")
        except Exception:
            pass


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
    def __init__(
        self,
        h5_path: str = "embeddings.h5",
        annotator_name: str = "",
        quantile_bias_exponent: float = 0.1,
        quantile_max_distance: Optional[float] = None,
        all_nearest: bool = False,
        parent=None,
    ):
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
        # In-memory proposals per query (preserve order as in file/UI)
        self._query_proposals: dict[int, List[LabeledProposal]] = {}
        # Timestamp of proposals list per query (latest ingested)
        self._proposal_timestamp: dict[int, float] = {}

        self.num_nn_idx = 20
        self.num_quantile_idx = 20

        # Sampling control
        self.bias_exponent: float = float(quantile_bias_exponent)
        # Optional upper bound for distances when sampling beyond top-k
        self.max_quantile_distance: Optional[float] = (
            float(quantile_max_distance)
            if quantile_max_distance is not None and np.isfinite(quantile_max_distance)
            else None
        )
        # Mode: when True, select top 40 nearest neighbours (no quantile sampling)
        self.all_nearest: bool = bool(all_nearest)

        # ----- Layout constants -----
        IMG_HEIGHT = 90  # half the previous height
        SIDEPANEL_WIDTH = 120  # fixed width for checkbox column

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Top bar (compact)  —— now with Annotator box
        topBar = QtWidgets.QHBoxLayout()
        topBar.setContentsMargins(0, 0, 0, 0)
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

        self.btnPrev = QtWidgets.QPushButton("◀ Previous")
        self.btnPrev.clicked.connect(self.prev_query)

        self.btnNext = QtWidgets.QPushButton("Next ▶")
        self.btnNext.clicked.connect(self.next_query)

        # Alignment toggle: when enabled, proposals are time-aligned to the query
        self.alignCheck = QtWidgets.QCheckBox("Align x-corr")
        self.alignCheck.setToolTip(
            "Temporally align proposals to the query using spectrogram cross-correlation"
        )
        self.alignCheck.setChecked(False)
        self.alignCheck.toggled.connect(self._on_align_toggled)

        topBar.addWidget(self.queryCombo, 1)
        topBar.addWidget(self.btnRandom)
        topBar.addStretch(1)
        topBar.addWidget(self.alignCheck)
        topBar.addWidget(self.btnSave)
        topBar.addWidget(self.btnPrev)
        topBar.addWidget(self.btnNext)

        # Query row: image + sidepanel with query-level controls
        queryRowW = QtWidgets.QWidget()
        queryRow = QtWidgets.QHBoxLayout(queryRowW)
        queryRow.setContentsMargins(2, 2, 2, 2)
        queryRow.setSpacing(6)

        self.queryImg = MplImage(height_px=IMG_HEIGHT)
        queryRow.addWidget(self.queryImg, 1)

        # Right-side panel for query-level checkbox (No matches)
        querySide = QtWidgets.QWidget()
        querySide.setFixedWidth(SIDEPANEL_WIDTH)
        qSideLay = QtWidgets.QVBoxLayout(querySide)
        qSideLay.setContentsMargins(0, 0, 0, 0)
        qSideLay.setSpacing(4)

        self.noMatchesCheck = QtWidgets.QCheckBox("No matches")
        self.noMatchesCheck.setChecked(False)
        self.noMatchesCheck.setToolTip("Mark this query as having no matching proposals")
        self.noMatchesCheck.toggled.connect(self._on_no_matches_toggled)
        qSideLay.addWidget(self.noMatchesCheck, 0, Qt.AlignTop)

        self.ignoreCheck = QtWidgets.QCheckBox("Ignore")
        self.ignoreCheck.setChecked(False)
        self.ignoreCheck.setToolTip("Ignore this query (skip during review)")
        self.ignoreCheck.toggled.connect(self._on_ignore_toggled)
        qSideLay.addWidget(self.ignoreCheck, 0, Qt.AlignTop)
        qSideLay.addStretch(1)

        queryRow.addWidget(querySide)

        # Scrollable proposals area
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollInner = QtWidgets.QWidget()
        self.scroll.setWidget(self.scrollInner)
        self.proposalsLayout = QtWidgets.QVBoxLayout(self.scrollInner)
        self.proposalsLayout.setContentsMargins(0, 0, 0, 0)
        self.proposalsLayout.setSpacing(6)

        # assemble
        root.addLayout(topBar)
        root.addWidget(queryRowW)  # query
        root.addWidget(self.scroll, 1)  # proposals

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
        QtGui.QShortcut(
            QtGui.QKeySequence("A"),
            self,
            activated=lambda: self.alignCheck.setChecked(not self.alignCheck.isChecked()),
        )

        # initialize
        self.current_query: Optional[int] = None
        self.IMG_HEIGHT = IMG_HEIGHT
        self.SIDEPANEL_WIDTH = SIDEPANEL_WIDTH
        # Per-query flag for explicit "no matches"
        self._no_match_queries: dict[int, Tuple[float, bool]] = {}
        # Per-query flag for ignoring a query
        self._ignored_queries: dict[int, Tuple[float, bool]] = {}
        # Load any existing annotations from JSON files in CWD
        try:
            self._load_annotations_from_dir(os.getcwd())
        except Exception as e:
            # Non-fatal: continue without preloaded annotations
            print(f"Warning: failed to load annotations: {e}")

        self.set_query(0)
        # Reflect annotations in query dropdown labels
        self._refresh_all_query_item_labels()
        # Internal flag mirrors checkbox
        self.align_by_xcorr: bool = False

    # -------------------- Data loading --------------------

    def _load_h5(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(
                self, "File not found", f"Cannot find HDF5 at: {path}"
            )
            sys.exit(1)
        with h5py.File(path, "r") as f:
            if "spectrogram" not in f or "embedding" not in f:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid file",
                    "HDF5 must contain datasets 'spectrogram' and 'embedding'.",
                )
                sys.exit(1)
            S = f["spectrogram"][:]
            E = f["embedding"][:]
        if len(S) != len(E):
            QtWidgets.QMessageBox.critical(
                self,
                "Length mismatch",
                f"'spectrogram' (len={len(S)}) and 'embedding' (len={len(E)}) differ.",
            )
            sys.exit(1)
        return S, E

    def open_h5_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open embeddings.h5", "", "HDF5 files (*.h5 *.hdf5)"
        )
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

        # Build proposals for this query (use saved list if available)
        proposals = self._build_proposals_for_query(self.current_query)
        # populate proposals according to provided order
        d = self.dist[self.current_query].copy()
        self._populate_proposals_from_items(proposals, d)
        # Ensure any saved matches reflected (redundant if already set via items)
        self._apply_saved_matches_for_query(self.current_query)
        # Restore query-level no-matches state
        nm = bool(self._no_match_queries.get(self.current_query, (0.0, False))[1])
        ig = bool(self._ignored_queries.get(self.current_query, (0.0, False))[1])
        self.noMatchesCheck.blockSignals(True)
        self.noMatchesCheck.setChecked(nm)
        self.noMatchesCheck.blockSignals(False)
        self.ignoreCheck.blockSignals(True)
        self.ignoreCheck.setChecked(ig)
        self.ignoreCheck.blockSignals(False)
        # Apply combined UI disabling rule
        self._apply_no_matches_ui_state(nm or ig)

    def _populate_proposals(
        self, nn_idx: List[int], q_idx: List[int], dvec: np.ndarray
    ):
        # clear
        while self.proposalsLayout.count():
            item = self.proposalsLayout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # add rows (20 nn then 20 quantiles)
        for i in nn_idx + q_idx:
            spec = self._get_display_spectrogram_for_index(i)
            row = ProposalRow(
                idx=i,
                spectrogram=spec,
                distance=float(dvec[i]),
                bucket=("nn" if i in nn_idx else "quantile"),
                img_height=self.IMG_HEIGHT,
                sidepanel_width=self.SIDEPANEL_WIDTH,
            )
            # Update Next button enabled and refresh in-memory state + labels
            row.labeledChanged.connect(self._on_any_row_changed)
            self.proposalsLayout.addWidget(row)

        self.proposalsLayout.addStretch(1)

    def _populate_proposals_from_items(
        self, items: List[LabeledProposal], dvec: np.ndarray
    ):
        # clear existing
        while self.proposalsLayout.count():
            item = self.proposalsLayout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        for p in items:
            i = int(p.index)
            # Prefer current distance matrix value, fallback to stored
            dist = (
                float(dvec[i])
                if 0 <= i < len(dvec) and np.isfinite(dvec[i])
                else float(p.distance)
            )
            spec = self._get_display_spectrogram_for_index(i)
            row = ProposalRow(
                idx=i,
                spectrogram=spec,
                distance=dist,
                bucket=p.bucket or "",
                img_height=self.IMG_HEIGHT,
                sidepanel_width=self.SIDEPANEL_WIDTH,
            )
            # Set initial match state from item
            row.set_match(bool(p.match))
            row.labeledChanged.connect(self._on_any_row_changed)
            self.proposalsLayout.addWidget(row)

        self.proposalsLayout.addStretch(1)

    def _build_proposals_for_query(self, q: int) -> List[LabeledProposal]:
        # If we have a saved ordered list and we're not forcing all-nearest mode,
        # use it but update matches from state
        if (not self.all_nearest) and (q in self._query_proposals):
            state = self._saved_match_state.get(q, {})
            items = []
            for p in self._query_proposals[q]:
                idx = int(p.index)
                match = state.get(idx, (0.0, p.match))[1] if state else p.match
                # Refresh distance from current matrix
                dist = (
                    float(self.dist[q, idx]) if 0 <= idx < self.N else float(p.distance)
                )
                items.append(
                    LabeledProposal(
                        index=idx,
                        distance=dist,
                        bucket=str(p.bucket),
                        match=bool(match),
                    )
                )
            # Order proposals by ascending distance
            items.sort(key=lambda x: x.distance)
            return items

        # Otherwise, compute proposals based on distances and current bias
        d = self.dist[q].copy()
        order = np.argsort(d)
        order = order[order != q]  # drop self

        # Build items depending on selection mode
        state = self._saved_match_state.get(q, {})
        items: List[LabeledProposal] = []

        if self.all_nearest:
            k = int(self.num_nn_idx + self.num_quantile_idx)
            topk = [int(i) for i in order[:k]]
            for i in topk:
                items.append(
                    LabeledProposal(
                        index=int(i),
                        distance=float(d[int(i)]),
                        bucket="nn",
                        match=bool(state.get(int(i), (0.0, False))[1] if state else False),
                    )
                )
        else:
            nn_idx = [int(i) for i in order[: self.num_nn_idx]]
            remainder = order[self.num_nn_idx :]
            # If a max distance is set, restrict the sampling pool
            if self.max_quantile_distance is not None:
                md = float(self.max_quantile_distance)
                # `order` is sorted by ascending distance; preserve order while filtering
                remainder = np.array(
                    [int(i) for i in remainder if float(d[int(i)]) <= md], dtype=int
                )
            quant_idx = quantile_indices_from_sorted(
                remainder, self.num_quantile_idx, bias_exponent=self.bias_exponent
            )

            # ensure uniqueness between buckets
            nn_set = set(nn_idx)
            q_unique = [int(i) for i in quant_idx if i not in nn_set]
            if len(q_unique) < self.num_quantile_idx:
                for i in remainder:
                    ii = int(i)
                    if ii not in nn_set and ii not in q_unique:
                        q_unique.append(ii)
                        if len(q_unique) == self.num_quantile_idx:
                            break

            # Build items with current match state
            for i in nn_idx:
                items.append(
                    LabeledProposal(
                        index=i,
                        distance=float(d[i]),
                        bucket="nn",
                        match=bool(state.get(i, (0.0, False))[1] if state else False),
                    )
                )
            for i in q_unique:
                items.append(
                    LabeledProposal(
                        index=i,
                        distance=float(d[i]),
                        bucket="quantile",
                        match=bool(state.get(i, (0.0, False))[1] if state else False),
                    )
                )

        # Order proposals by ascending distance
        items.sort(key=lambda x: x.distance)

        # Cache proposals for this query so future visits remain consistent
        self._query_proposals[q] = items
        self._proposal_timestamp[q] = time.time()
        return items

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
        # If any proposal is checked, clear the query-level no-matches flag
        any_true = False
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if isinstance(w, ProposalRow) and bool(w.chkMatch.isChecked()):
                any_true = True
                break
        if any_true and self.noMatchesCheck.isChecked():
            self.noMatchesCheck.blockSignals(True)
            self.noMatchesCheck.setChecked(False)
            self.noMatchesCheck.blockSignals(False)
            self._apply_no_matches_ui_state(False)
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
                labels.append(
                    LabeledProposal(
                        index=w.idx,
                        distance=w.distance,
                        bucket=w.bucket,
                        match=bool(w.chkMatch.isChecked()),
                    )
                )
        return labels

    def next_query(self):
        # Cache current UI state, including query-level no-matches flag
        self._cache_current_query_labels()
        rec = QueryResult(
            query_index=int(self.current_query),
            timestamp=time.time(),
            annotator=self.annotator_name(),
            proposals=self._collect_labels(),
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

    def prev_query(self):
        # Cache current labels before switching; do not append to results
        if self.current_query is not None:
            self._cache_current_query_labels()
        prev = (int(self.current_query) - 1) % self.N if self.current_query is not None else 0
        self.set_query(prev)

    # -------------------- Alignment helpers --------------------

    def _get_display_spectrogram_for_index(self, idx: int) -> np.ndarray:
        """
        Return the spectrogram for display. If alignment is enabled, shift the
        proposal spectrogram along time to best match the current query.
        """
        S = self.spectrograms[idx]
        if not getattr(self, "align_by_xcorr", False):
            return S
        try:
            q_idx = int(self.current_query) if self.current_query is not None else None
        except Exception:
            q_idx = None
        if q_idx is None or not (0 <= q_idx < self.N):
            return S
        if idx == q_idx:
            return S
        # Compute profiles and best shift
        prof_q = _time_profile_from_spectrogram(self.spectrograms[q_idx])
        prof_p = _time_profile_from_spectrogram(S)
        # Limit maximum tested shift to 25% of the shorter profile length
        max_shift = int(max(1, min(len(prof_q), len(prof_p)) * 0.25))
        s = _best_time_shift(prof_q, prof_p, max_shift=max_shift)
        return _shift_along_time(S, s)

    def _on_align_toggled(self, checked: bool):
        self.align_by_xcorr = bool(checked)
        # Refresh all proposal images in-place to reflect alignment state
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if not isinstance(w, ProposalRow):
                continue
            spec = self._get_display_spectrogram_for_index(w.idx)
            try:
                w.img.show_spectrogram(spec)
            except Exception:
                pass

    # -------------------- No matches (query-level) --------------------

    def _apply_no_matches_ui_state(self, _=None):
        # Disable/enable proposal list when marking no matches or ignored
        try:
            nm = bool(self.noMatchesCheck.isChecked()) if hasattr(self, "noMatchesCheck") else False
            ig = bool(self.ignoreCheck.isChecked()) if hasattr(self, "ignoreCheck") else False
            self.scrollInner.setDisabled(bool(nm or ig))
        except Exception:
            pass

    def _on_no_matches_toggled(self, checked: bool):
        # If marking no matches, clear all proposal matches and disable UI
        if bool(checked):
            for i in range(self.proposalsLayout.count()):
                w = self.proposalsLayout.itemAt(i).widget()
                if isinstance(w, ProposalRow):
                    w.set_match(False)
        self._apply_no_matches_ui_state(bool(checked))
        # Cache immediately and enable Next
        self._noop_update_next()
        self._cache_current_query_labels()
        if self.current_query is not None:
            self._update_query_item_label(int(self.current_query))

    def _on_ignore_toggled(self, checked: bool):
        # Ignoring a query disables proposals; also clear matches for consistency
        if bool(checked):
            for i in range(self.proposalsLayout.count()):
                w = self.proposalsLayout.itemAt(i).widget()
                if isinstance(w, ProposalRow):
                    w.set_match(False)
        self._apply_no_matches_ui_state(bool(checked))
        self._noop_update_next()
        self._cache_current_query_labels()
        if self.current_query is not None:
            self._update_query_item_label(int(self.current_query))

    def save_results(self):
        # Ensure any pending UI events are processed, then cache current state
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        self._cache_current_query_labels()
        self.results_path = self._make_results_filename()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save results JSON", self.results_path, "JSON (*.json)"
        )
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
        # Include any query that has proposals cached or any saved state
        queries = sorted(
            set(self._query_proposals.keys())
            | set(self._saved_match_state.keys())
            | set(self._no_match_queries.keys())
            | set(self._ignored_queries.keys())
        )
        for q in queries:
            state = self._saved_match_state.get(q, {})
            # Base proposals come from cached ordered list if available
            base_props = list(self._query_proposals.get(q, []))
            # If none, synthesize from saved state indices
            if not base_props and state:
                for idx in sorted(state.keys()):
                    _ts, match = state[idx]
                    dist = float(self.dist[q, idx]) if 0 <= idx < self.N else 0.0
                    base_props.append(
                        LabeledProposal(
                            index=idx, distance=dist, bucket="", match=bool(match)
                        )
                    )

            if not base_props and not state:
                # Still serialize query if it has explicit no-matches/ignored mark
                if not (
                    bool(self._no_match_queries.get(q, (0.0, False))[1])
                    or bool(self._ignored_queries.get(q, (0.0, False))[1])
                ):
                    continue

            # Timestamp preference: proposals ts, otherwise latest label ts
            ts_q = self._proposal_timestamp.get(q, 0.0)
            if ts_q <= 0.0:
                ts_q = max(
                    (
                        [ts for ts, _ in state.values()]
                        + ([self._no_match_queries[q][0]] if q in self._no_match_queries else [])
                        + ([self._ignored_queries[q][0]] if q in self._ignored_queries else [])
                    )
                    or [time.time()]
                )

            # Merge match values from state into proposal list; refresh distances
            merged: List[LabeledProposal] = []
            for p in base_props:
                idx = int(p.index)
                match = state.get(idx, (0.0, p.match))[1] if state else p.match
                dist = (
                    float(self.dist[q, idx]) if 0 <= idx < self.N else float(p.distance)
                )
                merged.append(
                    LabeledProposal(
                        index=idx,
                        distance=dist,
                        bucket=p.bucket,
                        match=bool(match),
                    )
                )

            rec = {
                "query_index": int(q),
                "timestamp": float(ts_q),
                "annotator": self.annotator_name(),
                "proposals": [asdict(p) for p in merged],
            }
            # Include explicit flags when set
            if bool(self._no_match_queries.get(q, (0.0, False))[1]):
                rec["no_matches"] = True
            if bool(self._ignored_queries.get(q, (0.0, False))[1]):
                rec["ignored"] = True
            out.append(rec)
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
        # Ingest optional no_matches flag if present on record
        if hasattr(rec, "no_matches"):
            try:
                nm = bool(getattr(rec, "no_matches"))
                prev_nm = self._no_match_queries.get(q)
                if prev_nm is None or ts >= prev_nm[0]:
                    self._no_match_queries[q] = (ts, nm)
            except Exception:
                pass
        # Ingest optional ignored flag if present on the record
        if hasattr(rec, "ignored"):
            try:
                ig = bool(getattr(rec, "ignored"))
                prev_ig = self._ignored_queries.get(q)
                if prev_ig is None or ts >= prev_ig[0]:
                    self._ignored_queries[q] = (ts, ig)
            except Exception:
                pass
        # Also store proposals list order if provided and newer
        if rec.proposals:
            if (q not in self._proposal_timestamp) or (
                ts >= self._proposal_timestamp[q]
            ):
                # Filter proposals within bounds
                props: List[LabeledProposal] = []
                for p in rec.proposals:
                    try:
                        idx = int(p.index)
                    except Exception:
                        continue
                    if 0 <= idx < self.N:
                        props.append(
                            LabeledProposal(
                                index=idx,
                                distance=float(p.distance),
                                bucket=str(p.bucket or ""),
                                match=bool(p.match),
                            )
                        )
                if props:
                    self._query_proposals[q] = props
                    self._proposal_timestamp[q] = ts

    def _load_annotations_from_dir(self, directory: str):
        # Load all *.json files that look like results produced by this tool
        try:
            files = [f for f in os.listdir(directory) if f.lower().endswith(".json")]
        except Exception as e:
            print(f"Warning: cannot list directory {directory}: {e}")
            return

        for fname in files:
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue  # skip non-JSON or unreadable

            if not isinstance(data, dict) or "results" not in data:
                continue

            file_mtime = 0.0
            try:
                file_mtime = os.path.getmtime(fpath)
            except Exception:
                pass

            results = data.get("results") or []
            if not isinstance(results, list):
                continue

            for rec in results:
                try:
                    q = int(rec.get("query_index"))
                except Exception:
                    continue
                ts = rec.get("timestamp")
                try:
                    ts = float(ts) if ts is not None else float(file_mtime)
                except Exception:
                    ts = float(file_mtime)
                annot = rec.get("proposals") or []
                if not isinstance(annot, list):
                    continue
                # Build temporary QueryResult-like object to reuse ingest logic
                props: List[LabeledProposal] = []
                for p in annot:
                    try:
                        idx = int(p.get("index"))
                    except Exception:
                        continue
                    match = bool(p.get("match", False))
                    bucket = str(p.get("bucket", ""))
                    dist = float(p.get("distance", 0.0)) if "distance" in p else 0.0
                    props.append(
                        LabeledProposal(
                            index=idx, distance=dist, bucket=bucket, match=match
                        )
                    )
                qr = QueryResult(
                    query_index=q,
                    timestamp=ts,
                    annotator=str(rec.get("annotator", "")),
                    proposals=props,
                )
                # Inject no_matches flag if present
                try:
                    setattr(qr, "no_matches", bool(rec.get("no_matches", False)))
                except Exception:
                    pass
                # Inject ignored flag if present
                try:
                    setattr(qr, "ignored", bool(rec.get("ignored", False)))
                except Exception:
                    pass
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
        # Also capture the current proposals list/order from UI
        ui_items: List[LabeledProposal] = []
        for i in range(self.proposalsLayout.count()):
            w = self.proposalsLayout.itemAt(i).widget()
            if not isinstance(w, ProposalRow):
                continue
            prev = current_state.get(w.idx)
            prev_match = prev[1] if prev is not None else None
            curr_match = bool(w.chkMatch.isChecked())
            ui_items.append(
                LabeledProposal(
                    index=w.idx,
                    distance=w.distance,
                    bucket=w.bucket,
                    match=curr_match,
                )
            )
            if prev_match is None:
                # New: only record True to avoid overwriting with False by default
                if curr_match:
                    changed.append(
                        LabeledProposal(
                            index=w.idx,
                            distance=w.distance,
                            bucket=w.bucket,
                            match=True,
                        )
                    )
            else:
                if curr_match != bool(prev_match):
                    changed.append(
                        LabeledProposal(
                            index=w.idx,
                            distance=w.distance,
                            bucket=w.bucket,
                            match=curr_match,
                        )
                    )

        if changed:
            rec = QueryResult(
                query_index=curr,
                timestamp=ts,
                annotator=self.annotator_name(),
                proposals=changed,
            )
            self._ingest_query_result_into_state(rec)
        # Always keep proposals list and timestamp in sync with UI
        if ui_items:
            self._query_proposals[curr] = ui_items
            self._proposal_timestamp[curr] = ts
        # Cache query-level no-matches flag
        try:
            nm_flag = bool(self.noMatchesCheck.isChecked())
        except Exception:
            nm_flag = False
        prev_nm = self._no_match_queries.get(curr)
        prev_val = prev_nm[1] if prev_nm is not None else None
        if prev_val is None or bool(prev_val) != nm_flag:
            self._no_match_queries[curr] = (ts, nm_flag)
        # Cache query-level ignored flag
        try:
            ig_flag = bool(self.ignoreCheck.isChecked())
        except Exception:
            ig_flag = False
        prev_ig = self._ignored_queries.get(curr)
        prev_ig_val = prev_ig[1] if prev_ig is not None else None
        if prev_ig_val is None or bool(prev_ig_val) != ig_flag:
            self._ignored_queries[curr] = (ts, ig_flag)
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
        # Marker precedence: ignored > no-matches > any-match
        ig = bool(self._ignored_queries.get(int(q), (0.0, False))[1])
        if ig:
            mark = "X "
        else:
            nm = bool(self._no_match_queries.get(int(q), (0.0, False))[1])
            if nm:
                mark = "∅ "
            else:
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
    parser.add_argument(
        "--annotator",
        dest="annotator",
        type=str,
        default="",
        help="Annotator name (prefills the GUI and is saved to JSON)",
    )
    parser.add_argument(
        "--h5",
        dest="h5_path",
        type=str,
        default="embeddings.h5",
        help="Path to embeddings HDF5 (default: embeddings.h5)",
    )
    parser.add_argument(
        "--quantile-bias",
        dest="quantile_bias",
        type=float,
        default=0.1,
        help=(
            "Bias exponent for quantile sampling (<1 overrepresents close neighbours; 1=even)"
        ),
    )
    parser.add_argument(
        "--quantile-max-distance",
        dest="quantile_max_distance",
        type=float,
        default=None,
        help=(
            "Upper bound on distance for proposals beyond the top-20; "
            "only candidates with distance <= this value are sampled"
        ),
    )
    parser.add_argument(
        "--all-nearest",
        dest="all_nearest",
        action="store_true",
        help=(
            "Select 40 proposals as the 40 nearest neighbours (ordered by distance)."
        ),
    )
    args, unknown = parser.parse_known_args()
    # Keep unknown args for Qt
    sys.argv = [sys.argv[0]] + unknown

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SpectrogramMatcher(
        h5_path=args.h5_path,
        annotator_name=args.annotator,
        quantile_bias_exponent=args.quantile_bias,
        quantile_max_distance=args.quantile_max_distance,
        all_nearest=args.all_nearest,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
