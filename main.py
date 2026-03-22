"""
Event Detection — PyQtGraph Desktop GUI
========================================
Signal:  10-second 5 Hz sine wave, 1000 Hz sample rate
Events:  N × 1-second bursts of 80 Hz high-frequency oscillation

Run:
    python event_detection_app.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys, io, time, warnings, threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import pywt
from sklearn.metrics import f1_score, precision_score, recall_score
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Qt / pyqtgraph ────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QProgressBar, QTextEdit,
    QFileDialog, QSplitter, QScrollArea, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg

# ── pyqtgraph global style ────────────────────────────────────────────────────
pg.setConfigOption("background", "#1e1e2e")
pg.setConfigOption("foreground", "#cdd6f4")
pg.setConfigOption("antialias", True)

# ── Palette constants ─────────────────────────────────────────────────────────
C_BG       = "#1e1e2e"
C_SURFACE  = "#313244"
C_BORDER   = "#45475a"
C_TEXT     = "#cdd6f4"
C_MUTED    = "#6c7086"
C_BLUE     = "#89b4fa"
C_GREEN    = "#a6e3a1"
C_RED      = "#f38ba8"
C_ORANGE   = "#fab387"
C_PURPLE   = "#cba6f7"
C_YELLOW   = "#f9e2af"


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    sample_rate:     int   = 1000
    signal_duration: float = 10.0
    baseline_freq:   float = 5.0
    event_freq:      float = 80.0
    event_duration:  float = 1.0
    noise_std:       float = 0.2
    num_events:      int   = 3
    event_amplitude: float = 2.0
    window_size:     int   = 256
    stride:          int   = 64
    wavelet:         str   = "db4"
    wavelet_levels:  int   = 4
    history_len:     int   = 10
    hidden_size:     int   = 64
    num_layers:      int   = 2
    dropout:         float = 0.3
    epochs:          int   = 30
    batch_size:      int   = 32
    learning_rate:   float = 1e-3
    pos_weight:      float = 4.0
    threshold:       float = 0.5
    smooth_window:   int   = 5

    def feature_dim(self) -> int:
        return 6 + 5 + self.wavelet_levels


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  2. SIGNAL GENERATOR                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def generate_signal(cfg: Config, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n   = int(cfg.signal_duration * cfg.sample_rate)
    t   = np.arange(n) / cfg.sample_rate
    sig = np.sin(2 * np.pi * cfg.baseline_freq * t)
    sig += rng.normal(0, cfg.noise_std, n)
    lbl = np.zeros(n, dtype=np.float32)
    ev_samples = int(cfg.event_duration * cfg.sample_rate)
    placed, attempts, used = 0, 0, []
    while placed < cfg.num_events and attempts < 1000:
        attempts += 1
        s = int(rng.integers(0, n - ev_samples))
        e = s + ev_samples
        if any(not (e + 200 <= us or s >= ue + 200) for us, ue in used):
            continue
        ev_t = np.arange(ev_samples) / cfg.sample_rate
        sig[s:e] += cfg.event_amplitude * np.sin(2 * np.pi * cfg.event_freq * ev_t)
        lbl[s:e]  = 1.0
        used.append((s, e)); placed += 1
    return sig.astype(np.float32), lbl


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  3. WINDOWING                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def sliding_windows(signal, labels, cfg):
    wins, wlbls = [], []
    i = 0
    while i + cfg.window_size <= len(signal):
        wins.append(signal[i: i + cfg.window_size])
        if labels is not None:
            wlbls.append(float(labels[i: i + cfg.window_size].mean() >= 0.5))
        i += cfg.stride
    W   = np.stack(wins)
    lbl = np.array(wlbls, dtype=np.float32) if labels is not None else None
    return W, lbl


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  4. FEATURE EXTRACTION                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _stat(w):
    return np.array([w.mean(), w.std(), float(skew(w)),
                     float(kurtosis(w)), w.min(), w.max()], dtype=np.float32)

def _freq(w, sr):
    c   = np.abs(rfft(w)); f = rfftfreq(len(w), 1.0/sr)
    eng = float(np.sum(c**2)) + 1e-8
    return np.array([
        eng, float(f[np.argmax(c)]),
        float(np.sum(f*c)/(np.sum(c)+1e-8)),
        float(np.sum(c[f<20]**2))/eng,
        float(np.sum(c[(f>=20)&(f<100)]**2))/eng,
    ], dtype=np.float32)

def _wav(w, wavelet, levels):
    return np.array([float(np.sum(c**2))
                     for c in pywt.wavedec(w, wavelet, level=levels)[:levels]],
                    dtype=np.float32)

def extract_features(w, cfg):
    return np.concatenate([_stat(w), _freq(w, cfg.sample_rate),
                           _wav(w, cfg.wavelet, cfg.wavelet_levels)])

def build_feature_matrix(windows, cfg):
    return np.stack([extract_features(w, cfg) for w in windows])


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  5. SEQUENCES + DATASET                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def build_sequences(features, labels, history):
    seqs, slbls = [], []
    for i in range(history-1, len(features)):
        seqs.append(features[i-history+1: i+1])
        if labels is not None: slbls.append(labels[i])
    X = np.stack(seqs)
    y = np.array(slbls, dtype=np.float32) if labels is not None else None
    return X, y

class EventDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):         return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  6. LSTM MODEL                                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class EventLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg.feature_dim(), cfg.hidden_size,
                            cfg.num_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.num_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden_size, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  7. TRAINING WORKER (runs in QThread)                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TrainWorker(QThread):
    progress  = pyqtSignal(int)          # 0-100
    log_line  = pyqtSignal(str)
    epoch_done = pyqtSignal(float, float, float)   # train_loss, val_loss, f1
    finished  = pyqtSignal(object, object)         # model, history

    def __init__(self, cfg, X_tr, y_tr, X_val, y_val):
        super().__init__()
        self.cfg   = cfg
        self.X_tr  = X_tr; self.y_tr  = y_tr
        self.X_val = X_val; self.y_val = y_val

    def run(self):
        cfg    = self.cfg
        model  = EventLSTM(cfg)
        loader = DataLoader(EventDataset(self.X_tr, self.y_tr),
                            batch_size=cfg.batch_size, shuffle=True)
        crit   = nn.BCEWithLogitsLoss(
                     pos_weight=torch.tensor([cfg.pos_weight]))
        opt    = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     opt, patience=4, factor=0.5)
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(cfg.epochs):
            model.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= len(self.X_tr)

            model.eval()
            with torch.no_grad():
                xv  = torch.from_numpy(self.X_val).float()
                yv  = torch.from_numpy(self.y_val).float()
                vl  = crit(model(xv), yv).item()
                prd = (torch.sigmoid(model(xv)) >= cfg.threshold).numpy()
                f1  = f1_score(self.y_val, prd, zero_division=0)

            sched.step(vl)
            history["train_loss"].append(ep_loss)
            history["val_loss"].append(vl)
            history["val_f1"].append(f1)
            self.progress.emit(int((epoch+1)/cfg.epochs*100))
            self.log_line.emit(
                f"Epoch {epoch+1:>3}/{cfg.epochs}  "
                f"train={ep_loss:.4f}  val={vl:.4f}  F1={f1:.3f}")
            self.epoch_done.emit(ep_loss, vl, f1)

        self.finished.emit(model, history)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  8. DETECTION WORKER                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class DetectWorker(QThread):
    finished = pyqtSignal(object, object, object)  # probs, smoothed, events

    def __init__(self, model, signal, cfg):
        super().__init__()
        self.model  = model
        self.signal = signal
        self.cfg    = cfg

    def run(self):
        cfg    = self.cfg
        model  = self.model
        model.eval()
        W, _   = sliding_windows(self.signal, None, cfg)
        feat   = build_feature_matrix(W, cfg)
        X, _   = build_sequences(feat, None, cfg.history_len)
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.from_numpy(X).float())).numpy()

        smoothed = np.zeros_like(probs)
        hw = cfg.smooth_window // 2
        for i in range(len(probs)):
            lo, hi = max(0, i-hw), min(len(probs), i+hw+1)
            smoothed[i] = float(probs[lo:hi].mean() >= cfg.threshold)

        events, in_ev, sw = [], False, 0
        for i, v in enumerate(smoothed):
            if v==1 and not in_ev:    in_ev, sw = True, i
            elif v==0 and in_ev:
                in_ev = False
                t0 = (sw + cfg.history_len-1)*cfg.stride/cfg.sample_rate
                t1 = (i  + cfg.history_len-1)*cfg.stride/cfg.sample_rate
                events.append((t0, t1))
        if in_ev:
            t0 = (sw            + cfg.history_len-1)*cfg.stride/cfg.sample_rate
            t1 = (len(smoothed) + cfg.history_len-1)*cfg.stride/cfg.sample_rate
            events.append((t0, t1))
        self.finished.emit(probs, smoothed, events)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  9. HELPER: window predictions → sample-level array                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def preds_to_samples(smoothed, signal_len, cfg):
    out    = np.zeros(signal_len, dtype=np.float32)
    offset = cfg.history_len - 1
    for i, v in enumerate(smoothed):
        if v == 1:
            s = (i + offset) * cfg.stride
            e = min(s + cfg.window_size, signal_len)
            out[s:e] = 1.0
    return out


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  10. STYLE HELPERS                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

GLOBAL_STYLE = f"""
QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "SF Pro Display", "Segoe UI", "Ubuntu", sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {C_BORDER};
    border-radius: 6px;
}}
QTabBar::tab {{
    background: {C_SURFACE};
    color: {C_MUTED};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {C_BG};
    color: {C_TEXT};
    border-bottom: 2px solid {C_BLUE};
}}
QGroupBox {{
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: {C_MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QPushButton {{
    background-color: {C_BLUE};
    color: {C_BG};
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}}
QPushButton:hover   {{ background-color: #b4d0ff; }}
QPushButton:pressed {{ background-color: #6090d0; }}
QPushButton:disabled {{ background-color: {C_BORDER}; color: {C_MUTED}; }}
QPushButton#danger  {{
    background-color: {C_RED};
}}
QSlider::groove:horizontal {{
    height: 4px; background: {C_BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_BLUE}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C_BLUE}; border-radius: 2px; }}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 4px; padding: 4px 8px;
    color: {C_TEXT};
}}
QComboBox::drop-down {{ border: none; }}
QProgressBar {{
    border: 1px solid {C_BORDER}; border-radius: 4px;
    background: {C_SURFACE}; text-align: center; color: {C_TEXT};
}}
QProgressBar::chunk {{ background: {C_BLUE}; border-radius: 4px; }}
QTextEdit {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 4px; font-family: monospace; font-size: 12px;
    color: {C_GREEN};
}}
QTableWidget {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    gridline-color: {C_BORDER}; border-radius: 4px;
}}
QTableWidget::item {{ padding: 4px 8px; }}
QTableWidget::item:selected {{ background: {C_BLUE}; color: {C_BG}; }}
QHeaderView::section {{
    background: {C_BG}; color: {C_MUTED};
    border: none; border-bottom: 1px solid {C_BORDER};
    padding: 6px 8px; font-weight: bold;
}}
QScrollBar:vertical {{
    background: {C_SURFACE}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER}; border-radius: 4px; min-height: 20px;
}}
QLabel#title {{
    font-size: 18px; font-weight: bold; color: {C_TEXT};
}}
QLabel#subtitle {{
    font-size: 12px; color: {C_MUTED};
}}
QLabel#metric_val {{
    font-size: 24px; font-weight: bold; color: {C_BLUE};
}}
QLabel#metric_lbl {{
    font-size: 11px; color: {C_MUTED};
}}
QFrame#card {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
"""

def make_plot_widget(title="", x_label="Time (s)", y_label="Amplitude",
                     height=None) -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pw.setBackground(C_BG)
    pw.showGrid(x=True, y=True, alpha=0.15)
    pw.setLabel("bottom", x_label,
                **{"color": C_MUTED, "font-size": "11px"})
    pw.setLabel("left",   y_label,
                **{"color": C_MUTED, "font-size": "11px"})
    if title:
        pw.setTitle(title, color=C_MUTED, size="11px")
    if height:
        pw.setFixedHeight(height)
    pw.getAxis("bottom").setPen(pg.mkPen(C_BORDER))
    pw.getAxis("left").setPen(pg.mkPen(C_BORDER))
    pw.getAxis("bottom").setTextPen(pg.mkPen(C_MUTED))
    pw.getAxis("left").setTextPen(pg.mkPen(C_MUTED))
    return pw

def metric_card(value: str, label: str) -> QFrame:
    card = QFrame(); card.setObjectName("card")
    lay  = QVBoxLayout(card); lay.setContentsMargins(16,12,16,12)
    v = QLabel(value); v.setObjectName("metric_val"); v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    l = QLabel(label); l.setObjectName("metric_lbl"); l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v); lay.addWidget(l)
    return card

def labeled_spin(label, min_v, max_v, default, step=1, decimals=0) -> Tuple[QLabel, QWidget]:
    lbl = QLabel(label)
    if decimals > 0:
        w = QDoubleSpinBox()
        w.setDecimals(decimals); w.setSingleStep(step)
    else:
        w = QSpinBox()
        w.setSingleStep(int(step))
    w.setMinimum(min_v); w.setMaximum(max_v); w.setValue(default)
    return lbl, w

def add_row(grid, row, label, widget):
    grid.addWidget(QLabel(label), row, 0)
    grid.addWidget(widget, row, 1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  11. MAIN WINDOW                                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Detection — PyQtGraph")
        self.resize(1400, 900)
        self.setStyleSheet(GLOBAL_STYLE)

        # ── state ─────────────────────────────────────────────────────────────
        self.signal   = None
        self.labels   = None
        self.model    = None
        self.history  = None
        self.probs    = None
        self.smoothed = None
        self.events   = []
        self.cfg      = Config()
        self._train_worker  = None
        self._detect_worker = None

        # ── root layout: sidebar | tabs ───────────────────────────────────────
        root   = QWidget(); self.setCentralWidget(root)
        h_lay  = QHBoxLayout(root); h_lay.setContentsMargins(0,0,0,0); h_lay.setSpacing(0)

        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(260)
        h_lay.addWidget(sidebar)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {C_BORDER};")
        h_lay.addWidget(sep)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        h_lay.addWidget(self.tabs, 1)

        self._build_tab_data()
        self._build_tab_train()
        self._build_tab_detect()
        self._build_tab_eval()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setStyleSheet(f"QWidget#sidebar {{ background:{C_SURFACE}; }}")
        lay = QVBoxLayout(panel); lay.setContentsMargins(12,16,12,12)

        title = QLabel("⚙️  Config"); title.setObjectName("title")
        lay.addWidget(title)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner_lay = QVBoxLayout(inner); inner_lay.setSpacing(8)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        def section(name):
            gb  = QGroupBox(name)
            grd = QGridLayout(gb); grd.setColumnStretch(1,1)
            inner_lay.addWidget(gb)
            return grd

        # ── Signal ─────────────────────────────────────────────────────────
        g = section("Signal")
        self._w_num_events = QSpinBox();    self._w_num_events.setRange(1,8);   self._w_num_events.setValue(3)
        self._w_noise      = QDoubleSpinBox(); self._w_noise.setRange(0,2);    self._w_noise.setValue(0.2); self._w_noise.setSingleStep(0.05)
        self._w_amp        = QDoubleSpinBox(); self._w_amp.setRange(0.5,5);    self._w_amp.setValue(2.0);   self._w_amp.setSingleStep(0.25)
        self._w_bfreq      = QDoubleSpinBox(); self._w_bfreq.setRange(1,20);   self._w_bfreq.setValue(5.0); self._w_bfreq.setSingleStep(1)
        self._w_efreq      = QDoubleSpinBox(); self._w_efreq.setRange(20,450); self._w_efreq.setValue(80);  self._w_efreq.setSingleStep(5)
        self._w_seed       = QSpinBox();    self._w_seed.setRange(0,9999);     self._w_seed.setValue(42)
        for r,(lbl,w) in enumerate([("Events",self._w_num_events),("Noise std",self._w_noise),
                                     ("Event amp",self._w_amp),("Baseline Hz",self._w_bfreq),
                                     ("Event Hz",self._w_efreq),("Seed",self._w_seed)]):
            add_row(g, r, lbl, w)

        # ── Windowing ──────────────────────────────────────────────────────
        g = section("Windowing")
        self._w_winsize = QComboBox(); [self._w_winsize.addItem(str(x)) for x in [64,128,256,512]]; self._w_winsize.setCurrentText("256")
        self._w_stride  = QComboBox(); [self._w_stride.addItem(str(x)) for x in [16,32,64,128]];   self._w_stride.setCurrentText("64")
        add_row(g, 0, "Window", self._w_winsize)
        add_row(g, 1, "Stride", self._w_stride)

        # ── Wavelet ────────────────────────────────────────────────────────
        g = section("Wavelet")
        self._w_wavelet = QComboBox(); [self._w_wavelet.addItem(x) for x in ["db4","db6","db8","sym4","coif2","haar"]]
        self._w_wavlvl  = QSpinBox(); self._w_wavlvl.setRange(1,6); self._w_wavlvl.setValue(4)
        add_row(g, 0, "Type",   self._w_wavelet)
        add_row(g, 1, "Levels", self._w_wavlvl)

        # ── LSTM ───────────────────────────────────────────────────────────
        g = section("LSTM")
        self._w_hist    = QSpinBox(); self._w_hist.setRange(2,20);  self._w_hist.setValue(10)
        self._w_hidden  = QComboBox(); [self._w_hidden.addItem(str(x)) for x in [32,64,128,256]]; self._w_hidden.setCurrentText("64")
        self._w_layers  = QSpinBox(); self._w_layers.setRange(1,4);  self._w_layers.setValue(2)
        self._w_drop    = QDoubleSpinBox(); self._w_drop.setRange(0,0.6); self._w_drop.setValue(0.3); self._w_drop.setSingleStep(0.05)
        for r,(lbl,w) in enumerate([("History",self._w_hist),("Hidden",self._w_hidden),
                                     ("Layers",self._w_layers),("Dropout",self._w_drop)]):
            add_row(g, r, lbl, w)

        # ── Training ───────────────────────────────────────────────────────
        g = section("Training")
        self._w_epochs  = QSpinBox(); self._w_epochs.setRange(5,200);  self._w_epochs.setValue(30)
        self._w_bsize   = QComboBox(); [self._w_bsize.addItem(str(x)) for x in [8,16,32,64]]; self._w_bsize.setCurrentText("32")
        self._w_lr      = QComboBox(); [self._w_lr.addItem(x) for x in ["0.0001","0.0005","0.001","0.005"]]; self._w_lr.setCurrentText("0.001")
        self._w_pw      = QDoubleSpinBox(); self._w_pw.setRange(1,10); self._w_pw.setValue(4.0); self._w_pw.setSingleStep(0.5)
        for r,(lbl,w) in enumerate([("Epochs",self._w_epochs),("Batch",self._w_bsize),
                                     ("LR",self._w_lr),("Pos weight",self._w_pw)]):
            add_row(g, r, lbl, w)

        # ── Detection ──────────────────────────────────────────────────────
        g = section("Detection")
        self._w_thresh  = QDoubleSpinBox(); self._w_thresh.setRange(0.1,0.9); self._w_thresh.setValue(0.5); self._w_thresh.setSingleStep(0.05)
        self._w_smooth  = QSpinBox(); self._w_smooth.setRange(1,15); self._w_smooth.setValue(5)
        add_row(g, 0, "Threshold", self._w_thresh)
        add_row(g, 1, "Smooth w",  self._w_smooth)

        inner_lay.addStretch()
        return panel

    def _read_cfg(self) -> Config:
        return Config(
            num_events     = self._w_num_events.value(),
            noise_std      = self._w_noise.value(),
            event_amplitude= self._w_amp.value(),
            baseline_freq  = self._w_bfreq.value(),
            event_freq     = self._w_efreq.value(),
            window_size    = int(self._w_winsize.currentText()),
            stride         = int(self._w_stride.currentText()),
            wavelet        = self._w_wavelet.currentText(),
            wavelet_levels = self._w_wavlvl.value(),
            history_len    = self._w_hist.value(),
            hidden_size    = int(self._w_hidden.currentText()),
            num_layers     = self._w_layers.value(),
            dropout        = self._w_drop.value(),
            epochs         = self._w_epochs.value(),
            batch_size     = int(self._w_bsize.currentText()),
            learning_rate  = float(self._w_lr.currentText()),
            pos_weight     = self._w_pw.value(),
            threshold      = self._w_thresh.value(),
            smooth_window  = self._w_smooth.value(),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_data(self):
        w   = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        self.tabs.addTab(w, "📂  Data")

        # Header row
        hdr = QHBoxLayout()
        info = QLabel("10-sec · 5 Hz baseline · 1-sec 80 Hz event bursts")
        info.setObjectName("subtitle"); hdr.addWidget(info, 1)
        self._btn_gen    = QPushButton("Generate signal")
        self._btn_upload = QPushButton("Upload file")
        self._btn_upload.setStyleSheet(f"QPushButton {{ background:{C_SURFACE}; color:{C_TEXT}; border:1px solid {C_BORDER}; }}")
        hdr.addWidget(self._btn_gen); hdr.addWidget(self._btn_upload)
        lay.addLayout(hdr)

        # Metric cards row
        self._card_dur   = metric_card("—", "Duration")
        self._card_sr    = metric_card("1000 Hz", "Sample rate")
        self._card_evts  = metric_card("—", "Event bursts")
        self._card_ratio = metric_card("—", "Event ratio")
        cards = QHBoxLayout()
        for c in [self._card_dur, self._card_sr, self._card_evts, self._card_ratio]:
            cards.addWidget(c)
        lay.addLayout(cards)

        # Signal plot (zoomable)
        self._plot_signal = make_plot_widget("Signal  (green = event, orange = prediction)",
                                             height=280)
        lay.addWidget(self._plot_signal)

        # FFT plot
        self._plot_fft = make_plot_widget("Frequency spectrum",
                                          x_label="Frequency (Hz)",
                                          y_label="Magnitude", height=200)
        lay.addWidget(self._plot_fft)

        # Status
        self._lbl_data_status = QLabel("No signal loaded.")
        self._lbl_data_status.setObjectName("subtitle")
        lay.addWidget(self._lbl_data_status)

        # Wiring
        self._btn_gen.clicked.connect(self._on_generate)
        self._btn_upload.clicked.connect(self._on_upload)

    def _on_generate(self):
        self.cfg = self._read_cfg()
        sig, lbl = generate_signal(self.cfg, seed=self._w_seed.value())
        self._load_signal(sig, lbl)

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open signal file", "", "NumPy / CSV (*.npy *.csv)")
        if not path: return
        try:
            if path.endswith(".npy"):
                arr = np.load(path)
            else:
                arr = np.loadtxt(path, delimiter=",")
            sig = arr[:,0].astype(np.float32) if arr.ndim==2 else arr.astype(np.float32)
            lbl = arr[:,1].astype(np.float32) if arr.ndim==2 and arr.shape[1]>=2 else None
            self._load_signal(sig, lbl)
        except Exception as e:
            self._lbl_data_status.setText(f"Error loading file: {e}")

    def _load_signal(self, sig, lbl):
        self.cfg    = self._read_cfg()
        self.signal = sig
        self.labels = lbl
        self.probs  = None; self.smoothed = None; self.events = []
        self._refresh_data_plots()
        n = len(sig)
        ev_s = int(lbl.sum()) if lbl is not None else 0
        self._card_dur.findChildren(QLabel)[0].setText(f"{n/self.cfg.sample_rate:.1f} s")
        self._card_evts.findChildren(QLabel)[0].setText(f"{ev_s/self.cfg.sample_rate:.1f} s")
        self._card_ratio.findChildren(QLabel)[0].setText(
            f"{lbl.mean()*100:.1f}%" if lbl is not None else "—")
        self._lbl_data_status.setText(
            f"Loaded {n:,} samples · {ev_s:,} event samples")

    def _refresh_data_plots(self):
        if self.signal is None: return
        cfg = self.cfg
        t   = np.arange(len(self.signal)) / cfg.sample_rate

        # ── signal plot ────────────────────────────────────────────────────
        self._plot_signal.clear()
        self._plot_signal.plot(t, self.signal,
                               pen=pg.mkPen(C_BLUE, width=1.2), name="Signal")

        # shade ground-truth events (green)
        if self.labels is not None:
            self._shade_regions(self._plot_signal, t, self.labels,
                                QColor(166, 227, 161, 55))

        # shade predicted events (orange)
        if self.smoothed is not None:
            pred_s = preds_to_samples(self.smoothed, len(self.signal), cfg)
            self._shade_regions(self._plot_signal, t, pred_s,
                                QColor(250, 179, 135, 70))

        # ── FFT plot ───────────────────────────────────────────────────────
        self._plot_fft.clear()
        c = np.abs(rfft(self.signal)) / len(self.signal)
        f = rfftfreq(len(self.signal), 1.0/cfg.sample_rate)
        self._plot_fft.plot(f, c,
                            pen=pg.mkPen(C_PURPLE, width=1.2))
        self._plot_fft.setXRange(0, cfg.sample_rate//2)
        for freq, color, name in [(cfg.baseline_freq, C_GREEN, "baseline"),
                                   (cfg.event_freq,    C_RED,   "event")]:
            line = pg.InfiniteLine(pos=freq, angle=90,
                                   pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine),
                                   label=f"{name} {freq:.0f} Hz",
                                   labelOpts={"color": color, "position": 0.85})
            self._plot_fft.addItem(line)

    @staticmethod
    def _shade_regions(plot_widget, t, labels, color: QColor):
        """Draw filled LinearRegionItems for contiguous label=1 blocks."""
        in_ev, start = False, 0.0
        for i in range(len(labels)):
            if labels[i] == 1 and not in_ev:
                in_ev, start = True, t[i]
            elif labels[i] == 0 and in_ev:
                in_ev = False
                region = pg.LinearRegionItem(
                    values=[start, t[i]],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(None),
                    movable=False)
                plot_widget.addItem(region)
        if in_ev:
            region = pg.LinearRegionItem(
                values=[start, t[-1]],
                brush=pg.mkBrush(color),
                pen=pg.mkPen(None), movable=False)
            plot_widget.addItem(region)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_train(self):
        w   = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        self.tabs.addTab(w, "🏋️  Train")

        # Controls row
        ctrl = QHBoxLayout()
        self._btn_train = QPushButton("Start training")
        self._btn_stop  = QPushButton("Stop"); self._btn_stop.setEnabled(False)
        self._btn_stop.setObjectName("danger")
        self._btn_save_model = QPushButton("Save model…"); self._btn_save_model.setEnabled(False)
        ctrl.addWidget(self._btn_train)
        ctrl.addWidget(self._btn_stop)
        ctrl.addWidget(self._btn_save_model)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # Progress bar
        self._train_progress = QProgressBar(); self._train_progress.setValue(0)
        lay.addWidget(self._train_progress)

        # Loss + F1 plots side by side
        plots_row = QHBoxLayout()
        self._plot_loss = make_plot_widget("Loss curves",
                                           x_label="Epoch", y_label="Loss", height=240)
        self._plot_f1   = make_plot_widget("Validation F1",
                                           x_label="Epoch", y_label="F1",   height=240)
        self._plot_loss.addLegend(offset=(10,10))
        plots_row.addWidget(self._plot_loss)
        plots_row.addWidget(self._plot_f1)
        lay.addLayout(plots_row)

        # Log
        self._train_log = QTextEdit(); self._train_log.setReadOnly(True)
        self._train_log.setFixedHeight(180)
        lay.addWidget(QLabel("Training log:")); lay.addWidget(self._train_log)

        # Internal data for live curve updates
        self._ep_train_loss = []; self._ep_val_loss = []; self._ep_f1 = []

        self._btn_train.clicked.connect(self._on_train)
        self._btn_stop.clicked.connect(self._on_stop_train)
        self._btn_save_model.clicked.connect(self._on_save_model)

    def _on_train(self):
        if self.signal is None:
            self._train_log.append("⚠  No signal loaded. Go to Data tab first.")
            return
        if self.labels is None:
            self._train_log.append("⚠  No labels. Generate a signal first.")
            return

        self.cfg = self._read_cfg()
        self._train_log.clear()
        self._train_log.append("Building features…")
        self._ep_train_loss.clear(); self._ep_val_loss.clear(); self._ep_f1.clear()
        self._plot_loss.clear(); self._plot_f1.clear()
        self._train_progress.setValue(0)

        W, Wl    = sliding_windows(self.signal, self.labels, self.cfg)
        feat     = build_feature_matrix(W, self.cfg)
        X, y     = build_sequences(feat, Wl, self.cfg.history_len)
        split    = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        self._train_log.append(
            f"Sequences: train {len(X_tr):,}  val {len(X_val):,}  "
            f"feat_dim {X.shape[-1]}\nTraining…")

        self._train_worker = TrainWorker(
            self.cfg, X_tr, y_tr, X_val, y_val)
        self._train_worker.progress.connect(self._train_progress.setValue)
        self._train_worker.log_line.connect(self._train_log.append)
        self._train_worker.epoch_done.connect(self._on_epoch_done)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.start()

        self._btn_train.setEnabled(False)
        self._btn_stop.setEnabled(True)

    def _on_epoch_done(self, tl, vl, f1):
        self._ep_train_loss.append(tl)
        self._ep_val_loss.append(vl)
        self._ep_f1.append(f1)
        ep = list(range(1, len(self._ep_train_loss)+1))
        self._plot_loss.clear()
        self._plot_loss.addLegend(offset=(10,10))
        self._plot_loss.plot(ep, self._ep_train_loss,
                             pen=pg.mkPen(C_BLUE,  width=2), name="Train")
        self._plot_loss.plot(ep, self._ep_val_loss,
                             pen=pg.mkPen(C_RED,   width=2), name="Val")
        self._plot_f1.clear()
        self._plot_f1.plot(ep, self._ep_f1,
                           pen=pg.mkPen(C_GREEN, width=2))

    def _on_train_done(self, model, history):
        self.model   = model
        self.history = history
        self._btn_train.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_save_model.setEnabled(True)
        best = max(history["val_f1"])
        self._train_log.append(f"\n✅  Training complete  ·  Best val F1: {best:.3f}")

    def _on_stop_train(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.terminate()
            self._train_log.append("⛔  Training stopped by user.")
            self._btn_train.setEnabled(True)
            self._btn_stop.setEnabled(False)

    def _on_save_model(self):
        if self.model is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model weights", "event_lstm.pt", "PyTorch (*.pt)")
        if path:
            torch.save(self.model.state_dict(), path)
            self._train_log.append(f"💾  Model saved to {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — DETECT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_detect(self):
        w   = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        self.tabs.addTab(w, "🔍  Detect")

        ctrl = QHBoxLayout()
        self._btn_detect = QPushButton("Run detection")
        ctrl.addWidget(self._btn_detect); ctrl.addStretch()
        lay.addLayout(ctrl)

        # Linked plots: signal + probability
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._plot_det_sig = make_plot_widget(
            "Signal  (green = truth · orange = predicted)")
        self._plot_det_prob = make_plot_widget(
            "Event probability", y_label="P(event)", height=180)
        self._plot_det_prob.setYRange(0, 1)

        # Link x-axes
        self._plot_det_prob.setXLink(self._plot_det_sig)

        splitter.addWidget(self._plot_det_sig)
        splitter.addWidget(self._plot_det_prob)
        splitter.setSizes([420, 180])
        lay.addWidget(splitter, 1)

        # Event table
        lay.addWidget(QLabel("Detected events:"))
        self._tbl_events = QTableWidget(0, 4)
        self._tbl_events.setHorizontalHeaderLabels(["#", "Start (s)", "End (s)", "Duration (s)"])
        self._tbl_events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_events.setFixedHeight(160)
        lay.addWidget(self._tbl_events)

        self._lbl_detect_status = QLabel("")
        self._lbl_detect_status.setObjectName("subtitle")
        lay.addWidget(self._lbl_detect_status)

        self._btn_detect.clicked.connect(self._on_detect)

    def _on_detect(self):
        if self.model is None:
            self._lbl_detect_status.setText("⚠  Train a model first.")
            return
        if self.signal is None:
            self._lbl_detect_status.setText("⚠  Load a signal first.")
            return
        self.cfg = self._read_cfg()
        self._btn_detect.setEnabled(False)
        self._lbl_detect_status.setText("Running inference…")
        self._detect_worker = DetectWorker(self.model, self.signal, self.cfg)
        self._detect_worker.finished.connect(self._on_detect_done)
        self._detect_worker.start()

    def _on_detect_done(self, probs, smoothed, events):
        self.probs    = probs
        self.smoothed = smoothed
        self.events   = events
        self._btn_detect.setEnabled(True)
        self._lbl_detect_status.setText(f"Detected {len(events)} event segment(s)")

        cfg    = self.cfg
        sig    = self.signal
        lbl    = self.labels
        t      = np.arange(len(sig)) / cfg.sample_rate
        pred_s = preds_to_samples(smoothed, len(sig), cfg)
        offset = cfg.history_len - 1
        prob_t = np.array([
            ((i + offset) * cfg.stride + cfg.window_size // 2) / cfg.sample_rate
            for i in range(len(probs))
        ])

        # Signal plot
        self._plot_det_sig.clear()
        if lbl is not None:
            self._shade_regions(self._plot_det_sig, t, lbl,
                                QColor(166, 227, 161, 55))
        self._shade_regions(self._plot_det_sig, t, pred_s,
                            QColor(250, 179, 135, 70))
        self._plot_det_sig.plot(t, sig, pen=pg.mkPen(C_BLUE, width=1.2))

        # Probability plot
        self._plot_det_prob.clear()
        self._plot_det_prob.addLine(y=0.5, pen=pg.mkPen(C_MUTED, width=1,
                                    style=Qt.PenStyle.DashLine))
        self._plot_det_prob.plot(prob_t, probs,
                                 pen=pg.mkPen(C_ORANGE, width=1.5))
        fill = pg.FillBetweenItem(
            self._plot_det_prob.plot(prob_t, np.zeros_like(probs),
                                     pen=pg.mkPen(None)),
            self._plot_det_prob.plot(prob_t, probs,
                                     pen=pg.mkPen(None)),
            brush=pg.mkBrush(QColor(250, 179, 135, 50))
        )
        self._plot_det_prob.addItem(fill)

        # Event table
        self._tbl_events.setRowCount(0)
        for i, (s, e) in enumerate(events):
            self._tbl_events.insertRow(i)
            for j, val in enumerate([str(i+1), f"{s:.3f}", f"{e:.3f}", f"{e-s:.3f}"]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._tbl_events.setItem(i, j, item)

        # Refresh data tab signal plot too
        self._refresh_data_plots()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — EVALUATE
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_eval(self):
        w   = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        self.tabs.addTab(w, "📊  Evaluate")

        self._btn_eval = QPushButton("Compute metrics")
        lay.addWidget(self._btn_eval)

        # Metric cards
        self._eval_f1   = metric_card("—", "F1 Score")
        self._eval_prec = metric_card("—", "Precision")
        self._eval_rec  = metric_card("—", "Recall")
        mrow = QHBoxLayout()
        for c in [self._eval_f1, self._eval_prec, self._eval_rec]:
            mrow.addWidget(c)
        lay.addLayout(mrow)

        # Confusion matrix
        lay.addWidget(QLabel("Sample-level confusion matrix:"))
        self._tbl_conf = QTableWidget(2, 3)
        self._tbl_conf.setHorizontalHeaderLabels(["", "True event", "True non-event"])
        self._tbl_conf.setVerticalHeaderLabels(["", ""])
        self._tbl_conf.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_conf.setFixedHeight(100)
        lay.addWidget(self._tbl_conf)

        # Training curves (replicated from train tab)
        curves = QHBoxLayout()
        self._plot_eval_loss = make_plot_widget("Loss curves",
                                                x_label="Epoch", y_label="Loss", height=220)
        self._plot_eval_f1   = make_plot_widget("Validation F1",
                                                x_label="Epoch", y_label="F1",   height=220)
        self._plot_eval_loss.addLegend(offset=(10,10))
        curves.addWidget(self._plot_eval_loss)
        curves.addWidget(self._plot_eval_f1)
        lay.addLayout(curves)
        lay.addStretch()

        self._btn_eval.clicked.connect(self._on_evaluate)

    def _on_evaluate(self):
        if self.smoothed is None:
            return
        if self.labels is None:
            return
        cfg    = self.cfg
        sig    = self.signal
        lbl    = self.labels
        pred_s = preds_to_samples(self.smoothed, len(sig), cfg)
        n      = min(len(lbl), len(pred_s))
        y_true = lbl[:n]; y_pred = pred_s[:n]

        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        self._eval_f1.findChildren(QLabel)[0].setText(f"{f1:.4f}")
        self._eval_prec.findChildren(QLabel)[0].setText(f"{prec:.4f}")
        self._eval_rec.findChildren(QLabel)[0].setText(f"{rec:.4f}")

        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        tn = int(((y_true==0)&(y_pred==0)).sum())

        self._tbl_conf.setRowCount(2)
        for r, (row_lbl, tp_val, fp_val) in enumerate([
            ("Predicted event",     tp, fp),
            ("Predicted non-event", fn, tn),
        ]):
            self._tbl_conf.setItem(r, 0, QTableWidgetItem(row_lbl))
            self._tbl_conf.setItem(r, 1, QTableWidgetItem(str(tp_val)))
            self._tbl_conf.setItem(r, 2, QTableWidgetItem(str(fp_val)))
            for c in range(3):
                if self._tbl_conf.item(r,c):
                    self._tbl_conf.item(r,c).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Training curves
        if self.history:
            ep = list(range(1, len(self.history["train_loss"])+1))
            self._plot_eval_loss.clear()
            self._plot_eval_loss.addLegend(offset=(10,10))
            self._plot_eval_loss.plot(ep, self.history["train_loss"],
                                      pen=pg.mkPen(C_BLUE,  width=2), name="Train")
            self._plot_eval_loss.plot(ep, self.history["val_loss"],
                                      pen=pg.mkPen(C_RED,   width=2), name="Val")
            self._plot_eval_f1.clear()
            self._plot_eval_f1.plot(ep, self.history["val_f1"],
                                    pen=pg.mkPen(C_GREEN, width=2))


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
