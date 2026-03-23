"""
Event Detection — PyQtGraph Desktop GUI  v4
============================================
Changes in v4
  • Neon dark theme  (cyan / magenta / electric-green on deep black)
  • Fonts bumped to 14px base, 18px headings, 24px metric values
  • Sidebar reorganised into workflow order:
      1. Signal   2. Feature Engineering   3. Model (LSTM)   4. Detection
  • New  🖊 Label  tab — manually draw event regions on the signal:
      - Toggle draw mode → click start → click end of region
      - Regions are draggable / resizable (pyqtgraph LinearRegionItem)
      - Undo last · Clear all · Apply to signal · Export .npy
"""

import sys, io, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
from numpy import (sin, cos, tan, exp, log, log2, log10,
                   sqrt, abs, pi, inf, sign, floor, ceil,
                   linspace, arange, zeros, ones)
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import pywt
from sklearn.metrics import f1_score, precision_score, recall_score
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QProgressBar, QTextEdit,
    QFileDialog, QSplitter, QScrollArea, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QCursor
import pyqtgraph as pg

pg.setConfigOption("background", "#050510")
pg.setConfigOption("foreground", "#e0e0ff")
pg.setConfigOption("antialias",  True)
pg.setConfigOption("useNumba",   False)

# ── Neon palette ──────────────────────────────────────────────────────────────
C_BG       = "#050510"
C_BG2      = "#0d0d1f"
C_SURFACE  = "#12122a"
C_BORDER   = "#2a2a50"
C_BORDER2  = "#3a3a70"
C_CYAN     = "#00f5ff"
C_MAGENTA  = "#ff00cc"
C_GREEN    = "#00ff88"
C_YELLOW   = "#ffe600"
C_PURPLE   = "#bb66ff"
C_ORANGE   = "#ff8800"
C_TEXT     = "#dde4ff"
C_MUTED    = "#6068a0"
C_DIM      = "#2e3060"


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
    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer:       str   = "Adam"      # Adam | AdamW | SGD | RMSprop | LBFGS
    weight_decay:    float = 0.0         # L2 regularisation
    momentum:        float = 0.9         # SGD / RMSprop only
    # ── lr scheduler ──────────────────────────────────────────────────────────
    scheduler:       str   = "ReduceLROnPlateau"
    sched_patience:  int   = 4           # ReduceLROnPlateau
    sched_factor:    float = 0.5         # ReduceLROnPlateau
    sched_t_max:     int   = 10          # CosineAnnealingLR restart period
    sched_pct_start: float = 0.3         # OneCycleLR warm-up fraction
    sched_max_lr:    float = 0.01        # OneCycleLR peak LR
    # ── gradient clipping ─────────────────────────────────────────────────────
    grad_clip:       float = 1.0         # 0.0 = disabled

    def feature_dim(self) -> int:
        return 6 + 5 + self.wavelet_levels


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  2. SIGNAL SOURCES                                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def generate_preset_signal(cfg: Config, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n   = int(cfg.signal_duration * cfg.sample_rate)
    t   = np.arange(n) / cfg.sample_rate
    sig = np.sin(2 * np.pi * cfg.baseline_freq * t)
    sig += rng.normal(0, cfg.noise_std, n)
    lbl = np.zeros(n, dtype=np.float32)
    ev  = int(cfg.event_duration * cfg.sample_rate)
    placed, attempts, used = 0, 0, []
    while placed < cfg.num_events and attempts < 2000:
        attempts += 1
        s = int(rng.integers(0, n - ev))
        e = s + ev
        if any(not (e + 200 <= us or s >= ue + 200) for us, ue in used):
            continue
        ev_t     = np.arange(ev) / cfg.sample_rate
        sig[s:e] += cfg.event_amplitude * np.sin(2 * np.pi * cfg.event_freq * ev_t)
        lbl[s:e]  = 1.0
        used.append((s, e)); placed += 1
    return sig.astype(np.float32), lbl


def evaluate_signal_expr(expr, lbl_expr, duration, sample_rate,
                          noise_std=0.0, seed=42):
    n  = int(duration * sample_rate)
    t  = np.arange(n) / sample_rate
    sr = sample_rate
    ns = {
        "__builtins__": {},
        "t": t, "n": n, "sr": sr, "np": np,
        "sin": sin, "cos": cos, "tan": tan,
        "exp": exp, "log": log, "log2": log2, "log10": log10,
        "sqrt": sqrt, "abs": abs, "pi": pi, "inf": inf,
        "sign": sign, "floor": floor, "ceil": ceil,
        "linspace": linspace, "arange": arange,
        "zeros": zeros, "ones": ones, "e": np.e,
    }
    sig = np.array(eval(expr, ns), dtype=np.float32)
    if sig.shape != (n,):
        raise ValueError(f"Shape {sig.shape} != ({n},)")
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        sig += rng.normal(0, noise_std, n).astype(np.float32)
    lbl = None
    if lbl_expr.strip():
        raw = np.array(eval(lbl_expr, ns), dtype=np.float32)
        lbl = (raw > 0).astype(np.float32)
        if lbl.shape != (n,):
            raise ValueError(f"Label shape {lbl.shape} != ({n},)")
    return sig, lbl


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  3. WINDOWING + FEATURES + SEQUENCES                                        ║
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

def _stat(w):
    return np.array([w.mean(), w.std(), float(skew(w)),
                     float(kurtosis(w)), w.min(), w.max()], dtype=np.float32)

def _freq(w, sr):
    c = np.abs(rfft(w)); f = rfftfreq(len(w), 1.0/sr)
    eng = float(np.sum(c**2)) + 1e-8
    return np.array([eng, float(f[np.argmax(c)]),
                     float(np.sum(f*c)/(np.sum(c)+1e-8)),
                     float(np.sum(c[f<20]**2))/eng,
                     float(np.sum(c[(f>=20)&(f<100)]**2))/eng], dtype=np.float32)

def _wav(w, wavelet, levels):
    return np.array([float(np.sum(c**2))
                     for c in pywt.wavedec(w, wavelet, level=levels)[:levels]],
                    dtype=np.float32)

def extract_features(w, cfg):
    return np.concatenate([_stat(w), _freq(w, cfg.sample_rate),
                           _wav(w, cfg.wavelet, cfg.wavelet_levels)])

def build_feature_matrix(windows, cfg):
    return np.stack([extract_features(w, cfg) for w in windows])

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
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  4. LSTM                                                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class EventLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg.feature_dim(), cfg.hidden_size, cfg.num_layers,
                            batch_first=True,
                            dropout=cfg.dropout if cfg.num_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden_size, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  5. WORKERS                                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TrainWorker(QThread):
    progress   = pyqtSignal(int)
    log_line   = pyqtSignal(str)
    epoch_done = pyqtSignal(float, float, float)
    lr_updated = pyqtSignal(float)           # emitted each epoch with current LR
    finished   = pyqtSignal(object, object)
    error      = pyqtSignal(str)

    def __init__(self, cfg, signal, labels, val_ratio=0.2):
        super().__init__()
        self.cfg=cfg; self.signal=signal; self.labels=labels
        self.val_ratio=val_ratio; self._stop=False

    def stop(self): self._stop = True

    def run(self):
        try:
            cfg = self.cfg
            self.log_line.emit("Windowing…")
            W, Wl  = sliding_windows(self.signal, self.labels, cfg)
            self.log_line.emit(f"Features from {len(W):,} windows…")
            feat   = build_feature_matrix(W, cfg)
            self.log_line.emit("Building sequences…")
            X, y   = build_sequences(feat, Wl, cfg.history_len)
            split  = int(len(X)*(1-self.val_ratio))
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]
            pos=int(y_tr.sum()); neg=len(y_tr)-pos
            self.log_line.emit(
                f"Train {len(X_tr):,} · Val {len(X_val):,} · "
                f"feat_dim {X.shape[-1]} · pos/neg {pos}/{neg}")
            model  = EventLSTM(cfg)
            loader = DataLoader(EventDataset(X_tr, y_tr),
                                batch_size=cfg.batch_size, shuffle=True)
            crit   = nn.BCEWithLogitsLoss(
                         pos_weight=torch.tensor([cfg.pos_weight]))

            # ── build optimizer ───────────────────────────────────────────────
            base_kw = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            if cfg.optimizer == "Adam":
                opt = torch.optim.Adam(model.parameters(), **base_kw)
            elif cfg.optimizer == "AdamW":
                opt = torch.optim.AdamW(model.parameters(), **base_kw)
            elif cfg.optimizer == "SGD":
                opt = torch.optim.SGD(model.parameters(),
                                      momentum=cfg.momentum, **base_kw)
            elif cfg.optimizer == "RMSprop":
                opt = torch.optim.RMSprop(model.parameters(),
                                          momentum=cfg.momentum, **base_kw)
            elif cfg.optimizer == "LBFGS":
                # LBFGS ignores weight_decay & momentum in standard form
                opt = torch.optim.LBFGS(model.parameters(),
                                        lr=cfg.learning_rate, max_iter=20)
            else:
                opt = torch.optim.Adam(model.parameters(), **base_kw)

            self.log_line.emit(
                f"Optimizer: {cfg.optimizer}  lr={cfg.learning_rate}"
                + (f"  wd={cfg.weight_decay}" if cfg.weight_decay>0 else "")
                + (f"  momentum={cfg.momentum}"
                   if cfg.optimizer in ("SGD","RMSprop") else ""))

            # ── build lr scheduler ────────────────────────────────────────────
            n_steps = len(loader) * cfg.epochs
            if cfg.scheduler == "ReduceLROnPlateau":
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            opt, patience=cfg.sched_patience,
                            factor=cfg.sched_factor, verbose=False)
            elif cfg.scheduler == "CosineAnnealingLR":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=cfg.sched_t_max, eta_min=1e-6)
            elif cfg.scheduler == "OneCycleLR":
                sched = torch.optim.lr_scheduler.OneCycleLR(
                            opt, max_lr=cfg.sched_max_lr,
                            total_steps=n_steps,
                            pct_start=cfg.sched_pct_start)
            elif cfg.scheduler == "StepLR":
                sched = torch.optim.lr_scheduler.StepLR(
                            opt, step_size=max(1, cfg.epochs//5), gamma=0.5)
            elif cfg.scheduler == "ExponentialLR":
                sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
            elif cfg.scheduler == "None":
                sched = None
            else:
                sched = None

            self.log_line.emit(f"Scheduler: {cfg.scheduler}  "
                               f"grad_clip={cfg.grad_clip if cfg.grad_clip>0 else 'off'}")

            history = {"train_loss":[], "val_loss":[], "val_f1":[], "lr":[]}
            step = 0
            for epoch in range(cfg.epochs):
                if self._stop:
                    self.log_line.emit("⛔  Stopped."); return
                model.train(); ep_loss=0.0
                for xb, yb in loader:
                    if cfg.optimizer == "LBFGS":
                        # LBFGS requires a closure
                        def closure():
                            opt.zero_grad()
                            out = model(xb)
                            l   = crit(out, yb)
                            l.backward()
                            return l
                        loss = opt.step(closure)
                        ep_loss += loss.item()*len(xb)
                    else:
                        opt.zero_grad()
                        loss = crit(model(xb), yb)
                        loss.backward()
                        if cfg.grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(),
                                                     cfg.grad_clip)
                        opt.step()
                        ep_loss += loss.item()*len(xb)
                    # step-level schedulers
                    if isinstance(sched,
                        torch.optim.lr_scheduler.OneCycleLR):
                        sched.step()
                    step += 1
                ep_loss /= len(X_tr)

                model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_val).float()
                    yv = torch.from_numpy(y_val).float()
                    vl = crit(model(xv), yv).item()
                    prd = (torch.sigmoid(model(xv)) >= cfg.threshold).numpy()
                    f1  = f1_score(y_val, prd, zero_division=0)

                # epoch-level schedulers
                if sched is not None and not isinstance(
                        sched, torch.optim.lr_scheduler.OneCycleLR):
                    if isinstance(sched,
                            torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched.step(vl)
                    else:
                        sched.step()

                cur_lr = opt.param_groups[0]["lr"]
                history["train_loss"].append(ep_loss)
                history["val_loss"].append(vl)
                history["val_f1"].append(f1)
                history["lr"].append(cur_lr)
                self.progress.emit(int((epoch+1)/cfg.epochs*100))
                self.log_line.emit(
                    f"Epoch {epoch+1:>3}/{cfg.epochs}  "
                    f"train={ep_loss:.4f}  val={vl:.4f}  "
                    f"F1={f1:.3f}  lr={cur_lr:.2e}")
                self.epoch_done.emit(ep_loss, vl, f1)
                self.lr_updated.emit(cur_lr)
            self.finished.emit(model, history)
        except Exception as e:
            self.error.emit(str(e))


class DetectWorker(QThread):
    finished = pyqtSignal(object, object, object)
    error    = pyqtSignal(str)

    def __init__(self, model, signal, cfg):
        super().__init__()
        self.model=model; self.signal=signal; self.cfg=cfg

    def run(self):
        try:
            cfg=self.cfg; model=self.model; model.eval()
            W, _  = sliding_windows(self.signal, None, cfg)
            feat  = build_feature_matrix(W, cfg)
            X, _  = build_sequences(feat, None, cfg.history_len)
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.from_numpy(X).float())).numpy()
            smoothed = np.zeros_like(probs)
            hw = cfg.smooth_window//2
            for i in range(len(probs)):
                lo,hi = max(0,i-hw), min(len(probs),i+hw+1)
                smoothed[i] = float(probs[lo:hi].mean() >= cfg.threshold)
            events, in_ev, sw = [], False, 0
            for i, v in enumerate(smoothed):
                if v==1 and not in_ev:   in_ev,sw=True,i
                elif v==0 and in_ev:
                    in_ev=False
                    t0=(sw+cfg.history_len-1)*cfg.stride/cfg.sample_rate
                    t1=(i +cfg.history_len-1)*cfg.stride/cfg.sample_rate
                    events.append((t0,t1))
            if in_ev:
                t0=(sw           +cfg.history_len-1)*cfg.stride/cfg.sample_rate
                t1=(len(smoothed)+cfg.history_len-1)*cfg.stride/cfg.sample_rate
                events.append((t0,t1))
            self.finished.emit(probs, smoothed, events)
        except Exception as e:
            self.error.emit(str(e))


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  6. HELPERS                                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def preds_to_samples(smoothed, signal_len, cfg):
    out = np.zeros(signal_len, dtype=np.float32)
    offset = cfg.history_len-1
    for i, v in enumerate(smoothed):
        if v==1:
            s = (i+offset)*cfg.stride
            e = min(s+cfg.window_size, signal_len)
            out[s:e] = 1.0
    return out


def shade_regions(pw, t, labels, color: QColor, movable=False):
    padded = np.concatenate([[0], labels.astype(np.int8), [0]])
    diff   = np.diff(padded)
    starts = np.where(diff== 1)[0]
    ends   = np.where(diff==-1)[0]
    brush  = pg.mkBrush(color)
    items  = []
    for s, e in zip(starts, ends):
        ts = t[min(s,   len(t)-1)]
        te = t[min(e-1, len(t)-1)]
        item = pg.LinearRegionItem(
            values=[ts, te], brush=brush,
            pen=pg.mkPen(None), movable=movable)
        pw.addItem(item); items.append(item)
    return items


def make_plot(title="", x_label="Time (s)", y_label="Amplitude",
              height=None) -> pg.PlotWidget:
    from PyQt6.QtGui import QFont as _QFont
    pw = pg.PlotWidget()
    pw.setBackground(C_BG)
    pw.showGrid(x=True, y=True, alpha=0.12)
    pw.setLabel("bottom", x_label, color=C_CYAN,  size="11pt")
    pw.setLabel("left",   y_label, color=C_CYAN,  size="11pt")
    if title: pw.setTitle(title, color=C_TEXT, size="11pt")
    if height: pw.setFixedHeight(height)
    tick_font = _QFont("Helvetica Neue", 10)
    for axis in ("bottom", "left"):
        ax = pw.getAxis(axis)
        ax.setPen(pg.mkPen(color=C_CYAN, width=1))   # spine + tick marks
        ax.setTextPen(pg.mkPen(color=C_TEXT))         # tick number colour
        ax.setTickFont(tick_font)                      # tick number size
        ax.setStyle(tickLength=-8)                     # inward ticks, 8 px
    pw.setClipToView(True)
    pw.setDownsampling(mode="peak", auto=True)
    return pw


def metric_card(value: str, label: str) -> QFrame:
    card = QFrame(); card.setObjectName("neon_card")
    lay  = QVBoxLayout(card); lay.setContentsMargins(16,14,16,14)
    v = QLabel(value); v.setObjectName("metric_val")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    l = QLabel(label); l.setObjectName("metric_lbl")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v); lay.addWidget(l)
    return card


def add_row(grid, row, label, widget):
    lbl = QLabel(label); lbl.setObjectName("sidebar_lbl")
    grid.addWidget(lbl, row, 0)
    grid.addWidget(widget, row, 1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  7. NEON STYLE                                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

STYLE = f"""
QWidget {{
    background:{C_BG}; color:{C_TEXT};
    font-family:"Helvetica Neue","Arial","Segoe UI",sans-serif;
    font-size:14px;
}}
QTabWidget::pane {{ border:1px solid {C_BORDER2}; border-radius:6px; background:{C_BG2}; }}
QTabBar::tab {{
    background:{C_BG2}; color:{C_MUTED}; padding:9px 22px;
    border-top-left-radius:6px; border-top-right-radius:6px;
    margin-right:2px; font-size:14px;
}}
QTabBar::tab:selected {{ background:{C_BG}; color:{C_CYAN}; border-bottom:2px solid {C_CYAN}; }}
QTabBar::tab:hover:!selected {{ color:{C_TEXT}; background:{C_SURFACE}; }}
QGroupBox {{
    border:1px solid {C_BORDER2}; border-radius:6px;
    margin-top:12px; padding-top:10px;
    color:{C_CYAN}; font-weight:bold; font-size:13px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background:{C_SURFACE}; border:1px solid {C_BORDER2};
    border-radius:4px; padding:5px 8px; color:{C_TEXT}; font-size:14px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border:1px solid {C_CYAN}; }}
QComboBox::drop-down {{ border:none; }}
QComboBox QAbstractItemView {{
    background:{C_SURFACE}; border:1px solid {C_BORDER2};
    color:{C_TEXT}; selection-background-color:{C_CYAN}44;
}}
QPushButton {{
    background:{C_SURFACE}; color:{C_CYAN}; border:1px solid {C_CYAN};
    border-radius:6px; padding:8px 18px; font-weight:bold; font-size:14px;
}}
QPushButton:hover   {{ background:{C_CYAN}22; }}
QPushButton:pressed {{ background:{C_CYAN}44; }}
QPushButton:disabled {{ background:{C_SURFACE}; color:{C_DIM}; border:1px solid {C_DIM}; }}
QPushButton#danger  {{ color:{C_MAGENTA}; border-color:{C_MAGENTA}; }}
QPushButton#danger:hover  {{ background:{C_MAGENTA}22; }}
QPushButton#success {{ color:{C_GREEN}; border-color:{C_GREEN}; }}
QPushButton#success:hover {{ background:{C_GREEN}22; }}
QPushButton#warn    {{ color:{C_YELLOW}; border-color:{C_YELLOW}; }}
QPushButton#warn:hover    {{ background:{C_YELLOW}22; }}
QProgressBar {{
    border:1px solid {C_BORDER2}; border-radius:4px;
    background:{C_SURFACE}; text-align:center; color:{C_TEXT};
    font-size:13px; height:18px;
}}
QProgressBar::chunk {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_CYAN}, stop:1 {C_PURPLE});
    border-radius:4px;
}}
QTextEdit {{
    background:{C_BG2}; border:1px solid {C_BORDER2}; border-radius:4px;
    font-family:"Menlo","Consolas",monospace; font-size:15px; color:{C_GREEN};
}}
QTableWidget {{
    background:{C_SURFACE}; border:1px solid {C_BORDER2};
    gridline-color:{C_BORDER}; border-radius:4px; font-size:14px;
}}
QTableWidget::item          {{ padding:5px 10px; color:{C_TEXT}; }}
QTableWidget::item:selected {{ background:{C_CYAN}33; color:{C_CYAN}; }}
QHeaderView::section {{
    background:{C_BG2}; color:{C_CYAN}; border:none;
    border-bottom:1px solid {C_BORDER2};
    padding:7px 10px; font-weight:bold; font-size:13px;
}}
QScrollBar:vertical {{ background:{C_BG2}; width:8px; border-radius:4px; }}
QScrollBar::handle:vertical {{ background:{C_BORDER2}; border-radius:4px; min-height:20px; }}
QScrollBar::handle:vertical:hover {{ background:{C_CYAN}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QLabel#title      {{ font-size:18px; font-weight:bold; color:{C_CYAN}; }}
QLabel#subtitle   {{ font-size:12px; color:{C_MUTED}; }}
QLabel#metric_val {{ font-size:24px; font-weight:bold; color:{C_CYAN}; }}
QLabel#metric_lbl {{ font-size:12px; color:{C_MUTED}; }}
QLabel#hint       {{ font-size:12px; color:{C_MUTED}; font-style:italic; }}
QLabel#sidebar_lbl{{ font-size:13px; color:{C_MUTED}; }}
QFrame#neon_card  {{ background:{C_BG2}; border:1px solid {C_CYAN}; border-radius:8px; }}
QSplitter::handle {{ background:{C_BORDER2}; height:2px; }}
"""


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  8. MAIN WINDOW                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Detector — Neon v4")
        self.resize(1500, 960)
        self.setStyleSheet(STYLE)

        self.signal        = None
        self.labels        = None
        self.model         = None
        self.history       = None
        self.probs         = None
        self.smoothed      = None
        self.events        = []
        self.cfg           = Config()
        self._train_worker = None
        self._det_worker   = None
        self._draw_mode    = False
        self._drag_start_t = None
        self._manual_regions: List[Tuple[float,float]] = []
        self._manual_items   = []

        root  = QWidget(); self.setCentralWidget(root)
        h_lay = QHBoxLayout(root)
        h_lay.setContentsMargins(0,0,0,0); h_lay.setSpacing(0)

        sidebar = self._build_sidebar(); sidebar.setFixedWidth(295)
        h_lay.addWidget(sidebar)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"background:{C_BORDER2}; max-width:1px;")
        h_lay.addWidget(sep)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        h_lay.addWidget(self.tabs, 1)

        self._build_tab_data()
        self._build_tab_label()
        self._build_tab_train()
        self._build_tab_detect()
        self._build_tab_eval()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — workflow order: Signal → Features → Model → Detection
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"QWidget{{background:{C_BG2};}}")
        lay = QVBoxLayout(panel); lay.setContentsMargins(12,16,12,12); lay.setSpacing(4)
        hdr = QLabel("⚙  Parameters"); hdr.setObjectName("title"); lay.addWidget(hdr)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner)
        il.setSpacing(4); il.setContentsMargins(0,4,0,4)
        scroll.setWidget(inner); lay.addWidget(scroll, 1)

        def section(icon, name):
            gb = QGroupBox(f"{icon}  {name}")
            g  = QGridLayout(gb); g.setColumnStretch(1,1)
            g.setVerticalSpacing(6); g.setHorizontalSpacing(8)
            il.addWidget(gb); return g

        # 1. Signal
        g = section("📡","Signal")
        self._w_dur     = QDoubleSpinBox(); self._w_dur.setRange(1,120);    self._w_dur.setValue(10.0); self._w_dur.setSingleStep(1)
        self._w_bfreq   = QDoubleSpinBox(); self._w_bfreq.setRange(1,20);   self._w_bfreq.setValue(5.0); self._w_bfreq.setSingleStep(1)
        self._w_noise   = QDoubleSpinBox(); self._w_noise.setRange(0,2);    self._w_noise.setValue(0.2); self._w_noise.setSingleStep(0.05)
        self._w_nevents = QSpinBox();       self._w_nevents.setRange(1,8);  self._w_nevents.setValue(3)
        self._w_efreq   = QDoubleSpinBox(); self._w_efreq.setRange(20,450); self._w_efreq.setValue(80); self._w_efreq.setSingleStep(5)
        self._w_amp     = QDoubleSpinBox(); self._w_amp.setRange(0.5,5);    self._w_amp.setValue(2.0); self._w_amp.setSingleStep(0.25)
        self._w_seed    = QSpinBox();       self._w_seed.setRange(0,9999);  self._w_seed.setValue(42)
        for r,(lbl,w) in enumerate([("Duration (s)",self._w_dur),("Baseline Hz",self._w_bfreq),
                                     ("Noise std",self._w_noise),("# events",self._w_nevents),
                                     ("Event Hz",self._w_efreq),("Event amp",self._w_amp),
                                     ("Seed",self._w_seed)]):
            add_row(g,r,lbl,w)

        # 2. Feature Engineering
        g = section("🔬","Feature Engineering")
        self._w_win = QComboBox(); [self._w_win.addItem(str(x)) for x in [64,128,256,512]]; self._w_win.setCurrentText("256")
        self._w_str = QComboBox(); [self._w_str.addItem(str(x)) for x in [16,32,64,128]];   self._w_str.setCurrentText("64")
        self._w_wav = QComboBox(); [self._w_wav.addItem(x) for x in ["db4","db6","db8","sym4","coif2","haar"]]
        self._w_wlv = QSpinBox();  self._w_wlv.setRange(1,6); self._w_wlv.setValue(4)
        for r,(lbl,w) in enumerate([("Window",self._w_win),("Stride",self._w_str),
                                     ("Wavelet",self._w_wav),("Wav levels",self._w_wlv)]):
            add_row(g,r,lbl,w)

        # 3. Model
        g = section("🧠","Model  (LSTM)")
        self._w_hist = QSpinBox();       self._w_hist.setRange(2,30);  self._w_hist.setValue(10)
        self._w_hid  = QComboBox();      [self._w_hid.addItem(str(x)) for x in [32,64,128,256]]; self._w_hid.setCurrentText("64")
        self._w_lay  = QSpinBox();       self._w_lay.setRange(1,4);   self._w_lay.setValue(2)
        self._w_drop = QDoubleSpinBox(); self._w_drop.setRange(0,0.6); self._w_drop.setValue(0.3); self._w_drop.setSingleStep(0.05)
        self._w_ep   = QSpinBox();       self._w_ep.setRange(5,200);  self._w_ep.setValue(30)
        self._w_bs   = QComboBox();      [self._w_bs.addItem(str(x)) for x in [8,16,32,64]]; self._w_bs.setCurrentText("32")
        self._w_lr   = QComboBox();      [self._w_lr.addItem(x) for x in ["0.0001","0.0005","0.001","0.005"]]; self._w_lr.setCurrentText("0.001")
        self._w_pw   = QDoubleSpinBox(); self._w_pw.setRange(1,10);   self._w_pw.setValue(4.0); self._w_pw.setSingleStep(0.5)
        for r,(lbl,w) in enumerate([("History",self._w_hist),("Hidden",self._w_hid),
                                     ("Layers",self._w_lay),("Dropout",self._w_drop),
                                     ("Epochs",self._w_ep),("Batch",self._w_bs),
                                     ("LR",self._w_lr),("Pos weight",self._w_pw)]):
            add_row(g,r,lbl,w)

        # Optimizer sub-section
        g2 = section("⚡","Optimizer & Scheduler")
        self._w_opt  = QComboBox()
        for x in ["Adam","AdamW","SGD","RMSprop","LBFGS"]: self._w_opt.addItem(x)
        self._w_wd   = QDoubleSpinBox(); self._w_wd.setRange(0,0.1);   self._w_wd.setValue(0.0);  self._w_wd.setSingleStep(0.0001); self._w_wd.setDecimals(5)
        self._w_mom  = QDoubleSpinBox(); self._w_mom.setRange(0,0.99); self._w_mom.setValue(0.9); self._w_mom.setSingleStep(0.05)
        self._w_clip = QDoubleSpinBox(); self._w_clip.setRange(0,10);  self._w_clip.setValue(1.0); self._w_clip.setSingleStep(0.5)
        self._w_sched = QComboBox()
        for x in ["ReduceLROnPlateau","CosineAnnealingLR","OneCycleLR",
                   "StepLR","ExponentialLR","None"]:
            self._w_sched.addItem(x)
        self._w_spat  = QSpinBox();       self._w_spat.setRange(1,20);   self._w_spat.setValue(4)
        self._w_sfact = QDoubleSpinBox(); self._w_sfact.setRange(0.1,0.9); self._w_sfact.setValue(0.5); self._w_sfact.setSingleStep(0.05)
        self._w_stmax = QSpinBox();       self._w_stmax.setRange(1,100);  self._w_stmax.setValue(10)
        self._w_spct  = QDoubleSpinBox(); self._w_spct.setRange(0.05,0.5); self._w_spct.setValue(0.3); self._w_spct.setSingleStep(0.05)
        self._w_smlr  = QDoubleSpinBox(); self._w_smlr.setRange(0.001,0.1); self._w_smlr.setValue(0.01); self._w_smlr.setSingleStep(0.005)
        for r,(lbl,w) in enumerate([
            ("Optimizer",  self._w_opt),
            ("Weight decay", self._w_wd),
            ("Momentum",   self._w_mom),
            ("Grad clip",  self._w_clip),
            ("Scheduler",  self._w_sched),
            ("Patience",   self._w_spat),
            ("LR factor",  self._w_sfact),
            ("T-max",      self._w_stmax),
            ("Pct start",  self._w_spct),
            ("Max LR",     self._w_smlr),
        ]):
            add_row(g2,r,lbl,w)
        # show/hide scheduler-specific rows based on selection
        self._sched_plateau_rows = [5,6]
        self._sched_cosine_rows  = [7]
        self._sched_onecycle_rows= [8,9]
        self._w_sched.currentTextChanged.connect(self._on_sched_changed)
        self._on_sched_changed(self._w_sched.currentText())  # init visibility
        self._sched_grid = g2

        # 4. Detection
        g = section("🔍","Detection")
        self._w_thr = QDoubleSpinBox(); self._w_thr.setRange(0.1,0.9); self._w_thr.setValue(0.5); self._w_thr.setSingleStep(0.05)
        self._w_smw = QSpinBox();       self._w_smw.setRange(1,15);   self._w_smw.setValue(5)
        add_row(g,0,"Threshold",self._w_thr)
        add_row(g,1,"Smooth w", self._w_smw)

        il.addStretch()
        return panel

    def _read_cfg(self) -> Config:
        return Config(
            signal_duration=self._w_dur.value(), baseline_freq=self._w_bfreq.value(),
            noise_std=self._w_noise.value(), num_events=self._w_nevents.value(),
            event_freq=self._w_efreq.value(), event_amplitude=self._w_amp.value(),
            window_size=int(self._w_win.currentText()), stride=int(self._w_str.currentText()),
            wavelet=self._w_wav.currentText(), wavelet_levels=self._w_wlv.value(),
            history_len=self._w_hist.value(), hidden_size=int(self._w_hid.currentText()),
            num_layers=self._w_lay.value(), dropout=self._w_drop.value(),
            epochs=self._w_ep.value(), batch_size=int(self._w_bs.currentText()),
            learning_rate=float(self._w_lr.currentText()), pos_weight=self._w_pw.value(),
            threshold=self._w_thr.value(), smooth_window=self._w_smw.value(),
            optimizer=self._w_opt.currentText(),
            weight_decay=self._w_wd.value(),
            momentum=self._w_mom.value(),
            grad_clip=self._w_clip.value(),
            scheduler=self._w_sched.currentText(),
            sched_patience=self._w_spat.value(),
            sched_factor=self._w_sfact.value(),
            sched_t_max=self._w_stmax.value(),
            sched_pct_start=self._w_spct.value(),
            sched_max_lr=self._w_smlr.value(),
        )

    def _apply_cfg_to_sidebar(self, cfg: Config):
        self._w_dur.setValue(cfg.signal_duration); self._w_bfreq.setValue(cfg.baseline_freq)
        self._w_noise.setValue(cfg.noise_std);     self._w_nevents.setValue(cfg.num_events)
        self._w_efreq.setValue(cfg.event_freq);    self._w_amp.setValue(cfg.event_amplitude)
        self._w_win.setCurrentText(str(cfg.window_size)); self._w_str.setCurrentText(str(cfg.stride))
        self._w_wav.setCurrentText(cfg.wavelet);   self._w_wlv.setValue(cfg.wavelet_levels)
        self._w_hist.setValue(cfg.history_len);    self._w_hid.setCurrentText(str(cfg.hidden_size))
        self._w_lay.setValue(cfg.num_layers);      self._w_drop.setValue(cfg.dropout)
        self._w_ep.setValue(cfg.epochs);           self._w_bs.setCurrentText(str(cfg.batch_size))
        self._w_lr.setCurrentText(str(cfg.learning_rate)); self._w_pw.setValue(cfg.pos_weight)
        self._w_thr.setValue(cfg.threshold);       self._w_smw.setValue(cfg.smooth_window)
        self._w_opt.setCurrentText(cfg.optimizer)
        self._w_wd.setValue(cfg.weight_decay);     self._w_mom.setValue(cfg.momentum)
        self._w_clip.setValue(cfg.grad_clip)
        self._w_sched.setCurrentText(cfg.scheduler)
        self._w_spat.setValue(cfg.sched_patience); self._w_sfact.setValue(cfg.sched_factor)
        self._w_stmax.setValue(cfg.sched_t_max);   self._w_spct.setValue(cfg.sched_pct_start)
        self._w_smlr.setValue(cfg.sched_max_lr)
        self._on_sched_changed(cfg.scheduler)

    def _on_sched_changed(self, name: str):
        """Show only the rows relevant to the selected scheduler."""
        plateau_widgets  = [self._w_spat,  self._w_sfact]
        cosine_widgets   = [self._w_stmax]
        onecycle_widgets = [self._w_spct,  self._w_smlr]
        all_extra = plateau_widgets + cosine_widgets + onecycle_widgets
        for w in all_extra:
            w.setEnabled(False)
            w.setStyleSheet(f"color:#2e3060; border-color:#2e3060;")
        if name == "ReduceLROnPlateau":
            for w in plateau_widgets:
                w.setEnabled(True); w.setStyleSheet("")
        elif name == "CosineAnnealingLR":
            for w in cosine_widgets:
                w.setEnabled(True); w.setStyleSheet("")
        elif name == "OneCycleLR":
            for w in onecycle_widgets:
                w.setEnabled(True); w.setStyleSheet("")
        # StepLR / ExponentialLR / None — no extra params needed

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_data(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)
        self.tabs.addTab(w, "📂  Data")

        pg_box = QGroupBox("Preset generator")
        pb = QHBoxLayout(pg_box)
        self._btn_gen    = QPushButton("Generate preset")
        self._btn_upload = QPushButton("Upload file…"); self._btn_upload.setObjectName("warn")
        pb.addWidget(self._btn_gen); pb.addWidget(self._btn_upload); pb.addStretch()
        lay.addWidget(pg_box)

        comp_box = QGroupBox("Signal composer  (numpy · variable  t  = seconds)")
        cl = QGridLayout(comp_box); cl.setColumnStretch(1,1)
        self._comp_expr = QLineEdit()
        self._comp_expr.setPlaceholderText("e.g.  sin(2*pi*5*t) + 2*sin(2*pi*80*t)*((t>=3)&(t<4))")
        self._comp_expr.setText(
            "sin(2*pi*5*t) + 2*sin(2*pi*80*(t-3))*((t>=3)&(t<4))"
            " - 1.5*sin(2*pi*80*(t-7))*((t>=7)&(t<8))")
        self._comp_lbl_expr = QLineEdit()
        self._comp_lbl_expr.setPlaceholderText("Label expr (optional)  e.g.  ((t>=3)&(t<4))|((t>=7)&(t<8))")
        self._comp_lbl_expr.setText("((t>=3)&(t<4))|((t>=7)&(t<8))")
        self._comp_dur   = QDoubleSpinBox(); self._comp_dur.setRange(1,300);   self._comp_dur.setValue(10.0); self._comp_dur.setSingleStep(1)
        self._comp_sr    = QSpinBox();       self._comp_sr.setRange(100,48000); self._comp_sr.setValue(1000); self._comp_sr.setSingleStep(100)
        self._comp_noise = QDoubleSpinBox(); self._comp_noise.setRange(0,2);   self._comp_noise.setValue(0.1); self._comp_noise.setSingleStep(0.05)
        hint = QLabel("sin  cos  exp  log  sqrt  abs  pi  e  sign  np  ·  gate with  ((t>=a)&(t<b))")
        hint.setObjectName("hint"); hint.setWordWrap(True)
        self._btn_compose = QPushButton("Build signal from expression")
        cl.addWidget(QLabel("Signal expr:"), 0,0); cl.addWidget(self._comp_expr,     0,1,1,3)
        cl.addWidget(QLabel("Label expr:"),  1,0); cl.addWidget(self._comp_lbl_expr, 1,1,1,3)
        cl.addWidget(QLabel("Duration:"),    2,0); cl.addWidget(self._comp_dur,      2,1)
        cl.addWidget(QLabel("Sample rate:"), 2,2); cl.addWidget(self._comp_sr,       2,3)
        cl.addWidget(QLabel("Noise std:"),   3,0); cl.addWidget(self._comp_noise,    3,1)
        cl.addWidget(hint,                   4,0,1,4)
        cl.addWidget(self._btn_compose,      5,0,1,4)
        lay.addWidget(comp_box)

        mrow = QHBoxLayout()
        self._card_dur   = metric_card("—","Duration")
        self._card_sr    = metric_card("—","Sample rate")
        self._card_evts  = metric_card("—","Event time")
        self._card_ratio = metric_card("—","Event ratio")
        for c in [self._card_dur,self._card_sr,self._card_evts,self._card_ratio]:
            mrow.addWidget(c)
        lay.addLayout(mrow)

        self._plot_sig = make_plot("Time domain  (cyan = signal · green = truth · magenta = predicted)")
        self._plot_fft = make_plot("Frequency domain", x_label="Frequency (Hz)", y_label="Magnitude")

        # vertical splitter so both plots are resizable
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        plot_splitter.addWidget(self._plot_sig)
        plot_splitter.addWidget(self._plot_fft)
        plot_splitter.setSizes([320, 220])
        lay.addWidget(plot_splitter, 1)
        self._lbl_status = QLabel("No signal loaded."); self._lbl_status.setObjectName("subtitle")
        lay.addWidget(self._lbl_status)

        self._btn_gen.clicked.connect(self._on_generate_preset)
        self._btn_upload.clicked.connect(self._on_upload)
        self._btn_compose.clicked.connect(self._on_compose)

    def _on_generate_preset(self):
        self.cfg = self._read_cfg()
        sig, lbl = generate_preset_signal(self.cfg, seed=self._w_seed.value())
        self._load_signal(sig, lbl)

    def _on_compose(self):
        expr = self._comp_expr.text().strip()
        if not expr:
            QMessageBox.warning(self,"Empty","Enter a signal expression."); return
        try:
            sig, lbl = evaluate_signal_expr(
                expr, self._comp_lbl_expr.text().strip(),
                self._comp_dur.value(), self._comp_sr.value(),
                self._comp_noise.value(), self._w_seed.value())
        except Exception as e:
            QMessageBox.critical(self,"Expression error",str(e)); return
        self.cfg = self._read_cfg(); self.cfg.sample_rate = self._comp_sr.value()
        self._load_signal(sig, lbl)

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self,"Open signal","","NumPy/CSV (*.npy *.csv)")
        if not path: return
        try:
            arr = np.load(path) if path.endswith(".npy") \
                  else np.loadtxt(path, delimiter=",")
            sig = arr[:,0].astype(np.float32) if arr.ndim==2 else arr.astype(np.float32)
            lbl = arr[:,1].astype(np.float32) if arr.ndim==2 and arr.shape[1]>=2 else None
            self._load_signal(sig, lbl)
        except Exception as e:
            QMessageBox.critical(self,"Load error",str(e))

    def _load_signal(self, sig, lbl):
        self.signal=sig; self.labels=lbl
        self.probs=None; self.smoothed=None; self.events=[]
        self._manual_regions.clear(); self._manual_items.clear()
        self._refresh_data_plots(); self._refresh_label_plot()
        n=len(sig); sr=self.cfg.sample_rate
        evs=int(lbl.sum()) if lbl is not None else 0
        self._card_dur.findChildren(QLabel)[0].setText(f"{n/sr:.1f} s")
        self._card_sr.findChildren(QLabel)[0].setText(f"{sr:,} Hz")
        self._card_evts.findChildren(QLabel)[0].setText(f"{evs/sr:.2f} s")
        self._card_ratio.findChildren(QLabel)[0].setText(
            f"{lbl.mean()*100:.1f}%" if lbl is not None else "—")
        self._lbl_status.setText(f"Loaded {n:,} samples · {evs:,} event samples")

    def _refresh_data_plots(self):
        if self.signal is None: return
        cfg=self.cfg; t=np.arange(len(self.signal))/cfg.sample_rate
        self._plot_sig.clear()
        if self.labels is not None:
            shade_regions(self._plot_sig, t, self.labels, QColor(0,255,136,50))
        if self.smoothed is not None:
            pred_s = preds_to_samples(self.smoothed, len(self.signal), cfg)
            shade_regions(self._plot_sig, t, pred_s, QColor(255,0,204,55))
        self._plot_sig.plot(t, self.signal, pen=pg.mkPen(C_CYAN, width=1.0))
        self._plot_fft.clear()
        c = np.abs(rfft(self.signal))/len(self.signal)
        f = rfftfreq(len(self.signal), 1.0/cfg.sample_rate)
        self._plot_fft.plot(f, c, pen=pg.mkPen(C_PURPLE, width=1.2))
        self._plot_fft.setXRange(0, cfg.sample_rate//2)
        for freq,color,name in [(cfg.baseline_freq,C_GREEN,f"baseline {cfg.baseline_freq:.0f} Hz"),
                                  (cfg.event_freq,C_MAGENTA,f"event {cfg.event_freq:.0f} Hz")]:
            self._plot_fft.addItem(pg.InfiniteLine(
                pos=freq, angle=90,
                pen=pg.mkPen(color,width=1.5,style=Qt.PenStyle.DashLine),
                label=name, labelOpts={"color":color,"position":0.85}))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — MANUAL LABELLING
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_label(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)
        self.tabs.addTab(w, "🖊  Label")

        toolbar = QHBoxLayout()
        self._btn_draw_toggle = QPushButton("Enable draw mode")
        self._btn_draw_toggle.setObjectName("success")
        self._btn_draw_toggle.setCheckable(True)
        self._btn_undo_lbl   = QPushButton("Undo last");    self._btn_undo_lbl.setObjectName("warn")
        self._btn_clear_lbl  = QPushButton("Clear all");    self._btn_clear_lbl.setObjectName("danger")
        self._btn_apply_lbl  = QPushButton("Apply labels to signal")
        self._btn_export_lbl = QPushButton("Export .npy");  self._btn_export_lbl.setObjectName("warn")
        toolbar.addWidget(self._btn_draw_toggle)
        toolbar.addWidget(self._btn_undo_lbl)
        toolbar.addWidget(self._btn_clear_lbl)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_apply_lbl)
        toolbar.addWidget(self._btn_export_lbl)
        lay.addLayout(toolbar)

        self._lbl_draw_hint = QLabel(
            "Click 'Enable draw mode' → click the START of a region → click the END.  "
            "Drag the yellow region edges to fine-tune.  "
            "Click 'Apply labels to signal' when done.")
        self._lbl_draw_hint.setObjectName("hint"); self._lbl_draw_hint.setWordWrap(True)
        lay.addWidget(self._lbl_draw_hint)

        self._plot_label = make_plot("Signal  —  yellow = manual labels")
        lay.addWidget(self._plot_label, 1)

        lay.addWidget(QLabel("Labelled regions:"))
        self._tbl_regions = QTableWidget(0, 4)
        self._tbl_regions.setHorizontalHeaderLabels(["#","Start (s)","End (s)","Duration (s)"])
        self._tbl_regions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_regions.setFixedHeight(160)
        lay.addWidget(self._tbl_regions)

        self._lbl_label_status = QLabel("No signal loaded.")
        self._lbl_label_status.setObjectName("subtitle")
        lay.addWidget(self._lbl_label_status)

        self._plot_label.scene().sigMouseClicked.connect(self._label_on_click)

        self._btn_draw_toggle.toggled.connect(self._on_draw_toggle)
        self._btn_undo_lbl.clicked.connect(self._on_undo_label)
        self._btn_clear_lbl.clicked.connect(self._on_clear_labels)
        self._btn_apply_lbl.clicked.connect(self._on_apply_labels)
        self._btn_export_lbl.clicked.connect(self._on_export_labels)

    def _refresh_label_plot(self):
        if self.signal is None: return
        cfg=self.cfg; t=np.arange(len(self.signal))/cfg.sample_rate
        self._plot_label.clear(); self._manual_items.clear()
        self._plot_label.plot(t, self.signal, pen=pg.mkPen(C_CYAN+"88", width=1.0))
        if self.labels is not None:
            shade_regions(self._plot_label, t, self.labels, QColor(0,255,136,35))
        for t0, t1 in self._manual_regions:
            item = pg.LinearRegionItem(
                values=[t0, t1],
                brush=pg.mkBrush(QColor(255,230,0,60)),
                pen=pg.mkPen(QColor(255,230,0,200), width=1.5),
                movable=True)
            self._plot_label.addItem(item)
            self._manual_items.append(item)
        self._lbl_label_status.setText(
            f"{len(self._manual_regions)} region(s)  ·  "
            f"{'🟢 Draw mode ON' if self._draw_mode else '⚫ Draw mode OFF'}")
        self._refresh_region_table()

    def _refresh_region_table(self):
        self._tbl_regions.setRowCount(0)
        for i,(t0,t1) in enumerate(self._manual_regions):
            self._tbl_regions.insertRow(i)
            for j,val in enumerate([str(i+1),f"{t0:.3f}",f"{t1:.3f}",f"{t1-t0:.3f}"]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._tbl_regions.setItem(i,j,item)

    def _on_draw_toggle(self, checked: bool):
        self._draw_mode = checked
        vb = self._plot_label.getViewBox()
        if checked:
            self._btn_draw_toggle.setText("Drawing…  (click to exit)")
            vb.setMouseEnabled(x=False, y=False)
            self._plot_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self._btn_draw_toggle.setText("Enable draw mode")
            vb.setMouseEnabled(x=True, y=True)
            self._plot_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self._drag_start_t = None
        self._lbl_label_status.setText(
            f"{len(self._manual_regions)} region(s)  ·  "
            f"{'🟢 Draw mode ON — click START then END of region' if checked else '⚫ Draw mode OFF'}")

    def _label_on_click(self, event):
        if not self._draw_mode or self.signal is None: return
        vb  = self._plot_label.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t   = float(pos.x())
        max_t = len(self.signal)/self.cfg.sample_rate
        t = max(0.0, min(t, max_t))
        if self._drag_start_t is None:
            self._drag_start_t = t
            self._lbl_label_status.setText(
                f"Start: {t:.3f} s  →  now click the END of the region")
        else:
            t0 = min(self._drag_start_t, t)
            t1 = max(self._drag_start_t, t)
            if t1 - t0 > 0.01:
                self._manual_regions.append((t0, t1))
                self._refresh_label_plot()
            self._drag_start_t = None

    def _on_undo_label(self):
        if self._manual_regions:
            self._manual_regions.pop(); self._refresh_label_plot()

    def _on_clear_labels(self):
        self._manual_regions.clear(); self._drag_start_t=None
        self._refresh_label_plot()

    def _on_apply_labels(self):
        if self.signal is None:
            QMessageBox.warning(self,"No signal","Load a signal first."); return
        if not self._manual_regions:
            QMessageBox.information(self,"No regions","Draw at least one region."); return
        # sync movable regions back from plot items
        for i, item in enumerate(self._manual_items):
            t0, t1 = item.getRegion()
            if i < len(self._manual_regions):
                self._manual_regions[i] = (min(t0,t1), max(t0,t1))
        cfg=self.cfg; n=len(self.signal); sr=cfg.sample_rate
        if self.labels is None:
            self.labels = np.zeros(n, dtype=np.float32)
        for t0,t1 in self._manual_regions:
            s = max(0, int(t0*sr)); e = min(n, int(t1*sr))
            self.labels[s:e] = 1.0
        evs = int(self.labels.sum())
        self._card_evts.findChildren(QLabel)[0].setText(f"{evs/sr:.2f} s")
        self._card_ratio.findChildren(QLabel)[0].setText(f"{self.labels.mean()*100:.1f}%")
        self._refresh_data_plots(); self._refresh_label_plot()
        self._lbl_label_status.setText(
            f"✅  Applied {len(self._manual_regions)} region(s)  ({evs:,} event samples)")

    def _on_export_labels(self):
        if self.signal is None:
            QMessageBox.warning(self,"No signal","Load a signal first."); return
        path, _ = QFileDialog.getSaveFileName(self,"Export","labels.npy","NumPy (*.npy)")
        if path:
            lbl = self.labels if self.labels is not None \
                  else np.zeros(len(self.signal), dtype=np.float32)
            np.save(path, np.column_stack([self.signal, lbl]))
            self._lbl_label_status.setText(f"💾  Exported → {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_train(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)
        self.tabs.addTab(w, "🏋  Train")

        ctrl = QHBoxLayout()
        self._btn_train    = QPushButton("Start training")
        self._btn_stop_tr  = QPushButton("Stop");          self._btn_stop_tr.setObjectName("danger");  self._btn_stop_tr.setEnabled(False)
        self._btn_save_mdl = QPushButton("Save model…");   self._btn_save_mdl.setObjectName("warn");   self._btn_save_mdl.setEnabled(False)
        self._btn_load_mdl = QPushButton("Load model…");   self._btn_load_mdl.setObjectName("warn")
        val_lbl = QLabel("Val split:"); val_lbl.setObjectName("sidebar_lbl")
        self._w_val = QDoubleSpinBox(); self._w_val.setRange(0.1,0.4); self._w_val.setValue(0.2); self._w_val.setSingleStep(0.05)
        ctrl.addWidget(self._btn_train); ctrl.addWidget(self._btn_stop_tr)
        ctrl.addWidget(self._btn_save_mdl); ctrl.addWidget(self._btn_load_mdl)
        ctrl.addStretch(); ctrl.addWidget(val_lbl); ctrl.addWidget(self._w_val)
        lay.addLayout(ctrl)

        self._train_prog = QProgressBar(); self._train_prog.setValue(0)
        lay.addWidget(self._train_prog)

        plots = QHBoxLayout()
        self._plot_loss = make_plot("Loss", x_label="Epoch", y_label="Loss", height=220)
        self._plot_f1   = make_plot("Validation F1", x_label="Epoch", y_label="F1", height=220)
        self._plot_lr   = make_plot("Learning rate", x_label="Epoch", y_label="LR", height=220)
        self._plot_loss.addLegend(offset=(10,10))
        plots.addWidget(self._plot_loss); plots.addWidget(self._plot_f1)
        plots.addWidget(self._plot_lr)
        lay.addLayout(plots)

        self._train_log = QTextEdit(); self._train_log.setReadOnly(True); self._train_log.setFixedHeight(190)
        lay.addWidget(QLabel("Log:")); lay.addWidget(self._train_log)

        self._ep_tl=[]; self._ep_vl=[]; self._ep_f1_hist=[]; self._ep_lr_hist=[]
        self._btn_train.clicked.connect(self._on_train)
        self._btn_stop_tr.clicked.connect(self._on_stop_train)
        self._btn_save_mdl.clicked.connect(self._on_save_model)
        self._btn_load_mdl.clicked.connect(self._on_load_model)

    def _on_train(self):
        if self.signal is None:
            self._train_log.append("⚠  No signal."); return
        if self.labels is None:
            self._train_log.append("⚠  No labels — generate preset or draw labels in the Label tab."); return
        self.cfg=self._read_cfg()
        self._ep_tl.clear(); self._ep_vl.clear(); self._ep_f1_hist.clear(); self._ep_lr_hist.clear()
        self._plot_loss.clear(); self._plot_f1.clear(); self._plot_lr.clear()
        self._train_prog.setValue(0); self._train_log.clear()
        self._btn_train.setEnabled(False); self._btn_stop_tr.setEnabled(True)
        self._btn_save_mdl.setEnabled(False)
        self._train_worker = TrainWorker(self.cfg, self.signal, self.labels, self._w_val.value())
        self._train_worker.progress.connect(self._train_prog.setValue)
        self._train_worker.log_line.connect(self._train_log.append)
        self._train_worker.epoch_done.connect(self._on_epoch)
        self._train_worker.lr_updated.connect(self._on_epoch_lr)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(lambda e: self._train_log.append(f"❌ {e}"))
        self._train_worker.start()

    def _on_epoch(self, tl, vl, f1):
        self._ep_tl.append(tl); self._ep_vl.append(vl); self._ep_f1_hist.append(f1)
        ep = list(range(1, len(self._ep_tl)+1))
        self._plot_loss.clear(); self._plot_loss.addLegend(offset=(10,10))
        self._plot_loss.plot(ep, self._ep_tl, pen=pg.mkPen(C_CYAN,    width=2), name="Train")
        self._plot_loss.plot(ep, self._ep_vl, pen=pg.mkPen(C_MAGENTA, width=2), name="Val")
        self._plot_f1.clear()
        self._plot_f1.plot(ep, self._ep_f1_hist, pen=pg.mkPen(C_GREEN, width=2))

    def _on_epoch_lr(self, lr: float):
        """Slot connected to TrainWorker lr signal — updates LR curve."""
        self._ep_lr_hist.append(lr)
        ep = list(range(1, len(self._ep_lr_hist)+1))
        self._plot_lr.clear()
        self._plot_lr.plot(ep, self._ep_lr_hist,
                           pen=pg.mkPen(C_YELLOW, width=2))

    def _on_train_done(self, model, history):
        self.model=model; self.history=history
        self._btn_train.setEnabled(True); self._btn_stop_tr.setEnabled(False)
        self._btn_save_mdl.setEnabled(True)
        best_f1  = max(history["val_f1"])
        best_ep  = history["val_f1"].index(best_f1) + 1
        best_loss= history["val_loss"][best_ep - 1]
        final_lr = history["lr"][-1] if history.get("lr") else self.cfg.learning_rate
        summary = (
            f"\n{'─'*52}\n"
            f"✅  Training complete\n"
            f"   Best val F1   : {best_f1:.4f}  (epoch {best_ep})\n"
            f"   Best val loss : {best_loss:.4f}\n"
            f"   Final LR      : {final_lr:.2e}\n"
            f"   Optimizer     : {self.cfg.optimizer}\n"
            f"   Scheduler     : {self.cfg.scheduler}\n"
            f"{'─'*52}\n"
            f"→ Switch to Detect tab to run inference."
        )
        self._train_log.append(summary)

    def _on_stop_train(self):
        if self._train_worker: self._train_worker.stop()
        self._btn_train.setEnabled(True); self._btn_stop_tr.setEnabled(False)

    def _on_save_model(self):
        if not self.model: return
        path, _ = QFileDialog.getSaveFileName(self,"Save","event_lstm.pt","PyTorch (*.pt)")
        if path:
            torch.save({"state_dict": self.model.state_dict(), "config": self.cfg}, path)
            self._train_log.append(f"💾  Saved → {path}")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self,"Load model","","PyTorch (*.pt)")
        if not path: return
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt,dict) and "state_dict" in ckpt:
                sd = ckpt["state_dict"]; saved_cfg = ckpt.get("config",None)
                if saved_cfg:
                    self._apply_cfg_to_sidebar(saved_cfg); self.cfg=saved_cfg
                    self._train_log.append("ℹ️  Config restored from checkpoint.")
            else:
                sd = ckpt; self.cfg=self._read_cfg()
                self._train_log.append("ℹ️  Legacy checkpoint — using current config.")
            model = EventLSTM(self.cfg); model.load_state_dict(sd); model.eval()
            self.model=model; self.history=None; self._btn_save_mdl.setEnabled(True)
            self._train_log.append(
                f"✅  Loaded  ·  feat_dim={self.cfg.feature_dim()}  "
                f"hidden={self.cfg.hidden_size}  layers={self.cfg.num_layers}\n"
                f"    Go to Detect tab.")
        except Exception as e:
            QMessageBox.critical(self,"Load failed",str(e))
            self._train_log.append(f"❌ {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — DETECT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_detect(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)
        self.tabs.addTab(w, "🔍  Detect")

        ctrl = QHBoxLayout()
        self._btn_detect = QPushButton("Run detection")
        ctrl.addWidget(self._btn_detect); ctrl.addStretch()
        lay.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._plot_det      = make_plot("Signal  (green = truth · magenta = predicted)")
        self._plot_det_prob = make_plot("P(event)", y_label="Probability", height=170)
        self._plot_det_prob.setYRange(0,1)
        self._plot_det_prob.setXLink(self._plot_det)
        splitter.addWidget(self._plot_det); splitter.addWidget(self._plot_det_prob)
        splitter.setSizes([450,170]); lay.addWidget(splitter,1)

        self._tbl_evt = QTableWidget(0,4)
        self._tbl_evt.setHorizontalHeaderLabels(["#","Start (s)","End (s)","Duration"])
        self._tbl_evt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_evt.setFixedHeight(155)
        lay.addWidget(QLabel("Detected event ranges:")); lay.addWidget(self._tbl_evt)

        self._lbl_det_status = QLabel(""); self._lbl_det_status.setObjectName("subtitle")
        lay.addWidget(self._lbl_det_status)
        self._btn_detect.clicked.connect(self._on_detect)

    def _on_detect(self):
        if self.model is None:
            self._lbl_det_status.setText("⚠  Train or load a model first."); return
        if self.signal is None:
            self._lbl_det_status.setText("⚠  Load a signal first."); return
        self.cfg=self._read_cfg(); self._btn_detect.setEnabled(False)
        self._lbl_det_status.setText("Running inference…")
        self._det_worker = DetectWorker(self.model, self.signal, self.cfg)
        self._det_worker.finished.connect(self._on_detect_done)
        self._det_worker.error.connect(lambda e: self._lbl_det_status.setText(f"❌ {e}"))
        self._det_worker.start()

    def _on_detect_done(self, probs, smoothed, events):
        self.probs=probs; self.smoothed=smoothed; self.events=events
        self._btn_detect.setEnabled(True)
        self._lbl_det_status.setText(f"Detected {len(events)} event(s)")
        cfg=self.cfg; sig=self.signal; lbl=self.labels
        t=np.arange(len(sig))/cfg.sample_rate
        pred_s = preds_to_samples(smoothed, len(sig), cfg)
        offset = cfg.history_len-1
        prob_t = np.array([
            ((i+offset)*cfg.stride+cfg.window_size//2)/cfg.sample_rate
            for i in range(len(probs))])
        self._plot_det.clear()
        if lbl is not None: shade_regions(self._plot_det, t, lbl,    QColor(0,255,136,45))
        shade_regions(self._plot_det, t, pred_s, QColor(255,0,204,55))
        self._plot_det.plot(t, sig, pen=pg.mkPen(C_CYAN, width=1.0))
        self._plot_det_prob.clear()
        self._plot_det_prob.addLine(
            y=0.5, pen=pg.mkPen(C_MUTED,width=1,style=Qt.PenStyle.DashLine))
        zero_c = self._plot_det_prob.plot(prob_t, np.zeros_like(probs), pen=pg.mkPen(None))
        prob_c = self._plot_det_prob.plot(prob_t, probs, pen=pg.mkPen(C_ORANGE, width=1.5))
        fill   = pg.FillBetweenItem(zero_c, prob_c,
                                    brush=pg.mkBrush(QColor(255,136,0,50)))
        self._plot_det_prob.addItem(fill)
        self._tbl_evt.setRowCount(0)
        for i,(s,e) in enumerate(events):
            self._tbl_evt.insertRow(i)
            for j,val in enumerate([str(i+1),f"{s:.3f}",f"{e:.3f}",f"{e-s:.3f} s"]):
                item=QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._tbl_evt.setItem(i,j,item)
        self._refresh_data_plots()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — EVALUATE
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_eval(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)
        self.tabs.addTab(w, "📊  Evaluate")

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._btn_eval        = QPushButton("Compute metrics")
        self._btn_export_eval = QPushButton("Export report (.txt)")
        self._btn_export_eval.setObjectName("warn"); self._btn_export_eval.setEnabled(False)
        tb.addWidget(self._btn_eval); tb.addWidget(self._btn_export_eval); tb.addStretch()
        lay.addLayout(tb)

        # ── sample-level metric cards (row 1) ─────────────────────────────────
        lay.addWidget(self._make_section_label("Sample-level metrics"))
        mrow = QHBoxLayout()
        self._ev_f1   = metric_card("—","F1 Score")
        self._ev_prec = metric_card("—","Precision")
        self._ev_rec  = metric_card("—","Recall")
        self._ev_acc  = metric_card("—","Accuracy")
        for c in [self._ev_f1,self._ev_prec,self._ev_rec,self._ev_acc]:
            mrow.addWidget(c)
        lay.addLayout(mrow)

        # ── segment-level metric cards (row 2) ────────────────────────────────
        lay.addWidget(self._make_section_label("Segment-level metrics  (IoU ≥ 0.5)"))
        srow = QHBoxLayout()
        self._ev_seg_tp  = metric_card("—", "Seg TP")
        self._ev_seg_fp  = metric_card("—", "Seg FP")
        self._ev_seg_fn  = metric_card("—", "Seg FN")
        self._ev_seg_iou = metric_card("—", "Mean IoU")
        for c in [self._ev_seg_tp,self._ev_seg_fp,self._ev_seg_fn,self._ev_seg_iou]:
            srow.addWidget(c)
        lay.addLayout(srow)

        # ── confusion matrix ──────────────────────────────────────────────────
        lay.addWidget(self._make_section_label("Sample-level confusion matrix"))
        self._tbl_conf = QTableWidget(2,3)
        self._tbl_conf.setHorizontalHeaderLabels(["","True event","True non-event"])
        self._tbl_conf.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_conf.setFixedHeight(95); lay.addWidget(self._tbl_conf)

        # ── training curves ───────────────────────────────────────────────────
        lay.addWidget(self._make_section_label("Training history"))
        curves = QHBoxLayout()
        self._plot_el  = make_plot("Loss",          x_label="Epoch", y_label="Loss", height=200)
        self._plot_ef  = make_plot("Validation F1", x_label="Epoch", y_label="F1",   height=200)
        self._plot_elr = make_plot("Learning rate", x_label="Epoch", y_label="LR",   height=200)
        self._plot_el.addLegend(offset=(10,10))
        curves.addWidget(self._plot_el)
        curves.addWidget(self._plot_ef)
        curves.addWidget(self._plot_elr)
        lay.addLayout(curves)
        lay.addStretch()

        self._btn_eval.clicked.connect(self._on_eval)
        self._btn_export_eval.clicked.connect(self._on_export_eval)

    @staticmethod
    def _make_section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{C_CYAN};"
            f" padding-top:6px;")
        return lbl

    @staticmethod
    def _segment_metrics(y_true: np.ndarray, events_pred: list,
                          sr: int, iou_thresh: float = 0.5):
        """
        Compute segment-level TP/FP/FN and mean IoU.
        Ground-truth segments are extracted from y_true binary array.
        Predicted segments come from the events list [(t0,t1),...].
        A prediction is a TP if IoU with any GT segment >= iou_thresh.
        """
        # extract GT segments from label array
        padded = np.concatenate([[0], y_true.astype(np.int8), [0]])
        diff   = np.diff(padded)
        gt_segs = [(int(s)/sr, int(e)/sr)
                   for s, e in zip(np.where(diff==1)[0], np.where(diff==-1)[0])]

        def iou(a, b):
            inter = max(0, min(a[1],b[1]) - max(a[0],b[0]))
            union = max(a[1],b[1]) - min(a[0],b[0])
            return inter/union if union > 0 else 0.0

        matched_gt  = set()
        matched_pred = set()
        ious = []
        for pi, pred in enumerate(events_pred):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gt_segs):
                v = iou(pred, gt)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= iou_thresh:
                matched_pred.add(pi)
                matched_gt.add(best_gi)
                ious.append(best_iou)

        tp  = len(matched_pred)
        fp  = len(events_pred) - tp
        fn  = len(gt_segs)     - len(matched_gt)
        mean_iou = float(np.mean(ious)) if ious else 0.0
        return tp, fp, fn, mean_iou, len(gt_segs)

    def _on_eval(self):
        if self.smoothed is None or self.labels is None: return
        cfg    = self.cfg
        pred_s = preds_to_samples(self.smoothed, len(self.signal), cfg)
        n      = min(len(self.labels), len(pred_s))
        yt, yp = self.labels[:n], pred_s[:n]

        # ── sample-level ──────────────────────────────────────────────────────
        f1   = f1_score(yt,   yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        tp_s = int(((yt==1)&(yp==1)).sum())
        fp_s = int(((yt==0)&(yp==1)).sum())
        fn_s = int(((yt==1)&(yp==0)).sum())
        tn_s = int(((yt==0)&(yp==0)).sum())
        acc  = (tp_s + tn_s) / max(1, n)

        self._ev_f1.findChildren(QLabel)[0].setText(f"{f1:.4f}")
        self._ev_prec.findChildren(QLabel)[0].setText(f"{prec:.4f}")
        self._ev_rec.findChildren(QLabel)[0].setText(f"{rec:.4f}")
        self._ev_acc.findChildren(QLabel)[0].setText(f"{acc:.4f}")

        for r,(rl,a,b) in enumerate([("Pred event",tp_s,fp_s),
                                      ("Pred non-ev",fn_s,tn_s)]):
            self._tbl_conf.setItem(r,0,QTableWidgetItem(rl))
            self._tbl_conf.setItem(r,1,QTableWidgetItem(str(a)))
            self._tbl_conf.setItem(r,2,QTableWidgetItem(str(b)))
            for c in range(3):
                if self._tbl_conf.item(r,c):
                    self._tbl_conf.item(r,c).setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter)

        # ── segment-level (IoU ≥ 0.5) ─────────────────────────────────────────
        seg_tp, seg_fp, seg_fn, mean_iou, n_gt = self._segment_metrics(
            yt, self.events, cfg.sample_rate)
        self._ev_seg_tp.findChildren(QLabel)[0].setText(str(seg_tp))
        self._ev_seg_fp.findChildren(QLabel)[0].setText(str(seg_fp))
        self._ev_seg_fn.findChildren(QLabel)[0].setText(str(seg_fn))
        self._ev_seg_iou.findChildren(QLabel)[0].setText(f"{mean_iou:.3f}")

        # ── store for export ──────────────────────────────────────────────────
        self._last_eval = dict(
            f1=f1, precision=prec, recall=rec, accuracy=acc,
            tp=tp_s, fp=fp_s, fn=fn_s, tn=tn_s,
            seg_tp=seg_tp, seg_fp=seg_fp, seg_fn=seg_fn,
            mean_iou=mean_iou, n_gt_segments=n_gt,
            optimizer=cfg.optimizer, scheduler=cfg.scheduler,
            lr=cfg.learning_rate, epochs=cfg.epochs,
            hidden=cfg.hidden_size, layers=cfg.num_layers,
        )
        self._btn_export_eval.setEnabled(True)

        # ── training history curves ───────────────────────────────────────────
        if self.history:
            ep = list(range(1, len(self.history["train_loss"])+1))
            self._plot_el.clear(); self._plot_el.addLegend(offset=(10,10))
            self._plot_el.plot(ep, self.history["train_loss"],
                               pen=pg.mkPen(C_CYAN,    width=2), name="Train")
            self._plot_el.plot(ep, self.history["val_loss"],
                               pen=pg.mkPen(C_MAGENTA, width=2), name="Val")
            self._plot_ef.clear()
            self._plot_ef.plot(ep, self.history["val_f1"],
                               pen=pg.mkPen(C_GREEN, width=2))
            self._plot_elr.clear()
            if "lr" in self.history and self.history["lr"]:
                self._plot_elr.plot(ep, self.history["lr"],
                                    pen=pg.mkPen(C_YELLOW, width=2))

    def _on_export_eval(self):
        """Export a plain-text evaluation report."""
        if not hasattr(self, "_last_eval"): return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "eval_report.txt", "Text (*.txt)")
        if not path: return
        d = self._last_eval
        lines = [
            "Event Detection — Evaluation Report",
            "=" * 44,
            "",
            "Sample-level metrics",
            f"  F1 Score   : {d['f1']:.4f}",
            f"  Precision  : {d['precision']:.4f}",
            f"  Recall     : {d['recall']:.4f}",
            f"  Accuracy   : {d['accuracy']:.4f}",
            "",
            "Sample-level confusion matrix",
            f"  TP={d['tp']}  FP={d['fp']}  FN={d['fn']}  TN={d['tn']}",
            "",
            "Segment-level metrics  (IoU threshold = 0.50)",
            f"  GT segments : {d['n_gt_segments']}",
            f"  TP          : {d['seg_tp']}",
            f"  FP          : {d['seg_fp']}",
            f"  FN          : {d['seg_fn']}",
            f"  Mean IoU    : {d['mean_iou']:.3f}",
            "",
            "Training config",
            f"  Optimizer   : {d['optimizer']}",
            f"  Scheduler   : {d['scheduler']}",
            f"  LR          : {d['lr']}",
            f"  Epochs      : {d['epochs']}",
            f"  Hidden size : {d['hidden']}",
            f"  LSTM layers : {d['layers']}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        self._btn_export_eval.setText("Exported ✓")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY                                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
