# Event-Detection
Train a LSTM neural network to detect a time series event


# 📡 Event Detection in Noisy Time-Series

> A desktop application for detecting high-frequency event bursts embedded in a noisy signal — built with a multi-resolution LSTM architecture and a PyQtGraph GUI.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-green?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Signal Definition](#signal-definition)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Signal Composer](#signal-composer)
- [Configuration Reference](#configuration-reference)
- [How It Works](#how-it-works)
- [GUI Walkthrough](#gui-walkthrough)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

<img width="1438" height="763" alt="image" src="https://github.com/user-attachments/assets/695ef0ce-53f6-4f87-95b2-076530a42196" />

## Overview

This project solves a common signal processing problem: **automatically detecting when and where short high-frequency events occur inside a longer, noisier baseline signal** — without hand-tuning thresholds.

Instead of rule-based detection, the system learns directly from labeled data using a **Long Short-Term Memory (LSTM)** neural network trained on multi-resolution features extracted from sliding windows of the signal. The full pipeline — signal generation, feature extraction, model training, inference, and evaluation — runs inside a single self-contained Python file with a responsive desktop GUI.

**Typical use case:**  
A 10-second sine wave sampled at 1000 Hz contains 3 short 1-second bursts of 80 Hz oscillation embedded at random positions. The model learns to distinguish these bursts from the noisy 5 Hz baseline and outputs both a probability trace and segmented event ranges `(t_start, t_end)`.

---

## Signal Definition

| Property | Value |
|---|---|
| Duration | 10 seconds (configurable) |
| Sample rate | 1000 Hz |
| Baseline | `sin(2π × 5 × t)` + Gaussian noise |
| Event type | 1-second bursts of `sin(2π × 80 × t)` |
| Event count | 3 (configurable, non-overlapping) |
| Event amplitude | 2× baseline (configurable) |

Events are guaranteed non-overlapping with a 200-sample gap between them. The event ratio is approximately 10% of the total signal — the `pos_weight` training parameter compensates for this class imbalance.

---

## Features

- **Self-contained** — the entire system (data, features, model, GUI) lives in one Python file
- **Signal composer** — type any NumPy expression in `t` to build arbitrary signals, including complex multi-component waveforms
- **Multi-resolution feature extraction** — statistical, FFT-based, and multi-level wavelet features per window
- **Sequence learning** — LSTM ingests N consecutive feature windows for temporal context
- **Non-blocking GUI** — training and inference run in `QThread` workers; the UI never freezes
- **Live training curves** — loss and F1 plots update after every epoch
- **Linked plots** — signal and probability plots share a synchronized x-axis; zooming one zooms both
- **Evaluation tab** — F1, precision, recall, and a sample-level confusion matrix
- **Dark theme** — Catppuccin Mocha palette throughout

---

## Architecture

```
Raw signal
    │
    ▼
Sliding windows  (window_size=256, stride=64)
    │
    ▼
Feature extraction per window
    ├── Statistical  (mean, std, skewness, kurtosis, min, max)       → 6 features
    ├── Frequency    (FFT energy, dominant freq, centroid, band ratios) → 5 features
    └── Wavelet      (multi-level DWT energy per level)               → 4 features
                                                                  ─────────────
                                                                   15 features/window
    │
    ▼
Sequence builder  (stack N=10 consecutive windows)
    │   Input shape per sample: (10, 15)
    ▼
LSTM  (2 layers, hidden=64, dropout=0.3)
    │
    ▼
Linear head  →  sigmoid  →  P(event)  ∈ [0, 1]
    │
    ▼
Sliding majority-vote smoother  (window=5)
    │
    ▼
Segment extractor  →  [(t_start, t_end), …]
```

---

## Project Structure

```
event_detection_pyqt_v2/
├── event_detection_app.py   ← entire system in one file
├── requirements.txt
├── run.sh                   ← one-click launcher
└── venv/                    ← isolated Python environment
```

### Internal file layout (`event_detection_app.py`)

| Section | Description |
|---|---|
| `Config` | Dataclass holding all parameters — single source of truth |
| `generate_preset_signal` | Synthetic 5 Hz baseline + 80 Hz event bursts |
| `evaluate_signal_expr` | NumPy expression evaluator for the signal composer |
| `sliding_windows` | Overlapping window slicer with majority-vote labeling |
| `extract_features` | Statistical + FFT + wavelet feature extraction per window |
| `build_sequences` | Stacks N windows into LSTM input sequences `(N, feature_dim)` |
| `EventDataset` | PyTorch `Dataset` wrapper |
| `EventLSTM` | Stacked LSTM + dropout + linear head |
| `TrainWorker` | `QThread` — runs full training pipeline off the main thread |
| `DetectWorker` | `QThread` — runs inference + smoothing + segmentation |
| `preds_to_samples` | Expands per-window predictions to per-sample array |
| `shade_regions` | Batched `LinearRegionItem` shading using `np.diff` |
| `MainWindow` | PyQt6 main window — sidebar config + 4-tab layout |

---

## Installation

### One-shot setup (recommended)

```bash
# Download and run the setup script
chmod +x setup_event_detection_pyqt_v2.sh
./setup_event_detection_pyqt_v2.sh

# Launch
cd event_detection_pyqt_v2
./run.sh
```

The script will:
1. Create an `event_detection_pyqt_v2/` project folder
2. Create an isolated Python virtual environment inside it
3. Install all dependencies with `numpy<2` pinned for PyTorch compatibility
4. Write `event_detection_app.py` and `run.sh`

### Manual setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/event-detection.git
cd event-detection

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies (numpy<2 is required)
pip install -r requirements.txt

# Run
python event_detection_app.py
```

> **Important — NumPy version:**  
> PyTorch 2.x wheels for Python 3.9 were compiled against NumPy 1.x.  
> Always install with `numpy<2` or you will get a `RuntimeError: Numpy is not available` at runtime.

---

## Usage

The workflow follows the four tabs in order:

### 1 · Data tab

Choose between three signal sources:

**Preset generator** — uses sidebar parameters to generate the standard 5 Hz + 80 Hz burst signal. Click **Generate preset**.

**Signal composer** — type any NumPy expression (see [Signal Composer](#signal-composer) below). Click **Build signal from expression**.

**File upload** — load a `.npy` or `.csv` file. A two-column file is interpreted as `[signal, labels]`; a one-column file loads signal only (no labels → evaluation disabled).

After loading, the Data tab shows:
- Metric cards (duration, sample rate, event time, event ratio)
- Interactive signal plot with green shading on ground-truth event regions
- FFT spectrum with markers at baseline and event frequencies

### 2 · Train tab

1. Optionally adjust **Val split** (default 20%)
2. Click **Start training**
3. Watch loss and F1 curves update live after every epoch
4. Click **Stop** at any time — training stops cleanly after the current epoch
5. Click **Save model…** to export weights as `.pt`

All heavy computation (windowing, feature extraction, LSTM training) runs in a background thread. The GUI stays responsive throughout.

### 3 · Detect tab

1. Click **Run detection**
2. The signal plot shows orange shading on predicted event regions overlaid over the green ground-truth shading
3. The probability plot below shows the raw `P(event)` trace and the 0.5 threshold line
4. Both plots are **linked** — zoom or pan one and the other follows
5. The event table lists each detected segment with start time, end time, and duration

### 4 · Evaluate tab

Click **Compute metrics** to see:

| Metric | Description |
|---|---|
| **F1 Score** | Harmonic mean of precision and recall at sample level |
| **Precision** | Of all samples predicted as event, how many were actually events |
| **Recall** | Of all true event samples, how many were correctly detected |

A confusion matrix (Predicted event / non-event × True event / non-event) is shown at sample resolution. Training curves are also reproduced here for reference.

---

## Signal Composer

The signal composer lets you define **any signal** as a NumPy expression in `t`, where `t` is a 1-D time array from `0` to `duration` seconds at the configured sample rate.

### Available symbols

| Symbol | Description |
|---|---|
| `t` | Time array in seconds, shape `(N,)` |
| `n` | Number of samples |
| `sr` | Sample rate (Hz) |
| `sin`, `cos`, `tan` | Trigonometric functions |
| `exp`, `log`, `log2`, `log10` | Exponential / logarithm |
| `sqrt`, `abs` | Square root, absolute value |
| `pi`, `e`, `inf` | Constants |
| `sign`, `floor`, `ceil` | Rounding / sign |
| `np` | Full NumPy namespace |

### Examples

**Your exact expression:**
```python
sin(t) + sin(10*(t-100)) - sin(10*(t-20)) + cos(t)
```

**Standard preset equivalent:**
```python
sin(2*pi*5*t) + 2*sin(2*pi*80*t)*((t>=3)&(t<4))
```

**Multi-event with labels:**
```
Signal: sin(2*pi*5*t) + 2*sin(2*pi*80*(t-3))*((t>=3)&(t<4)) - 1.5*sin(2*pi*80*(t-7))*((t>=7)&(t<8))
Labels: ((t>=3)&(t<4)) | ((t>=7)&(t<8))
```

**Chirp event (frequency sweep):**
```python
sin(2*pi*5*t) + 2*sin(2*pi*(10+40*t)*t)*((t>=2)&(t<3))
```

**AM modulation burst:**
```python
sin(2*pi*5*t) + (1 + 0.5*sin(2*pi*2*t)) * sin(2*pi*80*t) * ((t>=4)&(t<5))
```

**Exponential decay event:**
```python
sin(2*pi*5*t) + 3*sin(2*pi*80*t)*exp(-5*(t-3))*((t>=3)&(t<5))
```

**Multiple asymmetric events:**
```python
sin(2*pi*5*t) \
  + 2.0*sin(2*pi*60*t)*((t>=1.0)&(t<2.0)) \
  - 1.5*sin(2*pi*80*t)*((t>=4.5)&(t<5.5)) \
  + 2.5*cos(2*pi*100*t)*((t>=8.0)&(t<9.0))
```

> **Tip:** Boolean masks like `((t>=3)&(t<4))` evaluate to a `0/1` array at each time step, making them ideal for gating a component to a specific time window.

---

## Configuration Reference

All parameters are accessible from the sidebar. Changes take effect when you click Generate / Start training / Run detection.

### Signal (preset)

| Parameter | Default | Description |
|---|---|---|
| Events | 3 | Number of 1-second event bursts to embed |
| Noise std | 0.2 | Standard deviation of additive Gaussian noise |
| Event amp | 2.0 | Amplitude multiplier of the event burst |
| Baseline Hz | 5.0 | Frequency of the background sine wave |
| Event Hz | 80.0 | Frequency of the embedded event burst |
| Duration (s) | 10.0 | Total signal length in seconds |
| Seed | 42 | Random seed for reproducible signal generation |

### Windowing

| Parameter | Default | Description |
|---|---|---|
| Window size | 256 | Samples per window (larger = more frequency resolution) |
| Stride | 64 | Samples between window starts (smaller = more overlap, more sequences) |

### Wavelet features

| Parameter | Default | Description |
|---|---|---|
| Wavelet type | db4 | Daubechies 4 — good general-purpose choice |
| Levels | 4 | Number of decomposition levels; each adds one feature per window |

### LSTM

| Parameter | Default | Description |
|---|---|---|
| History | 10 | Number of consecutive windows fed as one sequence to the LSTM |
| Hidden size | 64 | LSTM hidden state dimensionality |
| Layers | 2 | Number of stacked LSTM layers |
| Dropout | 0.3 | Dropout rate applied between LSTM layers and before the output head |

### Training

| Parameter | Default | Description |
|---|---|---|
| Epochs | 30 | Number of full passes through the training set |
| Batch size | 32 | Sequences per gradient update |
| Learning rate | 0.001 | Initial Adam learning rate; reduced automatically on plateau |
| Pos weight | 4.0 | BCEWithLogitsLoss weight for positive class — increase if recall is low |
| Val split | 0.2 | Fraction of sequences held out for validation |

### Detection

| Parameter | Default | Description |
|---|---|---|
| Threshold | 0.5 | Probability cutoff for classifying a window as an event |
| Smooth w | 5 | Majority-vote smoothing window in number of windows |

---

## How It Works

### Feature extraction

Each window of 256 samples is transformed into a 15-dimensional feature vector:

**Statistical (6):** mean, standard deviation, skewness, kurtosis, min, max — capture the overall amplitude distribution of the window.

**Frequency (5):** total FFT energy, dominant frequency, spectral centroid, fraction of energy below 20 Hz, fraction of energy in 20–100 Hz. The 20–100 Hz band directly captures the 80 Hz event energy, making it the most discriminative single feature.

**Wavelet (4):** energy per decomposition level from a 4-level Daubechies 4 wavelet transform. Wavelet decomposition captures transient, non-stationary structure that a plain FFT averages over.

### Sequence learning

Rather than classifying each window independently, the LSTM receives 10 consecutive feature vectors as a sequence `(10, 15)` and outputs a single probability for the last (most recent) window. This gives the model temporal context — it can learn that a window is more likely to be an event if the preceding windows also showed elevated high-frequency energy.

### Class imbalance

With 3 events of 1 second each in a 10-second signal, only ~30% of windows (and far fewer sequences) are labeled positive. `BCEWithLogitsLoss` with `pos_weight=4.0` upweights the loss contribution of positive samples, preventing the model from collapsing to always predicting "no event".

### Smoothing and segmentation

Raw per-window probabilities are noisy. A sliding majority-vote smoother (window=5) reduces isolated false positives. Contiguous runs of smoothed predictions above threshold are then merged into `(t_start, t_end)` event ranges.

---

## GUI Walkthrough

```
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar (config)  │  Tabs                                       │
│                    │  ┌──────────┬──────────┬──────────┬──────┐ │
│  Preset signal     │  │  📂 Data │ 🏋️ Train │ 🔍 Detect│ 📊  │ │
│  Windowing         │  │          │          │          │ Eval │ │
│  Wavelet           │  │ [expr]   │ [loss ↗] │ [signal] │      │ │
│  LSTM              │  │ [signal] │ [F1   ↗] │ [P(ev) ] │      │ │
│  Training          │  │ [fft   ] │ [log    ]│ [table ] │      │ │
│  Detection         │  │          │          │          │      │ │
└─────────────────────────────────────────────────────────────────┘
```

**Keyboard shortcuts:**  
- Scroll wheel on any plot to zoom in/out  
- Right-click drag to zoom a region  
- Middle-click drag to pan  
- `A` key to auto-fit the view  

---

## Requirements

```
numpy<2             # pinned — torch 2.x on Python 3.9 requires numpy 1.x
torch>=2.1.0
PyQt6>=6.5.0
pyqtgraph>=0.13.3
scipy>=1.11.0
PyWavelets>=1.5.0
scikit-learn>=1.3.0
```

**Python version:** 3.9 or higher  
**OS:** macOS, Linux, Windows (tested on macOS 14 and Ubuntu 22.04)

---

## Troubleshooting

### `RuntimeError: Numpy is not available`

NumPy 2.x is installed but your PyTorch wheel was built against NumPy 1.x.

```bash
source venv/bin/activate
pip install "numpy<2"
```

### `torch._C._get_default_device` crash on import

Your PyTorch version is too old for Python 3.9.

```bash
pip install --upgrade torch
```

### Font warning: `SF Pro Display not found`

Harmless on non-macOS systems. The app falls back to `Helvetica Neue` / `Arial` automatically. No action needed.

### `QThread: Destroyed while thread is still running`

Close the app only after training completes, or click **Stop** first to cleanly terminate the worker thread.

### Plots are slow on very long signals (> 100k samples)

pyqtgraph's auto-downsampling handles most cases. If you need even better performance for very long signals, reduce the sample rate or increase the stride in the sidebar.

### `ModuleNotFoundError: No module named 'PyQt6'`

Make sure you have activated the virtual environment before running:

```bash
source venv/bin/activate
python event_detection_app.py
```

---

## Roadmap

- [ ] Multi-channel signal support
- [ ] Load and apply a previously saved `.pt` model to a new signal without retraining
- [ ] Manual annotation mode — click the signal plot to add/remove event regions and export corrected labels
- [ ] Export detected events to CSV
- [ ] Transformer-based sequence model as alternative to LSTM
- [ ] Real-time detection from a live audio or serial stream

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with PyTorch · PyQtGraph · PyQt6 · NumPy · SciPy · PyWavelets · scikit-learn*
