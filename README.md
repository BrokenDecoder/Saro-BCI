<div align="center">

<img width="1052" height="436" alt="Saro BCI Banner" src="https://github.com/user-attachments/assets/db6e1cab-3b23-4900-817b-ff3c918c670a" />

# 🧠 Saro BCI
### *A Real-Time Brain-Computer Interface for Neuroscience Research & Clinical Monitoring*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CEBRA](https://img.shields.io/badge/CEBRA-0.4.0-8A2BE2)](https://cebra.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-Real--Time-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Saro BCI** fuses cutting-edge deep learning, topological data analysis, and contrastive neural embeddings into a unified platform — bridging the gap between academic hyperscanning research and real-time clinical-grade neural monitoring.

[**Quickstart**](#-getting-started) · [**Architecture**](#-system-architecture) · [**Research**](#-scientific-goals) · [**Results**](#-results--outputs)

</div>

---

## 🔭 What Is This Project?

Saro BCI is a dual-purpose neuroscience platform built around two intertwined goals:

| Axis | Description |
|------|-------------|
| 🔬 **Research** | Extract and interpret low-dimensional neural embeddings from **EEG hyperscanning** of dyadic (two-person) social interactions using the [CEBRA](https://cebra.ai) contrastive learning framework |
| 🏥 **Clinical** | Provide a **real-time BCI pipeline** capable of streaming EEG, detecting anomalies (e.g. seizure onset), and generating clinical alerts using EEGMamba + Topological Data Analysis |

---

## 🎯 Scientific Goals

### 🤝 Inter-Brain Synchrony
We simultaneously record 64-channel EEG from a *speaker* and *listener* during naturalistic conversation. CEBRA learns a shared latent space revealing how two brains co-fluctuate across social turns — a window into **neural coupling and social cognition**.

### 🧩 Clinical vs Neurotypical Comparison
CEBRA embeddings are used to distinguish dyads involving **autistic participants** from neurotypical pairs, probing whether the neural manifold geometry differs across clinical populations.

### 📊 Behavioral Correlates
Continuous psychometric measures — **AQ-10**, **PRCA**, **RSAS**, self-report ratings — are woven into the decoding pipeline to decode individual traits directly from the joint neural manifold.

### 💡 Biomarker Discovery
We systematically identify embedding features (latent dimensions, synchrony metrics, topological signatures) that reliably index social-communication differences in **Autism Spectrum Disorder**.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SARO BCI PLATFORM                               │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐   │
│  │   DATA INPUT  │    │           REAL-TIME PIPELINE                 │   │
│  │              │    │                                              │   │
│  │  PhysioNet   │───▶│  EEG Chunks (128ch @ 250Hz)                 │   │
│  │  LSL Stream  │    │       │                                      │   │
│  │  Synthetic   │    │       ▼                                      │   │
│  └──────────────┘    │  ┌─────────────┐   ┌───────────────────┐   │   │
│                      │  │  Ring Buffer │──▶│  Stateful Filter  │   │   │
│                      │  │  (10s / 128ch)│  │  (1-100Hz + notch)│   │   │
│                      │  └─────────────┘   └────────┬──────────┘   │   │
│                      │                             │               │   │
│                      │                             ▼               │   │
│                      │                    ┌────────────────┐       │   │
│                      │                    │  EEGMamba      │       │   │
│                      │                    │  Inference     │       │   │
│                      │                    │  Engine        │       │   │
│                      │                    └───────┬────────┘       │   │
│                      │                            │                │   │
│                      │              ┌─────────────┼──────────────┐ │   │
│                      │              ▼             ▼              ▼ │   │
│                      │      ┌───────────┐  ┌──────────┐  ┌────────┐│   │
│                      │      │  AI Twin  │  │  TDA /   │  │ MedPalm││   │
│                      │      │  (Healthy │  │  CEBRA   │  │ Alert  ││   │
│                      │      │  State)   │  │ Detector │  │Formatter│   │
│                      │      └───────────┘  └──────────┘  └────────┘│   │
│                      └──────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   RESEARCH PIPELINE (Offline)                    │   │
│  │                                                                  │   │
│  │  EDF Files ──▶ Preprocessing ──▶ CEBRA Training ──▶ Embeddings  │   │
│  │                  (cut_60s)      (Supervised &        + Metrics   │   │
│  │                                  Unsupervised)                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │        FastAPI WebSocket Server  +  Web Dashboard                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
saro-bci/
│
├── 📂 src/
│   ├── 📂 core/                    # Real-time BCI pipeline engine
│   │   ├── pipeline.py             # BCIPipeline — main orchestrator
│   │   ├── ring_buffer.py          # Lock-free circular EEG buffer
│   │   ├── stateful_filter.py      # Butterworth + notch filtering
│   │   ├── inference_engine.py     # EEGMamba forward pass (ONNX / PyTorch)
│   │   └── lsl_bridge.py           # LabStreamingLayer hardware interface
│   │
│   ├── 📂 ai_twin/
│   │   └── eeg_mamba.py            # EEGMamba healthy-state prediction model
│   │
│   ├── 📂 topology/
│   │   └── cebra_tda_detector.py   # CEBRA + Wasserstein divergence detector
│   │
│   ├── 📂 alerting/
│   │   └── medpalm_formatter.py    # Clinical alert JSON generation (MedPaLM-style)
│   │
│   ├── 📂 backend/
│   │   └── server.py               # FastAPI WebSocket server
│   │
│   ├── 📂 preprocessing/
│   │   ├── signal_processor.py     # Band-power, IIR filtering helpers
│   │   └── physionet_fetcher.py    # PhysioNet EEG dataset downloader
│   │
│   ├── 📂 frontend/                # Web dashboard (real-time EEG viewer)
│   └── 📂 dashboard/               # Panel / Bokeh dashboard components
│
├── 📂 EEGMamba/                    # Forked EEGMamba repository (pre-training)
│   ├── models/                     # Model architecture (model_for_bciciv2a.py, ...)
│   ├── pretrained_weights/         # BCI Competition IV 2a pre-trained weights
│   └── finetune_main.py            # Fine-tuning entry point
│
├── 📂 PreprocessingPipeline/       # Offline EEG preprocessing utilities
├── 📂 GenderRuns/                  # Demographic stratification experiments
│
├── prepare_cebra_input.py          # EDF → NumPy data preparation grid
├── train_cebra_cut_supervised.py   # Supervised CEBRA training (5 runs × pairings)
├── train_cebra_cut_unsupervised.py # Unsupervised CEBRA training (time-contrast)
│
├── 📂 models/                      # Saved CEBRA model checkpoints (.pt)
├── 📂 results/                     # Embedding arrays, GoF history, metrics CSVs
├── 📂 data/                        # Raw & preprocessed EEG arrays (.npy / .edf)
│
└── requirements.txt
```

---

## 🔄 Research Pipeline — Offline CEBRA Training

### 1. Data Preparation
EEG recordings are trimmed to the **middle 60 seconds** ("cut_60") per trial, the two 64-channel streams (speaker + listener) are stacked into a **128-channel array**, and saved as raw NumPy arrays — no normalization, which proved most effective empirically.

**Pairings studied:**
- `spk9-lst10` — Participant 9 speaks, Participant 10 listens  
- `lst9-spk10` — Participant 9 listens, Participant 10 speaks
- `stacked` — Full stacked dyad representation
- `patient-ai` — Clinical participant paired with AI-twin reference

### 2. CEBRA Training

```
For each pairing × mode:
  5 independent GPU runs
       │
       ├── train_cebra_cut_supervised.py
       │   Uses speaker/listener role as auxiliary label
       │   → Learns role-conditioned neural manifold
       │
       └── train_cebra_cut_unsupervised.py
           Time-contrastive learning only
           → Learns temporal structure without labels
```

**Key Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Latent dims | 3 |
| Training steps | 5,000 |
| Runs per pairing | 5 |
| Batch size | Auto (CEBRA default) |
| Optimizer | Adam |

### 3. Metrics & Evaluation

| Metric | Description |
|--------|-------------|
| **Variability** | `1 − mean(consistency across runs)` — lower is more stable |
| **Goodness-of-Fit (GoF)** | Info-NCE loss (bits) trajectory over training |
| **Decoding R²** | KNN regression on latent coordinates → speaker/listener label recovery |
| **Topological divergence** | Wasserstein distance between patient and healthy-state manifolds |

---

## ⚡ Real-Time Pipeline — Clinical Monitoring

The `BCIPipeline` class wires all components into a multi-threaded, low-latency processing engine:

```
push_chunk(128ch EEG)
    │
    ▼
[Thread 1 — bci-filter]
  RingBuffer.write() ──▶ RealtimeEEGFilter.process()
    │                    (1-100Hz bandpass + 50Hz notch)
    ▼
[Thread 2 — bci-infer]  (every 1.0s)
  EEGMambaInferenceEngine.infer()      ← anomaly_score, features
    │
    ├──▶ AI Twin healthy-state reconstruction
    │         (from EEGMamba features, no extra forward pass)
    │
    ├──▶ TopologyDetector.measure_divergence()
    │         (CEBRA embeddings + Wasserstein distance)
    │         → is_anomaly (bool), divergence_score (z-score)
    │
    ├──▶ Band power computation
    │         (δ θ α β γ via single FFT — 10× faster than IIR)
    │
    └──▶ MedPalmAlertFormatter.generate_alert_json()
              (Seizure onset zone localisation → structured clinical alert)
```

### Alert System
When `divergence_score > threshold`, the pipeline:
1. Localises the **Seizure Onset Zone (SOZ)** from peak-RMS channel
2. Maps channel index → neuroanatomical region (frontal, temporal, parietal, etc.)
3. Emits a structured JSON alert (MedPaLM-compatible format)
4. Broadcasts via WebSocket to connected dashboard clients

---

## 🤖 EEGMamba Integration

We integrate [EEGMamba](https://github.com/AdaptiveMotorControlLab/EEGMamba) — a state-space model (Mamba architecture) pre-trained on **BCI Competition IV Dataset 2a** — as the core neural feature extractor.

```
128-channel EEG input
    │
    ▼ (auto-select 19 highest-power channels)
EEGMamba forward pass
    │
    ├── anomaly_score  (scalar, 0–1)
    ├── features       (1 × 19 × n_seg × patch_len)
    └── latency_ms     (inference time)
```

> The inference engine supports both **PyTorch native** and **ONNX runtime** backends, auto-selecting based on hardware availability.

---

## 🌐 FastAPI WebSocket Server

`src/backend/server.py` exposes a real-time API:

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Health check + pipeline status |
| `GET /status` | HTTP | Live pipeline metrics (fill ratio, mode, uptime) |
| `WS /ws` | WebSocket | Real-time EEG results stream (JSON, 1Hz) |

The server manages a `lifespan` context that spawns:
- `BCIPipeline` background thread
- Optional `LSLBridge` for real hardware streaming
- `PhysioNet` replay thread for demo/validation mode

---

## 📈 Results & Outputs

After training, outputs are organized as:

```
models/
  {pair}/run{n}/
    {pair}_run{n}.pt          # CEBRA model checkpoint

results/
  embeddings_{pair}_run{n}.npy    # (T × 3) latent coordinates
  gof_history_{pair}_run{n}.csv   # GoF (bits) per training step
  variability_{pair}.csv          # Cross-run consistency metrics
  decoding_r2_{pair}.csv          # KNN decoding R² scores
  plots/
    embedding_scatter_{pair}.png
    loss_curves_{pair}.png
    combined_overview_{pair}.png
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repo
git clone https://github.com/BrokenDecoder/Saro-BCI.git
cd Saro-BCI

# Create a virtual environment (Python 3.10–3.12 recommended for GPU support)
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# OR
.venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users on WSL:** The `.venv_wsl` environment is pre-configured with CUDA-enabled PyTorch. Activate it inside WSL with `source .venv_wsl/bin/activate` for full GPU acceleration.

### Run the Research Pipeline

```bash
# Step 1: Prepare CEBRA inputs from EDF recordings
python prepare_cebra_input.py

# Step 2: Train supervised embeddings (5 runs × all pairings)
python train_cebra_cut_supervised.py

# Step 3: Train unsupervised embeddings
python train_cebra_cut_unsupervised.py

# Outputs saved to models/ and results/
```

### Start the Real-Time Server

```bash
python src/backend/server.py
# → FastAPI server at http://localhost:8000
# → WebSocket stream at ws://localhost:8000/ws
```

### Run the Pipeline Self-Test

```bash
python src/core/pipeline.py
# Feeds 8s of synthetic 128ch EEG and validates every subsystem
```

---

## 📚 Technology Stack

| Component | Technology |
|-----------|-----------|
| Neural embeddings | [CEBRA](https://cebra.ai) 0.4+ |
| Deep learning | [PyTorch](https://pytorch.org) 2.0+ / ONNX Runtime |
| EEG backbone | [EEGMamba](https://github.com/AdaptiveMotorControlLab/EEGMamba) (Mamba SSM) |
| Topology / TDA | Wasserstein distance + persistent homology |
| Signal processing | MNE-Python, SciPy |
| Real-time server | FastAPI + WebSockets |
| Hardware interface | Lab Streaming Layer (LSL) |
| Data format | EDF+, NumPy (.npy), HDF5 |
| GPU acceleration | CUDA (via WSL2 + NVIDIA driver) |

---

## 🧭 Roadmap

- [x] EEG preprocessing pipeline (cut_60, stacking, raw arrays)
- [x] CEBRA supervised & unsupervised training with 5-run variability metrics
- [x] Real-time BCI pipeline (ring buffer → filter → EEGMamba → TDA → alerts)
- [x] FastAPI WebSocket server with live broadcasting
- [x] EEGMamba AI Twin integration for healthy-state prediction
- [x] Topological anomaly detection (Wasserstein divergence)
- [x] Clinical alert formatting (MedPaLM-style structured JSON)
- [x] GPU-accelerated training via WSL2 + CUDA
- [ ] AQ-10 / PRCA / RSAS behavioral label decoding
- [ ] ICA-cleaned and frequency-band-filtered embedding comparison
- [ ] Web dashboard (real-time EEG + alert visualisation)
- [ ] Multi-dyad generalization & dataset plug-in interface
- [ ] EEGMamba fine-tuning on hyperscanning data

---

## 📖 Citations

```bibtex
@inproceedings{barde2019interbrain,
  author    = {Barde, A. and Saffaryazdi, N. and Withana, P. and Patel, N. and Sasikumar, P. and Billinghurst, M.},
  title     = {Inter-brain connectivity: Comparisons between real and virtual environments using hyperscanning},
  booktitle = {IEEE ISMAR-Adjunct},
  year      = {2019},
  doi       = {10.1109/ISMAR-Adjunct.2019.00-17}
}

@article{jazayeri2017navigating,
  author  = {Jazayeri, M. and Afraz, A.},
  title   = {Navigating the neural space in search of the neural code},
  journal = {Neuron},
  volume  = {93},
  number  = {5},
  year    = {2017},
  doi     = {10.1016/j.neuron.2017.02.019}
}

@article{algumaei2023neuroscience,
  author  = {Algumaei, A. and Hettiarachchi, I.T. and Farghaly, M. and Bhatti, A.},
  title   = {The neuroscience of team dynamics: Exploring neurophysiological measures for assessing team performance},
  journal = {IEEE Access},
  volume  = {11},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3332907}
}
```

---

<div align="center">

Built with ❤️ for neuroscience · Open an [issue](https://github.com/BrokenDecoder/Saro-BCI/issues) or [PR](https://github.com/BrokenDecoder/Saro-BCI/pulls) anytime!

</div>
