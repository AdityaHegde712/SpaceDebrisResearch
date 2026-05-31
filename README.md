# Space Debris Research: High-Fidelity Probabilistic Risk Assessment

This repository contains a high-fidelity research framework and pipeline to model space debris collision risk and trajectory propagation. By leveraging 13.59 million Vector Covariance Message (VCM) snapshots across 19,300 objects, the system transitions from deterministic Keplerian point-models to uncertainty-aware 3D probability volumes.

---

## 📂 Repository Architecture

The project is structured logically into data pipelines, shared source modules, trained models, metrics reports, and documentation:

```
├── data/                         # Intermediate and ML-ready Parquet files (gitignored)
├── celestrak_data/               # Celestrak fragmentation event CSV logs (gitignored)
├── src/                          # Shared source modules
│   ├── models/                   # Neural network architectures
│   │   └── tcn.py                # PyTorch Temporal Convolutional Network model
│   ├── constants.py              # Astro-dynamics & training constants (GM, R_Earth, SEQ_LEN)
│   ├── feature_utils.py          # Columns selection, categorical encoding, scaling utilities
│   ├── io_utils.py               # Optimized file I/O & logging timers
│   ├── polars_utils.py           # Polars helper functions (altitude bands, gaps calculation)
│   ├── scaler.py                 # Custom standard scaling module
│   ├── vcm_parser.py             # Raw VCM text file parsing functions
│   └── window_utils.py           # Memory-efficient sliding window extraction
├── scripts/                      # Data pipeline execution stages & baselines
│   ├── pipeline_phase1.py        # Stage 0-5 sequential data preparation
│   ├── phase2_feature_engineering.py # Compiles 41 features across 7 categories
│   ├── t3a1_create_targets.py    # Generates sliding-window targets
│   ├── t3a2_rf_baseline.py       # Random Forest classifier baseline
│   ├── t3a4_tcn_baseline.py      # PyTorch TCN sequence classifier baseline
│   ├── phase3b_physics_baseline.py # RK4 J2 perturbation numerical propagator
│   └── analyze_errors.py         # Physics propagation error analyzer
├── models/                       # Saved model checkpoints & scalers
│   ├── physics_propagator.py     # Core RK4 numerical integration routine
│   ├── feature_list.json         # Grouped feature columns metadata
│   ├── scaler.pkl                # Standard Scaler parameters
│   ├── rf_*.pkl                  # Saved Random Forest classifiers
│   └── tcn_*.pt                  # PyTorch TCN checkpoints
├── reports/                      # Evaluation reports and performance metrics
│   ├── physics_baseline_errors.* # Baseline numerical propagation errors
│   ├── rf_*_metrics.json         # Random Forest evaluation output metrics
│   ├── tcn_*_metrics.json        # TCN evaluation output metrics
│   └── t3a1_target_summary.json  # Data splits & risk thresholds summary
└── docs/                         # Project proposals, diagnostics, and logs
    ├── research_proposal_2.0.md  # Main Research Proposal
    ├── proposal_summary.md       # Executive Proposal Summary (conceptual, non-ML)
    ├── diagnosis.md              # Detailed diagnosis of the TCN classification failure
    ├── vcm_work.md               # Work log and initial Keplerian case study
    ├── ML_Variables_bySize.docx.md # Sizing feature guide (5-10cm, 10-25cm, 25-50cm)
    └── roadmap_for_others.md     # Reference ML curriculum guidelines
```

---

## ⚡ Execution Pipeline

The workflow runs in sequential phases to ingest raw observations and generate models/reports:

### 1. Phase 1: Data Preparation
```bash
python scripts/pipeline_phase1.py
```
*   **Stage 0:** Projects J2K (ECI) coordinates into Radial, Along-Track, and Cross-Track (RTN/RSW) frames.
*   **Stage 1:** Derives Keplarian orbital elements from raw coordinates.
*   **Stage 2:** Merges Celestrak satellite databases to identify object classification classes.
*   **Stage 3:** Organizes observation data into sliding historical sequences of length $L = 20$ (168-hour lookback).
*   **Stage 4:** Splits objects temporally into train ($70\%$), val ($15\%$), and test ($15\%$) sets.
*   **Stage 5:** Generates a data profiling report saved to `data/quality_report.html`.

### 2. Phase 2: Feature Engineering
```bash
python scripts/phase2_feature_engineering.py
```
*   Calculates atmospheric density using the Harris-Priester model.
*   Engineers 41 distinct features (AMR, Decay Rates, Uncertainty Volumes, Solar Flux Indices).
*   Generates size-specific subsets (5–10 cm, 10–25 cm, 25–50 cm) and fits standard scalers.

### 3. Phase 3: Target Creation
```bash
python scripts/t3a1_create_targets.py
```
*   Thresholds the 3D uncertainty volume per altitude band to establish classification targets (`collision_risk`).
*   Extracts multi-task regression targets (`future_pos_x/y/z` and `future_sigma_r/t/w`) representing coordinates at timestep $t+1$.

### 4. Phase 4: Baseline Modeling & Evaluation
*   **Physics baseline:** Computes orbital propagation errors using RK4 J2 numerical propagation:
    ```bash
    python scripts/phase3b_physics_baseline.py
    ```
*   **Random Forest baseline:** Evaluates snapshot classifiers on sequence-end data:
    ```bash
    python scripts/t3a2_rf_baseline.py
    ```
*   **TCN sequence baseline:** Trains sequence convolutional neural networks on PyTorch:
    ```bash
    python scripts/t3a4_tcn_baseline.py
    ```

---

## 📊 Baseline Findings

*   **Numerical Propagation Drift:** Position errors grow exponentially without correction due to unmodeled upper-atmosphere swell. The RK4 baseline median error increases from **$41.87\text{ km}$** at 1 day to **$2,244.46\text{ km}$** at 7 days.
*   **Machine Learning Classification Mismatch:**
    *   The Random Forest snapshot classifier achieved **$100\%$ accuracy** ($1.0$ ROC-AUC) due to tautological leakage: the synthetic risk target was defined directly by thresholding `log_uncertainty_volume`, which was provided as an input feature.
    *   The sequence-based TCN failed completely, scoring **$52.52\%$ accuracy** (random guessing). Global average pooling diluted the final-timestep risk signal over the 20-epoch sequence window.

---

## 💡 Research Pivot: Hybrid Physics-ML Trajectory Correction

Based on baseline diagnostics, the project is pivoting away from classification toward **multi-task trajectory regression**. 

Instead of expecting sequence networks to learn absolute coordinates directly (which breaks Newtonian constraints), a **hybrid propagator** is being developed:
1.  A standard physical numerical propagator (RK4 + J2) predicts a nominal gravity-driven orbital trajectory.
2.  A temporal sequence model utilizes solar flux data ($F10.7$), local atmospheric density ($\rho$), and historical residual errors to predict the *drag and solar wind deviation vector* ($\mathbf{r}_{true} - \mathbf{r}_{physics}$) and uncertainty growth ($\sigma_{R, T, W}$).
3.  The predicted correction is added back to the nominal gravity track to yield high-accuracy, uncertainty-aware positions, feeding into a voxel-based Dynamic Probabilistic Occupancy Map (DPOM) to optimize launch windows.
