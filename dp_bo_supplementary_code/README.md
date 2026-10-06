# Supplementary Code: Data-Space Dirichlet Process Bayesian Neural Networks (DP-BNNs)

Official code package accompanying the ICLR 2026 submission:  
**"Data-Space Dirichlet Process Bayesian Neural Networks: Resolving Epistemic Uncertainty Collapse in the Data Void"**

---

## 1. Overview

This supplementary material package provides clean, modular, self-contained implementations of:
1. **Single-Network DP Thompson Sampling (DP-TS)**: Algorithm 1 from the paper. Optimizing a single deterministic neural network on an atomic realization of the Dirichlet Process posterior measure generates an exact functional Thompson sample without multi-head or ensemble overhead.
2. **Baselines**:
   - Standard BNN with Monte Carlo Dropout ($S=20$ stochastic passes, Gal & Ghahramani, 2016)
   - Exact Gaussian Process Regression (Matérn / RBF kernel, Rasmussen & Williams, 2006)
   - Multi-Head DP-BO Ensemble ($K=4$ neural heads under Dirichlet draws with LCB)
   - Uniform Random Search
3. **Benchmarks**:
   - **4-Benchmark Multimodal Suite**: Ackley 2D, Levy 2D, Rosenbrock 4D, Rastrigin 4D (Table 1 & Figure 1)
   - **High-Dimensional Robotics ($d=60$)**: 60-dimensional Rover trajectory planning around circular obstacle discs (Table D.1 & Figure D.1)
   - **Continuous Policy Optimization ($d=32$)**: 32-parameter neural controller for OpenAI Gym `Pendulum-v1` (Table E.1)

---

## 2. Directory Structure

```
dp_bo_supplementary_code/
├── README.md                      # This documentation
├── requirements.txt               # Lightweight Python dependencies
├── run_all_reproductions.py       # 1-Click master reproduction script
│
├── models/                        # Surrogate models
│   ├── __init__.py
│   ├── dp_surrogate.py            # SingleNetworkDPTS (Alg. 1) & MultiHeadDPBO
│   ├── bnn_dropout.py             # Standard BNN (MC-Dropout)
│   └── gp_surrogate.py            # Exact Gaussian Process Regression
│
├── benchmarks/                    # Benchmark environments and objective functions
│   ├── __init__.py
│   ├── multimodal_landscapes.py   # Ackley 2D, Levy 2D, Rosenbrock 4D, Rastrigin 4D
│   ├── rover_60d.py               # 60D Rover trajectory planning environment
│   └── pendulum_32d.py            # 32D Pendulum continuous policy search
│
├── scripts/                       # Reproduction & plotting scripts
│   ├── run_table1_multimodal.py   # Reproduce Table 1
│   ├── run_rover_60d.py           # Reproduce Table D.1
│   ├── run_pendulum_32d.py        # Reproduce Table E.1
│   ├── plot_figure1_showdown.py   # Regenerate Figure 1 (figure_dp_bo_showdown.png)
│   └── plot_figure_rover_60d.py   # Regenerate Figure D.1 (figure_rover_60d.png)
│
├── results/                       # Precomputed experimental results (JSON)
│   ├── dp_bo_results.json         # Table 1 & Figure 1 raw data
│   ├── results_rover_60d_bo.json  # Table D.1 & Figure D.1 raw data
│   └── results_nn_policy_bo.json  # Table E.1 raw data
│
└── figures/                       # Publication figures
    ├── figure_dp_bo_showdown.png  # 4-panel multimodal showdown
    └── figure_rover_60d.png       # 2-panel 60D Rover trajectory & obstacle map
```

---

## 3. Installation & Setup

We recommend creating a clean virtual environment using Python 3.8+:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- `torch >= 2.0.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.5.0`
- `gym >= 0.21.0` (for Pendulum-v1 continuous control)

---

## 4. Quick Start: Reproducing Results

### 1-Click Master Reproduction
To run and display all quantitative tables (Table 1, Table D.1, Table E.1) and re-render all publication figures (Figure 1, Figure D.1):

```bash
python run_all_reproductions.py
```

### Reproducing Individual Tables & Figures

1. **Table 1: 4-Benchmark Multimodal Suite**
   ```bash
   python scripts/run_table1_multimodal.py
   # To launch fresh live optimization runs (35 steps each):
   python scripts/run_table1_multimodal.py --live
   ```

2. **Table D.1: 60D Rover Robotics Trajectory Planning**
   ```bash
   python scripts/run_rover_60d.py
   ```

3. **Table E.1: 32D Pendulum Continuous Control Policy Search**
   ```bash
   python scripts/run_pendulum_32d.py
   ```

4. **Figure 1: Multimodal Showdown (4 Panels)**
   ```bash
   python scripts/plot_figure1_showdown.py
   # Output saved to: figures/figure_dp_bo_showdown.png
   ```

5. **Figure D.1: 60D Rover Trajectory & Obstacle Map (2 Panels)**
   ```bash
   python scripts/plot_figure_rover_60d.py
   # Output saved to: figures/figure_rover_60d.png
   ```

---

## 5. Hardware Specifications & Computational Complexity

All experiments reported in the paper and reproduced here were conducted on standard consumer hardware (Apple M-series CPU, 16 GB RAM).
- **Single-Network DP-TS**: Linear time complexity $\mathcal{O}(t)$ per optimization step, averaging **21.6 ms/step** on 2D/4D benchmarks and **21.8 ms/step** on 60D Rover.
- **Model Memory**: Maintains a single network in memory ($\mathcal{O}(1)$ model memory, $<15$ MB VRAM vs. $>60$ MB for ensembles).
