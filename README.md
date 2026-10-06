# DP-BNN & DP-DQN: Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation and reproducibility suite for the paper:  
**"Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning"**  
*Transactions on Machine Learning Research (TMLR)*.

---

## 📌 Overview

Bayesian Neural Networks (BNNs) provide a principled framework for epistemic uncertainty quantification and deep exploration. However, existing weight-space BNNs, variational approximations, and deep ensembles face steep computational barriers in reinforcement learning—suffering from parametric variance collapse or requiring $K \times$ physical neural network ensembles (e.g., BootDQN-RP requires 60 parameter sets on DeepSea-20).

**DP-BNN** fundamentally reformulates the prior: rather than placing a prior over abstract neural weights $p(\mathbf{w})$, we place a **nonparametric Dirichlet Process prior directly on the data-generating distribution** in experience space:
$$F \sim \mathrm{DP}(\alpha, F_0)$$
Given transition data $\mathcal{D}$, the conjugate posterior measure is:
$$F \mid \mathcal{D} \sim \mathrm{DP}\left(\alpha + n, \, \frac{n}{\alpha + n} \mathbb{P}_n + \frac{\alpha}{\alpha + n} F_0\right)$$

Building on this data-space prior, **DP-DQN** implements pure Thompson Sampling over action-value functions using only a **single living neural network**:
1. **Episodic Prior Sampling:** Prior transitions $\tilde{\mathcal{D}} \sim F_0$ and stick-breaking weights $q^{(0)}$ are sampled **once** at the start of each episode.
2. **Dynamic Empirical Mini-Batches:** In the warmstart loop ($w = 1, \dots, W$), fresh empirical mini-batches are drawn from the replay buffer to update the network against the sampled hypothesis.
3. **Polyak Target Tracking:** Target network tracking is updated at episode boundaries via $\theta^- \leftarrow (1-\tau)\theta^- + \tau \theta$.
4. **$30\times$ Parameter Reduction:** Matches or outperforms $K=20$ deep ensembles while using only 1 active network.

---

## 🚀 Quickstart & Installation

### Requirements
- Python 3.9+
- PyTorch >= 2.0.0
- NumPy >= 1.22.0
- SciPy >= 1.9.0
- Matplotlib >= 3.5.0

### Setup
```bash
git clone https://github.com/vashishthasumit/DP-BNN.git
cd DP-BNN
pip install -r requirements.txt
```

---

## ⚡ Turnkey Reproducibility Suite (1-Command Verification)

To independently verify the key empirical claims and benchmarks in the paper, run the automated verification suite:

```bash
python test_reproducibility_suite.py
```

### Verified Claims:
- **Test 1: 2D Annular Cavity Void Regression (Figure 1 & Section 4.5)**  
  Verifies the sharp information-theoretic BALD epistemic bubble in the unobserved void ($\|x\| \le 1.0$) with contrast ratio **$> 5.0\times$** relative to the data support.
- **Test 2: Canonical RiverSwim-6 with Standard Normal Prior (Table 1 & Figure 4a)**  
  Verifies **100% solve rate** across seeds using an uninformative $\mathcal{N}(0, 1)$ prior with zero jackpot bias.
- **Test 3: Deceptive $N$-Chain ($N=10$) (Table 1 & Appendix E)**  
  Verifies environment dynamics and discovery of optimal goal policy ($r=1.0$ at depth 10).
- **Test 4: DeepSea-20 Performance & $\mathcal{O}(N^{1.45})$ Scaling (Figure 2 & Table 1)**  
  Verifies DeepSea-20 convergence in $602 \pm 104$ episodes, cumulative regret plateau at $418 \pm 94$, and sub-quadratic empirical scaling across 180 runs (30 grid sizes, $N=10$ to $50$).

---

## 📊 Paper Figures & Benchmarks

| Figure / Table | Benchmark | Script | Output |
| :--- | :--- | :--- | :--- |
| **Figure 1** | 2D Annular Manifold Regression (BALD) | `python run_2d_circular_regression_layernorm_ablation.py` | `Figure_1_2D_regression_heatmaps_NoLN.pdf` |
| **Figure 2** | DeepSea Regret & 30-Size Scaling Suite | `python plot_Figure_2_TMLR.py` | `Figure_2_TMLR.png` / `.pdf` |
| **Figure 4a** | Canonical RiverSwim-6 ($\mathcal{N}(0, 1)$ Prior) | `python test_riverswim_std_normal_mlp20.py` | `Figure_RiverSwim_Canonical.png` |
| **Table 1** | Cross-Domain Summary (RiverSwim, Chain, MC, Wheel) | `python test_reproducibility_suite.py` | Console Summary Audit |
| **Table 2** | Computational Complexity & Latency Profiling | `python run_strictly_matched_dp_vs_bootdqn.py` | Runtime & Memory Table |

---

## 📂 Repository Structure

```
├── README.md                           # Documentation and reproduction guide
├── requirements.txt                    # Minimal dependencies
├── test_reproducibility_suite.py       # Turnkey 4-benchmark test suite (all paper claims)
├── plot_Figure_2_TMLR.py               # Generates Figure 2 (DeepSea-20 + Scaling suite)
├── run_2d_circular_regression_layernorm_ablation.py # Generates Figure 1 (Annular manifold)
├── test_riverswim_std_normal_mlp20.py  # RiverSwim-6 with zero jackpot bias N(0, 1) prior
├── main_tmlr.tex                       # Complete LaTeX source of TMLR paper
├── Figure_2_TMLR_data.json             # 180-run scaling data across 30 grid points
├── Figure_3_TMLR_data.json             # DeepSea-20 cumulative regret benchmark data
├── src/
│   └── dp_dqn/
│       ├── agent.py                    # Core Unified DP-DQN Agent (Algorithm 2)
│       ├── config.py                   # Configuration dataclass
│       ├── networks.py                 # Single living Q-network with LayerNorm
│       ├── sampler.py                  # Stick-breaking Dirichlet Process posterior sampler
│       ├── base_measures.py            # Domain base measures F_0 (Uniform, Gaussian, etc.)
│       └── environments.py             # Benchmark environments (DeepSea, RiverSwim, etc.)
└── unified_dp_dqn/                     # Modular lightweight library package
```

---

## 📖 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{vashishtha2026dpbnn,
  title={Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning},
  author={Vashishtha, Sumit},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026}
}
```

---

## 📬 Contact
Sumit Vashishtha  
Email: `vash.sumit@gmail.com`