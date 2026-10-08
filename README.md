# DP-BNN & DP-DQN: Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Reproducibility CI](https://github.com/sumilations/Dirichlet-Process-Bayesian-Neural-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/sumilations/Dirichlet-Process-Bayesian-Neural-networks/actions/workflows/ci.yml)
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

## 🗺️ Paper-to-Code Traceability Table

Every theoretical object, equation, and algorithmic step in the paper corresponds directly to self-contained code modules:

| Paper Reference | Theoretical Component | Implementation Location | Description |
| :--- | :--- | :--- | :--- |
| **Equations (2)–(3)** | Dirichlet Process Prior $F \sim \mathrm{DP}(\alpha, F_0)$ | [`src/dp_dqn/base_measures.py`](src/dp_dqn/base_measures.py) | Nonparametric base measure generators $F_0$ |
| **Equation (4)** | Conjugate Posterior Measure $F \mid \mathcal{D}$ | [`src/dp_dqn/sampler.py`](src/dp_dqn/sampler.py) | Posterior update balancing empirical data and $F_0$ |
| **Equation (5)** | Stick-Breaking Construction $\mathrm{GEM}(\alpha)$ | [`src/dp_dqn/sampler.py`](src/dp_dqn/sampler.py) | Sethuraman stick-breaking weights $q_k$ |
| **Algorithm 2** | Unified DP-DQN Algorithm | [`src/dp_dqn/agent.py`](src/dp_dqn/agent.py) | Episodic prior sampling & dynamic empirical mini-batches |
| **Equation (7)** | Target Network Warmstart Update | [`src/dp_dqn/agent.py`](src/dp_dqn/agent.py) | Gradient descent on mixed posterior sample |
| **Section 4.5 & Fig. 1** | BALD Information Gain in Cavity Void | [`run_2d_circular_regression_layernorm_ablation.py`](run_2d_circular_regression_layernorm_ablation.py) | Sharp epistemic bubble in unobserved void |
| **Section 5.1 & Fig. 2** | DeepSea Regret & $\mathcal{O}(N^{1.45})$ Scaling | [`plot_Figure_2_TMLR.py`](plot_Figure_2_TMLR.py) | 30-size grid scalability ($N=10$ to $50$) across 180 runs |
| **Section 5.1 & Fig. 3** | DeepSea-20 Benchmark Showdown | [`plot_Figure_3_TMLR.py`](plot_Figure_3_TMLR.py) | Comparison against BootDQN-RP, BDQN, Bootstrap Limit |
| **Table 1 & Fig. 4a** | RiverSwim-6 with Standard Normal Prior | [`test_riverswim_std_normal_mlp20.py`](test_riverswim_std_normal_mlp20.py) | 100% solve rate under uninformative $\mathcal{N}(0, 1)$ prior |
| **Table 1 & App. E** | Deceptive $N$-Chain Environment | [`src/dp_dqn/environments.py`](src/dp_dqn/environments.py) | Deceptive reward structure with delayed goal |

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
git clone https://github.com/sumilations/Dirichlet-Process-Bayesian-Neural-networks.git
cd Dirichlet-Process-Bayesian-Neural-networks
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

| Figure / Table | Benchmark | Reproduction Script | Output |
| :--- | :--- | :--- | :--- |
| **Figure 1** | 2D Annular Manifold Regression (BALD) | `python run_2d_circular_regression_layernorm_ablation.py` | `Figure_1_2D_regression_heatmaps_NoLN.pdf` |
| **Figure 2** | DeepSea Regret & 30-Size Scaling Suite | `python plot_Figure_2_TMLR.py --cached` | `Figure_2_TMLR.png` / `.pdf` |
| **Figure 3** | DeepSea-20 Regret Comparison | `python plot_Figure_3_TMLR.py --cached` | `Figure_3_TMLR.png` / `.pdf` |
| **Figure 4a** | Canonical RiverSwim-6 ($\mathcal{N}(0, 1)$ Prior) | `python test_riverswim_std_normal_mlp20.py` | `Figure_RiverSwim_Canonical.png` |
| **Verification Suite** | Cross-Domain Summary (RiverSwim, Chain, DeepSea) | `python test_reproducibility_suite.py` | Automated 4-test verification |

---

## 📂 Repository Structure

```
.github/
└── workflows/
    └── ci.yml                          # Continuous Integration (Python 3.9-3.12 automated tests)
├── README.md                           # Documentation and paper-to-code traceability
├── requirements.txt                    # Minimal dependencies (torch, numpy, scipy, matplotlib)
├── test_reproducibility_suite.py       # Turnkey 4-benchmark test suite (all paper claims)
├── plot_Figure_2_TMLR.py               # Generates Figure 2 (DeepSea-20 + Scaling suite)
├── plot_Figure_3_TMLR.py               # Generates Figure 3 (DeepSea-20 Regret Comparison)
├── run_2d_circular_regression_layernorm_ablation.py # Generates Figure 1 (Annular manifold)
├── test_riverswim_std_normal_mlp20.py  # RiverSwim-6 with zero jackpot bias N(0, 1) prior
├── Figure_2_TMLR_data.json             # 180-run scaling data across 30 grid points
├── Figure_3_TMLR_data.json             # DeepSea-20 cumulative regret benchmark data
├── Figure_RiverSwim_Canonical.json     # RiverSwim-6 benchmark data
├── results_option_c_dag_vs_nondag.json # DeepSea Option C verified run data
└── src/
    └── dp_dqn/
        ├── agent.py                    # Core Unified DP-DQN Agent (Algorithm 2)
        ├── config.py                   # Configuration dataclass
        ├── networks.py                 # Single living Q-network with LayerNorm
        ├── sampler.py                  # Stick-breaking Dirichlet Process posterior sampler
        ├── base_measures.py            # Domain base measures F_0 (Uniform, Gaussian, etc.)
        ├── environments.py             # Environments (DeepSea, RiverSwim, Gym wrappers)
        ├── trainer.py                  # Polyak target tracking & training loop
        └── bdqn.py                     # Bootstrapped DQN baseline implementation
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