# Research Blueprint: DP-MCTS (Bayesian Deep Monte Carlo Tree Search via Dirichlet Process Value Posteriors)

**Author:** Sumit Vashishtha  
**Status:** Saved for execution post-TMLR camera-ready submission  
**Target Venues:** ICML / UAI / NeurIPS / ICLR  

---

## 1. Executive Summary & Vision

Modern deep Monte Carlo Tree Search (MCTS), popularized by AlphaZero and MuZero, relies on two critical yet mathematically unprincipled heuristics:
1. **Deterministic Point-Estimate Leaf Evaluation:** The value network $V_\theta(s)$ lacks epistemic uncertainty; when tree search encounters rare or out-of-distribution (OOD) states, $V_\theta$ hallucinates arbitrary confidence, corrupting the tree backup.
2. **The Root Dirichlet Noise Hack:** DeepMind artificially injects heuristic Dirichlet noise $\eta \sim \operatorname{Dir}(\alpha)$ solely at the root node because parameter-space networks cannot maintain coherent functional uncertainty across the search tree.

**DP-MCTS** resolves this foundational limitation by placing a Bayesian nonparametric **Dirichlet Process (DP) prior** directly over the state-action-reward-nextstate experience distribution:
$$\mathcal{P} \sim \mathrm{DP}(\alpha, F_0)$$

By coupling the DP stick-breaking representation with MCTS, DP-MCTS enables **pure Bayesian Thompson Sampling across the tree depth using a single neural network**, eliminating heuristic exploration hyperparameter tuning ($c_{\text{puct}}$) while preventing out-of-distribution value hallucination.

Furthermore, this framework directly addresses the emerging frontier of **test-time compute scaling and reasoning in Large Language Models (LLMs)**, where Process Reward Models (PRMs) suffer from reward hacking and verifier exploitation.

---

## 2. Mathematical & Algorithmic Framework

### 2.1 The DP Experience Prior & Stick-Breaking
Let the environment transition-value tuples be $z = (s, a, r, s')$. We place:
$$P \sim \mathrm{DP}(\alpha, F_0), \quad \text{where } F_0 = F_{0s} \times F_{0a} \times F_{0r} \times F_{0s'}$$
Conditioned on observed search tree trajectories $\mathcal{D}_n$, the posterior random measure is generated via stick-breaking:
$$P^{(m)} = \sum_{i=1}^B p_i \delta_{z_i} + \sum_{j=1}^T \tilde{p}_j \delta_{\tilde{z}_j}, \quad z_i \sim \mathcal{D}_n, \quad \tilde{z}_j \sim F_0$$
where $(p_1, \dots, p_B) \sim \operatorname{Dir}(1, \dots, 1)$ and $\tilde{p}_j = V_j \prod_{l=1}^{j-1}(1 - V_l)$ with $V_j \sim \operatorname{Beta}(1, \alpha)$.

### 2.2 Algorithmic Phases of DP-MCTS

```
Algorithm: DP-MCTS (Pure Thompson Sampling in Deep Tree Search)
--------------------------------------------------------------------------------
Input: Root state s_0, simulation budget M, DP prior (alpha, F_0), single network Q(s, a; theta)
For simulation m = 1, ..., M:
    1. Posterior Model Draw:
       Draw stick-breaking measure P^(m) ~ DP(alpha + n, F_bar_n).
       Obtain temporary episode network parameters theta^(m) via warmstart update under P^(m).
       Freeze theta^(m) for the entirety of simulation trajectory m.
       
    2. Selection (Pure Thompson Sampling):
       Traverse tree from root using sampled posterior values:
           a_t = argmax_a Q(s_t, a; theta^(m))
       (Zero reliance on heuristic c_puct tuning).
       
    3. Expansion:
       Expand leaf node s_L.
       
    4. Nonparametric Epistemic Leaf Evaluation:
       If s_L is in data-sparse region (N(s_L) approx 0):
           Q(s_L, a; theta^(m)) automatically regularized by base measure F_0 (prevents hallucination).
           
    5. Backup:
       Propagate returns along the selected tree path to update visit counts N(s, a) and mean Q(s, a).
       
Output: Execute action a* = argmax_a N(s_0, a).
```

### 2.3 Key Advantages over Existing Paradigms
| Feature | Classical AlphaZero MCTS | Deep Ensemble MCTS | **DP-MCTS (Ours)** |
| :--- | :--- | :--- | :--- |
| **Exploration Mechanism** | Heuristic PUCT + Root Noise | Ensemble Variance Penalty | **Pure Thompson Sampling** |
| **Epistemic Uncertainty** | None (Deterministic $V_\theta$) | Parameter-space spread | **Data-Space Nonparametric** |
| **Network Count** | 1 network | 10–20 parallel networks | **1 network** |
| **Search Latency** | Baseline ($1\times$) | $10\times$ slower | **$\approx 1.2\times$ baseline** |
| **Deep Tree Hallucinations** | Severe | Moderate | **Physically Bounded by $F_0$** |

---

## 3. Empirical Evaluation Suite

1. **Deceptive Tree Search & Planning Puzzles:**
   * **Sokoban & MiniHack / Key-Door Gridworlds:** Environments with deceptive local dead-ends where standard MCTS value networks hallucinate non-existent escapes.
   * Metric: Success rate, search depth, number of node expansions to solution.
2. **Tactical & Strategic Games:**
   * **Connect Four, Gomoku, Mini-Chess:**
   * Direct match-play showdown against AlphaZero baselines under fixed simulation budgets ($M \in \{50, 100, 400, 800\}$).
3. **Continuous Control Tree Search:**
   * Continuous-action planning on MuJoCo / Gym benchmarks with discrete tree discretization.
4. **Computational Latency & Memory Profiling:**
   * Wallclock search time per move comparing DP-MCTS vs. Deep Ensemble MCTS (5, 10, 20 heads).

---

## 4. Strategic Discussion: The LLM & Process Reward Model (PRM) Bridge

To connect the paper directly to the current frontier of AI (test-time compute scaling, OpenAI o1, DeepSeek-R1, AlphaProof):

* **The Problem in LLM Reasoning:**  
  Multi-step mathematical and code reasoning is formulated as MCTS over intermediate thought steps guided by a Process Reward Model (PRM) $V_\theta(s)$. Current PRMs suffer from *verifier exploitation* (Goodhart's Law): the search engine discovers out-of-distribution, flawed reasoning steps that fool the PRM into predicting high rewards.
* **The DP-PRM Solution:**  
  By replacing deterministic PRMs with **DP-BNN Process Reward Models**, intermediate thought steps with low empirical coverage automatically incur a non-parametric pessimistic value penalty via $F_0$. This bounds verifier hallucination, ensuring that test-time search remains grounded in rigorous logic without requiring prohibitive PRM ensembles.

---

## 5. Working Titles & Target Deadlines

* **Recommended Title:**  
  *Bayesian Deep Monte Carlo Tree Search via Dirichlet Process Value Posteriors*
* **Alternative Title:**  
  *DP-MCTS: Nonparametric Thompson Sampling in Deep Monte Carlo Tree Search*
* **Submission Venues:**  
  * **ICML (Late January)** — Primary Target
  * **UAI (Mid-February)** — Direct Bayesian / UQ Community Target
  * **TMLR (Rolling)** — Fast journal track with conference presentation
