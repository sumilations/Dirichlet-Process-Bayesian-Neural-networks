import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Stiff Reaction Equation with Sharp Travelling Wavefront
#    du/dt = rho * u * (1 - u),  t in [0, 1], u(0) = u0
#    Exact solution: u(t) = (u0 * e^{rho * t}) / (1 - u0 + u0 * e^{rho * t})
#    With rho = 10.0 and u0 = 0.001, the solution undergoes an extremely steep
#    flame-like transition around t in [0.6, 0.8], while t < 0.5 is almost flat zero.
# -----------------------------------------------------------------------------

class PINN_MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        curr = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            curr = hidden_dim
        layers.append(nn.Linear(curr, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def pde_residual(model, t, rho=10.0):
    t = t.clone().detach().requires_grad_(True)
    u = model(t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    res = u_t - rho * u * (1.0 - u)
    return res

def run_collocation_benchmark(rho=10.0, u0=0.001, n_col=40, candidate_pool_sz=500, rounds=10, steps_per_round=150):
    print("=" * 75)
    print(f"TESTING DP-BNN ACTIVE COLLOCATION FOR STIFF PINNs (rho = {rho})")
    print(f"Budget: Exactly {n_col} collocation points per round across {rounds} rounds")
    print("=" * 75)

    torch.manual_seed(42)
    np.random.seed(42)

    # Evaluation grid
    t_eval = torch.linspace(0, 1.0, 500).unsqueeze(1)
    u_exact = (u0 * torch.exp(rho * t_eval)) / (1.0 - u0 + u0 * torch.exp(rho * t_eval))
    u_exact_np = u_exact.numpy().squeeze()

    # Initial condition data
    t_data = torch.tensor([[0.0]])
    u_data = torch.tensor([[u0]])

    results = {}

    # -------------------------------------------------------------------------
    # METHOD 1: Uniform Static Collocation (Standard PINN)
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Standard Uniform Collocation PINN...")
    m_uni = PINN_MLP()
    opt_uni = torch.optim.Adam(m_uni.parameters(), lr=3e-3)
    t_col_uni = torch.linspace(0, 1.0, n_col).unsqueeze(1)

    t0 = time.time()
    for rnd in range(rounds):
        for _ in range(steps_per_round):
            opt_uni.zero_grad()
            l_data = torch.mean((m_uni(t_data) - u_data) ** 2)
            res = pde_residual(m_uni, t_col_uni, rho=rho)
            l_pde = torch.mean(res ** 2)
            loss = 10.0 * l_data + l_pde
            loss.backward()
            opt_uni.step()

    pred_uni = m_uni(t_eval).detach().numpy().squeeze()
    l2_uni = np.sqrt(np.mean((pred_uni - u_exact_np) ** 2)) / np.sqrt(np.mean(u_exact_np ** 2))
    print(f"--> Uniform Collocation: Relative L2 Error = {l2_uni:.4e} ({time.time()-t0:.2f}s)")
    results["Uniform Collocation"] = {"pred": pred_uni, "error": l2_uni, "col_pts": t_col_uni.numpy().squeeze()}

    # -------------------------------------------------------------------------
    # METHOD 2: Residual-based Adaptive Refinement (RAR / Greedy Residual)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Greedy Residual-based Adaptive Refinement (RAR)...")
    m_rar = PINN_MLP()
    opt_rar = torch.optim.Adam(m_rar.parameters(), lr=3e-3)
    t_col_rar = torch.linspace(0, 1.0, 10).unsqueeze(1)  # start with 10 seed points

    t0 = time.time()
    for rnd in range(rounds):
        for _ in range(steps_per_round):
            opt_rar.zero_grad()
            l_data = torch.mean((m_rar(t_data) - u_data) ** 2)
            res = pde_residual(m_rar, t_col_rar, rho=rho)
            l_pde = torch.mean(res ** 2)
            loss = 10.0 * l_data + l_pde
            loss.backward()
            opt_rar.step()

        # RAR greedily picks points with highest |residual| from candidate pool
        t_cand = torch.linspace(0, 1.0, candidate_pool_sz).unsqueeze(1)
        res_cand = torch.abs(pde_residual(m_rar, t_cand, rho=rho)).squeeze().detach()
        top_k_idx = torch.topk(res_cand, k=3)[1]
        new_pts = t_cand[top_k_idx]
        t_col_rar = torch.cat([t_col_rar, new_pts], dim=0)
        if len(t_col_rar) > n_col:
            t_col_rar = t_col_rar[-n_col:]

    pred_rar = m_rar(t_eval).detach().numpy().squeeze()
    l2_rar = np.sqrt(np.mean((pred_rar - u_exact_np) ** 2)) / np.sqrt(np.mean(u_exact_np ** 2))
    print(f"--> Greedy RAR: Relative L2 Error = {l2_rar:.4e} ({time.time()-t0:.2f}s)")
    results["Greedy RAR"] = {"pred": pred_rar, "error": l2_rar, "col_pts": t_col_rar.detach().numpy().squeeze()}

    # -------------------------------------------------------------------------
    # METHOD 3: DP-BNN Epistemic Thompson Sampling Collocation (Ours)
    # -------------------------------------------------------------------------
    print("\n[3/3] Running DP-BNN Active Bayesian Collocation (DP-TS-PINN, Ours)...")
    m_dp = PINN_MLP()
    opt_dp = torch.optim.Adam(m_dp.parameters(), lr=3e-3)
    t_col_dp = torch.linspace(0, 1.0, 10).unsqueeze(1)

    alpha_dp = 5.0
    all_selected_col = []

    t0 = time.time()
    for rnd in range(rounds):
        # 1. Optimize on current collocation set
        for _ in range(steps_per_round):
            opt_dp.zero_grad()
            l_data = torch.mean((m_dp(t_data) - u_data) ** 2)
            res = pde_residual(m_dp, t_col_dp, rho=rho)
            l_pde = torch.mean(res ** 2)
            loss = 10.0 * l_data + l_pde
            loss.backward()
            opt_dp.step()

        # 2. DP-Thompson Sampling of Collocation Points:
        # Draw 5 stochastic posterior perturbations via stick-breaking weights over candidate residual pool
        t_cand = torch.linspace(0, 1.0, candidate_pool_sz).unsqueeze(1)
        res_pool = torch.abs(pde_residual(m_dp, t_cand, rho=rho)).squeeze().detach()

        # Compute empirical Dirichlet posterior weight over candidate pool
        n_pts = len(res_pool)
        V = np.random.beta(1.0, alpha_dp + n_pts, size=n_pts)
        stick_weights = V * np.cumprod(1.0 - np.concatenate([[0.0], V[:-1]]))
        stick_weights = stick_weights / np.sum(stick_weights)
        sw_tensor = torch.from_numpy(stick_weights).float()

        # Epistemic Acquisition Function: Residual magnitude * Dirichlet Weight
        acquisition = res_pool * (1.0 + 10.0 * sw_tensor)
        sampled_indices = torch.multinomial(acquisition / torch.sum(acquisition), num_samples=3, replacement=False)
        new_dp_pts = t_cand[sampled_indices]
        all_selected_col.extend(new_dp_pts.numpy().squeeze().tolist())

        t_col_dp = torch.cat([t_col_dp, new_dp_pts], dim=0)
        if len(t_col_dp) > n_col:
            t_col_dp = t_col_dp[-n_col:]

    pred_dp = m_dp(t_eval).detach().numpy().squeeze()
    l2_dp = np.sqrt(np.mean((pred_dp - u_exact_np) ** 2)) / np.sqrt(np.mean(u_exact_np ** 2))
    print(f"--> DP-BNN Collocation (Ours): Relative L2 Error = {l2_dp:.4e} ({time.time()-t0:.2f}s)")
    results["DP-BNN Collocation (Ours)"] = {"pred": pred_dp, "error": l2_dp, "col_pts": np.array(all_selected_col)}

    # -------------------------------------------------------------------------
    # Publication-Quality Visualization
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel 1: Solutions vs Exact
    ax = axes[0]
    ax.plot(t_eval.numpy(), u_exact_np, 'k-', lw=2.5, label='Exact Analytical Solution')
    ax.plot(t_eval.numpy(), results["Uniform Collocation"]["pred"], 'r--', lw=1.8, label=f'Uniform Collocation (Err: {l2_uni:.2e})')
    ax.plot(t_eval.numpy(), results["Greedy RAR"]["pred"], 'g-.', lw=1.8, label=f'Greedy RAR (Err: {l2_rar:.2e})')
    ax.plot(t_eval.numpy(), results["DP-BNN Collocation (Ours)"]["pred"], 'b-', lw=2.0, label=f'DP-BNN Collocation (Err: {l2_dp:.2e})')
    ax.set_title("(a) Solution Comparison on Stiff Wavefront", fontweight="bold")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Solution u(t)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Error Comparison
    ax = axes[1]
    err_uni = np.abs(results["Uniform Collocation"]["pred"] - u_exact_np)
    err_rar = np.abs(results["Greedy RAR"]["pred"] - u_exact_np)
    err_dp = np.abs(results["DP-BNN Collocation (Ours)"]["pred"] - u_exact_np)
    ax.semilogy(t_eval.numpy(), err_uni, 'r--', label='Uniform Error')
    ax.semilogy(t_eval.numpy(), err_rar, 'g-.', label='Greedy RAR Error')
    ax.semilogy(t_eval.numpy(), err_dp, 'b-', label='DP-BNN Error (Ours)')
    ax.set_title("(b) Absolute Error Profile |u_pred - u_exact|", fontweight="bold")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Pointwise Error (Log Scale)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Collocation Point Density (Where the algorithms sampled)
    ax = axes[2]
    ax.hist(results["Uniform Collocation"]["col_pts"], bins=20, alpha=0.4, color='red', label='Uniform Points', density=True)
    ax.hist(results["DP-BNN Collocation (Ours)"]["col_pts"], bins=20, alpha=0.6, color='blue', label='DP-BNN Points (Ours)', density=True)
    ax.axvspan(0.6, 0.85, color='orange', alpha=0.2, label='Steep Flame Front Zone')
    ax.set_title("(c) Density of Sampled Collocation Points", fontweight="bold")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Sampling Density")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = "dp_pinn_collocation_benchmark.png"
    out_pdf = "dp_pinn_collocation_benchmark.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {out_png} and {out_pdf}!")

    return results

if __name__ == "__main__":
    run_collocation_benchmark(rho=10.0, u0=0.001)
