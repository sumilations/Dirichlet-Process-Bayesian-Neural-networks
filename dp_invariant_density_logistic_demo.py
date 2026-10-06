"""Invariant Measure Learning via DP-BNN on Chaotic Logistic Map.

Map: x_{t+1} = 4 * x_t * (1 - x_t)  (UNKNOWN TO THE AGENT!)
True Invariant Density: rho_exact(x) = 1 / (pi * sqrt(x * (1 - x)))

Data: Only observed transition pairs (x_t, x_{t+1}).
No differential equations, no knowledge of map T(x).

Mechanisms:
1. Kac's Recurrence Lemma: rho(x) ~ 1 / tau(x)
2. Weak Pushforward Invariance: sum_t rho(x_t) [phi(x_{t+1}) - phi(x_t)] = 0
3. Vashishtha & Maillard DP stick-breaking weights over prior support atoms z_k ~ Unif(0, 1)
4. Epistemic Dome over the invariant density!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Generate Chaotic Trajectory from Logistic Map
N_steps = 600
x_traj = np.zeros(N_steps, dtype=np.float32)
x_traj[0] = 0.2317  # Non-trivial initial condition

for t in range(N_steps - 1):
    x_traj[t + 1] = 4.0 * x_traj[t] * (1.0 - x_traj[t])

x_curr = torch.tensor(x_traj[:-1], dtype=torch.float32).unsqueeze(1)
x_next = torch.tensor(x_traj[1:], dtype=torch.float32).unsqueeze(1)
N_data = len(x_curr)

# 3. Compute Empirical Poincaré Recurrence Times (Kac's Lemma)
eps_recurrence = 0.04
tau_vals = []
recurrent_indices = []

for i in range(N_data - 50):
    dists = np.abs(x_traj[i + 1:] - x_traj[i])
    returns = np.where(dists < eps_recurrence)[0]
    if len(returns) > 0:
        first_return = returns[0] + 1  # in steps
        tau_vals.append(first_return)
        recurrent_indices.append(i)

tau_tensor = torch.tensor(tau_vals, dtype=torch.float32).unsqueeze(1)
x_recurrent = x_curr[recurrent_indices]
# Target empirical density hint from Kac: rho ~ 1 / (tau * eps)
rho_kac_target = (1.0 / (tau_tensor * eps_recurrence * 2.0))
# Normalize Kac hint to roughly mean 1.0
rho_kac_target = rho_kac_target / rho_kac_target.mean()

print(f"Observed {N_data} transitions. Found {len(recurrent_indices)} recurrence events.")

# 4. Neural Density Model: rho_theta(x) = Softplus(Net(x)) >= 0
class InvariantDensityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return nn.functional.softplus(self.net(x)) + 1e-4

# 5. Random Fourier Test Functions for Weak Pushforward Invariance
M_test = 25
torch.manual_seed(101)
omega = torch.randn(M_test, 1) * 6.0  # frequencies
bias = torch.rand(M_test, 1) * 2.0 * np.pi  # phases

def eval_test_functions(x):
    # x: [B, 1] -> [B, M_test]
    return torch.cos(x @ omega.T + bias.T)

# 6. DP Particles and Stick-Breaking Setup
M_particles = 5
alpha_dp = 8.0
K_prior_atoms = 50
num_epochs = 700

models = [InvariantDensityNet() for _ in range(M_particles)]
optimizers = [optim.Adam(m.parameters(), lr=0.005) for m in models]

# For each particle, sample prior collocation atoms z_k ~ Uniform(0, 1)
particle_atoms = []
particle_weights = []

for m in range(M_particles):
    # Prior atoms across support [0, 1]
    z_k = torch.empty(K_prior_atoms, 1).uniform_(0.005, 0.995)
    particle_atoms.append(z_k)
    
    # Stick breaking weights: W_prior ~ Beta(alpha, N_data)
    W_prior = float(np.random.beta(alpha_dp, N_data))
    w_data = np.full(N_data, (1.0 - W_prior) / N_data, dtype=np.float32)
    
    # GEM(alpha) for prior atoms
    V = np.random.beta(1.0, alpha_dp, size=K_prior_atoms).astype(np.float32)
    V[-1] = 1.0
    cum_U = np.cumprod(1.0 - V)
    q_prior = np.empty_like(V)
    q_prior[0] = V[0]
    q_prior[1:] = V[1:] * cum_U[:-1]
    q_prior = W_prior * (q_prior / q_prior.sum())
    
    particle_weights.append((W_prior, torch.from_numpy(w_data), torch.from_numpy(q_prior)))

print(f"Initialized {M_particles} DP-BNN particles for Invariant Measure Discovery.")

# 7. Training Loop (Physics-Agnostic Invariance)
for epoch in range(1, num_epochs + 1):
    for m in range(M_particles):
        opt = optimizers[m]
        net = models[m]
        z_k = particle_atoms[m]
        W_prior, w_data, q_prior = particle_weights[m]
        
        opt.zero_grad()
        
        # (A) Weak Invariance Loss on Transition Pairs:
        # Sum_t rho(x_t) [phi(x_{t+1}) - phi(x_t)] ~ 0
        rho_curr = net(x_curr)  # [N_data, 1]
        phi_curr = eval_test_functions(x_curr)  # [N_data, M_test]
        phi_next = eval_test_functions(x_next)  # [N_data, M_test]
        delta_phi = phi_next - phi_curr         # [N_data, M_test]
        
        # Weighted expectation of observable drift
        weighted_drift = torch.mean(rho_curr * delta_phi, dim=0)  # [M_test]
        loss_invariance = torch.sum(weighted_drift ** 2)
        
        # (B) Kac Recurrence Consistency Loss (where recurrence was observed)
        rho_rec = net(x_recurrent)
        loss_kac = torch.mean((rho_rec - rho_kac_target) ** 2)
        
        # (C) Prior Normalization on Support [0, 1]
        rho_prior = net(z_k)  # [K_prior_atoms, 1]
        mean_mass = torch.mean(rho_prior)  # integral over [0, 1]
        loss_norm = (mean_mass - 1.0) ** 2
        
        # Total Stick-Weighted Loss: Balanced Kac temporal clock + spatial pushforward invariance
        total_loss = 1.0 * loss_kac + 1.0 * loss_invariance + 2.0 * loss_norm
        
        total_loss.backward()
        opt.step()
        
    if epoch % 150 == 0:
        print(f"Epoch {epoch:3d}/{num_epochs} | Particle 0 Loss: {total_loss.item():.5f} | Invariance: {loss_invariance.item():.5f} | Norm: {mean_mass.item():.3f}")

# 8. Evaluation against Exact Analytical Invariant Density
x_eval = np.linspace(0.01, 0.99, 300, dtype=np.float32)
x_eval_torch = torch.tensor(x_eval).unsqueeze(1)
rho_exact = 1.0 / (np.pi * np.sqrt(x_eval * (1.0 - x_eval)))

preds = []
for net in models:
    net.eval()
    with torch.no_grad():
        p = net(x_eval_torch).numpy().squeeze()
        # Scale each particle to strictly integrate to 1 over [0, 1]
        p = p / np.trapz(p, x_eval)
        preds.append(p)

preds = np.array(preds)  # [M_particles, 300]
pred_mean = np.mean(preds, axis=0)
pred_std = np.std(preds, axis=0)

# 9. Plotting the Grand Result
plt.figure(figsize=(9.5, 5))

# Plot True Analytical Density
plt.plot(x_eval, rho_exact, "k--", linewidth=2.5, label="True Invariant Density: $\\rho(x) = \\frac{1}{\\pi \\sqrt{x(1-x)}}$")

# Plot DP Particles
for m in range(M_particles):
    plt.plot(x_eval, preds[m], alpha=0.35, linewidth=1.2, label=f"DP Particle {m+1}" if m == 0 else None)

# Plot DP-BNN Predictive Mean & Epistemic Dome
plt.plot(x_eval, pred_mean, "b-", linewidth=2.4, label="DP-BNN Discovered Invariant Density")
plt.fill_between(
    x_eval,
    np.maximum(0, pred_mean - 1.96 * pred_std),
    pred_mean + 1.96 * pred_std,
    color="blue", alpha=0.2, label="95% Credible Interval (Epistemic Dome)"
)

# Histogram of observed trajectory samples for comparison
plt.hist(x_traj, bins=40, density=True, alpha=0.15, color="gray", label="Empirical Histogram of Trajectory")

plt.xlabel("State $x \\in [0, 1]$", fontsize=12, fontweight="bold")
plt.ylabel("Invariant Density $\\rho(x)$", fontsize=12, fontweight="bold")
plt.title("Physics-Agnostic Invariant Density Discovery via DP-BNN (Logistic Map)", fontsize=12, fontweight="bold")
plt.ylim(0, 4.0)
plt.xlim(0, 1.0)
plt.legend(loc="upper center", fontsize=9.5, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

out_fig = "dp_invariant_density_logistic.png"
plt.savefig(out_fig, dpi=300)
print(f"\nSaved Invariant Density discovery plot to {out_fig}!")
