"""Minimal Standalone DP-PINN Demo.

ODE: u''(t) + u(t) = 0,  t in [0, 2*pi]
Exact Solution: u(t) = sin(t)

Data: Only 2 points:
  t = 0.0      -> u = 0.0
  t = pi / 2   -> u = 1.0

The remaining domain t in (pi/2, 2*pi] has ZERO data!
Demonstrates:
1. Exact autograd: u_tt = d^2 u / dt^2
2. Drawing different random collocation points for each DP particle
3. Stick-breaking weights (alpha = 10.0)
4. Epistemic Dome emergence in the unmonitored void!
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Ultra-Sparse Data (N = 2)
t_data = torch.tensor([[0.0], [np.pi / 2.0]], dtype=torch.float32)
u_data = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
N_data = 2

# 3. Simple Solution Network: t -> u(t)
class SimplePINNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t):
        return self.net(t)

# 4. DP Posterior Sampler parameters
alpha = 10.0          # Prior mass
K_colloc = 30         # Collocation atoms per particle
M_particles = 5       # Number of Bayesian particles
num_epochs = 600

# 5. Initialize M particles
models = [SimplePINNNet() for _ in range(M_particles)]
optimizers = [torch.optim.Adam(m.parameters(), lr=0.01) for m in models]

# 6. For each particle, draw its own random measure:
#    - Different random collocation points t_c ~ Unif(0, 2*pi)
#    - Stick-breaking weights q
particle_collocs = []
particle_weights = []

for m in range(M_particles):
    # (a) Draw fresh random collocation points from base measure F_0
    t_c = torch.empty(K_colloc, 1).uniform_(0.0, 2.0 * np.pi)
    t_c.requires_grad_(True)
    particle_collocs.append(t_c)
    
    # (b) Vashishtha & Maillard stick-breaking weights
    # W_prior ~ Beta(alpha, N_data)
    W_prior = float(np.random.beta(alpha, N_data))
    w_data = np.full(N_data, (1.0 - W_prior) / N_data, dtype=np.float32)
    
    # GEM(alpha) for collocation atoms
    V = np.random.beta(1.0, alpha, size=K_colloc).astype(np.float32)
    V[-1] = 1.0
    cum_U = np.cumprod(1.0 - V)
    q_colloc = np.empty_like(V)
    q_colloc[0] = V[0]
    q_colloc[1:] = V[1:] * cum_U[:-1]
    q_colloc = W_prior * (q_colloc / q_colloc.sum())
    
    q_total = np.concatenate([w_data, q_colloc]).astype(np.float32)
    particle_weights.append(torch.from_numpy(q_total))

print(f"Initialized {M_particles} DP-PINN particles.")
print(f"Data weight: {1.0 - W_prior:.1%} | Prior Physics Collocation weight: {W_prior:.1%}")

# 7. Training loop using PyTorch Automatic Differentiation
for epoch in range(1, num_epochs + 1):
    for m in range(M_particles):
        opt = optimizers[m]
        net = models[m]
        t_c = particle_collocs[m]
        weights = particle_weights[m]  # [N_data + K_colloc]
        
        opt.zero_grad()
        
        # (1) Data Loss: (u(t_data) - u_data)^2
        u_pred_data = net(t_data)
        loss_data = (u_pred_data - u_data) ** 2  # [N_data, 1]
        
        # (2) Physics Residual Loss via Exact Autograd: u''(t) + u(t) = 0
        u_colloc = net(t_c)
        u_t = torch.autograd.grad(u_colloc, t_c, grad_outputs=torch.ones_like(u_colloc), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_c, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        residual = u_tt + u_colloc  # The ODE operator
        loss_phys = residual ** 2    # [K_colloc, 1]
        
        # (3) Stick-Weighted DP Expected Loss
        all_losses = torch.cat([loss_data, loss_phys], dim=0).squeeze(-1)
        total_loss = torch.sum(weights * all_losses)
        
        total_loss.backward()
        opt.step()

    if epoch % 150 == 0:
        print(f"Epoch {epoch:3d}/{num_epochs} | Particle 0 Loss: {total_loss.item():.6f}")

# 8. Evaluation & Visualization across full domain t in [0, 2*pi]
t_test = torch.linspace(0.0, 2.0 * np.pi, 200).unsqueeze(-1)
u_true = np.sin(t_test.numpy().squeeze())

preds = []
for net in models:
    net.eval()
    with torch.no_grad():
        preds.append(net(t_test).numpy().squeeze())

preds = np.array(preds)  # [M_particles, 200]
pred_mean = np.mean(preds, axis=0)
pred_std = np.std(preds, axis=0)

plt.figure(figsize=(9, 4.5))
# Plot ground truth
plt.plot(t_test.numpy().squeeze(), u_true, "k--", linewidth=2.0, label="True Exact Solution: $\sin(t)$")

# Plot individual DP particles
for m in range(M_particles):
    plt.plot(t_test.numpy().squeeze(), preds[m], alpha=0.35, linewidth=1.5, label=f"DP Particle {m+1}" if m == 0 else None)

# Plot predictive mean & 95% Credible Interval
plt.plot(t_test.numpy().squeeze(), pred_mean, "b-", linewidth=2.2, label="DP-PINN Mean Prediction")
plt.fill_between(
    t_test.numpy().squeeze(),
    pred_mean - 1.96 * pred_std,
    pred_mean + 1.96 * pred_std,
    color="blue", alpha=0.18, label="95% Credible Interval (Epistemic Dome)"
)

# Mark the 2 sparse training points
plt.scatter(t_data.numpy().squeeze(), u_data.numpy().squeeze(), color="red", s=100, zorder=5, label="Observed Data (N=2 only!)")

# Add vertical shading for unobserved void
plt.axvspan(np.pi / 2.0, 2.0 * np.pi, color="yellow", alpha=0.08, label="Unobserved Data Void (t > $\pi$/2)")

plt.xlabel("Time $t$", fontsize=12, fontweight="bold")
plt.ylabel("State $u(t)$", fontsize=12, fontweight="bold")
plt.title("Minimal DP-PINN Demo: Solving $u''(t) + u(t) = 0$ with N=2 Points via Autograd", fontsize=12, fontweight="bold")
plt.legend(loc="lower left", fontsize=8.5, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_fig = "simple_dp_pinn_demo.png"
plt.savefig(output_fig, dpi=300)
print(f"\nSaved transparent demo figure to {output_fig}!")
