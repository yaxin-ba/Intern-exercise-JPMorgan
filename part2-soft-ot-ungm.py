import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Resampler Modules
# ==========================================

class SoftResample(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, particles, weights):
        B, N, D = particles.shape
        uniform = torch.ones_like(weights) / N
        soft_weights = self.alpha * weights + (1 - self.alpha) * uniform

        # Multinomial sampling (Gradients blocked)
        if soft_weights.dim() > 2: soft_weights = soft_weights.squeeze(-1)
        indices = torch.multinomial(soft_weights, num_samples=N, replacement=True)

        batch_idx = torch.arange(B, device=particles.device).unsqueeze(1).expand_as(indices)
        new_particles = particles[batch_idx, indices]

        # Importance Correction (Gradients flow)
        weights_selected = weights[batch_idx, indices]
        soft_selected = soft_weights[batch_idx, indices]
        new_weights = weights_selected / (soft_selected + 1e-10)
        new_weights = new_weights / new_weights.sum(dim=1, keepdim=True)

        return new_particles, new_weights

class OTResample(nn.Module):
    def __init__(self, epsilon=0.1, n_iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, particles, weights):
        B, N, D = particles.shape
        x_i = particles.unsqueeze(2)
        x_j = particles.unsqueeze(1)
        C = torch.sum((x_i - x_j)**2, dim=-1)

        with torch.no_grad():
            C_mean = C.mean(dim=(1,2), keepdim=True) + 1e-8
        C_scaled = C / C_mean

        log_mu = torch.log(weights + 1e-16)
        log_nu = torch.log(torch.ones_like(weights) / N)
        f = torch.zeros_like(weights)
        g = torch.zeros_like(weights)
        log_K = -C_scaled / self.epsilon

        for _ in range(self.n_iters):
            term_g = g.unsqueeze(1) + log_K
            f = log_mu - torch.logsumexp(term_g, dim=2)
            term_f = f.unsqueeze(2) + log_K
            g = log_nu - torch.logsumexp(term_f, dim=1)

        log_P = f.unsqueeze(2) + g.unsqueeze(1) + log_K
        P = torch.exp(log_P)
        new_particles = N * torch.matmul(P.transpose(1, 2), particles)
        new_weights = torch.ones_like(weights) / N
        return new_particles, new_weights

# ==========================================
# 2. Filter Models
# ==========================================

class UNGM_Model(nn.Module):
    """
    Univariate Non-linear Growth Model (Benchmark for PFs)
    x_t = 0.5*x_{t-1} + 25*x_{t-1}/(1+x_{t-1}^2) + 8*cos(1.2*t) + v_t
    y_t = x_t^2 / 20 + w_t
    """
    def __init__(self, N=50):
        super().__init__()
        self.N = N
        self.q_std = 1.0 # Process noise std
        self.r_std = 1.0 # Obs noise std

    def transition(self, particles, t):
        # particles: (B, N, D=1)
        term1 = 0.5 * particles
        term2 = 25 * particles / (1 + particles**2)
        term3 = 8 * np.cos(1.2 * t)
        x_next = term1 + term2 + term3 + torch.randn_like(particles) * self.q_std
        return x_next

    def observation_likelihood(self, particles, obs):
        # y_t = x_t^2 / 20 + w_t
        pred_obs = (particles**2) / 20.0
        dist = torch.sum((pred_obs - obs.unsqueeze(1))**2, dim=2)
        log_lik = -0.5 * dist / (self.r_std**2)
        return torch.exp(log_lik) + 1e-10

class Generic_Filter_UNGM(UNGM_Model):
    def __init__(self, resampler, N=50):
        super().__init__(N)
        self.resampler = resampler

    def forward(self, observations):
        B, T, D = observations.shape
        # Initialize from N(0, 5^2) as per standard benchmark
        particles = torch.randn(B, self.N, D, device=observations.device) * np.sqrt(5)
        weights = torch.ones(B, self.N, device=observations.device) / self.N

        est_states = []
        ess_history = []

        for t in range(T):
            obs = observations[:, t]

            if t > 0 and self.resampler is not None:
                particles, weights = self.resampler(particles, weights)

            # Predict
            particles = self.transition(particles, t)

            # Update
            lik = self.observation_likelihood(particles, obs)
            weights = weights * lik
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

            # Estimate (Mean)
            est = (weights.unsqueeze(2) * particles).sum(dim=1)
            est_states.append(est)

            ess = 1.0 / (weights ** 2).sum(dim=1)
            ess_history.append(ess)

        return torch.stack(est_states, dim=1), torch.stack(ess_history, dim=1)

# --- Tuning Function ---
def tune_ot_parameters():
    print("=== TUNING OT RESAMPLER ===")
    print(f"{'Eps':<6} | {'Iters':<5} | {'Cost':<8} | {'Bias(Var_Ratio)':<15} | {'Speed(ms)':<8}")
    print("-" * 55)

    B, N, D = 10, 100, 2
    # Source: Mixture of Gaussians
    particles = torch.randn(B, N, D) * 2.0
    weights = F.softmax(torch.randn(B, N), dim=1)

    configs = [(1.0, 5), (0.5, 10), (0.1, 10), (0.1, 20), (0.01, 50)]

    for eps, iters in configs:
        resampler = OTResample(epsilon=eps, n_iters=iters)

        start = time.time()
        new_parts, _ = resampler(particles, weights)
        elapsed = (time.time() - start) * 1000

        # Bias Metric: Variance Ratio (New Var / Old Var)
        # If < 1.0, particles are shrinking (High Bias)
        old_var = particles.var().item()
        new_var = new_parts.var().item()
        ratio = new_var / old_var

        # Cost Metric: Transport distance
        # Ideally should be low but not zero (zero means no movement)
        cost = torch.norm(new_parts.mean(1) - particles.mean(1)).item()

        print(f"{eps:<6.2f} | {iters:<5} | {cost:<8.4f} | {ratio:<15.4f} | {elapsed:<8.2f}")
    print("\n")

# --- Experiment Runner ---
def run_full_experiment():
    # 1. Tuning
    tune_ot_parameters()

    # 2. Main Experiment (UNGM)
    print("=== MAIN EXPERIMENT: UNGM BENCHMARK ===")
    T = 50
    true_x = torch.zeros(1, 1) # Start at 0
    obs_list, state_list = [], []

    # Generate UNGM Data
    for t in range(T):
        term1 = 0.5 * true_x
        term2 = 25 * true_x / (1 + true_x**2)
        term3 = 8 * np.cos(1.2 * t)
        true_x = term1 + term2 + term3 + torch.randn(1, 1) # q_std=1
        y = (true_x**2) / 20.0 + torch.randn(1, 1) # r_std=1
        state_list.append(true_x)
        obs_list.append(y)

    obs_batch = torch.stack(obs_list, dim=1)
    true_states = torch.stack(state_list, dim=1)

    # Models
    N_part = 100
    models = {
        'Soft-PF': Generic_Filter_UNGM(SoftResample(alpha=0.5), N=N_part),
        'OT-DPF (Eps=0.1)': Generic_Filter_UNGM(OTResample(0.1, 15), N=N_part)
    }

    results = {}

    for name, model in models.items():
        # Enable grad on obs to simulate "input gradient" check
        obs_batch.requires_grad_(True)

        est, ess = model(obs_batch)
        loss = F.mse_loss(est, true_states)

        loss.backward()
        grad_norm = obs_batch.grad.norm().item()

        results[name] = {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'est': est.detach().squeeze().numpy(),
            'ess': ess.detach().mean(dim=0).numpy()
        }

        obs_batch.grad.zero_()
        obs_batch.requires_grad_(False)

    # 3. Display Results
    print(f"{'Method':<20} | {'RMSE':<10} | {'Grad Norm (Obs)':<15}")
    print("-" * 50)
    for name, res in results.items():
        rmse = np.sqrt(res['loss'])
        print(f"{name:<20} | {rmse:<10.4f} | {res['grad_norm']:<15.4f}")

    # 4. Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Trajectory
    axs[0].plot(true_states.squeeze().numpy(), 'k-', lw=2, label='True')
    for name, res in results.items():
        axs[0].plot(res['est'], '--', label=name)
    axs[0].set_title('UNGM Tracking (Non-linear)')
    axs[0].legend()

    # ESS
    for name, res in results.items():
        axs[1].plot(res['ess'], label=name)
    axs[1].set_title('Effective Sample Size')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_experiment()