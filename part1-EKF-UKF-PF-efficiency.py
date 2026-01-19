import numpy as np
import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. System Setup ---
np.random.seed(42)
N_steps = 500  # Long trajectory for better time measurement
alpha = 0.91
sigma = 1.0
beta = 0.5
Q = sigma ** 2

# Generate Data
X_true = np.zeros(N_steps)
Y_obs = np.zeros(N_steps)
init_variance = sigma ** 2 / (1 - alpha ** 2)
X_true[0] = np.random.normal(0, np.sqrt(init_variance))

for n in range(1, N_steps):
    X_true[n] = alpha * X_true[n - 1] + sigma * np.random.normal(0, 1)
    Y_obs[n] = beta * np.exp(X_true[n] / 2) * np.random.normal(0, 1)

# Pre-transform for EKF/UKF
Z_obs_sq = Y_obs ** 2


# --- 2. Filter Implementations (Wrapped for Benchmarking) ---

def run_ekf():
    X_est = np.zeros(N_steps)
    P_est = np.zeros(N_steps)
    X_est[0] = 0.0
    P_est[0] = init_variance

    for n in range(1, N_steps):
        # Predict
        X_pred = alpha * X_est[n - 1]
        P_pred = alpha ** 2 * P_est[n - 1] + Q

        # Update
        h_x = beta ** 2 * np.exp(X_pred)
        H = h_x
        R = 2 * h_x ** 2 + 1e-6

        S = H ** 2 * P_pred + R
        K = P_pred * H / S
        X_est[n] = X_pred + K * (Z_obs_sq[n] - h_x)
        P_est[n] = (1 - K * H) * P_pred
    return X_est


def run_ukf():
    X_est = np.zeros(N_steps)
    P_est = np.zeros(N_steps)
    X_est[0] = 0.0
    P_est[0] = init_variance

    # UKF Params
    alpha_u = 1e-3;
    beta_u = 2.0;
    kappa_u = 0.0;
    n_L = 1
    lambda_u = alpha_u ** 2 * (n_L + kappa_u) - n_L
    Wm = np.full(3, 0.5 / (n_L + lambda_u))
    Wc = np.full(3, 0.5 / (n_L + lambda_u))
    Wm[0] = lambda_u / (n_L + lambda_u)
    Wc[0] = Wm[0] + (1 - alpha_u ** 2 + beta_u)

    for n in range(1, N_steps):
        # Predict (Linear shortcut for speed, still valid)
        X_pred = alpha * X_est[n - 1]
        P_pred = alpha ** 2 * P_est[n - 1] + Q

        # Update (Sigma Points)
        sqrt_P = np.sqrt((n_L + lambda_u) * P_pred)
        sigmas = np.array([X_pred, X_pred + sqrt_P, X_pred - sqrt_P])

        Z_sigmas = beta ** 2 * np.exp(sigmas)
        Z_mean = np.sum(Wm * Z_sigmas)
        R_dyn = 2 * Z_mean ** 2 + 1e-6

        S = np.sum(Wc * (Z_sigmas - Z_mean) ** 2) + R_dyn
        Pxz = np.sum(Wc * (sigmas - X_pred) * (Z_sigmas - Z_mean))

        K = Pxz / S
        X_est[n] = X_pred + K * (Z_obs_sq[n] - Z_mean)
        P_est[n] = P_pred - K * S * K
    return X_est

def systematic_resample(weights):
    """ Performs systematic resampling of the particles """
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
def run_particle_filter(n_particles=1000):
    particles = np.random.normal(0, np.sqrt(init_variance), n_particles)
    weights = np.ones(n_particles) / n_particles
    X_est = np.zeros(N_steps)

    for n in range(1, N_steps):
        # Predict
        particles = alpha * particles + sigma * np.random.normal(0, 1, n_particles)

        # Update
        std_devs = beta * np.exp(particles / 2.0)
        variances = std_devs ** 2 + 1e-9
        obs = Y_obs[n]
        likelihoods = (1.0 / np.sqrt(variances)) * np.exp(- (obs ** 2) / (2 * variances))

        weights *= likelihoods
        weights /= np.sum(weights)

        X_est[n] = np.sum(particles * weights)

        # # Resample
        # N_eff = 1.0 / np.sum(weights ** 2)
        # if N_eff < n_particles / 2:
        #     cumulative_sum = np.cumsum(weights)
        #     cumulative_sum[-1] = 1.0  # Ensure last is exactly 1
        #     indexes = np.searchsorted(cumulative_sum, np.random.random(n_particles))
        #     particles = particles[indexes]
        #     weights.fill(1.0 / n_particles)

        # --- D. Resampling ---
        # Resample if effective sample size is too low
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < n_particles / 2:
            indexes = systematic_resample(weights)
            particles = particles[indexes]
            weights.fill(1.0 / n_particles)

    return X_est


# --- 3. Benchmarking Loop ---

methods = [
    ("EKF (Square)", run_ekf),
    ("UKF (Square)", run_ukf),
    ("PF (N=100)", lambda: run_particle_filter(100)),
    ("PF (N=1000)", lambda: run_particle_filter(1000)),
    ("PF (N=5000)", lambda: run_particle_filter(5000))
]

results = []

for name, func in methods:
    # 1. Reset memory tracking
    tracemalloc.start()

    # 2. Start Timer
    start_time = time.perf_counter()

    # 3. Run Algorithm
    func()

    # 4. Stop Timer
    end_time = time.perf_counter()

    # 5. Measure Peak Memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_ms = (end_time - start_time) * 1000
    memory_kb = peak / 1024

    results.append({
        "Method": name,
        "Runtime (ms)": runtime_ms,
        "Peak Memory (KB)": memory_kb
    })

df = pd.DataFrame(results)

# --- 4. Visualization ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Filter Method', fontsize=12)
ax1.set_ylabel('Runtime (ms) - Lower is Better', color=color, fontsize=12)
ax1.bar(df["Method"], df["Runtime (ms)"], color=color, alpha=0.6, label='Runtime')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Peak Memory (KB) - Lower is Better', color=color, fontsize=12)
ax2.plot(df["Method"], df["Peak Memory (KB)"], color=color, marker='o', linewidth=2, label='Memory')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f"Performance Comparison (N={N_steps} steps)", fontsize=14)
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

print(df)