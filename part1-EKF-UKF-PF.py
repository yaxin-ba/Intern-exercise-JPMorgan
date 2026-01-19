import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
import pandas as pd

# --- 1. Simulation Setup ---
np.random.seed(3)
N = 500
alpha = 0.91
sigma = 1.0
beta = 0.5
Q = sigma ** 2

# Generate Truth (SV Model) and Observations
X_true = np.zeros(N)
Y_obs = np.zeros(N)
init_variance = sigma ** 2 / (1 - alpha ** 2)
X_true[0] = np.random.normal(0, np.sqrt(init_variance))

for n in range(1, N):
    X_true[n] = alpha * X_true[n - 1] + np.random.normal(0, sigma)
    Y_obs[n] = beta * np.exp(X_true[n] / 2) * np.random.normal(0, 1)

# TRANSFORM: Use Square of Observations
Z_obs = Y_obs ** 2

# --- 2. EKF Implementation (Square Transform) ---
X_ekf = np.zeros(N)
P_ekf = np.zeros(N)
X_ekf[0] = 0.0
P_ekf[0] = init_variance

# 1. Reset memory tracking
tracemalloc.start()

# 2. Start Timer
start_time = time.perf_counter()

for n in range(1, N):
    # Prediction
    X_pred = alpha * X_ekf[n - 1]
    P_pred = alpha ** 2 * P_ekf[n - 1] + Q

    # Update
    # h(x) = beta^2 * exp(X)
    h_x = beta ** 2 * np.exp(X_pred)

    # Jacobian H = h(x) (since derivative of e^x is e^x)
    H = h_x

    # Dynamic Noise Variance R = 2 * (Expected Z)^2
    R_dyn = 2 * (h_x ** 2) + 1e-6

    S = H ** 2 * P_pred + R_dyn
    K = (P_pred * H) / S

    X_ekf[n] = X_pred + K * (Z_obs[n] - h_x)
    P_ekf[n] = (1 - K * H) * P_pred

# 4. Stop Timer
end_time = time.perf_counter()

# 5. Measure Peak Memory
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_ms = (end_time - start_time) * 1000
memory_kb = peak / 1024

print("EKF")
print("Runtime (ms)", runtime_ms)
print("Peak Memory (KB)", memory_kb)

# --- 3. UKF Implementation (Square Transform) ---
X_ukf = np.zeros(N)
P_ukf = np.zeros(N)
X_ukf[0] = 0.0
P_ukf[0] = init_variance

# UKF Parameters (Merwe)
alpha_u = 1e-3;
beta_u = 2.0;
kappa_u = 0.0
n_L = 1  # Dimension of state
lambda_u = alpha_u ** 2 * (n_L + kappa_u) - n_L

# Weights
Wm = np.full(2 * n_L + 1, 0.5 / (n_L + lambda_u))
Wc = np.full(2 * n_L + 1, 0.5 / (n_L + lambda_u))
Wm[0] = lambda_u / (n_L + lambda_u)
Wc[0] = Wm[0] + (1 - alpha_u ** 2 + beta_u)

# 1. Reset memory tracking
tracemalloc.start()

# 2. Start Timer
start_time = time.perf_counter()

for n in range(1, N):
    # --- Prediction (Standard Linear Shortcut for Efficiency) ---
    X_pred = alpha * X_ukf[n - 1]
    P_pred = alpha ** 2 * P_ukf[n - 1] + Q

    # --- Update (Sigma Points for Measurement) ---
    # 1. Generate Sigma Points around Predicted State
    sqrt_P = np.sqrt((n_L + lambda_u) * P_pred)
    sigmas = np.array([X_pred, X_pred + sqrt_P, X_pred - sqrt_P])

    # 2. Transform through Measurement Function h(x) = beta^2 * exp(x)
    Z_sigmas = beta ** 2 * np.exp(sigmas)

    # 3. Predicted Mean and Covariance
    Z_mean = np.sum(Wm * Z_sigmas)

    # Dynamic Noise R based on predicted mean
    R_dyn = 2 * (Z_mean ** 2) + 1e-6

    S = np.sum(Wc * (Z_sigmas - Z_mean) ** 2) + R_dyn

    # 4. Cross Covariance
    Pxz = np.sum(Wc * (sigmas - X_pred) * (Z_sigmas - Z_mean))

    # 5. Kalman Gain and State Update
    K = Pxz / S
    X_ukf[n] = X_pred + K * (Z_obs[n] - Z_mean)
    P_ukf[n] = P_pred - K * S * K

# 4. Stop Timer
end_time = time.perf_counter()

# 5. Measure Peak Memory
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_ms = (end_time - start_time) * 1000
memory_kb = peak / 1024

print("UKF")
print("Runtime (ms)", runtime_ms)
print("Peak Memory (KB)", memory_kb)

# --- 4. Visualization ---
plt.figure(figsize=(12, 4))
# Plot 0: Synthetic Data Generation
plt.subplot(1, 1, 1)
plt.plot(X_true, 'b-', linewidth=2, alpha=0.4, label='Volatility')
plt.plot(Y_obs, 'r*', alpha=0.4, label='Observations')
plt.title('Simulated Volatility Sequence')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 10))

# Plot 1: Tracking Performance
plt.subplot(2, 1, 1)
plt.plot(X_true, 'k-', linewidth=2, alpha=0.4, label='True Volatility')
plt.plot(X_ekf, 'b--', linewidth=1.5, label='EKF Estimate)')
plt.plot(X_ukf, 'g-', linewidth=1.5, alpha=0.8, label='UKF Estimate)')
# plt.plot(Y_obs, 'r*', label='Observations')
plt.title('EKF vs UKF Tracking Performance (Square Transform)')
plt.legend()
plt.grid(True)

# Plot 2: Error Analysis
plt.subplot(2, 1, 2)
err_ekf = np.abs(X_true - X_ekf)
err_ukf = np.abs(X_true - X_ukf)
plt.plot(err_ekf, 'b-', alpha=0.4, label=f'EKF Error (RMSE: {np.sqrt(np.mean(err_ekf ** 2)):.4f})')
plt.plot(err_ukf, 'g-', alpha=0.6, label=f'UKF Error (RMSE: {np.sqrt(np.mean(err_ukf ** 2)):.4f})')
plt.title('Estimation Error Comparison')
plt.legend()
plt.grid(True)
plt.ylabel('Absolute Error')
plt.xlabel('Time Step')

plt.tight_layout()
plt.show()



n_particles = 1000  # Number of particles
N_steps = N

# 2. Generate Synthetic Data (Raw SV Model)
# X_true = np.zeros(N_steps)
# Y_obs = np.zeros(N_steps)
init_variance = sigma ** 2 / (1 - alpha ** 2)
# X_true[0] = np.random.normal(0, np.sqrt(init_variance))

# for n in range(1, N_steps):
#     X_true[n] = alpha * X_true[n - 1] + sigma * np.random.normal(0, 1)
#     # Raw observation with Zero Mean
#     Y_obs[n] = beta * np.exp(X_true[n] / 2) * np.random.normal(0, 1)

# 3. Particle Filter Implementation
# Initialize Particles
particles = np.random.normal(0, np.sqrt(init_variance), n_particles)
weights = np.ones(n_particles) / n_particles

X_est_pf = np.zeros(N_steps)
X_est_pf[0] = np.mean(particles)


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

# 1. Reset memory tracking
tracemalloc.start()

# 2. Start Timer
start_time = time.perf_counter()

for n in range(1, N_steps):
    # --- A. Prediction Step ---
    # Propagate particles through state equation
    noise = np.random.normal(0, 1, n_particles)
    particles = alpha * particles + sigma * noise

    # --- B. Update Step (Likelihood) ---
    # We evaluate p(Y_n | Particle_i)
    # The observation Y comes from N(0, std^2) where std = beta * exp(X/2)
    obs = Y_obs[n]

    # Calculate std dev for each particle
    std_devs = beta * np.exp(particles / 2.0)

    # Calculate Gaussian Likelihood (PDF)
    # L = (1 / sqrt(2*pi*std^2)) * exp(-y^2 / 2*std^2)
    # We can ignore sqrt(2*pi) as it cancels out during normalization
    variances = std_devs ** 2
    likelihoods = (1.0 / np.sqrt(variances)) * np.exp(- (obs ** 2) / (2 * variances))

    # # Handle numerical stability (avoid divide by zero)
    # likelihoods += 1.e-300

    # Update weights
    weights *= likelihoods
    weights /= np.sum(weights)  # Normalize

    # --- C. Estimation ---
    # Weighted Mean
    X_est_pf[n] = np.sum(particles * weights)

    # --- D. Resampling ---
    # Resample if effective sample size is too low
    N_eff = 1.0 / np.sum(weights ** 2)
    if N_eff < n_particles / 2:
        indexes = systematic_resample(weights)
        particles = particles[indexes]
        weights.fill(1.0 / n_particles)
    # Resample
    # N_eff = 1.0 / np.sum(weights ** 2)
    # if N_eff < n_particles / 2:
    #     cumulative_sum = np.cumsum(weights)
    #     cumulative_sum[-1] = 1.0  # Ensure last is exactly 1
    #     indexes = np.searchsorted(cumulative_sum, np.random.random(n_particles))
    #     particles = particles[indexes]
    #     weights.fill(1.0 / n_particles)

# 4. Stop Timer
end_time = time.perf_counter()

# 5. Measure Peak Memory
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_ms = (end_time - start_time) * 1000
memory_kb = peak / 1024

print("PF")
print("Runtime (ms)", runtime_ms)
print("Peak Memory (KB)", memory_kb)
# # 4. Visualization
# plt.figure(figsize=(12, 6))
#
# plt.plot(X_true, 'k-', linewidth=2, alpha=0.5, label='True Volatility $X_n$')
# plt.plot(X_est_pf, 'r--', linewidth=1.5, label=f'Particle Filter Estimate (N={n_particles})')
#
# plt.title('Particle Filter Tracking on Raw SV Model (No Transformation)')
# plt.xlabel('Time Step')
# plt.ylabel('Log-Volatility')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # RMSE Calculation
rmse_pf = np.sqrt(np.mean((X_true - X_est_pf) ** 2))
print(f"Particle Filter RMSE: {rmse_pf:.4f}")

plt.figure(figsize=(12, 10))

# Plot 1: Tracking Performance
plt.subplot(2, 1, 1)
plt.plot(X_true, 'k-', linewidth=2, alpha=0.4, label='True Volatility')
plt.plot(X_est_pf, 'b--', linewidth=1.5, label='PF Estimate')
# plt.plot(X_ukf, 'g-', linewidth=1.5, alpha=0.8, label='UKF Estimate ($Y^2$)')
# plt.plot(Y_obs, 'r*', label='Observations')
plt.title('PF Tracking Performance')
plt.legend()
plt.grid(True)

# Plot 2: Error Analysis
plt.subplot(2, 1, 2)
# err_ekf = np.abs(X_true - X_ekf)
err_pf = np.abs(X_true - X_est_pf)
# plt.plot(err_ekf, 'b-', alpha=0.4, label=f'EKF Error (RMSE: {np.sqrt(np.mean(err_ekf ** 2)):.3f})')
plt.plot(err_pf, 'g-', alpha=0.6, label=f'PF Error (RMSE: {rmse_pf:.4f})')
plt.title('Estimation Error Comparison')
plt.legend()
plt.grid(True)
plt.ylabel('Absolute Error')
plt.xlabel('Time Step')

plt.show()