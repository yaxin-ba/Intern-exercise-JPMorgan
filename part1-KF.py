import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# Set a seed for reproducibility of the synthetic data
np.random.seed(42)

# --- 1. Define Model Parameters (Matrices and Noise Covariances) ---
dt = 0.1  # Time step
N = 100 # Number of time steps

# State Transition Matrix A (Constant Velocity Model)
A = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
])

# Process Noise Matrix B and Covariance Q
B = np.array([
    [dt**2/2, 0],
    [dt, 0],
    [0, dt**2/2],
    [0, dt]
])

# q_factor = 1000.0
Q = (B @ B.T)

# Observation Matrix C
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

# Observation Noise Matrix D and Covariance R
D = np.array([[1.0, 0.0],
              [0.0, 1.0]])
R = D @ D.T # Observation Noise Covariance R

nx = A.shape[0]  # State dimension
ny = C.shape[0]  # Observation dimension
nv = B.shape[1]  # Process noise dimension
nw = D.shape[1]  # Measurement noise dimension (Computed automatically)
I = np.eye(nx)

# --- 2. Generate Synthetic Data ---
X_true = np.zeros((N, nx))
Y_obs = np.zeros((N, ny))

# True Initial State (Stochastic initial state for the *system*)
mean_true = np.zeros(nx)
scale = 1.0
Sigma_true = scale*np.eye(4)
# Generate the true initial state X_1 by sampling from N(0, Sigma)
X_true[0] = np.random.multivariate_normal(mean_true, Sigma_true)
# mean_X0_true = np.zeros(nx)
# P0_true = 1.0 * np.eye(4)
# L_P0_true = np.linalg.cholesky(P0_true)
# X_true[0] = mean_X0_true + L_P0_true @ np.random.randn(nx)
P0_true = Sigma_true.copy()

# Initial Filter State and Covariance (used for both filter runs)
# X0_filter = np.array([0.0, 0.0, 0.0, 0.0]) # Filter's initial state estimate, can be other values
X0_filter = np.random.multivariate_normal(mean_true, Sigma_true)
# P0_filter = 10.0 * np.eye(4) # Initial Covariance P_{0|0}, high uncertainty, different from Sigma_true
P0_filter = Sigma_true.copy()

# Simulate the process
def synthetic_data_generation(A,B,C,D,N,nx,ny,nv,nw):
    X_true = np.zeros((N, nx))
    Y_obs = np.zeros((N, ny))
    X_true[0] = np.random.multivariate_normal(mean_true, Sigma_true)
    for n in range(1, N):
        # print('n',n)
        process_noise = (B @ np.random.randn(nv))
        X_true[n] = A @ X_true[n-1] + process_noise

        observation_noise = (D @ np.random.randn(nw))
        Y_obs[n] = C @ X_true[n] + observation_noise
    return X_true, Y_obs

# --- 3. Kalman Filter Function ---

def run_kalman_filter(Y_obs, A, Q, C, R, X0_filter, P0_filter, update_type):
    N = Y_obs.shape[0]
    nx = A.shape[0]

    X_hat = np.zeros((N, nx))
    P_history = np.zeros((N, nx, nx))

    P = P0_filter
    X_hat[0] = X0_filter
    P_history[0] = P0_filter

    cond_numbers = []

    for n in range(1, N):
        X_hat_prior = A @ X_hat[n-1]
        P_prior = A @ P @ A.T + Q
        S = C @ P_prior @ C.T + R

        try:
            K = P_prior @ C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            X_hat[n:] = np.nan
            break

        X_hat[n] = X_hat_prior + K @ (Y_obs[n] - C @ X_hat_prior)

        if update_type == 'standard':
            P = (I - K @ C) @ P_prior
        elif update_type == 'joseph':
            IC_KC = I - K @ C
            P = IC_KC @ P_prior @ IC_KC.T + K @ R @ K.T

        P_history[n] = P
        cond_number = np.linalg.cond(P)
        cond_numbers.append(cond_number)

    return X_hat, P_history, cond_numbers

# --- Helper Function for Plotting Covariance Ellipses ---
CHI2_VAL_95 = chi2.ppf(0.9, 2)

def plot_covariance_ellipse(ax, mean, P_submatrix, color, show_rho=False, label=None, filled=False, alpha=0.15):
    eigenvalues, eigenvectors = np.linalg.eigh(P_submatrix)

    if np.any(eigenvalues <= 0):
        ax.plot(mean[0], mean[1], marker='o', color=color, markersize=3)
        return

    s1 = np.sqrt(CHI2_VAL_95 * eigenvalues[0])
    s2 = np.sqrt(CHI2_VAL_95 * eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(xy=(mean[0], mean[1]), width=2 * s1, height=2 * s2,
                      angle=angle, color=color, linewidth=2 if not filled else 0, fill=filled, alpha=alpha)

    if label:
        ellipse.set_label(label)

    ax.add_patch(ellipse)

    if show_rho:
         ax.plot(mean[0], mean[1], 'ro', markersize=5, zorder=10)


# --- 4. Run Comparisons ---
X_true, Y_obs = synthetic_data_generation(A,B,C,D,N,nx,ny,nv,nw)
X_hat_std, P_std, cond_numbers_std1 = run_kalman_filter(Y_obs, A, Q, C, R, X0_filter, P0_filter, 'standard')
X_hat_joseph, P_joseph, cond_numbers_joseph = run_kalman_filter(Y_obs, A, Q, C, R, X0_filter, P0_filter, 'joseph')
# print('P_joseph',P_joseph)

q_factor = 10
B_factor = q_factor*B
Q_factor = B_factor @ B_factor.T
# Q = q_factor * (B @ B.T)
X_true1, Y_obs1 = synthetic_data_generation(A,B_factor,C,D,N,nx,ny,nv,nw)
X_hat_joseph1, P_joseph1, cond_numbers_joseph1 = run_kalman_filter(Y_obs1, A, Q_factor, C, R, X0_filter, P0_filter, 'joseph')


# print('P_joseph1',P_joseph1)

# --- 5. Plotting Results ---

# --- Plot 0: Mean comparison ---
plt.figure(figsize=(12, 5))
t = np.arange(N)
print('X_true',X_true.shape)
state_idx = 0  # Visualizing first state dimension
state_idx2 = 2  # Visualizing first state dimension
est_error = X_true[:, state_idx] - X_hat_joseph[:, state_idx]
est_error2 = X_true[:, state_idx2] - X_hat_joseph[:, state_idx2]
# rmse_est = np.sqrt(np.mean((X_true[:, state_idx] - X_hat_joseph[:, state_idx]) ** 2))
est_error_factor = X_true1[:, state_idx] - X_hat_joseph1[:, state_idx]
sigma = np.sqrt(P_joseph[:, state_idx, state_idx])
sigma2 = np.sqrt(P_joseph[:, state_idx2, state_idx2])
# plt.plot(t, est_error_factor, 'g-', label=f'Estimation Error of Joseph Covariance Update (q_factor={q_factor})')
# plt.plot(t, est_error, 'r-.', label='Estimation Error on $p_x$')
# plt.plot(t, est_error2, 'r:', label='Estimation Error on $p_y$')

plt.plot(t, est_error, color='navy', linestyle='-', linewidth=1.5, alpha=0.8,
        label=r'Error $p_x$ ($X_{true} - \hat{X}$)')
plt.plot(t, 2*sigma, color='blue', linestyle='--', alpha=0.3)
plt.plot(t, -2*sigma, color='blue', linestyle='--', alpha=0.3)
plt.fill_between(t, -2*sigma, 2*sigma, color='blue', alpha=0.1,
                label=r'$p_x$ 95% Confidence ($\pm 2\sigma$)')
# plt.plot(t, 2 * sigma, 'k--', alpha=0.5, label='$\pm 2\sigma_{p_x}$ Theoretical Bounds on $p_x$')
# plt.plot(t, 2 * sigma, 'k--', alpha=0.5, label='$p_x$ 95% Confidence ($\pm 2\sigma$)')
# plt.plot(t, -2 * sigma, 'k--', alpha=0.5)
# plt.fill_between(t, -2 * sigma, 2 * sigma, color='gray', alpha=0.2)
plt.plot(t, est_error2, color='darkorange', linestyle='-', linewidth=1.5, alpha=0.8,
        label=r'Error $p_y$ ($Y_{true} - \hat{Y}$)')
plt.plot(t, 2*sigma2, color='orange', linestyle='--', alpha=0.3)
plt.plot(t, -2*sigma2, color='orange', linestyle='--', alpha=0.3)
plt.fill_between(t, -2*sigma2, 2*sigma2, color='orange', alpha=0.1,
                label=r'$p_x$ 95% Confidence ($\pm 2\sigma$)')
# plt.plot(t, 2 * sigma2, 'b--', alpha=0.5, label='$\pm 2\sigma_{p_y}$ Theoretical Bounds on $p_y$')
# plt.plot(t, -2 * sigma2, 'b--', alpha=0.5)
# plt.fill_between(t, -2 * sigma2, 2 * sigma, color='blue', alpha=0.1)
# plt.title(r'Optimality Analysis: $X_{true}-\hat{X}_{n|n}$')
# 4. Styling
plt.title(r'Optimality Analysis: Error Residuals vs Theoretical Bounds', fontsize=16)
plt.xlabel('Time Step $n$', fontsize=14)
plt.ylabel('Estimation Error (m)', fontsize=14)
# ax.axhline(0, color='black', linewidth=0.5, alpha=0.5) # Zero line reference
# ax.grid(True, linestyle='--', alpha=0.6)

# Improved Legend: clear separation of concepts
plt.legend(loc='upper right', frameon=True, fontsize=12, fancybox=True, shadow=True)
plt.xlim([0, N])

plt.tight_layout()

# plt.legend()
plt.grid(True)
plt.show()

# --- Plot 1: Condition Number Comparison (Numerical Stability) ---
plt.figure(figsize=(12, 6))
for q_factor in [0.001,0.01,0.1,1,10]:
    # q_factor = 0.1
    B_factor = q_factor * B
    Q_factor = B_factor @ B_factor.T
    # Q = q_factor * (B @ B.T)
    X_true1, Y_obs1 = synthetic_data_generation(A, B_factor, C, D, N, nx, ny, nv, nw)
    X_hat_joseph1, P_joseph1, cond_numbers_joseph1 = run_kalman_filter(Y_obs1, A, Q_factor, C, R, X0_filter, P0_filter,
                                                                       'joseph')
    plt.plot(range(1, N), cond_numbers_joseph1, label=f'Joseph Form Update (q_factor={q_factor})', linewidth=2)
plt.title('Condition Number of Covariance Matrix $P_{n|n}$', fontsize=16)
plt.xlabel('Time Step $n$', fontsize=14)
plt.ylabel('Condition Number $\kappa(P_{n|n})$', fontsize=14)
plt.legend(fontsize=12)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Observation Noise Matrix D and Covariance R
D_modify = np.array([[1.0, 0.0],
              [0.0, 1e-4]])
R_modify = D_modify @ D_modify.T # Observation Noise Covariance R
plt.figure(figsize=(12, 6))
for q_factor in [0.001,0.05,1,10,100]:
    # q_factor = 0.1
    B_factor = q_factor * B
    Q_factor = B_factor @ B_factor.T
    # Q = q_factor * (B @ B.T)
    X_true1, Y_obs1 = synthetic_data_generation(A, B_factor, C, D, N, nx, ny, nv, nw)
    X_hat_joseph1, P_joseph1, cond_numbers_joseph1 = run_kalman_filter(Y_obs1, A, Q_factor, C, R_modify, X0_filter, P0_filter,
                                                                       'joseph')
    plt.plot(range(1, N), cond_numbers_joseph1, label=f'Joseph Form Update (q_factor={q_factor})', linewidth=2)
plt.title("Condition Number of Covariance Matrix $P_{n|n}$ with Unbalance Observation Noise Matrix $\\tilde{D}$", fontsize=16)
plt.xlabel('Time Step $n$', fontsize=14)
plt.ylabel('Condition Number $\kappa(P_{n|n})$', fontsize=14)
plt.legend(fontsize=12)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


# --- Plot 2: Tracking Comparison with Uncertainty Ellipses (Position only) ---
fig, ax = plt.subplots(figsize=(10, 8))

# True State (Position)
ax.plot(X_true[:, 0], X_true[:, 2], 'k-', linewidth=3, label='True State $X_{true}$ (Position)')
# Noisy Observations
ax.plot(Y_obs[:, 0], Y_obs[:, 1], 'rx', alpha=0.4, label='Observations $Y_{obs}$')
# Kalman Filter Estimates
ax.plot(X_hat_std[:, 0], X_hat_std[:, 2], 'b--', linewidth=1.5, label='Standard KF Estimate')
ax.plot(X_hat_joseph[:, 0], X_hat_joseph[:, 2], 'g-.', linewidth=1.5, label='Joseph Form KF Estimate')
# ax.plot(X_hat_joseph1[:, 0], X_hat_joseph1[:, 2], 'b--', linewidth=1.5, label=f'Joseph Form KF Estimate (q_factor={q_factor})')
# --- Draw Covariance Ellipses (Position Only) ---
plot_steps_pos = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

for n in plot_steps_pos:
    mean_std_pos = X_hat_std[n, [0, 2]] # [px, py] mean estimate
    P_pos_std = P_std[n, [0, 2], :][:, [0, 2]] # 2x2 position covariance block

    label_std_pos = f'95% Standard Ellipses (Pos)' if n == plot_steps_pos[0] else None
    plot_covariance_ellipse(ax, mean_std_pos, P_pos_std, 'blue', label=label_std_pos, show_rho=True)

    mean_joseph_pos = X_hat_joseph[n, [0, 2]] # [px, py] mean estimate
    P_pos_joseph = P_joseph[n, [0, 2], :][:, [0, 2]] # 2x2 position covariance block

    label_joseph_pos = '95% Joseph Ellipses (Pos)' if n == plot_steps_pos[0] else None
    plot_covariance_ellipse(ax, mean_joseph_pos, P_pos_joseph, 'green', label=label_joseph_pos, show_rho=True)


ax.set_title('2D Position Tracking with 95% Uncertainty Ellipses', fontsize=16)
ax.set_xlabel('X-Position $p_x$', fontsize=14)
ax.set_ylabel('Y-Position $p_y$', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True)
ax.axis('equal')
plt.tight_layout()
plt.show()

# # --- Plot 3: Position (px) vs Position (py) Uncertainty Ellipses (The fix for inconsistency) ---
# fig_pp, ax_pp = plt.subplots(figsize=(10, 8))
#
# # Use plot_steps_pos to ensure coverage across the entire trajectory
# plot_steps_pp = plot_steps_pos
#
# # Indices for Px and Py in the 4-state vector [px, vx, py, vy]
# px_idx = 0
# py_idx = 2
# color_pp = 'darkred'
#
# ax_pp.set_title('Uncertainty Ellipses for Position $p_x$ vs. Position $p_y$ (Full Trajectory)', fontsize=16)
# ax_pp.set_xlabel('Position $p_x$', fontsize=14)
# ax_pp.set_ylabel('Position $p_y$', fontsize=14)
# ax_pp.grid(True, linestyle='--', alpha=0.7)
#
# # Plot the full trajectory as a thin line for context
# ax_pp.plot(X_hat_joseph[:, px_idx], X_hat_joseph[:, py_idx], color_pp, linewidth=1, alpha=0.5, label='KF Estimated Trajectory')

#
# # Plot the Joseph Form ellipses across the full trajectory
# for n in plot_steps_pp:
#     # Extract mean for (px, py)
#     mean_pp = X_hat_joseph[n, [px_idx, py_idx]]
#
#     # Extract 2x2 covariance sub-matrix for (px, py)
#     P_pp = P_joseph[n, [px_idx, py_idx], :][:, [px_idx, py_idx]]
#     # print('P_joseph',P_joseph)
#
#     # Plot ellipse with mean (red dot) and correlation rho (hollow ellipse)
#     plot_covariance_ellipse(ax_pp, mean_pp, P_pp, color_pp, show_rho=True)
#
# # Create custom proxy artists for the final legend
# ax_pp.plot([], [], 'ro', markersize=5, label='Estimated state ($\hat{\mathbf{X}}_{n|n}$)')
# ax_pp.plot([], [], color=color_pp, linewidth=2, label='95% Covariance Ellipse')
# # ax_pp.plot([], [], 'w', label='$\\rho$: Pearson correlation coefficient')
#
# ax_pp.legend(fontsize=10, loc='upper left')
#
# # Set limits to match the full extent of the trajectory
# ax_pp.set_xlim(np.min(X_hat_joseph[:, px_idx]) - 10, np.max(X_hat_joseph[:, px_idx]) + 10)
# ax_pp.set_ylim(np.min(X_hat_joseph[:, py_idx]) - 10, np.max(X_hat_joseph[:, py_idx]) + 10)
#
# plt.tight_layout()
# plt.show()