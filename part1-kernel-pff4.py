import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. CORE MATH FUNCTIONS
# ==========================================

def get_all_gradients(particles, y, R_inv, H_idx):
    """
    Calculates gradients for ALL particles (Source Gradients).
    Equation 13 in the paper.
    """
    N, dim = particles.shape
    grads = np.zeros((N, dim))
    for i in range(N):
        # Innovation: y - H(x)
        innov = y - particles[i, H_idx]
        # Gradient: H^T R^-1 (y - Hx)
        grads[i, H_idx] = innov @ R_inv
    return grads

def pff_matrix_kernel_update(particles, grads, B, alpha=0.05):
    """
    (a) Matrix-Valued Kernel Update (Corrected).
    - Uses component-wise distance to maintain repulsion in observed dimensions.
    """
    N, dim = particles.shape
    x = particles.copy()
    ds = 0.01
    diags = np.diag(B) # Variances per dimension

    # Pre-calculate bandwidth scale
    width_scale = alpha * diags

    for step in range(50):
        flow = np.zeros_like(x)
        for i in range(N):
            # diff: Vector from Source (x_j) to Target (x_i)
            # Shape (N, d)
            diff = x[i] - x

            # 1. KERNEL (Component-wise Independent)
            # CRITICAL: No summation over dimensions here.
            # We calculate a separate weight for dimension 1, dimension 2, etc.
            # Shape (N, d)
            k_vals = np.exp(-0.5 * (diff**2) / width_scale)

            # 2. DIVERGENCE (Repelling Force)
            # CRITICAL: Sign must be positive relative to 'diff' (Target - Source)
            # to push particles APART.
            div_K = (diff / width_scale) * k_vals

            # 3. FLOW SMOOTHING
            # Use gradients from source particles ('grads'), weighted by kernel
            term1 = k_vals * grads

            # Integral approximation
            integral = np.mean(term1 + div_K, axis=0)

            # Apply Preconditioner D (matrix B)
            flow[i] = B @ integral

        x += ds * flow
    return x

def pff_scalar_kernel_update(particles, grads, B, alpha=0.05):
    """
    (b) Scalar Kernel Update (The Failure Case).
    - Uses global distance, causing repulsion to vanish if ANY dimension is far.
    """
    N, dim = particles.shape
    x = particles.copy()
    ds = 0.01
    B_inv = np.linalg.inv(B + np.eye(dim)*1e-6)

    for step in range(50):
        flow = np.zeros_like(x)
        for i in range(N):
            diff = x[i] - x

            # 1. KERNEL (Global Scalar)
            # CRITICAL: Sum over axis=1 collapses all dimensions to one number.
            # If x is far from neighbor in dim 19, dist_sq is huge, k_scalar -> 0.
            dist_sq = np.sum((diff @ B_inv) * diff, axis=1)
            k_scalar = np.exp(-0.5 * dist_sq / alpha) # Shape (N,)

            # 2. DIVERGENCE
            # The tiny scalar 'k_scalar' kills the force in ALL dimensions.
            div_K = (diff @ B_inv) * k_scalar[:, None] / alpha

            # 3. FLOW SMOOTHING
            term1 = k_scalar[:, None] * grads

            integral = np.mean(term1 + div_K, axis=0)
            flow[i] = B @ integral

        x += ds * flow
    return x

# ==========================================
# 2. FIGURE 2: REPELLING FORCE PHYSICS
# ==========================================
def plot_figure_2():
    print("Generating Figure 2 (Physics of Repulsion)...")
    # Grid setup
    x = np.linspace(-3, 3, 20) # Unobserved (Wide)
    y = np.linspace(-1, 1, 20) # Observed (Narrow)
    X, Y = np.meshgrid(x, y)

    sigma = np.array([2.0, 0.2])
    alpha = 1.0
    A = np.diag(1.0 / sigma**2)

    # --- Scalar Calculation ---
    U_sca, V_sca = np.zeros_like(X), np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i,j], Y[i,j]])
            # Global distance
            dist_sq = pos.T @ A @ pos
            k_val = np.exp(-0.5 * dist_sq / alpha)
            # Force (Fixed Sign: Positive = Repulsion)
            f = (A @ pos) * k_val
            U_sca[i,j], V_sca[i,j] = f[0], f[1]
    print('U_sca, V_sca', U_sca.shape, V_sca.shape)
    plt.imshow(U_sca)
    plt.show()
    plt.imshow(V_sca)
    plt.show()
    # --- Matrix Calculation ---
    U_mat, V_mat = np.zeros_like(X), np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i,j], Y[i,j]])
            # Independent Dimensions
            k1 = np.exp(-0.5 * pos[0]**2 / (alpha*sigma[0]**2))
            f1 = (pos[0] / (alpha*sigma[0]**2)) * k1

            k2 = np.exp(-0.5 * pos[1]**2 / (alpha*sigma[1]**2))
            f2 = (pos[1] / (alpha*sigma[1]**2)) * k2

            U_mat[i,j], V_mat[i,j] = f1, f2
    print('U_mat, V_mat', U_mat.shape, V_mat.shape, U_mat, V_mat)
    plt.imshow(U_mat)
    plt.show()
    plt.imshow(V_mat)
    plt.show()
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Matrix (Success)
    ax[0].quiver(X, Y, U_mat, V_mat, color='r')
    ax[0].set_title("(a) Matrix-Valued Kernel Repulsion\n(Strong in Y despite X distance)")
    ax[0].set_xlabel("Unobserved Dim (x1)"); ax[0].set_ylabel("Observed Dim (x2)")

    # Right: Scalar (Failure)
    ax[1].quiver(X, Y, U_sca, V_sca, color='b')
    ax[1].set_title("(b) Scalar Kernel Repulsion\n(Weak in Y because X is far)")
    ax[1].set_xlabel("Unobserved Dim (x1)"); ax[1].set_ylabel("Observed Dim (x2)")

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. FIGURE 3 & EXAMPLE 1: LORENZ 96 SIM
# ==========================================
def run_lorenz_experiment():
    print("Running Lorenz 96 Experiment...")
    np.random.seed(42)
    d = 40
    N = 20
    H_idx = np.arange(0, d, 4) # Observe 0, 4, 8... (Index 20 is observed)

    # 1. Truth & Obs
    x_true = np.random.rand(d) + 8.0
    y_obs = x_true[H_idx] + np.random.normal(0, 0.5, len(H_idx))

    # 2. Prior (Wide spread)
    prior = x_true + np.random.normal(0, 2.0, (N, d))

    # 3. Model Matrices
    raw_cov = np.cov(prior.T)
    # Simple Localization for B
    C = np.array([[np.exp(-(min(abs(i-j), d-abs(i-j))/4.0)**2) for j in range(d)] for i in range(d)])
    B = raw_cov * C

    R_inv = np.eye(len(H_idx)) * (1.0/0.5**2)

    # 4. Get Gradients (Source)
    grads = get_all_gradients(prior, y_obs, R_inv, H_idx)

    # 5. Run Filters
    # Use alpha ~ 2.0/N (roughly 0.1) to ensure connection
    alpha_tuned = 2.0 / N

    post_matrix = pff_matrix_kernel_update(prior, grads, B, alpha=alpha_tuned)
    post_scalar = pff_scalar_kernel_update(prior, grads, B, alpha=alpha_tuned)
    print('post_matrix',post_matrix.shape)
    print('post_scalar', post_scalar.shape)

    # --- PLOT FIGURE 3 (Marginals) ---
    plt.figure(figsize=(14, 6))

    # (a) Matrix (Success)
    plt.subplot(1, 2, 1)
    plt.scatter(prior[:, 19], prior[:, 20], facecolors='none', edgecolors='k', alpha=0.5, label='Prior')
    plt.scatter(post_matrix[:, 19], post_matrix[:, 20], color='r', marker='o', label='Posterior')
    plt.axhline(y_obs[5], color='g', linestyle='--', alpha=0.5, label='Observation (x20)')
    plt.title("(a) Matrix Kernel (Correct)\nVariance Preserved (Cloud)")
    plt.xlabel("Unobserved x19"); plt.ylabel("Observed x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (b) Scalar (Failure)
    plt.subplot(1, 2, 2)
    plt.scatter(prior[:, 19], prior[:, 20], facecolors='none', edgecolors='k', alpha=0.5, label='Prior')
    plt.scatter(post_scalar[:, 19], post_scalar[:, 20], color='r', marker='x', label='Posterior')
    plt.axhline(y_obs[5], color='g', linestyle='--', alpha=0.5, label='Observation (x20)')
    plt.title("(b) Scalar Kernel (Failure)\nCollapse to Line")
    plt.xlabel("Unobserved x19"); plt.ylabel("Observed x20")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- PLOT EXAMPLE 1 (Snapshot) ---
    plt.figure(figsize=(12, 6))
    plt.plot(x_true, 'k-', linewidth=2, label='Truth')
    plt.plot(prior.T, 'b.', alpha=0.1, markersize=3) # Prior as background
    plt.plot(post_matrix.T, 'r.', alpha=0.6, markersize=4) # Posterior
    plt.plot(H_idx, y_obs, 'go', markersize=8, fillstyle='none', markeredgewidth=1.5) # Obs

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Truth'),
        Line2D([0], [0], marker='.', color='w', label='Prior', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='.', color='w', label='Posterior (Matrix)', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Observations', markerfacecolor='none', markeredgecolor='g', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Example 1: Lorenz 96 Tracking Snapshot (Matrix PFF)")
    plt.xlabel("State Dimension Index"); plt.ylabel("State Value")
    plt.grid(True, alpha=0.3)
    plt.show()

# Run
if __name__ == "__main__":
    plot_figure_2()
    run_lorenz_experiment()