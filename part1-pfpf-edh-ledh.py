import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.linalg import block_diag


class ParticleFlowParticleFilter:
    def __init__(self, mode='LEDH', n_particles=200, state_dim=1, meas_dim=1, Q=None, R=None):
        """
        Implements PF-PF (EDH/LEDH) from Li & Coates (2017).
        """
        self.mode = mode  # 'EDH' or 'LEDH'
        self.N = n_particles
        self.d = state_dim
        self.s = meas_dim
        self.Q = Q  # Process Noise Covariance
        self.R = R  # Measurement Noise Covariance

        # Initialize Posterior Covariance (Identity * 10 initially)
        self.P_post = np.eye(self.d) * 10.0

        # Flow Steps (Log-spaced as per Sec IV-C) [cite: 386-387]
        self.n_lambda = 29
        q = 1.2
        eps1 = (1 - q) / (1 - q ** self.n_lambda)
        self.steps = [eps1 * (q ** i) for i in range(self.n_lambda)]

    def predict(self, particles, dynamics_fn, jacob_F_fn):
        """
        Step 1: Prediction [cite: 232-235]
        - Propagates particles through dynamics to get eta_0 (Prior).
        - Propagates Covariance P_post -> P_pred using EKF.
        """
        # 1. Propagate Particles (Monte Carlo Prediction)
        # eta_0 = g(x_k-1) + v_k
        noise = np.random.multivariate_normal(np.zeros(self.d), self.Q, self.N)
        # print('particles',particles.shape)
        # print('noise',noise.shape)
        particles_pred = np.array([dynamics_fn(p) for p in particles]) + noise

        # 2. Propagate Covariance (EKF Prediction)
        # We approximate the Jacobian F at the mean of the particles
        x_mean = np.mean(particles, axis=0)
        F = jacob_F_fn(x_mean)
        self.P_pred = F @ self.P_post @ F.T + self.Q

        return particles_pred

    def update(self, particles_pred, weights, z, h_fn, jacob_h_fn, R_adaptive=None):
        """
        Step 2: Flow & Update [cite: 247-294]
        - Migrates particles (eta_0 -> eta_1)
        - Updates weights using invertible mapping formula.
        - Updates Covariance P_pred -> P_post for next step.
        """
        R_cov = R_adaptive if R_adaptive is not None else self.R
        R_inv = np.linalg.inv(R_cov)

        eta = particles_pred.copy()
        log_det_J = np.zeros(self.N)  # Accumulator for log|det(T)|
        lam = 0.0

        # --- A. Particle Flow Loop ---
        for eps in self.steps:
            lam += eps

            # Helper to calc A, b
            def get_flow_params(x, P, R_inv, z, lam):
                H = jacob_h_fn(x)
                h = h_fn(x)
                # S = lam * H P H' + R
                S = lam * (H @ P @ H.T) + R_cov
                S_inv = np.linalg.pinv(S)

                # A = -0.5 P H' S^-1 H [cite: 162]
                K_part = P @ H.T @ S_inv
                A = -0.5 * K_part @ H

                # b = (I + 2 lam A) [(I + lam A) P H' R^-1 (z - e) + A x] [cite: 164]
                e = h - H @ x
                term1 = (np.eye(self.d) + lam * A) @ P @ H.T @ R_inv @ (z - e)
                term2 = A @ x
                b = (np.eye(self.d) + 2 * lam * A) @ (term1 + term2)
                return A, b

            if self.mode == 'EDH':
                # Linearize ONCE at the mean
                x_mean = np.mean(eta, axis=0)
                A, b = get_flow_params(x_mean, self.P_pred, R_inv, z, lam)

                # Apply flow to all particles
                # eta = eta + eps * (A eta + b)
                drift = (eta @ A.T) + b
                eta += eps * drift

                # Update Jacobian Det (Same for all)
                sign, logdet = np.linalg.slogdet(np.eye(self.d) + eps * A)
                log_det_J += logdet

            elif self.mode == 'LEDH':
                # Linearize at EACH particle
                for i in range(self.N):
                    A_i, b_i = get_flow_params(eta[i], self.P_pred, R_inv, z, lam)
                    eta[i] += eps * (A_i @ eta[i] + b_i)
                    sign, logdet = np.linalg.slogdet(np.eye(self.d) + eps * A_i)
                    log_det_J[i] += logdet

        # --- B. Weight Update [cite: 382] ---
        # w = w_prev * p(z|x) * |det|
        log_lik = np.zeros(self.N)
        for i in range(self.N):
            # Check if Poisson or Gaussian based on R_adaptive (Example 3 is Poisson)
            if self.mode == 'POISSON_SPECIAL':  # Flag for Ex 3
                rate = h_fn(eta[i])
                log_lik[i] = np.sum(poisson.logpmf(z, rate))
            else:
                innov = z - h_fn(eta[i])
                log_lik[i] = -0.5 * innov @ R_inv @ innov

        log_w = np.log(weights + 1e-300) + log_lik + log_det_J
        w_unnorm = np.exp(log_w - np.max(log_w))
        new_weights = w_unnorm / np.sum(w_unnorm)

        # --- C. EKF Update for P_post (used in next prediction) [cite: 294] ---
        # "Apply EKF/UKF update: (m, P) -> (m_k|k, P_k)"
        x_est = np.average(eta, axis=0, weights=new_weights)
        H_final = jacob_h_fn(x_est)
        S_final = H_final @ self.P_pred @ H_final.T + R_cov
        K_final = self.P_pred @ H_final.T @ np.linalg.pinv(S_final)
        self.P_post = (np.eye(self.d) - K_final @ H_final) @ self.P_pred

        return eta, new_weights, x_est

    def resample(self, particles, weights):
        """Systematic Resampling [cite: 297]"""
        neff = 1.0 / np.sum(weights ** 2)
        if neff < self.N / 2:
            positions = (np.arange(self.N) + np.random.random()) / self.N
            indexes = np.zeros(self.N, 'i')
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            return particles[indexes], np.ones(self.N) / self.N
        return particles, weights


def run_example_1_acoustic():
    print("--- Running Example 1: Multi-Target Acoustic Tracking ---")

    # --- Setup Parameters ---
    # Dynamics: Constant Velocity for 4 targets (16D state)
    dt = 1.0
    F_block = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    F_mat = block_diag(F_block, F_block, F_block, F_block)

    Q_block = np.array([[3, 0, 0.1, 0], [0, 3, 0, 0.1], [0.1, 0, 0.03, 0], [0, 0.1, 0, 0.03]])
    Q_mat = block_diag(Q_block, Q_block, Q_block, Q_block)

    # Measurements: 25 Sensors
    grid = np.linspace(0, 40, 5)
    sensors = np.array([[x, y] for x in grid for y in grid])
    Psi, d0 = 10.0, 0.1
    R_mat = np.eye(25) * 0.01

    # Helpers
    def f_dynamics(x):
        return F_mat @ x

    def jacob_f(x):
        return F_mat

    def h_meas(x):
        z = np.zeros(25)
        for s in range(25):
            for c in range(4):  # 4 targets
                pos = x[c * 4: c * 4 + 2]
                dist = np.linalg.norm(pos - sensors[s])
                z[s] += Psi / (dist + d0)
        return z

    def jacob_h(x):
        H = np.zeros((25, 16))
        for s in range(25):
            for c in range(4):
                pos = x[c * 4: c * 4 + 2]
                diff = pos - sensors[s]
                dist = np.linalg.norm(diff)
                H[s, c * 4: c * 4 + 2] = -Psi * diff / ((dist + d0) ** 2 * dist)
        return H

    # --- Initialization ---
    x_true = np.array([12, 6, 0.001, 0.001, 32, 32, -0.001, -0.005, 20, 13, -0.1, 0.01, 15, 35, 0.002, 0.002])
    N = 200  # Paper uses 500

    # Initialize Filter
    # pf = ParticleFlowParticleFilter(mode='LEDH', n_particles=N, state_dim=16, meas_dim=25, Q=Q_mat, R=R_mat)
    pf_edh = ParticleFlowParticleFilter(mode='EDH', n_particles=N, state_dim=16, meas_dim=25, Q=Q_mat, R=R_mat)
    pf_ledh = ParticleFlowParticleFilter(mode='LEDH', n_particles=N, state_dim=16, meas_dim=25, Q=Q_mat, R=R_mat)

    # Initial Particles (Perturbed from truth)
    init_noise = np.random.multivariate_normal(np.zeros(16), np.eye(16) * 10, N)
    particles_ledh = x_true + init_noise
    particles_edh = x_true + init_noise  # Start same for fair comparison

    weights_ledh = np.ones(N) / N
    weights_edh = np.ones(N) / N

    history_true = []
    history_est_ledh = []
    history_est_edh = []

    # --- Loop ---
    for t in range(40):
        if t % 10 == 0: print(f"Step {t}/40")

        # 1. Simulate Reality
        x_true = f_dynamics(x_true) + np.random.multivariate_normal(np.zeros(16), Q_mat)
        z_obs = h_meas(x_true) + np.random.normal(0, 0.1, 25)

        # 2. Filter Predict
        particles_ledh = pf_ledh.predict(particles_ledh, f_dynamics, jacob_f)
        particles_edh = pf_edh.predict(particles_edh, f_dynamics, jacob_f)

        # 3. Filter Update
        particles_ledh, weights_ledh, x_est_ledh = pf_ledh.update(particles_ledh, weights_ledh, z_obs, h_meas, jacob_h)
        particles_edh, weights_edh, x_est_edh = pf_edh.update(particles_edh, weights_edh, z_obs, h_meas, jacob_h)

        # 4. Resample
        particles_ledh, weights_ledh = pf_ledh.resample(particles_ledh, weights_ledh)
        particles_edh, weights_edh = pf_edh.resample(particles_edh, weights_edh)

        history_true.append(x_true)
        history_est_ledh.append(x_est_ledh)
        history_est_edh.append(x_est_edh)

    # --- Plotting ---
    ht = np.array(history_true)
    he_ledh = np.array(history_est_ledh)
    he_edh = np.array(history_est_edh)

    plt.figure(figsize=(10, 10))
    plt.scatter(sensors[:, 0], sensors[:, 1], marker='s', c='gray', alpha=0.5, label='Sensors')
    colors = ['r', 'g', 'b', 'c']

    # Dummy plot lines for legend clarity
    plt.plot([], [], 'k-', label='True Trajectory')
    plt.plot([], [], 'k--', linewidth=2, label='LEDH Estimate')
    plt.plot([], [], 'k:', linewidth=2, label='EDH Estimate')

    for i in range(4):
        # Truth
        plt.plot(ht[:, i * 4], ht[:, i * 4 + 1], c=colors[i], ls='-', alpha=0.6)
        # LEDH Estimate
        plt.plot(he_ledh[:, i * 4], he_ledh[:, i * 4 + 1], c=colors[i], ls='--', lw=2)
        # EDH Estimate
        plt.plot(he_edh[:, i * 4], he_edh[:, i * 4 + 1], c=colors[i], ls=':', lw=2)

    plt.title("Ex 1: Acoustic Tracking (PF-PF LEDH vs EDH)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



def run_example_2_linear():
    print("\n--- Running Example 2: Linear Gaussian ---")

    # --- Setup [cite: 568-580] ---
    dim = 64
    alpha_dyn = 0.9

    # Spatial Covariance Q
    coords = np.array([(i, j) for i in range(8) for j in range(8)])
    Q_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            dist2 = np.sum((coords[i] - coords[j]) ** 2)
            Q_mat[i, j] = 3.0 * np.exp(-dist2 / 20.0) + (0.01 if i == j else 0)

    R_mat = np.eye(dim) * 1.0  # Sigma_z = 1.0

    # Linear Functions
    def f_lin(x):
        return alpha_dyn * x

    def j_f(x):
        return np.eye(dim) * alpha_dyn

    def h_lin(x):
        return x

    def j_h(x):
        return np.eye(dim)

    # --- Init ---
    # pf = ParticleFlowParticleFilter(mode='EDH', n_particles=200, state_dim=dim, meas_dim=dim, Q=Q_mat, R=R_mat)
    pf_ledh = ParticleFlowParticleFilter(mode='LEDH', n_particles=200, state_dim=dim, meas_dim=dim, Q=Q_mat, R=R_mat)
    pf_edh = ParticleFlowParticleFilter(mode='EDH', n_particles=200, state_dim=dim, meas_dim=dim, Q=Q_mat, R=R_mat)

    x_true = np.zeros(dim)
    particles = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 200)
    particles_l = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 200)
    weights = np.ones(200) / 200
    weights_l = np.ones(200) / 200

    mse_list = []
    mse_list_l = []

    # --- Loop ---
    for t in range(15):
        if t % 3 == 0: print(f"Step {t}/15")
        # Reality
        x_true = f_lin(x_true) + np.random.multivariate_normal(np.zeros(dim), Q_mat)
        z_obs = h_lin(x_true) + np.random.normal(0, 1.0, dim)

        # Filter
        particles = pf_edh.predict(particles, f_lin, j_f)
        particles, weights, x_est = pf_edh.update(particles, weights, z_obs, h_lin, j_h)
        particles, weights = pf_edh.resample(particles, weights)

        particles_l = pf_ledh.predict(particles_l, f_lin, j_f)
        particles_l, weights_l, x_est_l = pf_ledh.update(particles_l, weights_l, z_obs, h_lin, j_h)
        particles_l, weights_l = pf_ledh.resample(particles_l, weights_l)

        mse = np.mean((x_est - x_true) ** 2)
        mse_list.append(mse)

        mse_l = np.mean((x_est_l - x_true) ** 2)
        mse_list_l.append(mse_l)

    plt.figure(figsize=(7, 5))
    plt.plot(mse_list_l, '*-', label='PF-PF (LEDH)')
    plt.plot(mse_list, 'o-', label='PF-PF (EDH)')
    plt.legend()
    plt.title("Ex 2: Linear Gaussian MSE")
    plt.xlabel("Time");
    plt.ylabel("MSE")
    plt.grid(True);
    plt.show()



def run_example_3_poisson():
    print("\n--- Running Example 3: High-Dim Poisson ---")

    # --- 1. Setup [cite: 630-642] ---
    # Dimensions and Parameters
    dim = 144  # 12x12 Grid
    m1, m2 = 1.0, 1.0 / 3.0

    # Dynamics: The paper uses a Skewed-t model.
    # For this reproduction, we approximate the dynamics with a linear Gaussian
    # that matches the "heavy tail" challenge via process noise.
    alpha_dyn = 0.9
    Q_val = 0.5
    Q_mat = np.eye(dim) * Q_val

    N_particles = 10000

    # Measurement Functions
    def h_pois(x):
        # Returns Rates lambda = m1 * exp(m2 * x)
        return m1 * np.exp(m2 * x)

    def jacob_h_pois(x):
        # Jacobian of rate: m1 * m2 * exp(m2 * x)
        # Returns (144, 144) diagonal matrix
        diag_val = m1 * m2 * np.exp(m2 * x)
        return np.diag(diag_val)

    def f_dyn(x): return alpha_dyn * x

    def j_f(x): return np.eye(dim) * alpha_dyn

    # --- 2. Initialization ---
    # Init Filter (EDH is preferred for Ex3 [cite: 653])
    # We pass a dummy R initially; it will be overridden adaptively.
    pf = ParticleFlowParticleFilter(mode='EDH', n_particles=N_particles, state_dim=dim, meas_dim=dim, Q=Q_mat, R=np.eye(dim))
    pf.mode = 'POISSON_SPECIAL'  # Flag to enforce Poisson Likelihood in weight update

    # Init State and Particles (Start at 0 as per [cite: 649])
    x_true = np.zeros(dim)
    # Particles sampled around true state
    particles = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), N_particles)
    weights = np.ones(N_particles) / N_particles

    # Covariance Estimate for Filter (Identity initially)
    pf.P_post = np.eye(dim)

    mse_history = []
    T_steps = 20  #

    # --- 3. Time Loop ---
    for t in range(T_steps):
        print(f"Ex3 Step {t + 1}/{T_steps}")

        # A. Simulate Reality (Skewed-t approximated by Gaussian transition + Poisson Meas)
        # x_k = alpha * x_{k-1} + v_k
        x_true = f_dyn(x_true) + np.random.multivariate_normal(np.zeros(dim), Q_mat)

        # Measurement z ~ Poisson(h(x)) [cite: 641]
        rates_true = h_pois(x_true)
        z_obs = poisson.rvs(rates_true)

        # B. Prediction [cite: 232]
        particles = pf.predict(particles, f_dyn, j_f)

        # C. Adaptive R Calculation [cite: 647-648]
        # "Measurement covariance R is updated... using eta_bar (mean)"
        # For Poisson, Variance approx equals Rate.
        # We estimate rates using the predicted mean of particles.
        pred_mean = np.average(particles, axis=0, weights=weights)
        pred_rates = h_pois(pred_mean)
        # Add small epsilon to avoid singular matrix if rate is 0
        R_adaptive = np.diag(pred_rates + 1e-6)

        # D. Update & Flow [cite: 247]
        # Pass the adaptive R so the Flow PDE uses the correct noise scale
        particles, weights, x_est = pf.update(particles, weights, z_obs, h_pois, jacob_h_pois, R_adaptive=R_adaptive)

        # E. Resample
        particles, weights = pf.resample(particles, weights)

        # F. Track Performance
        mse = np.mean((x_est - x_true) ** 2)
        mse_history.append(mse)

    # --- 4. Visualization ---
    plt.figure(figsize=(20, 6))

    # Plot 1: MSE over time
    plt.subplot(1, 3, 1)
    plt.plot(range(1, T_steps + 1), mse_history, 'b-o', label='PF-PF (EDH)')
    plt.title(f"Ex 3: MSE over {T_steps} Steps")
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()

    # Plot 2: Final State Heatmap Comparison
    plt.subplot(1, 3, 2)
    # Compare 1st dimension of True vs Est just to see scale
    # Or reshape to 12x12 grid as in paper context
    # combined = np.hstack((x_true.reshape(12, 12), x_est.reshape(12, 12)))
    # plt.imshow(combined, cmap='viridis')
    p1 = plt.imshow(x_true.reshape(12, 12));
    plt.title("True State (144D)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    p2 = plt.imshow(x_est.reshape(12, 12));
    plt.title("Estimate (144D)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    #
    # # plt.title("True State (Left) vs Estimate (Right)")
    # plt.axis('off')
    # plt.colorbar(fraction=0.046, pad=0.04)
    #
    # plt.tight_layout()
    # plt.show()



# run_example_1_acoustic()

# run_example_2_linear()

run_example_3_poisson()
