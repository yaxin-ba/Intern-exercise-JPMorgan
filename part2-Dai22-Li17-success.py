import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class HamiltonianScheduleSolver:
    def __init__(self, P_norm_ref, R_norm_ref, target_reg=50.0):
        self.P_ref = P_norm_ref
        self.R_ref = R_norm_ref
        self.target_reg = target_reg
        self.current_reg = target_reg # Will vary during continuation
        self.schedule_interp = None

        # Trigger the solve sequence
        self.solve_with_continuation()

    def hamiltonian_system(self, t, y):
        beta = y[0]
        p = y[1]

        # Stiffness Constant J ~ C * exp(beta)
        C = self.P_ref / (self.R_ref + 1e-9)

        # Gradients
        # Uses self.current_reg which updates in the loop
        dV_db = (C * np.exp(beta)) + (2 * self.current_reg * beta)

        # Equations: beta' = p/2reg, p' = -dV/db
        d_beta = p / (2 * self.current_reg)
        d_p = -dV_db

        return np.vstack((d_beta, d_p))

    def bc(self, ya, yb):
        return np.array([ya[0], yb[0]])

    def solve_with_continuation(self):
        print(f"Solving Hamiltonian Sequence (Target Reg={self.target_reg})...")

        # 1. Initial Guess (Flat)
        x_mesh = np.linspace(0, 1, 20)
        y_guess = np.zeros((2, 20)) # [beta, p]

        # 2. Homotopy Schedule: Start Easy (High Reg) -> End Hard (Target Reg)
        # We start at 2000.0 (Very flat/easy) and step down to target
        reg_steps = [2000.0, 1000.0, 500.0, 200.0, 100.0, self.target_reg]

        for reg in reg_steps:
            self.current_reg = reg

            # Solve using previous mesh/guess
            res = solve_bvp(self.hamiltonian_system, self.bc, x_mesh, y_guess, tol=1e-4, max_nodes=2000)

            if res.success:
                # Update guess for next harder step
                x_mesh = res.x
                y_guess = res.y
                # print(f"  > Converged at Reg={reg}")
            else:
                print(f"  > Warning: Failed at Reg={reg}. Retrying or stopping.")
                break

        if res.success:
            print(f"  > Success! Hamiltonian Schedule Solved.")
            self.schedule_interp = interp1d(res.x, res.y[0], kind='cubic', fill_value="extrapolate")
        else:
            print("CRITICAL: BVP Solver failed completely. Using fallback (0.0).")
            self.schedule_interp = lambda x: 0.0

    def get_beta(self, lam, current_P_norm):
        if self.schedule_interp is None: return 0.0

        base_beta = self.schedule_interp(lam)

        # Use the "Cubic Decay" logic we discussed to vanish the schedule quickly
        ratio = np.clip(current_P_norm / (self.P_ref + 1e-9), 0.0, 1.0)
        scale = ratio ** 3

        return base_beta * scale

class LinearScheduleSolver:
    def get_beta(self, lam, p_norm):
        return 0.0

# ==========================================
# 2. Strict LEDH Update
# ==========================================
class TrackingSystem:
    def __init__(self):
        self.sensors = np.array([[-150, 0], [150, 0]])
        self.R_val = 0.0005
        self.R = np.eye(2) * self.R_val
        self.dt = 1.0

    def get_true_state(self, t):
        freq = 0.05
        px = 100 * np.sin(freq * t)
        py = 100 * np.sin(2 * freq * t)
        vx = 100 * freq * np.cos(freq * t)
        vy = 100 * 2 * freq * np.cos(2 * freq * t)
        return np.array([px, py, vx, vy])

    def propagate(self, particles):
        F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return (particles @ F.T) + np.random.normal(0, 0.2, particles.shape)

    def measure(self, x):
        dx = x[..., 0:1] - self.sensors[:, 0]
        dy = x[..., 1:2] - self.sensors[:, 1]
        return np.arctan2(dy, dx)

    def jacobian_localized(self, particles):
        px, py = particles[:, 0], particles[:, 1]
        N = len(particles)
        H_all = np.zeros((N, 2, 4))
        for i, s in enumerate(self.sensors):
            sx, sy = s
            dx, dy = px - sx, py - sy
            r2 = dx**2 + dy**2
            H_all[:, i, 0] = -dy / r2
            H_all[:, i, 1] = dx / r2
        return H_all

def update_particles_ledh(particles, z_obs, system, scheduler):
    N, dim = particles.shape
    x = particles.copy()

    # Measure uncertainty for Adaptive Scaling
    P_cov = np.cov(x, rowvar=False)
    P_norm_curr = np.trace(P_cov)

    # 50 Steps for smooth flow
    dt_flow = 0.02
    lambdas = np.arange(0, 1.0 + dt_flow/2, dt_flow)

    for lam in lambdas:
        if lam > 1.0: break

        # --- CRITICAL FIX: ADAPTIVE BETA ---
        beta = scheduler.get_beta(lam, P_norm_curr)
        R_eff = system.R * np.exp(-beta)

        x_mean = np.mean(x, axis=0)
        x_centered = x - x_mean
        P = (x_centered.T @ x_centered) / (N - 1)
        H_all = system.jacobian_localized(x)

        drift = np.zeros_like(x)

        # Batch Inverse for Speed/Stability
        # We can actually vectorize this loop for speed if needed,
        # but the logic here is clear.
        for i in range(N):
            H_i = H_all[i]
            S_i = H_i @ P @ H_i.T + R_eff

            # Robust Solve
            try:
                K_i = (np.linalg.solve(S_i, H_i @ P)).T
            except:
                K_i = P @ H_i.T @ np.linalg.pinv(S_i)

            z_pred = system.measure(x[i:i+1]).flatten()
            innov = (z_obs - z_pred + np.pi) % (2*np.pi) - np.pi
            drift[i] = K_i @ innov

        x = x + drift * dt_flow

    return x

# ==========================================
# 3. Comparison Execution (Corrected)
# ==========================================
sys = TrackingSystem()
np.random.seed(42)

# Initial Uncertainty
P_init_val = np.diag([20**2, 20**2, 5**2, 5**2])
P_norm_init = np.trace(P_init_val)
start_state = sys.get_true_state(0)

# 1. Create Baseline Solver
base_solver = LinearScheduleSolver()

# 2. Create Optimal Solver (Corrected Parameter Name)
# We use target_reg=50.0. The solver will walk from 2000 -> 50 automatically.
opt_solver = HamiltonianScheduleSolver(P_norm_init, np.trace(sys.R), target_reg=50.0)

def run_track(scheduler):
    particles = np.random.multivariate_normal(start_state, P_init_val, 500)
    traj = []
    errs = []
    # Run for 50 steps with dt=4.0 (Hard problem)
    for t in range(50):
        true_x = sys.get_true_state(t)

        # Propagate
        particles = sys.propagate(particles)

        # Measure
        z_obs = sys.measure(true_x) + np.random.multivariate_normal([0,0], sys.R)

        # Update
        particles = update_particles_ledh(particles, z_obs, sys, scheduler)

        # Estimate
        est = np.mean(particles, axis=0)
        traj.append(est)
        errs.append(np.linalg.norm(est[:2] - true_x[:2]))

        if t % 10 == 0: print(f"Step {t}")

    return np.array(traj), np.array(errs)

print("--- Running Baseline (Standard LEDH) ---")
traj_base, err_base = run_track(base_solver)

print("\n--- Running Optimal (Hamiltonian + Cubic Decay) ---")
traj_opt, err_opt = run_track(opt_solver)

# --- PLOTTING ---
true_path = np.array([sys.get_true_state(t) for t in range(50)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Trajectory
ax1.plot(true_path[:,0], true_path[:,1], 'k-', lw=3, alpha=0.3, label='Truth')
ax1.plot(traj_base[:,0], traj_base[:,1], 'b--', label='Baseline')
ax1.plot(traj_opt[:,0], traj_opt[:,1], 'r-', lw=2, label='Optimal')
ax1.set_title(f"Tracking with Sparse Updates (dt={sys.dt})")
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Error
ax2.plot(err_base, 'b--', alpha=0.5, label='Baseline')
ax2.plot(err_opt, 'r-', lw=2, label='Optimal')
ax2.set_title("RMSE Error (m)")
ax2.legend()
ax2.grid(True)

plt.show()

print(f"Avg RMSE Baseline: {np.mean(err_base):.2f}")
print(f"Avg RMSE Optimal:  {np.mean(err_opt):.2f}")