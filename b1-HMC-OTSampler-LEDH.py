import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. OT RESAMPLER (Sinkhorn)
# ==========================================
class OTResample(tf.Module):
    def __init__(self, epsilon=0.1, n_iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def __call__(self, particles, weights):
        B, N, D = tf.shape(particles)[0], tf.shape(particles)[1], tf.shape(particles)[2]
        N_float = tf.cast(N, tf.float32)

        # Stability: Replace NaNs with uniform weights if any appear
        weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights) / N_float, weights)

        # 1. Cost Matrix (Squared Euclidean)
        x_i = tf.expand_dims(particles, 2)
        x_j = tf.expand_dims(particles, 1)
        C = tf.reduce_sum(tf.square(x_i - x_j), axis=-1)

        # Normalize Cost for numerical stability
        C_mean = tf.reduce_mean(C, axis=[1, 2], keepdims=True) + 1e-8
        C_scaled = C / tf.stop_gradient(C_mean)

        # 2. Sinkhorn (Log Domain)
        log_mu = tf.math.log(weights + 1e-16)
        log_nu = tf.math.log(tf.ones_like(weights) / N_float)
        log_K = -C_scaled / self.epsilon

        f = tf.zeros_like(weights)
        g = tf.zeros_like(weights)

        for _ in range(self.n_iters):
            f = log_mu - tf.reduce_logsumexp(tf.expand_dims(g, 1) + log_K, axis=2)
            g = log_nu - tf.reduce_logsumexp(tf.expand_dims(f, 2) + log_K, axis=1)

        # 3. Transport & Apply
        log_P = tf.expand_dims(f, 2) + tf.expand_dims(g, 1) + log_K
        P = tf.exp(log_P)

        new_particles = N_float * tf.matmul(P, particles, transpose_a=True)
        new_weights = tf.ones_like(weights) / N_float
        return new_particles, new_weights


# ==========================================
# 2. UNGM MODEL & DIFFERENTIABLE FILTER
# ==========================================
class UNGM_Model(tf.Module):
    def __init__(self):
        super().__init__()
        # Initialize with "Wrong" values to test learning
        self.log_Q = tf.Variable(tf.math.log(2.0), name='log_Q')
        self.log_R = tf.Variable(tf.math.log(5.0), name='log_R')

    def get_Q(self): return tf.exp(self.log_Q)

    def get_R(self): return tf.exp(self.log_R)

    def transition(self, x, t, noise=None):
        if noise is None:
            noise = tf.random.normal(tf.shape(x)) * tf.sqrt(self.get_Q())
        term1 = 0.5 * x
        term2 = 25.0 * x / (1.0 + x ** 2)
        term3 = 8.0 * tf.cos(1.2 * tf.cast(t, tf.float32))
        return term1 + term2 + term3 + noise

    def observation(self, x): return (x ** 2) / 20.0

    def jacobian_h(self, x): return x / 10.0


class Differentiable_Li17_Filter(tf.Module):
    def __init__(self, model_class, resampler):
        super().__init__()
        self.model = model_class()
        self.resampler = resampler
        self.n_flow = 15  # Standard Li17 Steps
        self.dt = 1.0 / float(self.n_flow)

    def flow_step(self, x_flow, x_aux, P_pred, z, lam):
        # Localized Exact Daum-Huang (LEDH) Equation
        R = self.model.get_R()
        H = self.model.jacobian_h(x_aux)
        S = R + lam * (H ** 2) * P_pred
        S_inv = 1.0 / (S + 1e-8)

        A = -0.5 * P_pred * H * S_inv * H
        e = self.model.observation(x_aux) - H * x_aux
        K = (1.0 + lam * A) * P_pred * H / R
        b = (1.0 + 2.0 * lam * A) * (K * (tf.reshape(z, [-1, 1, 1]) - e) + A * x_aux)

        # Update
        x_new = x_flow + self.dt * (A * x_flow + b)
        aux_new = x_aux + self.dt * (A * x_aux + b)
        log_det = tf.math.log(tf.abs(1.0 + self.dt * A) + 1e-16)
        return x_new, aux_new, log_det

    @tf.function
    def __call__(self, observations, N=50):
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]

        x = tf.random.normal([B, N, 1]) * tf.sqrt(self.model.get_Q())
        weights = tf.ones([B, N]) / float(N)
        log_likelihood_sum = tf.constant(0.0)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            # FIX: Correct variable name matching in shape_invariants
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (weights, tf.TensorShape([None, None])),
                    (x, tf.TensorShape([None, None, None])),
                    (log_likelihood_sum, tf.TensorShape([]))
                ]
            )

            obs = observations[:, t]
            if t > 0:
                x, weights = self.resampler(x, weights)

            # Predict
            noise = tf.random.normal([B, N, 1]) * tf.sqrt(self.model.get_Q())
            mu_pred_det = self.model.transition(x, t, noise=0.0)
            x_pred = mu_pred_det + noise
            mean_pred = tf.reduce_mean(x_pred, axis=1, keepdims=True)
            P_pred = tf.reduce_mean(tf.square(x_pred - mean_pred), axis=1, keepdims=True) + 1e-4

            # Standard Li17 Flow
            x_flow = x_pred
            x_aux = mu_pred_det
            log_det_sum = tf.zeros([B, N, 1])
            for k in range(self.n_flow):
                lam = float(k) * self.dt
                x_flow, x_aux, log_det = self.flow_step(x_flow, x_aux, P_pred, obs, lam)
                log_det_sum += log_det

            # Weighting
            innov = tf.reshape(obs, [-1, 1, 1]) - self.model.observation(x_flow)
            log_lik = -0.5 * tf.square(innov) / self.model.get_R()
            diff_new = x_flow - mu_pred_det
            diff_old = x_pred - mu_pred_det
            log_prior_ratio = -0.5 * (tf.square(diff_new) - tf.square(diff_old)) / self.model.get_Q()

            log_w = tf.math.log(weights + 1e-16) + tf.squeeze(log_lik + log_prior_ratio + log_det_sum, -1)
            log_w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
            weights = tf.exp(log_w - log_w_max)
            w_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weights = weights / (w_sum + 1e-16)

            # Accumulate Likelihood
            step_log_lik = tf.squeeze(log_w_max) + tf.math.log(tf.squeeze(w_sum) + 1e-16)
            log_likelihood_sum += tf.reduce_mean(step_log_lik)

            # Estimate State
            est = tf.reduce_sum(weights[:, :, tf.newaxis] * x_flow, axis=1)
            est_states = est_states.write(t, est)
            x = x_flow

        return tf.transpose(est_states.stack(), [1, 0, 2]), log_likelihood_sum


# ==========================================
# 3. EXECUTION & HMC
# ==========================================
# Generate Data (Fixed Ground Truth)
T_steps = 40


def generate_data(T=40):
    x, y = np.zeros(T), np.zeros(T)
    x[0] = 0.1
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + 25 * x[t - 1] / (1 + x[t - 1] ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.random.normal(0,
                                                                                                                   np.sqrt(
                                                                                                                       10.0))
        y[t] = (x[t] ** 2) / 20.0 + np.random.normal(0, np.sqrt(1.0))
    return x, y


gt_states, obs_np = generate_data(T_steps)
obs_tensor = tf.convert_to_tensor(obs_np[np.newaxis, :], dtype=tf.float32)

# Init Filter
filter_module = Differentiable_Li17_Filter(UNGM_Model, OTResample(epsilon=0.1))


# HMC Target
@tf.function
def target_log_prob_fn(log_Q, log_R):
    filter_module.model.log_Q.assign(log_Q)
    filter_module.model.log_R.assign(log_R)
    # Filter Likelihood + Weak Priors
    _, log_lik = filter_module(obs_tensor, N=40)
    prior = -0.5 * ((log_Q - 2.3) ** 2 + (log_R - 0.0) ** 2)
    return log_lik + prior


# Robust HMC Config
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn,
    step_size=0.05,
    num_leapfrog_steps=10
)
kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=40
)

print("Running HMC...")
# FIX: Capture raw output, do NOT unpack immediately
raw_samples, _ = tfp.mcmc.sample_chain(
    num_results=80,
    num_burnin_steps=40,
    current_state=[filter_module.model.log_Q, filter_module.model.log_R],
    kernel=kernel,
    trace_fn=None
)

# Extract properly
est_log_Q = raw_samples[0]
est_log_R = raw_samples[1]
final_Q = np.mean(np.exp(est_log_Q.numpy()[-20:]))
final_R = np.mean(np.exp(est_log_R.numpy()[-20:]))

print(f"\nFinal Learned Q: {final_Q:.4f} (True: 10.0)")
print(f"Final Learned R: {final_R:.4f} (True: 1.0)")

# ==========================================
# 4. PLOTTING
# ==========================================
filter_module.model.log_Q.assign(np.log(final_Q))
filter_module.model.log_R.assign(np.log(final_R))
est_states_tf, _ = filter_module(obs_tensor, N=200)
est_states = est_states_tf.numpy()[0, :, 0]

plt.figure(figsize=(12, 6))
plt.plot(gt_states, 'k-', alpha=0.8, label='True State')
plt.plot(est_states, 'r--', linewidth=2, label=f'Li17 Estimate (Q={final_Q:.1f})')
obs_mag = np.sqrt(20 * np.abs(obs_np))
plt.scatter(np.arange(T_steps), obs_mag, c='g', s=15, alpha=0.3, label='Observed Roots')
plt.scatter(np.arange(T_steps), -obs_mag, c='g', s=15, alpha=0.3)
plt.title(f"UNGM Tracking with Learned Parameters (Q={final_Q:.2f}, R={final_R:.2f})")
plt.legend();
plt.grid(True, alpha=0.3)
plt.show()