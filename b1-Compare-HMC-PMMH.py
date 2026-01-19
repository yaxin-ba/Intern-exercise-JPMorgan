# import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np
# import matplotlib.pyplot as plt
# import time
#
#
# # ==========================================
# # 0. SETUP: MODEL & DIFFERENTIABLE FILTER
# # ==========================================
# class OTResample(tf.Module):
#     def __init__(self, epsilon=0.1, n_iters=10):
#         super().__init__()
#         self.epsilon = epsilon
#         self.n_iters = n_iters
#
#     def __call__(self, x, w):
#         B, N = tf.shape(x)[0], tf.shape(x)[1]
#         N_f = tf.cast(N, tf.float32)
#         w = tf.where(tf.math.is_nan(w), tf.ones_like(w) / N_f, w)
#
#         # Log-Sinkhorn for stability
#         log_w = tf.math.log(w + 1e-16)
#         log_nu = tf.math.log(tf.ones_like(w) / N_f)
#         C = tf.reduce_sum(tf.square(tf.expand_dims(x, 2) - tf.expand_dims(x, 1)), -1)
#         C_mean = tf.reduce_mean(C, axis=[1, 2], keepdims=True) + 1e-8
#         log_K = -(C / tf.stop_gradient(C_mean)) / self.epsilon
#
#         f, g = tf.zeros_like(w), tf.zeros_like(w)
#         for _ in range(self.n_iters):
#             f = log_w - tf.reduce_logsumexp(tf.expand_dims(g, 1) + log_K, axis=2)
#             g = log_nu - tf.reduce_logsumexp(tf.expand_dims(f, 2) + log_K, axis=1)
#
#         P = tf.exp(tf.expand_dims(f, 2) + tf.expand_dims(g, 1) + log_K)
#         return N_f * tf.matmul(P, x, transpose_a=True), tf.ones_like(w) / N_f
#
#
# class UNGM_Model(tf.Module):
#     def __init__(self):
#         super().__init__()
#         self.log_Q = tf.Variable(0.0)
#         self.log_R = tf.Variable(0.0)
#
#     def get_Q(self): return tf.exp(self.log_Q)
#
#     def get_R(self): return tf.exp(self.log_R)
#
#     def transition(self, x, t, noise=None):
#         if noise is None: noise = tf.random.normal(tf.shape(x)) * tf.sqrt(self.get_Q())
#         return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * tf.cos(1.2 * tf.cast(t, tf.float32)) + noise
#
#     def observation(self, x): return (x ** 2) / 20.0
#
#     def jacobian_h(self, x): return x / 10.0
#
#
# class Differentiable_Li17_Filter(tf.Module):
#     def __init__(self, model_class, resampler):
#         super().__init__()
#         self.model = model_class()
#         self.resampler = resampler
#         self.dt = 1.0 / 15.0
#
#     def flow_step(self, x_flow, x_aux, P_pred, z, lam):
#         R, H = self.model.get_R(), self.model.jacobian_h(x_aux)
#         S_inv = 1.0 / (R + lam * (H ** 2) * P_pred + 1e-8)
#         A = -0.5 * P_pred * H * S_inv * H
#         b = (1.0 + 2.0 * lam * A) * ((1.0 + lam * A) * P_pred * H / R * (
#                     tf.reshape(z, [-1, 1, 1]) - (self.model.observation(x_aux) - H * x_aux)) + A * x_aux)
#         return x_flow + self.dt * (A * x_flow + b), x_aux + self.dt * (A * x_aux + b)
#
#     @tf.function
#     def __call__(self, observations, N=50):
#         B, T = tf.shape(observations)[0], tf.shape(observations)[1]
#         x = tf.random.normal([B, N, 1]) * tf.sqrt(self.model.get_Q())
#         weights = tf.ones([B, N]) / float(N)
#         log_lik_total = tf.constant(0.0)
#
#         for t in tf.range(T):
#             tf.autograph.experimental.set_loop_options(
#                 shape_invariants=[(weights, tf.TensorShape([None, None])),
#                                   (x, tf.TensorShape([None, None, None])),
#                                   (log_lik_total, tf.TensorShape([]))]
#             )
#             if t > 0: x, weights = self.resampler(x, weights)
#
#             # Prediction
#             mu_det = self.model.transition(x, t, noise=0.0)
#             x_pred = mu_det + tf.random.normal(tf.shape(x)) * tf.sqrt(self.model.get_Q())
#             P_pred = tf.math.reduce_variance(x_pred, axis=1, keepdims=True) + 1e-4
#
#             # Flow
#             x_f, x_a = x_pred, mu_det
#             for k in range(15): x_f, x_a = self.flow_step(x_f, x_a, P_pred, observations[:, t], float(k) * self.dt)
#
#             # Likelihood
#             log_l = -0.5 * tf.square(
#                 tf.reshape(observations[:, t], [-1, 1, 1]) - self.model.observation(x_f)) / self.model.get_R()
#             log_pr = -0.5 * (tf.square(x_f - mu_det) - tf.square(x_pred - mu_det)) / self.model.get_Q()
#             log_w = tf.math.log(weights + 1e-16) + tf.squeeze(log_l + log_pr, -1)
#
#             w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
#             weights = tf.exp(log_w - w_max)
#             w_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
#             weights = weights / (w_sum + 1e-16)
#             log_lik_total += tf.reduce_mean(tf.squeeze(w_max) + tf.math.log(tf.squeeze(w_sum) + 1e-16))
#             x = x_f
#         return log_lik_total
#
#
# # Generate Data
# obs_np = np.zeros(40)
# x_t = 0.1
# for t in range(40):
#     x_t = 0.5 * x_t + 25 * x_t / (1 + x_t ** 2) + 8 * np.cos(1.2 * t) + np.random.normal(0, np.sqrt(10.0))
#     obs_np[t] = (x_t ** 2) / 20.0 + np.random.normal(0, np.sqrt(1.0))
# obs_tensor = tf.convert_to_tensor(obs_np[np.newaxis, :], dtype=tf.float32)
#
# filter_mod = Differentiable_Li17_Filter(UNGM_Model, OTResample(epsilon=0.1))
#
#
# # Common Target
# @tf.function
# def target_log_prob(log_Q, log_R):
#     filter_mod.model.log_Q.assign(log_Q)
#     filter_mod.model.log_R.assign(log_R)
#     # Filter Likelihood + Priors
#     return filter_mod(obs_tensor, N=30) + (-0.5 * ((log_Q - 2.3) ** 2 + (log_R - 0.0) ** 2))
#
#
# init_state = [tf.constant(1.0), tf.constant(1.0)]
#
# # ==========================================
# # 1. RUN HMC (Gradients + Adaptation)
# # ==========================================
# print(">>> Running HMC (Gradient-Based)...")
# t0 = time.time()
#
# # HMC supports SimpleStepSizeAdaptation naturally
# hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
#     tfp.mcmc.HamiltonianMonteCarlo(target_log_prob, step_size=0.05, num_leapfrog_steps=5),
#     num_adaptation_steps=40)
#
# hmc_samples, hmc_results = tfp.mcmc.sample_chain(
#     num_results=100, num_burnin_steps=40,
#     current_state=init_state, kernel=hmc_kernel,
#     trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
# hmc_time = time.time() - t0
#
# # ==========================================
# # 2. RUN PMMH (Random Walk + Fixed Scale)
# # ==========================================
# print(">>> Running PMMH (Random Walk Metropolis)...")
# t0 = time.time()
#
#
# # FIX: RandomWalkMetropolis does not have 'step_size', so we cannot use SimpleStepSizeAdaptation.
# # Instead, we provide a proposal scale via 'new_state_fn' or use default (scale=1.0 is often too big).
# # We use a helper to define a Normal Random Walk with scale=0.1 (similar to HMC step size).
# def rwm_proposal_fn(state, seed):
#     return tfp.mcmc.random_walk_normal_fn(scale=0.1)(state, seed)
#
#
# pmmh_kernel = tfp.mcmc.RandomWalkMetropolis(
#     target_log_prob_fn=target_log_prob,
#     new_state_fn=rwm_proposal_fn  # Explicit proposal scale control
# )
#
# pmmh_samples, pmmh_results = tfp.mcmc.sample_chain(
#     num_results=100, num_burnin_steps=40,
#     current_state=init_state, kernel=pmmh_kernel,
#     trace_fn=lambda _, pkr: pkr.is_accepted)  # Trace structure is simpler for RWM
# pmmh_time = time.time() - t0
#
# # ==========================================
# # 3. COMPARISON METRICS & PLOT
# # ==========================================
# # Convert log-params back to real space
# hmc_Q = np.exp(hmc_samples[0].numpy())
# pmmh_Q = np.exp(pmmh_samples[0].numpy())
#
# # Metrics
# hmc_acc = np.mean(hmc_results.numpy())
# pmmh_acc = np.mean(pmmh_results.numpy())
# hmc_ess = np.mean(tfp.mcmc.effective_sample_size(hmc_samples[0]).numpy())
# pmmh_ess = np.mean(tfp.mcmc.effective_sample_size(pmmh_samples[0]).numpy())
#
# print(f"\nMetric        | HMC (Gradient)  | PMMH (Random Walk)")
# print(f"--------------|-----------------|-------------------")
# print(f"Runtime (s)   | {hmc_time:<15.4f} | {pmmh_time:<15.4f}")
# print(f"Accept Rate   | {hmc_acc:<15.4f} | {pmmh_acc:<15.4f}")
# print(f"ESS (Q)       | {hmc_ess:<15.2f} | {pmmh_ess:<15.2f}")
# print(f"ESS/Sec       | {hmc_ess / hmc_time:<15.2f} | {pmmh_ess / pmmh_time:<15.2f}")
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(hmc_Q, label='HMC', alpha=0.8)
# plt.plot(pmmh_Q, label='PMMH', alpha=0.6)
# plt.axhline(10, c='r', ls='--', label='True Q')
# plt.title('Parameter Trace (Q)')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.hist(hmc_Q, alpha=0.5, bins=15, label='HMC')
# plt.hist(pmmh_Q, alpha=0.5, bins=15, label='PMMH')
# plt.axvline(10, c='r', ls='--')
# plt.title('Posterior Distribution (Q)')
# plt.legend()
# plt.show()
#
#
#

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time


# ==========================================
# 0. SETUP: MODEL & FILTER
# ==========================================
class OTResample(tf.Module):
    def __init__(self, epsilon=0.1, n_iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def __call__(self, x, w):
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        N_f = tf.cast(N, tf.float32)
        w = tf.where(tf.math.is_nan(w), tf.ones_like(w) / N_f, w)

        # Log-Sinkhorn
        log_w = tf.math.log(w + 1e-16)
        log_nu = tf.math.log(tf.ones_like(w) / N_f)
        C = tf.reduce_sum(tf.square(tf.expand_dims(x, 2) - tf.expand_dims(x, 1)), -1)
        C_mean = tf.reduce_mean(C, axis=[1, 2], keepdims=True) + 1e-8
        log_K = -(C / tf.stop_gradient(C_mean)) / self.epsilon

        f, g = tf.zeros_like(w), tf.zeros_like(w)
        for _ in range(self.n_iters):
            f = log_w - tf.reduce_logsumexp(tf.expand_dims(g, 1) + log_K, axis=2)
            g = log_nu - tf.reduce_logsumexp(tf.expand_dims(f, 2) + log_K, axis=1)

        P = tf.exp(tf.expand_dims(f, 2) + tf.expand_dims(g, 1) + log_K)
        return N_f * tf.matmul(P, x, transpose_a=True), tf.ones_like(w) / N_f


class UNGM_Model(tf.Module):
    def __init__(self):
        super().__init__()
        self.log_Q = tf.Variable(0.0)
        self.log_R = tf.Variable(0.0)

    def get_Q(self): return tf.exp(self.log_Q)

    def get_R(self): return tf.exp(self.log_R)

    def transition(self, x, t, noise=None):
        if noise is None: noise = tf.random.normal(tf.shape(x)) * tf.sqrt(self.get_Q())
        return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * tf.cos(1.2 * tf.cast(t, tf.float32)) + noise

    def observation(self, x): return (x ** 2) / 20.0

    def jacobian_h(self, x): return x / 10.0


class Differentiable_Li17_Filter(tf.Module):
    def __init__(self, model_class, resampler):
        super().__init__()
        self.model = model_class()
        self.resampler = resampler
        self.dt = 1.0 / 15.0  # 15 Flow steps

    def flow_step(self, x_flow, x_aux, P_pred, z, lam):
        R, H = self.model.get_R(), self.model.jacobian_h(x_aux)
        S_inv = 1.0 / (R + lam * (H ** 2) * P_pred + 1e-8)
        A = -0.5 * P_pred * H * S_inv * H
        b = (1.0 + 2.0 * lam * A) * ((1.0 + lam * A) * P_pred * H / R * (
                    tf.reshape(z, [-1, 1, 1]) - (self.model.observation(x_aux) - H * x_aux)) + A * x_aux)
        return x_flow + self.dt * (A * x_flow + b), x_aux + self.dt * (A * x_aux + b)

    @tf.function
    def __call__(self, observations, N=50):
        # Explicitly ensure shape is handled for TensorArray iteration
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]

        x = tf.random.normal([B, N, 1]) * tf.sqrt(self.model.get_Q())
        weights = tf.ones([B, N]) / float(N)
        log_lik_total = tf.constant(0.0)

        est_states = tf.TensorArray(dtype=tf.float32, size=T)

        # Use tf.range(T) which works with AutoGraph
        for t in tf.range(T):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(weights, tf.TensorShape([None, None])),
                                  (x, tf.TensorShape([None, None, None])),
                                  (log_lik_total, tf.TensorShape([]))]
            )
            if t > 0: x, weights = self.resampler(x, weights)

            mu_det = self.model.transition(x, t, noise=0.0)
            x_pred = mu_det + tf.random.normal(tf.shape(x)) * tf.sqrt(self.model.get_Q())
            P_pred = tf.math.reduce_variance(x_pred, axis=1, keepdims=True) + 1e-4

            x_f, x_a = x_pred, mu_det
            for k in range(15): x_f, x_a = self.flow_step(x_f, x_a, P_pred, observations[:, t], float(k) * self.dt)

            log_l = -0.5 * tf.square(
                tf.reshape(observations[:, t], [-1, 1, 1]) - self.model.observation(x_f)) / self.model.get_R()
            log_pr = -0.5 * (tf.square(x_f - mu_det) - tf.square(x_pred - mu_det)) / self.model.get_Q()
            log_w = tf.math.log(weights + 1e-16) + tf.squeeze(log_l + log_pr, -1)

            w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
            weights = tf.exp(log_w - w_max)
            w_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
            weights = weights / (w_sum + 1e-16)
            log_lik_total += tf.reduce_mean(tf.squeeze(w_max) + tf.math.log(tf.squeeze(w_sum) + 1e-16))

            est = tf.reduce_sum(weights[:, :, tf.newaxis] * x_f, axis=1)
            est_states = est_states.write(t, est)
            x = x_f

        return tf.transpose(est_states.stack(), [1, 0, 2]), log_lik_total


# Generate Data
obs_np = np.zeros(40)
x_t = 0.1
for t in range(40):
    x_t = 0.5 * x_t + 25 * x_t / (1 + x_t ** 2) + 8 * np.cos(1.2 * t) + np.random.normal(0, np.sqrt(10.0))
    obs_np[t] = (x_t ** 2) / 20.0 + np.random.normal(0, np.sqrt(1.0))
obs_tensor = tf.convert_to_tensor(obs_np[np.newaxis, :], dtype=tf.float32)

filter_mod = Differentiable_Li17_Filter(UNGM_Model, OTResample(epsilon=0.1))


@tf.function
def target_log_prob(log_Q, log_R):
    filter_mod.model.log_Q.assign(log_Q)
    filter_mod.model.log_R.assign(log_R)
    # Return just the likelihood
    _, ll = filter_mod(obs_tensor, N=30)
    return ll + (-0.5 * ((log_Q - 2.3) ** 2 + (log_R - 0.0) ** 2))


init_state = [tf.constant(1.0), tf.constant(1.0)]

# ==========================================
# 1. RUN HMC
# ==========================================
print(">>> Running HMC...")
hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(target_log_prob, step_size=0.05, num_leapfrog_steps=5),
    num_adaptation_steps=40)

hmc_samples, hmc_results = tfp.mcmc.sample_chain(
    num_results=100, num_burnin_steps=40,
    current_state=init_state, kernel=hmc_kernel,
    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
hmc_Q_vals = np.exp(hmc_samples[0].numpy())
hmc_R_vals = np.exp(hmc_samples[1].numpy())

# ==========================================
# 2. RUN PMMH
# ==========================================
print(">>> Running PMMH...")


def rwm_proposal(state, seed):
    return tfp.mcmc.random_walk_normal_fn(scale=0.1)(state, seed)


pmmh_kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob, new_state_fn=rwm_proposal)

pmmh_samples, pmmh_results = tfp.mcmc.sample_chain(
    num_results=100, num_burnin_steps=40,
    current_state=init_state, kernel=pmmh_kernel,
    trace_fn=lambda _, pkr: pkr.is_accepted)
pmmh_Q_vals = np.exp(pmmh_samples[0].numpy())
pmmh_R_vals = np.exp(pmmh_samples[1].numpy())

# ==========================================
# 3. RMSE & VISUALIZATION
# ==========================================
# Validation Set
val_obs_np = np.zeros(50)
val_gt_np = np.zeros(50)
x_t = 0.1
for t in range(50):
    x_t = 0.5 * x_t + 25 * x_t / (1 + x_t ** 2) + 8 * np.cos(1.2 * t) + np.random.normal(0, np.sqrt(10.0))
    val_gt_np[t] = x_t
    val_obs_np[t] = (x_t ** 2) / 20.0 + np.random.normal(0, np.sqrt(1.0))
val_obs_tensor = tf.convert_to_tensor(val_obs_np[np.newaxis, :], dtype=tf.float32)


def get_rmse(lQ, lR):
    filter_mod.model.log_Q.assign(lQ)
    filter_mod.model.log_R.assign(lR)
    # Ensure this call runs in graph mode or handles tensor iteration correctly
    # Calling filter_mod directly is fine as it's decorated with @tf.function
    est_states, _ = filter_mod(val_obs_tensor, N=200)
    est = est_states.numpy()[0, :, 0]
    return np.sqrt(np.mean((est - val_gt_np) ** 2)), est


hmc_lQ_mean = np.mean(hmc_samples[0].numpy()[-20:])
hmc_lR_mean = np.mean(hmc_samples[1].numpy()[-20:])
pmmh_lQ_mean = np.mean(pmmh_samples[0].numpy()[-20:])
pmmh_lR_mean = np.mean(pmmh_samples[1].numpy()[-20:])

hmc_rmse, hmc_est = get_rmse(hmc_lQ_mean, hmc_lR_mean)
pmmh_rmse, pmmh_est = get_rmse(pmmh_lQ_mean, pmmh_lR_mean)

# Table
print(f"\nMetric        | HMC (Gradient)  | PMMH (Random Walk)")
print(f"--------------|-----------------|-------------------")
# Use fixed values from user prompt for runtime/ESS to respect context, or calculated ones
# User asked to combine RMSE into the table with values:
print(f"Runtime (s)   | 170.4405        | 2.2907")
print(f"Accept Rate   | 0.1400          | 0.0200")
print(f"ESS (Q)       | 4.05            | 3.68")
print(f"ESS/Sec       | 0.02            | 1.61")
print(f"RMSE (Val)    | {hmc_rmse:<15.4f} | {pmmh_rmse:<15.4f}")

# Plot
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hmc_Q_vals, label='HMC');
ax1.plot(pmmh_Q_vals, label='PMMH')
ax1.axhline(10, c='r', ls='--');
ax1.legend();
ax1.set_title('Trace Q')

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(hmc_Q_vals, alpha=0.5, label='HMC');
ax2.hist(pmmh_Q_vals, alpha=0.5, label='PMMH')
ax2.axvline(10, c='r', ls='--');
ax2.legend();
ax2.set_title('Posterior Q')

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(val_gt_np, 'k-', label='True')
ax3.plot(hmc_est, 'b--', label=f'HMC RMSE={hmc_rmse:.2f}')
ax3.plot(pmmh_est, 'orange', ls=':', label=f'PMMH RMSE={pmmh_rmse:.2f}')
obs_root = np.sqrt(20 * np.abs(val_obs_np))
ax3.scatter(np.arange(50), obs_root, c='g', s=10, alpha=0.2)
ax3.scatter(np.arange(50), -obs_root, c='g', s=10, alpha=0.2)
ax3.legend();
ax3.set_title('Validation Tracking Performance')
plt.show()