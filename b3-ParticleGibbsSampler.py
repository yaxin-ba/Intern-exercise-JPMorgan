# import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # ==========================================
# # 1. PURE TF LSTM TRANSITION MODEL
# # ==========================================
# class TransitionLSTM(tf.Module):
#     def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#
#         # Initialize Weights manually
#         # Concatenated weights for [Forget, Input, Candidate, Output] gates
#         # Shape: [input_dim + hidden_dim, 4 * hidden_dim]
#         w_init = tf.random.normal([input_dim + hidden_dim, 4 * hidden_dim], stddev=0.1)
#         self.W_lstm = tf.Variable(w_init, name='W_lstm')
#         self.b_lstm = tf.Variable(tf.zeros([4 * hidden_dim]), name='b_lstm')
#
#         # Output Projection Layer (Hidden -> Residual Mean)
#         w_out_init = tf.random.normal([hidden_dim, output_dim], stddev=0.1)
#         self.W_out = tf.Variable(w_out_init, name='W_out')
#         self.b_out = tf.Variable(tf.zeros([output_dim]), name='b_out')
#
#         # Learnable Noise Parameter
#         self.log_sigma = tf.Variable(tf.math.log(1.0), name='log_sigma')
#
#     def get_initial_state(self, batch_size):
#         # State tuple: (h, c)
#         h = tf.zeros([batch_size, self.hidden_dim])
#         c = tf.zeros([batch_size, self.hidden_dim])
#         return (h, c)
#
#     def __call__(self, x_t, state):
#         """
#         Runs one time step of the LSTM equations.
#         x_t: [Batch, 1]
#         state: (h_prev, c_prev)
#         """
#         h_prev, c_prev = state
#
#         # 1. Concatenate input and previous hidden state
#         # x_t: [B, 1], h_prev: [B, H] -> combined: [B, 1+H]
#         combined = tf.concat([x_t, h_prev], axis=1)
#
#         # 2. Compute Gates (Vectorized Matrix Mul)
#         # gates: [B, 4*H]
#         gates = tf.matmul(combined, self.W_lstm) + self.b_lstm
#
#         # Split into 4 chunks: Forget, Input, Candidate, Output
#         f, i, c_tilde, o = tf.split(gates, num_or_size_splits=4, axis=1)
#
#         # Apply Activations
#         f = tf.sigmoid(f)
#         i = tf.sigmoid(i)
#         c_tilde = tf.tanh(c_tilde)
#         o = tf.sigmoid(o)
#
#         # 3. Update Cell State
#         c_new = f * c_prev + i * c_tilde
#
#         # 4. Update Hidden State
#         h_new = o * tf.tanh(c_new)
#
#         # 5. Output Projection (Residual)
#         mu_residual = tf.matmul(h_new, self.W_out) + self.b_out
#         mu_pred = x_t + mu_residual  # Residual connection x_t = x_{t-1} + NN(x_{t-1})
#
#         return mu_pred, (h_new, c_new)
#
#
# # ==========================================
# # 2. PARTICLE GIBBS SAMPLER (PGAS)
# # ==========================================
# class ParticleGibbsSampler:
#     def __init__(self, model, num_particles=50, R_known=1.0):
#         self.model = model
#         self.N = num_particles
#         self.R = R_known
#
#     def run_csmc(self, observations, ref_traj=None):
#         """
#         Conditional SMC Step.
#         """
#         T = len(observations)
#
#         # Initialize containers
#         particles_hist = []  # List of [N, 1] tensors
#         weights_hist = []  # List of [N] tensors
#
#         # Initialize State
#         curr_state = self.model.get_initial_state(batch_size=self.N)
#
#         # Initial particles (Prior)
#         x_curr = tf.random.normal([self.N, 1])
#
#         # If Reference exists, clamp the last particle at t=0
#         if ref_traj is not None:
#             # Replace N-th particle with ref_traj[0]
#             ref_val = tf.reshape(ref_traj[0], [1, 1])
#             x_curr = tf.concat([x_curr[:-1], ref_val], axis=0)
#
#         weights = tf.ones(self.N) / float(self.N)
#
#         for t in range(T):
#             obs = tf.cast(observations[t], tf.float32)
#
#             # --- 1. Resampling ---
#             # Standard systematic/multinomial resampling
#             # Log-weights for stability
#             logits = tf.math.log(weights + 1e-16)
#             indices = tf.random.categorical(tf.reshape(logits, [1, -1]), self.N)[0]
#
#             # Gather particles and states
#             x_resampled = tf.gather(x_curr, indices)
#             h_res = tf.gather(curr_state[0], indices)
#             c_res = tf.gather(curr_state[1], indices)
#             state_resampled = (h_res, c_res)
#
#             # --- 2. Propagation (LSTM) ---
#             mu, next_state = self.model(x_resampled, state_resampled)
#             sigma = tf.exp(self.model.log_sigma)
#
#             # Sample Transition
#             noise = tf.random.normal(tf.shape(mu)) * sigma
#             x_next = mu + noise
#
#             # --- 3. Conditional Step (Clamp Reference) ---
#             if ref_traj is not None and t < T - 1:
#                 # We force the last particle to match the NEXT step of the reference
#                 ref_next = tf.reshape(ref_traj[t + 1], [1, 1])
#                 x_next = tf.concat([x_next[:-1], ref_next], axis=0)
#
#                 # Note: In a rigorous PGAS implementation, we would perform "Ancestral Sampling"
#                 # to link this forced particle to a specific parent index.
#                 # Here, for simplicity in demonstration, we assume the N-th particle tracks the N-th ancestry.
#
#             # --- 4. Weighting ---
#             # Likelihood: y ~ N(x^2/20, R)
#             innov = obs - (x_next ** 2) / 20.0
#             log_lik = -0.5 * tf.square(innov) / self.R
#
#             # Normalize Weights
#             log_w = tf.squeeze(log_lik)
#             max_log_w = tf.reduce_max(log_w)
#             w = tf.exp(log_w - max_log_w)
#             w = w / (tf.reduce_sum(w) + 1e-16)
#
#             # Store
#             particles_hist.append(x_next)
#             weights_hist.append(w)
#
#             x_curr = x_next
#             weights = w
#             curr_state = next_state
#
#         return particles_hist, weights_hist
#
#     def backward_simulation(self, particles_hist, weights_hist):
#         """
#         Backward Simulation to sample a smoothed trajectory.
#         """
#         T = len(particles_hist)
#         traj = []
#
#         # 1. Sample Final State
#         w_final = weights_hist[-1]
#         idx = tf.random.categorical(tf.math.log(tf.reshape(w_final, [1, -1])), 1)[0, 0]
#         x_next = particles_hist[-1][idx]
#         traj.append(x_next)
#
#         sigma = tf.exp(self.model.log_sigma)
#         sigma_sq = sigma ** 2
#
#         # 2. Backward Loop
#         for t in range(T - 2, -1, -1):
#             particles = particles_hist[t]
#             weights = weights_hist[t]
#
#             # Transition Probability: P(x_{t+1} | x_t)
#             # Approximated by Euclidean distance since f(x_t) is hard to invert
#             # and requires the LSTM state history.
#             # PGAS usually stores full state history or recomputes.
#             # Approx: x_{t+1} approx x_t (for calculating transition likelihood weight)
#
#             # More rigorous: re-run LSTM on all 'particles' to get their means
#             # For this demo, we use the simplified Gaussian kernel on distance
#             dist = tf.square(x_next - particles)
#             log_trans = -0.5 * dist / sigma_sq
#
#             # Backward Weight = Filter Weight * Transition Prob
#             log_bw = tf.math.log(weights + 1e-16) + tf.squeeze(log_trans)
#             idx = tf.random.categorical(tf.reshape(log_bw, [1, -1]), 1)[0, 0]
#
#             x_next = particles[idx]
#             traj.append(x_next)
#
#         return tf.stack(traj[::-1])
#
#
# # ==========================================
# # 3. DATA & TRAINING LOOP
# # ==========================================
# # Generate Ground Truth Data (UNGM)
# def generate_ungm_data(T=60):
#     x = np.zeros(T)
#     y = np.zeros(T)
#     x[0] = 0.1
#     for t in range(1, T):
#         x[t] = 0.5 * x[t - 1] + 25 * x[t - 1] / (1 + x[t - 1] ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.random.normal(0,
#                                                                                                                    np.sqrt(
#                                                                                                                        10.0))
#         y[t] = (x[t] ** 2) / 20.0 + np.random.normal(0, np.sqrt(1.0))
#     return x.astype(np.float32), y.astype(np.float32)
#
#
# gt_x, obs_y = generate_ungm_data(60)
#
# # Instantiate Components
# lstm_model = TransitionLSTM(hidden_dim=16)
# pg_sampler = ParticleGibbsSampler(lstm_model, num_particles=100)
# optimizer = tf.optimizers.Adam(learning_rate=0.02)
#
# current_ref_traj = None
# loss_history = []
#
# print("Running Particle Gibbs with Pure TF LSTM...")
#
# for k in range(500):
#     # --- E-STEP: Sample Trajectory (Inference) ---
#     parts, weights = pg_sampler.run_csmc(obs_y, ref_traj=current_ref_traj)
#     new_traj = pg_sampler.backward_simulation(parts, weights)
#     current_ref_traj = new_traj
#
#     # --- M-STEP: Update Parameters (Learning) ---
#     # Maximize log P(x_{1:T} | theta)
#     # Equivalent to minimizing NLL of the transitions in 'new_traj'
#
#     x_input = new_traj[:-1]
#     x_target = new_traj[1:]
#
#     with tf.GradientTape() as tape:
#         loss = 0.0
#
#         # We need to run the LSTM sequence manually inside the tape
#         # to track gradients through the pure TF variables
#         state = lstm_model.get_initial_state(batch_size=1)
#         sigma = tf.exp(lstm_model.log_sigma)
#
#         for t in range(len(x_input)):
#             inp = tf.reshape(x_input[t], [1, 1])
#             tar = tf.reshape(x_target[t], [1, 1])
#
#             # Call Pure TF Model
#             mu_pred, state = lstm_model(inp, state)
#
#             # Negative Log Likelihood
#             nll = 0.5 * tf.square((tar - mu_pred) / sigma) + tf.math.log(sigma)
#             loss += nll
#
#     # Compute and Apply Gradients
#     vars_to_train = lstm_model.trainable_variables
#     grads = tape.gradient(loss, vars_to_train)
#     optimizer.apply_gradients(zip(grads, vars_to_train))
#
#     loss_history.append(loss.numpy()[0, 0])
#
#     if k % 10 == 0:
#         print(f"Iter {k}: Loss={loss_history[-1]:.4f}")
#
# # ==========================================
# # 4. VISUALIZATION
# # ==========================================
# plt.figure(figsize=(12, 8))
#
# # Training Loss
# plt.subplot(2, 1, 1)
# plt.plot(loss_history)
# plt.title("M-Step Loss (Transition NLL)")
# plt.ylabel("Loss")
# plt.xlabel("EM Iteration")
# plt.grid(True, alpha=0.3)
#
# # Final Trajectory
# plt.subplot(2, 1, 2)
# plt.plot(gt_x, 'k-', linewidth=3, alpha=0.5, label='Ground Truth')
# plt.plot(current_ref_traj.numpy(), 'r--', linewidth=2, label='PG Sampled Trajectory')
#
# # Observations
# roots = np.sqrt(20 * np.abs(obs_y))
# plt.scatter(range(60), roots, c='g', s=15, alpha=0.4, label='Obs Roots')
# plt.scatter(range(60), -roots, c='g', s=15, alpha=0.4)
#
# plt.title("Joint Posterior Sampling: Learned LSTM Dynamics on UNGM")
# plt.legend()
# plt.tight_layout()
# plt.show()


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. PURE TF LSTM (Transition Model)
# ==========================================
class PureTFLSTM(tf.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Init weights (Xavier-like)
        std = 1.0 / np.sqrt(hidden_dim)
        # Input is dim 1. Concatenated input [x, h] is dim 1+hidden_dim
        w_shape = [1 + hidden_dim, 4 * hidden_dim]
        self.W_lstm = tf.Variable(tf.random.normal(w_shape, stddev=std), name='W_lstm')
        self.b_lstm = tf.Variable(tf.zeros([4 * hidden_dim]), name='b_lstm')

        # Output projection (Hidden -> Mean)
        self.W_out = tf.Variable(tf.random.normal([hidden_dim, 1], stddev=std), name='W_out')
        self.b_out = tf.Variable(tf.zeros([1]), name='b_out')

        # Learnable Noise (Initialize small to encourage trusting the LSTM)
        self.log_sigma = tf.Variable(tf.math.log(0.5), name='log_sigma')

    def get_initial_state(self, batch_size):
        return (tf.zeros([batch_size, self.hidden_dim]), tf.zeros([batch_size, self.hidden_dim]))

    def __call__(self, x, state):
        h, c = state
        # Concatenate Input [B, 1] and Hidden [B, H]
        combined = tf.concat([x, h], axis=1)

        # Gates
        gates = tf.matmul(combined, self.W_lstm) + self.b_lstm
        # Split: Input, New, Forget, Output
        i, j, f, o = tf.split(gates, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + 1.0)) + (tf.sigmoid(i) * tf.tanh(j))  # f+1 bias for stability
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        # Output (Residual Connection: x_t = x_{t-1} + LSTM(x_{t-1}))
        mu_update = tf.matmul(new_h, self.W_out) + self.b_out
        mu = x + mu_update

        return mu, (new_h, new_c)


# ==========================================
# 2. CORRECTED PARTICLE GIBBS SAMPLER
# ==========================================
class ParticleGibbsSampler:
    def __init__(self, model, num_particles=100):
        self.model = model
        self.N = num_particles
        self.R = 1.0  # Known measurement noise for simplicity

    def run_csmc(self, observations, ref_traj=None):
        T = len(observations)
        particles_hist = []
        weights_hist = []

        curr_state = self.model.get_initial_state(self.N)

        # Initialize Particles
        # If ref_traj exists, force N-th particle to be ref_traj[0]
        x_curr = tf.random.normal([self.N, 1])
        if ref_traj is not None:
            ref_val = tf.reshape(ref_traj[0], [1, 1])
            # Replace the last particle
            x_curr = tf.concat([x_curr[:-1], ref_val], axis=0)

        weights = tf.ones(self.N) / self.N

        for t in range(T):
            obs = tf.cast(observations[t], tf.float32)

            # --- 1. RESAMPLING WITH ANCESTRY CLAMPING ---
            logits = tf.math.log(weights + 1e-16)
            indices = tf.random.categorical(tf.reshape(logits, [1, -1]), self.N)[0]

            # CRITICAL FIX: If we have a reference, the N-th particle MUST survive.
            # We force the parent of the N-th particle to be the N-th particle from t-1.
            if ref_traj is not None and t > 0:
                # Create a tensor update for the last index
                indices = tf.concat([indices[:-1], [self.N - 1]], axis=0)

            x_resampled = tf.gather(x_curr, indices)
            h_res = tf.gather(curr_state[0], indices)
            c_res = tf.gather(curr_state[1], indices)

            # --- 2. PROPAGATION ---
            mu, next_state = self.model(x_resampled, (h_res, c_res))
            sigma = tf.exp(self.model.log_sigma)
            x_next = mu + tf.random.normal(tf.shape(mu)) * sigma

            # --- 3. CLAMP VALUE ---
            if ref_traj is not None and t < T - 1:
                ref_val_next = tf.reshape(ref_traj[t + 1], [1, 1])
                x_next = tf.concat([x_next[:-1], ref_val_next], axis=0)
                # Note: We don't overwrite LSTM state here, assuming the "forced" parentage
                # carried the correct hidden state forward.

            # --- 4. WEIGHTING ---
            # Likelihood y ~ N(x^2/20, R)
            innov = obs - (x_next ** 2) / 20.0
            log_lik = -0.5 * tf.square(innov) / self.R
            log_w = tf.squeeze(log_lik)
            w = tf.exp(log_w - tf.reduce_max(log_w))
            w /= (tf.reduce_sum(w) + 1e-16)

            particles_hist.append(x_next)
            weights_hist.append(w)

            x_curr = x_next
            weights = w
            curr_state = next_state

        return particles_hist, weights_hist

    def backward_simulation(self, particles_hist, weights_hist):
        """ Sample a smoothed trajectory backwards """
        T = len(particles_hist)
        traj = []

        # Sample final
        idx = tf.random.categorical(tf.math.log(tf.reshape(weights_hist[-1], [1, -1])), 1)[0, 0]
        x_next = particles_hist[-1][idx]
        traj.append(x_next)

        sigma = tf.exp(self.model.log_sigma)

        for t in range(T - 2, -1, -1):
            parts = particles_hist[t]
            w = weights_hist[t]

            # Transition PDF P(x_{t+1} | x_t) approx by distance
            dist = tf.square(x_next - parts)
            log_trans = -0.5 * dist / (sigma ** 2)

            log_bw = tf.math.log(w + 1e-16) + tf.squeeze(log_trans)
            idx = tf.random.categorical(tf.reshape(log_bw, [1, -1]), 1)[0, 0]

            x_next = parts[idx]
            traj.append(x_next)

        return tf.stack(traj[::-1])


# ==========================================
# 3. TRAINING LOOP
# ==========================================
# Generate Data
def gen_data(T=60):
    x, y = np.zeros(T), np.zeros(T)
    x[0] = 0.1
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + 25 * x[t - 1] / (1 + x[t - 1] ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.random.normal(0,
                                                                                                                   np.sqrt(
                                                                                                                       10))
        y[t] = x[t] ** 2 / 20 + np.random.normal(0, 1)
    return x.astype(np.float32), y.astype(np.float32)


gt_x, obs_y = gen_data(80)

lstm = PureTFLSTM(hidden_dim=32)
pg = ParticleGibbsSampler(lstm, num_particles=500)
opt = tf.optimizers.Adam(learning_rate=0.001)

# --- PHASE 1: WARM START (Optional but helps) ---
# Train LSTM to just predict sqrt(20*y) for a few steps to initialize weights
print("Warm-starting LSTM...")
rough_x = np.sqrt(20 * np.abs(obs_y))  # Rough proxy
rough_x = tf.reshape(tf.convert_to_tensor(rough_x), [-1, 1])

for i in range(20):
    with tf.GradientTape() as tape:
        loss = 0
        state = lstm.get_initial_state(1)
        for t in range(len(rough_x) - 1):
            inp = rough_x[t:t + 1]
            tar = rough_x[t + 1:t + 2]
            mu, state = lstm(inp, state)
            loss += tf.square(tar - mu)
    grads = tape.gradient(loss, lstm.trainable_variables)
    opt.apply_gradients(zip(grads, lstm.trainable_variables))

# --- PHASE 2: PARTICLE GIBBS ---
print("Starting PGAS...")
loss_hist = []
curr_traj = None

for k in range(800):
    # E-Step
    parts, weights = pg.run_csmc(obs_y, ref_traj=curr_traj)
    new_traj = pg.backward_simulation(parts, weights)
    curr_traj = new_traj  # Update reference

    # M-Step (Train on sampled trajectory)
    # We detach the trajectory from graph to treat as "Ground Truth" for this step
    x_train = tf.stop_gradient(new_traj)

    with tf.GradientTape() as tape:
        loss = 0.0
        state = lstm.get_initial_state(1)
        sigma = tf.exp(lstm.log_sigma)

        for t in range(len(x_train) - 1):
            inp = tf.reshape(x_train[t], [1, 1])
            tar = tf.reshape(x_train[t + 1], [1, 1])

            mu, state = lstm(inp, state)

            # Loss: NLL
            nll = 0.5 * tf.square((tar - mu) / sigma) + tf.math.log(sigma)
            loss += nll

    grads = tape.gradient(loss, lstm.trainable_variables)
    opt.apply_gradients(zip(grads, lstm.trainable_variables))
    loss_hist.append(loss.numpy()[0, 0])

    if k % 10 == 0:
        print(f"Iter {k}: Loss={loss_hist[-1]:.4f}")

# Plot
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(gt_x, 'k', label='True')
plt.plot(curr_traj.numpy(), 'r--', label='PGAS Sampled')
plt.title("Particle Gibbs: LSTM Learned Trajectory")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(loss_hist)
plt.title("M-Step Loss")
plt.show()