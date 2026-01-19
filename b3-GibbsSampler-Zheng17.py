import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(101)
tf.random.set_seed(101)


# ==========================================
# 1. MODEL: STATE SPACE LSTM
# ==========================================
class StateSpaceLSTM(tf.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Initialization
        std = 1.0 / np.sqrt(hidden_dim)

        # LSTM Parameters
        # Input [z_{t-1}] (1) + Hidden [h_{t-1}] (H) -> Gates (4H)
        self.W_lstm = tf.Variable(tf.random.normal([1 + hidden_dim, 4 * hidden_dim], stddev=std), name='W_lstm')
        self.b_lstm = tf.Variable(tf.zeros([4 * hidden_dim]), name='b_lstm')

        # Output Projection: h_t -> mu_t
        self.W_out = tf.Variable(tf.random.normal([hidden_dim, 1], stddev=std), name='W_out')
        self.b_out = tf.Variable(tf.zeros([1]), name='b_out')

        # Learnable Noise (Initialize larger to prevent early collapse without warm start)
        self.log_sigma = tf.Variable(tf.math.log(2.0), name='log_sigma')

    def get_initial_state(self, batch_size):
        return (tf.zeros([batch_size, self.hidden_dim]), tf.zeros([batch_size, self.hidden_dim]))

    def __call__(self, z_prev, state):
        """
        Run LSTM Cell and Predict Next Mean
        """
        h_prev, c_prev = state

        # 1. Concatenate
        combined = tf.concat([z_prev, h_prev], axis=1)  # [B, 1+H]

        # 2. Gates
        gates = tf.matmul(combined, self.W_lstm) + self.b_lstm
        i, j, f, o = tf.split(gates, num_or_size_splits=4, axis=1)

        f = tf.sigmoid(f + 1.0)  # Forget bias
        i = tf.sigmoid(i)
        c_tilde = tf.tanh(j)
        o = tf.sigmoid(o)

        # 3. Update
        c_new = f * c_prev + i * c_tilde
        h_new = o * tf.tanh(c_new)

        # 4. Predict Mean (Residual connection)
        mu_update = tf.matmul(h_new, self.W_out) + self.b_out
        mu_t = z_prev + mu_update

        return mu_t, (h_new, c_new)


# ==========================================
# 2. INFERENCE: PARTICLE GIBBS w/ ANCESTOR SAMPLING (PGAS)
# ==========================================
class PGAS_Sampler:
    def __init__(self, model, num_particles=100, R=1.0):
        self.model = model
        self.N = num_particles
        self.R = R

    def run_pgas(self, observations, ref_traj=None):
        """
        Runs one iteration of PGAS.
        Returns: A new sampled trajectory (indices and values).
        """
        T = len(observations)

        # Storage
        particles_hist = []  # z_t for all particles
        indices_hist = []  # Parent indices
        weights_hist = []  # Normalized weights

        # To support Ancestor Sampling, we need to cache the 'predicted means'
        # of all particles from the previous step.
        prev_mus = None

        # Initialize
        curr_state = self.model.get_initial_state(self.N)
        z_curr = tf.random.normal([self.N, 1]) * 2.0  # Wide prior

        # If ref exists, set N-th particle
        if ref_traj is not None:
            ref_val = tf.reshape(ref_traj[0], [1, 1])
            z_curr = tf.concat([z_curr[:-1], ref_val], axis=0)

        weights = tf.ones(self.N) / self.N

        for t in range(T):
            obs = tf.cast(observations[t], tf.float32)

            # --- 1. RESAMPLING & ANCESTOR SAMPLING ---
            if t == 0:
                # Initial step: Standard resampling
                indices = tf.random.categorical(tf.math.log(weights[None, :]), self.N)[0]
                if ref_traj is not None:
                    # Force N-th particle to be N-th (trivial at t=0)
                    indices = tf.concat([indices[:-1], [self.N - 1]], axis=0)
            else:
                # A. Resample first N-1 particles normally
                indices_n_minus_1 = tf.random.categorical(tf.math.log(weights[None, :]), self.N - 1)[0]

                # B. ANCESTOR SAMPLING for the N-th particle (Reference)
                if ref_traj is not None:
                    # We need P(z_t* | z_{t-1}^{(i)})
                    # This is N(z_t* | mu_{t-1}^{(i)}, sigma)
                    # We use cached 'prev_mus' which are [N, 1] means predicted at t-1

                    z_ref_t = ref_traj[t]
                    sigma = tf.exp(self.model.log_sigma)

                    # Log Prob of transition
                    dist = tf.square(z_ref_t - prev_mus)
                    log_trans = -0.5 * dist / (sigma ** 2)

                    # Combine with weights from t-1
                    # w_{AS} propto w_{t-1} * p(z_t | z_{t-1})
                    log_as_weights = tf.math.log(weights + 1e-16) + tf.squeeze(log_trans)

                    # Sample ONE ancestor index for the reference
                    ref_ancestor_idx = tf.random.categorical(log_as_weights[None, :], 1)[0]

                    # Combine indices
                    indices = tf.concat([indices_n_minus_1, ref_ancestor_idx], axis=0)
                else:
                    # Standard resampling if no reference (first iter)
                    indices = tf.random.categorical(tf.math.log(weights[None, :]), self.N)[0]

            # Re-order states based on indices
            z_resampled = tf.gather(z_curr, indices)
            h_res = tf.gather(curr_state[0], indices)
            c_res = tf.gather(curr_state[1], indices)

            # --- 2. PROPAGATION ---
            # Run LSTM on all particles
            mu_pred, next_state = self.model(z_resampled, (h_res, c_res))
            sigma = tf.exp(self.model.log_sigma)

            # Sample new particles z_t
            noise = tf.random.normal(tf.shape(mu_pred))
            z_next = mu_pred + noise * sigma

            # --- 3. REFERENCE FORCE ---
            if ref_traj is not None:
                ref_val = tf.reshape(ref_traj[t], [1, 1])
                # Overwrite N-th particle value
                z_next = tf.concat([z_next[:-1], ref_val], axis=0)

            # --- 4. WEIGHTING ---
            innov = obs - (z_next ** 2) / 20.0
            log_lik = -0.5 * tf.square(innov) / self.R
            log_w = tf.squeeze(log_lik)
            w = tf.exp(log_w - tf.reduce_max(log_w))
            w /= (tf.reduce_sum(w) + 1e-16)

            # Store
            particles_hist.append(z_next)
            indices_hist.append(indices)  # Store PARENTS
            weights_hist.append(w)

            # Update loop vars
            z_curr = z_next
            weights = w
            curr_state = next_state
            prev_mus = mu_pred  # Cache for next step's AS

        return particles_hist, indices_hist, weights_hist

    def extract_trajectory(self, particles_hist, indices_hist, weights_hist):
        """
        Samples one trajectory from the particle system to be the new reference.
        Standard PGAS ends by sampling ONE trajectory from the final weights.
        """
        T = len(particles_hist)
        traj = []

        # Sample final index
        idx = tf.random.categorical(tf.math.log(weights_hist[-1][None, :]), 1)[0, 0]

        # Trace back lineage
        for t in range(T - 1, -1, -1):
            z_val = particles_hist[t][idx]
            traj.append(z_val)

            # Look up parent index
            # indices_hist[t] contains the parents of particles at t
            # Note: indices_hist[0] is just 0..N, meaningful ancestry starts at t=1
            if t > 0:
                idx = indices_hist[t][idx]

        return tf.stack(traj[::-1])


# ==========================================
# 3. EXPERIMENT: UNGM
# ==========================================
def generate_data(T=100):
    x, y = np.zeros(T), np.zeros(T)
    x[0] = 0.1
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + 25 * x[t - 1] / (1 + x[t - 1] ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.random.normal(0,
                                                                                                                   np.sqrt(
                                                                                                                       10))
        y[t] = x[t] ** 2 / 20 + np.random.normal(0, 1)
    return x.astype(np.float32), y.astype(np.float32)


gt_z, obs_y = generate_data(60)

# Init
lstm = StateSpaceLSTM(hidden_dim=32)
pgas = PGAS_Sampler(lstm, num_particles=1500)  # Need many particles for cold start
opt = tf.optimizers.Adam(learning_rate=0.005)  # Slower LR for stability

# Tracking
loss_history = []
ref_traj = None

print("Running PGAS (Strict Algorithm, No Warm Start)...")

for k in range(500):  # More iterations needed without warm start
    # --- E-STEP: PGAS ---
    # 1. Run SMC with Ancestor Sampling
    parts, inds, weights = pgas.run_pgas(obs_y, ref_traj=ref_traj)

    # 2. Sample New Reference Trajectory
    new_traj = pgas.extract_trajectory(parts, inds, weights)
    ref_traj = new_traj

    # --- M-STEP: Stochastic Gradient Ascent ---
    # Maximize log p(z_{1:T}, y_{1:T} | theta)
    # Equivalent to minimizing NLL of transitions z_t | z_{t-1}
    z_train = tf.stop_gradient(new_traj)

    with tf.GradientTape() as tape:
        loss = 0.0
        state = lstm.get_initial_state(1)
        sigma = tf.exp(lstm.log_sigma)

        # Calculate NLL of the sampled trajectory
        for t in range(len(z_train) - 1):
            inp = tf.reshape(z_train[t], [1, 1])
            tar = tf.reshape(z_train[t + 1], [1, 1])

            mu, state = lstm(inp, state)

            nll = 0.5 * tf.square((tar - mu) / sigma) + lstm.log_sigma
            loss += nll

    grads = tape.gradient(loss, lstm.trainable_variables)
    # Clip gradients to prevent explosion during early chaotic phase
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    opt.apply_gradients(zip(grads, lstm.trainable_variables))

    loss_val = tf.squeeze(loss).numpy().item()
    loss_history.append(loss_val)

    if k % 10 == 0:
        print(f"Iter {k}: Loss={loss_val:.4f}, Sigma={sigma.numpy():.4f}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 10))

# Trajectory
plt.subplot(2, 1, 1)
plt.plot(gt_z, 'k-', linewidth=2, alpha=0.6, label='Ground Truth')
plt.plot(ref_traj.numpy(), 'r--', linewidth=2, label='PGAS Recovered')

# Obs
roots = np.sqrt(20 * np.abs(obs_y))
plt.scatter(range(60), roots, c='g', s=15, alpha=0.2)
plt.scatter(range(60), -roots, c='g', s=15, alpha=0.2, label='Obs Roots')

plt.title("Strict PGAS Result (No Warm Start)")
plt.legend()

# Loss
plt.subplot(2, 1, 2)
plt.plot(loss_history)
plt.title("M-Step Loss (Transition NLL)")
plt.xlabel("Iteration")
plt.show()