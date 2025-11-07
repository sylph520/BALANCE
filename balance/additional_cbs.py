import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback


# ─────────────────────────────────────────────
# Example helper for workload-frequency drift
# ─────────────────────────────────────────────
def workload_changed(prev_freqs, new_freqs, threshold=0.15):
    """
    Detect workload change using cosine distance.
    Return True if frequencies differ beyond threshold.
    """
    if prev_freqs is None:
        return False
    # normalize to unit length
    p = prev_freqs / (np.linalg.norm(prev_freqs) + 1e-8)
    n = new_freqs / (np.linalg.norm(new_freqs) + 1e-8)
    cosine_dist = 1.0 - np.dot(p, n)
    return cosine_dist > threshold


# ─────────────────────────────────────────────
# Adaptive PPO2 callback
# ─────────────────────────────────────────────
class AdaptivePhaseCallback(BaseCallback):
    def __init__(self, env, check_freq=10_000, adaptation_steps=50_000):
        super().__init__()
        self.env = env
        self.check_freq = check_freq
        self.adaptation_steps = adaptation_steps
        self.prev_freqs = None
        self.adapting = False
        self.remaining_steps = 0

        self.stable_lr = 5e-5
        self.stable_clip = 0.3
        self.stable_epochs = 30
        self.adapt_lr = 8e-5
        self.adapt_clip = 0.35
        self.adapt_epochs = 15

    def _get_base_env(self):
        """Return the first underlying environment."""
        if hasattr(self.env, "envs"):
            return self.env.envs[0]
        return self.env

    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            base_env = self._get_base_env()
            if hasattr(base_env, "get_query_frequencies"):
                new_freqs = np.array(base_env.get_query_frequencies())
            else:
                raise AttributeError(
                    "Base env must implement get_query_frequencies()."
                )

            if workload_changed(self.prev_freqs, new_freqs):
                print(f"[AdaptivePhase] Workload changed at {self.num_timesteps}")
                self._enter_adaptation_phase()
            self.prev_freqs = new_freqs

        if self.adapting:
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self._exit_adaptation_phase()
        return True


class PPODiagnosticsCallback(BaseCallback):
    """
    Logs and warns when PPO training shows unhealthy patterns:
    - critic collapse (explained variance < 0)
    - entropy too high/low (relative entropy out of [0.05, 0.2])
    - frozen updates (KL < 0.001 or clipfrac < 0.05)
    Adds text summaries to TensorBoard as well.
    """

    def __init__(self, action_space_size, verbose=1):
        super(PPODiagnosticsCallback, self).__init__(verbose)
        self.A = float(action_space_size)
        self.warning_log = []
        self.logger_ref = None  # Will be set during training

    def _init_callback(self):
        # Called once training starts — model.logger now exists
        self.logger_ref = getattr(self.model, "logger", None)

    def _record_warning(self, message):
        """Add warning to rolling log + TensorBoard."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.warning_log.append(full_msg)
        self.warning_log = self.warning_log[-15:]  # keep last 15 warnings

        # Print to console
        if self.verbose > 0:
            print(f"\033[93m⚠️ {full_msg}\033[0m")

        # Log to TensorBoard if available
        if self.logger_ref is not None:
            tb_text = "\n".join(self.warning_log)
            self.logger_ref.record("diagnostics/warnings_text", tb_text)

    def _on_step(self) -> bool:
        if self.logger_ref is None:
            return True  # Skip until logger is ready

        logs = getattr(self.logger_ref, "name_to_value", {}) or {}

        ev = logs.get("train/explained_variance")
        ent = logs.get("loss/entropy_loss")
        kl = logs.get("loss/approximate_kullback-leibler")
        clipfrac = logs.get("loss/clip_factor")

        # ---- Critic health ----
        if ev is not None and ev < 0:
            self._record_warning(f"Critic collapse detected (EV={ev:.3f})")

        # ---- Entropy checks ----
        if ent is not None:
            rel_ent = ent / math.log(self.A)
            self.logger_ref.record("diagnostics/relative_entropy", rel_ent)
            if rel_ent > 0.3:
                self._record_warning(f"Entropy too high (rel={rel_ent:.2f}) — reduce ent_coef")
            elif rel_ent < 0.03:
                self._record_warning(f"Entropy too low (rel={rel_ent:.2f}) — exploration dying")

        # ---- PPO update health ----
        if kl is not None and kl < 0.001:
            self._record_warning(f"PPO nearly frozen (KL={kl:.4f})")

        if clipfrac is not None and clipfrac < 0.05:
            self._record_warning(f"Updates too conservative (clipfrac={clipfrac:.3f})")

        return True

class VecNormDiagCallback(BaseCallback):
    """
    Print VecNormalize stats every few updates.
    """
    def __init__(self, check_freq=5000, verbose=0):
        super(VecNormDiagCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self):
        # 'self.training_env' is VecNormalize(env, ...)
        if self.num_timesteps % self.check_freq == 0:
            try:
                vec = self.training_env
                print(f"\n[Diag] step={self.num_timesteps}")
                print("Reward RMS mean:", vec.ret_rms.mean)
                print("Reward RMS var :", vec.ret_rms.var)
                print("Obs RMS mean:", vec.obs_rms.mean.mean())
                print("Obs RMS var :", vec.obs_rms.var.mean())
            except Exception as e:
                print("[Diag] Could not read VecNormalize stats:", e)
        return True

class EntropySchedulerCallback(BaseCallback):
    """
    CORRECTED: A custom callback that linearly decays the entropy coefficient (ent_coef)
    by directly accessing the model.ent_coef attribute, which is correct for PPO2 in SB2.
    """
    def __init__(self, start_value, end_value, total_timesteps, verbose=0):
        super(EntropySchedulerCallback, self).__init__(verbose)
        self.start_value = start_value
        self.end_value = end_value
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Calculate the fraction of total training completed
        progress = self.num_timesteps / self.total_timesteps
        
        # Linear decay formula
        new_ent_coef = self.start_value - (self.start_value - self.end_value) * progress
        
        # **CORRECTED ACTION:** Set the new entropy coefficient by modifying the attribute directly
        self.model.ent_coef = new_ent_coef

        return True # Continue training
