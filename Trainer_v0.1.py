from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------
# Utility math
# -------------------------
def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = (x - np.max(x)) / max(tau, 1e-8)
    e = np.exp(z)
    return e / np.sum(e)

def kl_diag_gaussians(mu: np.ndarray, logvar: np.ndarray,
                      mu0: np.ndarray, logvar0: np.ndarray) -> float:
    """
    KL( N(mu, σ^2) || N(mu0, σ0^2) ) for diagonal Gaussians.
    """
    # Ensure float64 for numerical stability in sums
    mu   = np.asarray(mu, dtype=np.float64)
    lv   = np.asarray(logvar, dtype=np.float64)
    mu0  = np.asarray(mu0, dtype=np.float64)
    lv0  = np.asarray(logvar0, dtype=np.float64)

    var  = np.exp(lv)
    var0 = np.exp(lv0)
    # 0.5 * sum( log(var0/var) + (var + (mu-mu0)^2)/var0 - 1 )
    term1 = (lv0 - lv)
    term2 = (var + (mu - mu0) ** 2) / (var0 + 1e-12)
    kl = 0.5 * float(np.sum(term1 + term2 - 1.0))
    return kl


# -------------------------------------------------------------
# Minimal, optional "decoder head" to make p_theta(S|Z,q) expressive
# -------------------------------------------------------------
class TinyDecoder:
    """
    Optional learnable mixer that combines genre_match and latent similarity.
    Keeps parameters as simple scalars so we can do MLE step without heavy deps.

    score = sigmoid(alpha) * genre_match + (1 - sigmoid(alpha)) * latent_sim
    tau   = softplus(raw_tau)  (temperature for softmax)
    """
    def __init__(self, alpha_init: float = 0.7, tau_init: float = 0.5, lr: float = 1e-2):
        # raw params (unconstrained)
        self.raw_alpha = float(np.log(alpha_init / (1 - alpha_init + 1e-8) + 1e-8))  # inverse-sigmoid
        self.raw_tau   = float(np.log(np.exp(tau_init) - 1.0 + 1e-8))                # inverse-softplus
        self.lr = float(lr)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _softplus(x: float) -> float:
        return math.log(1.0 + math.exp(x))

    def get_alpha_tau(self) -> Tuple[float, float]:
        alpha = self._sigmoid(self.raw_alpha)
        tau   = max(self._softplus(self.raw_tau), 1e-3)
        return float(alpha), float(tau)

    def score_mix(self, genre_match: float, latent_sim: float) -> float:
        a, _ = self.get_alpha_tau()
        return a * genre_match + (1.0 - a) * latent_sim

    def mle_step(self, probs: np.ndarray, gt_index: int,
                 d_scores_d_alpha: np.ndarray,
                 d_scores_d_tau: Optional[np.ndarray] = None):
        """
        One simple gradient step on log p_gt w.r.t. raw_alpha and raw_tau.
        We use:
            log p_gt = scores_gt/tau - logsumexp(scores/tau)
        so grads come via provided partials.

        For simplicity:
          - We only update alpha here using chain rule through sigmoid.
          - tau often better scheduled externally; but we include a tiny update.

        Args:
          probs: softmax(scores / tau) over candidates
          gt_index: index of ground-truth candidate
          d_scores_d_alpha: ∂scores/∂alpha for each candidate (shape [C])
          d_scores_d_tau: optional ∂log p_gt / ∂tau helper (scalar); if None, do a tiny heuristic step
        """
        # ∂ log p_gt / ∂ scores_j = (1{j=gt} - probs[j]) / tau
        alpha = self._sigmoid(self.raw_alpha)
        tau   = self._softplus(self.raw_tau)
        inv_tau = 1.0 / max(tau, 1e-8)

        grad_logp_wrt_scores = np.zeros_like(probs)
        grad_logp_wrt_scores[gt_index] = inv_tau
        grad_logp_wrt_scores -= probs * inv_tau

        # Chain: ∂ log p_gt / ∂ alpha = sum_j ∂ log p_gt / ∂ scores_j * ∂ scores_j / ∂ alpha
        dlogp_dalpha = float(np.sum(grad_logp_wrt_scores * d_scores_d_alpha))

        # Through sigmoid: ∂alpha/∂raw_alpha = alpha*(1-alpha)
        dlogp_draw_alpha = dlogp_dalpha * alpha * (1.0 - alpha)
        self.raw_alpha += self.lr * dlogp_draw_alpha

        # A tiny (optional) tau update; by default skip or do very small heuristic
        if d_scores_d_tau is not None:
            # If provided, treat as direct gradient signal (already aggregated)
            self.raw_tau += 0.1 * self.lr * float(d_scores_d_tau)  # very small step
        else:
            # Heuristic nudge: if probs[gt] is low, slightly reduce tau (sharpen)
            self.raw_tau += 0.01 * self.lr * (0.5 - float(probs[gt_index]))


# -------------------------------------------------------------
# Meta-CoT Trainer
# -------------------------------------------------------------
class MetaCoTTrainer:
    """
    Wraps LLMPolicy with:
      - E-RL^2 updates (verifier reward + KL-to-reference)
      - q-STaR updates (self-training reward log p(gt|Z,q) + KL-to-ref, plus optional decoder MLE step)

    Notes:
      * We DO NOT modify LLM. We only update per-user latent tokens (mu, logvar).
      * Optionally, we update a TinyDecoder to make the softmax over candidates expressive.
    """

    def __init__(self,
                 policy,                              # your existing LLMPolicy instance
                 beta: float = 0.02,                  # KL strength
                 clip: float = 0.5,                   # latent step clip (same as your rl_step)
                 ema_tau: float = 0.0,                # EMA for reference; 0.0 => frozen ref
                 use_decoder_head: bool = True,       # enable TinyDecoder (MLE part of Eq. 20)
                 dec_lr: float = 5e-3,
                 # NEW: online traces config
                 enable_traces: bool = False,
                 trace_every: int = 10,          # sample a trace every N trainer steps
                 prm_mix: float = 0.5):          # mix coef: reward = prm_mix*old + (1-prm_mix)*PRM

        self.pi = policy
        self.beta = float(beta)
        self.clip = float(clip)
        self.ema_tau = float(ema_tau)
        self.decoder = TinyDecoder(lr=dec_lr) if use_decoder_head else None

        # Reference snapshot of user latent distributions (mu, logvar) ONLY
        self.ref_profiles = self._snapshot_ref_profiles()

        self.history = {"qstar": [], "e_rl2": [], "traces": []}
        self.print_every = 0  # 0 = silent
        self._global_step = 0   # NEW: step counter for trace cadence

        self.enable_traces = bool(enable_traces)
        self.trace_every = max(1, int(trace_every))
        self.prm_mix = float(np.clip(prm_mix, 0.0, 1.0))
        self.beta_min = 1e-3
        self.beta_max = 0.5
        self.target_kl = 1.0  # aim per-pack KL around this


    # optional: enable/disable periodic printing
    def set_logging(self, print_every: int = 0):
        self.print_every = int(print_every)

    def _adapt_beta(self, kl_mean: float):
        if kl_mean > 1.5 * self.target_kl:
           self.beta = min(self.beta * 1.5, self.beta_max)
        elif kl_mean < 0.5 * self.target_kl:
           self.beta = max(self.beta * 0.9, self.beta_min)

    def set_traces(self, enable: bool = True, every: int = 10, prm_mix: float = 0.5):
        self.enable_traces = bool(enable)
        self.trace_every = max(1, int(every))
        self.prm_mix = float(np.clip(prm_mix, 0.0, 1.0))

    # Easy access to recent stats
    def recent_stats(self, mode: str = "qstar", last: int = 5):
        seq = self.history.get(mode, [])
        return seq[-last:]

    # Peek at a user's current traits and top prefs
    def user_snapshot(self, uid: int) -> Dict[str, Any]:
        prof = self.pi.user_profiles[uid]
        prefs = dict(sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5])
        return {
            "traits": dict(prof["traits"]),
            "top_prefs": prefs,
            "latent_norms": [float(np.linalg.norm(t.mu)) for t in prof["latents"]],
        }

    # Simple descriptor generator for personality updates
    def _desc_from_adv(self, adv_val: float) -> str:
        # Positive adv → curiosity/engagement; Negative → resistant/distracted
        if adv_val >= 0.25:
            return "curious new deep active cooperative confident"
        elif adv_val >= 0.0:
            return "curious balanced engage polite stable"
        elif adv_val >= -0.25:
            return "familiar hesitant indifferent distracted"
        else:
            return "resistant avoid uncooperative discouraged"

    def kl_to_ref_for_pack(self, uid: int, seq: list) -> float:
        """
        KL between a sampled latent pack 'seq' (list of np arrays, one per token)
        and the reference Gaussian over latents. Uses ref logvar as covariance.
        """
        self._ensure_user_in_ref(uid)
        total = 0.0
        for k, z in enumerate(seq):
           mu0   = self.ref_profiles[uid][k]["mu"]
           lv0   = self.ref_profiles[uid][k]["logvar"]
           var0  = np.exp(lv0)
           # KL(N(z, var0_current) || N(mu0, var0)) ≈ 0.5 * ||z - mu0||^2 / var0  (ignore var mismatch)
           total += 0.5 * float(np.sum((z - mu0) ** 2 / (var0 + 1e-12)))
        return total

    # -------------------------
    # Reference management
    # -------------------------
    def _snapshot_ref_profiles(self):
        ref = {}
        for uid, prof in self.pi.user_profiles.items():
            latents = []
            for t in prof["latents"]:
                latents.append({"mu": t.mu.copy(), "logvar": t.logvar.copy()})
            ref[uid] = latents
        return ref

    def _ensure_user_in_ref(self, uid: int):
        if uid not in self.ref_profiles:
            prof = self.pi.user_profiles[uid]
            self.ref_profiles[uid] = [{"mu": t.mu.copy(), "logvar": t.logvar.copy()} for t in prof["latents"]]

    # --------- PRM: score compact LLM trace JSON to [-1,1] ---------
    def _prm_reward(self, trace: Dict[str, Any]) -> float:
        """
        Very lightweight PRM scoring:
          + keywords in latent_thoughts (intent/novelty/mood)
          + brevity & presence of a single chosen movie_id in reasoning_steps
        Returns a score in [-1, 1].
        """

        """
        Robust PRM scoring in [-1,1]. Accepts both dict and list forms for keys.
        """
        try:
           if not isinstance(trace, dict):
              return 0.0

           lt = trace.get("latent_thoughts", [])
           if isinstance(lt, dict):   # <- tolerate dict form
              lt = [lt]
           elif lt is None:
              lt = []

           rs = trace.get("reasoning_steps", [])
           if isinstance(rs, dict):   # <- tolerate dict form
              rs = [rs]
           elif rs is None:
              rs = []

           score = 0.0

           # latent thoughts
           for obj in lt:
               if not isinstance(obj, dict):
                  continue
               name = str(obj.get("name", "")).lower()
               content = str(obj.get("content", "")).lower()

               if "intent" in name or name == "intent":
                  for w in ["clear", "specific", "goal", "match", "relevant"]:
                     if w in content: score += 0.15
               if "novelty" in name or name == "novelty":
                  if any(x in content for x in ["medium", "balanced"]): score += 0.05
                  if "low" in content: score -= 0.05
               if "mood" in name or name == "mood":
                  if any(w in content for w in ["thoughtful", "focused", "engaged", "empathetic"]):
                     score += 0.1

           # reasoning: prefer exactly one chosen id + brevity
           if len(rs) > 0 and isinstance(rs[0], dict):
              step0 = rs[0]
              if "item_id" in step0: score += 0.25
              reason = str(step0.get("reasoning", ""))
              if 0 < len(reason) <= 200: score += 0.15
              elif len(reason) > 500:    score -= 0.05

           return float(np.clip(score, -1.0, 1.0))

        except Exception:
           return 0.0

    def _maybe_trace_and_mix(self, uid: int, base_reward: float, mode: str) -> float:
        use_trace = self.enable_traces and (self._global_step % self.trace_every == 0)
        if not use_trace:
           # print("No user trace")  # too chatty; keep silent or gate behind a debug flag
           return base_reward

        try:
           trace = self.pi.generate_trace(uid)   # Azure call
           # Defensive coercion for dict/list forms
           if isinstance(trace, dict):
              lt = trace.get("latent_thoughts", [])
              if isinstance(lt, dict): trace["latent_thoughts"] = [lt]
              rs = trace.get("reasoning_steps", [])
              if isinstance(rs, dict): trace["reasoning_steps"] = [rs]

           prm  = self._prm_reward(trace)        # [-1,1]
           # Optional: smooth scale the base reward to [-1,1] before mixing
           base_scaled = float(np.tanh(base_reward))
           mixed_scaled = self.prm_mix * base_scaled + (1.0 - self.prm_mix) * prm
           # If you want to return on the original scale, map back; otherwise just return scaled
           # Here we stay scaled (advantages are relative; scaling is fine):
           mixed = mixed_scaled

           # compact log
           short = {
               "step": self._global_step,
               "mode": mode,
               "uid": int(uid),
               "base_reward": float(base_reward),
               "base_scaled": base_scaled,
               "prm": float(prm),
               "mixed": float(mixed),
               }
           try:
               short["latent_thoughts"] = (trace.get("latent_thoughts") or [])[:3]
               short["reasoning_head"]  = (trace.get("reasoning_steps") or [])[:1]
           except Exception:
               pass
           self.history.setdefault("traces", []).append(short)

          #  # Optional: lightweight print every few steps
          #  if self.print_every and (self._global_step % self.print_every == 0):
          #     print(f"[trace] step={self._global_step} uid={uid} base={base_reward:.4f} "
          #         f"prm={prm:+.3f} mixed={mixed:+.3f}")

           return float(mixed)

        except Exception as e:
           # Log the exception type so we know why it failed (rate limit, parsing, etc.)
           print("LLM failed in _maybe_trace_and_mix:", repr(e))
           return base_reward
    
    def _batch_prm_for_uids(
        self,
        uids: List[int],
        mode: str = "qstar",
        max_workers: int = 4,
    ) -> Dict[int, float]:
        """
        Parallel PRM scoring for a batch of user IDs.

        Returns:
            uid -> prm_score  in [-1, 1]

        Also logs a compact trace summary into self.history["traces"].
        """
        # make uids unique but keep order
        unique_uids = list(dict.fromkeys(int(u) for u in uids))
        results: Dict[int, float] = {}

        def worker(uid: int):
            trace = self.pi.generate_trace(uid)  # Azure call
            # normalize dict/list forms as in _maybe_trace_and_mix
            if isinstance(trace, dict):
                lt = trace.get("latent_thoughts", [])
                if isinstance(lt, dict):
                    trace["latent_thoughts"] = [lt]
                rs = trace.get("reasoning_steps", [])
                if isinstance(rs, dict):
                    trace["reasoning_steps"] = [rs]

            prm = self._prm_reward(trace)  # [-1, 1]

            short = {
                "step": self._global_step,
                "mode": mode,
                "uid": int(uid),
                "prm": float(prm),
            }
            try:
                short["latent_thoughts"] = (trace.get("latent_thoughts") or [])[:3]
                short["reasoning_head"] = (trace.get("reasoning_steps") or [])[:1]
            except Exception:
                pass

            return uid, prm, short

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, uid) for uid in unique_uids]
            for fut in as_completed(futures):
                try:
                    uid, prm, short = fut.result()
                    results[uid] = prm
                    self.history.setdefault("traces", []).append(short)
                except Exception as e:
                    print("LLM failed in _batch_prm_for_uids:", repr(e))

        return results



    def ema_update_ref(self):
        """
        EMA refresh of the reference latents (if ema_tau>0). Otherwise leave frozen.
        """
        if self.ema_tau <= 0.0:
            return
        tau = self.ema_tau

        for uid, prof in self.pi.user_profiles.items():
            self._ensure_user_in_ref(uid)
            for k, t in enumerate(prof["latents"]):
                mu_ref = self.ref_profiles[uid][k]["mu"]
                lv_ref = self.ref_profiles[uid][k]["logvar"]
                # EMA
                self.ref_profiles[uid][k]["mu"]     = (1.0 - tau) * mu_ref + tau * t.mu
                self.ref_profiles[uid][k]["logvar"] = (1.0 - tau) * lv_ref + tau * t.logvar

    # -------------------------
    # KL-to-reference per user
    # -------------------------
    def kl_to_ref_for_user(self, uid: int) -> float:
        """
        Sum KL over all K latent tokens for this user between current latents and reference latents.
        """
        self._ensure_user_in_ref(uid)
        prof = self.pi.user_profiles[uid]
        total_kl = 0.0
        for k, t in enumerate(prof["latents"]):
            mu, lv = t.mu, t.logvar
            mu0, lv0 = self.ref_profiles[uid][k]["mu"], self.ref_profiles[uid][k]["logvar"]
            total_kl += kl_diag_gaussians(mu, lv, mu0, lv0)
        return float(total_kl)

    # -------------------------
    # Candidate scoring + log-prob for q-STaR
    # -------------------------
    def _genre_match(self, user_top: set, item_genres: List[str]) -> float:
        gset = set(item_genres)
        denom = len(user_top | gset)
        if denom <= 0:
            return 0.0
        return float(len(user_top & gset)) / float(denom)

    def _latent_vector(self, uid: int) -> np.ndarray:
        return self.pi._latent_vector(uid)

    def _item_vec(self, genres: List[str]) -> np.ndarray:
        return self.pi._genre_vector(genres)
    
    def _get_item_genres(self, mid: int) -> List[str]:
        """
        Get item genres using dynamic column names from Data_Structure.
        Works for both item_id and movie_id style datasets.
        """

        data = self.pi.data

        row = data.movies[
            data.movies[data.item_id_col] == int(mid)
        ]

        if row.empty:
            return []
  
        r = row.iloc[0]

        if "genre_set" in r and isinstance(r["genre_set"], (set, list)):
            return sorted(list(r["genre_set"]))

        genre_text = str(r[data.item_genre_col])

        return [
            g.strip()
            for g in genre_text.replace(" | ", "|").split("|")
            if g.strip()
        ]

    def item_logprob_given_Z(self, uid: int, candidates: List[int], gt_item: int) -> Tuple[float, np.ndarray, np.ndarray, int]:
        """
        Build a proper p_theta(i | Z, q) by softmax over scores and return log p(gt), probs, raw scores, gt_index.
        If a TinyDecoder is present, use it to mix genre_match and latent_sim (learnable).
        """
        prof = self.pi.user_profiles[uid]
        user_top = set(sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5])

        R = self._latent_vector(uid)
        genre_terms = []
        sim_terms = []

        # Compute components
        for mid in candidates:
            genres = self._get_item_genres(mid)

            if not genres:
               genre_terms.append(0.0)
               sim_terms.append(-1e9)
               continue

            gm = self._genre_match(user_top, genres)
            I = self._item_vec(genres)
            sim = float(np.dot(R, I))

            genre_terms.append(gm)
            sim_terms.append(sim)

        genre_terms = np.array(genre_terms, dtype=float)
        sim_terms = np.array(sim_terms, dtype=float)

        if self.decoder is None:
            # Fall back to your original fixed mix
            raw_scores = 0.7 * np.array(genre_terms) + 0.3 * np.array(sim_terms)
            tau = 0.5
        else:
            a, tau = self.decoder.get_alpha_tau()
            raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)

        probs = softmax_np(raw_scores, tau=tau)

        # Find index of ground-truth item within candidates (if present)
        try:
            gt_index = candidates.index(gt_item)
        except ValueError:
            # If GT not in candidates, treat as a 'miss' (very low logprob)
            return -20.0, probs, raw_scores, -1

        log_p = float(np.log(probs[gt_index] + 1e-12))
        return log_p, probs, raw_scores, gt_index

    # -------------------------
    # E-RL^2 Step (Eq. 15)
    # -------------------------
    def rl_step_e_rl2(self,
                      uids: List[int],
                      K_samples: int = 4,
                      beta: Optional[float] = None):
        """
        For each uid:
          - sample K latent packs (first = means)
          - reward_i  = verifier_reward(uid, pack_i) - beta * KL(current || ref)
          - use batch baseline on first-sample rewards
          - move means toward best-adv sample; shape variance based on adv sign (as in your baseline)
        Verifier reward uses your policy._ppl_reward() on compact (prompt, target) proxy by default.
        """
        if beta is None:
           beta = self.beta

        groups = []
        first_rewards = []

        # 1) Sample & score
        for uid in uids:
           packs = self.pi._sample_latent_pack(uid, K_samples=K_samples)
           rewards = []
           original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]

           for seq in packs:
               for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                   t.mu = seq[k].astype(np.float32)

               prompt, target = self.pi._format_prompt_for_scoring(uid, seq)
               r_star = self.pi._ppl_reward(prompt, target)
              #  kl_pen = self.kl_to_ref_for_user(uid)
               kl_pen = self.kl_to_ref_for_pack(uid, seq)
               kl_pen = min(kl_pen, 5.0)  # trust-region cap
               rewards.append(r_star - beta * kl_pen)

           # restore
           for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
               t.mu = original_latents[k]

           groups.append((uid, packs, rewards))
           first_rewards.append(rewards[0])

        # 2) Batch baseline
        sbar = float(np.mean(first_rewards))
        denom = np.linalg.norm(np.array(first_rewards) - sbar) + 1e-8

        # 3) Updates + stats
        step_stats = {"mode": "e_rl2", "n_users": len(uids), "baseline": sbar}
        all_best = []
        all_adv = []
        all_kl = []

        for uid, packs, rewards in groups:
            advs = [(sk - sbar) / denom for sk in rewards]
            best_idx = int(np.argmax(advs))
            best_seq = packs[best_idx]
            prof = self.pi.user_profiles[uid]

            # record
            all_best.append(rewards[best_idx])
            all_adv.append(advs[best_idx])
            all_kl.append(self.kl_to_ref_for_user(uid))

            # finite-diff update
            for k, tok in enumerate(prof["latents"]):
                delta = best_seq[k] - tok.mu
                step = np.clip(advs[best_idx] * delta, -self.clip, self.clip)
                tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)

                if max(advs) < 0:
                   tok.logvar = np.clip(tok.logvar - 0.03, -6.0, -0.2)
                else:
                   tok.logvar = np.clip(tok.logvar + 0.02, -6.0, -0.2)

            # ---- Personality update (uses normalized reward in [-1,1]) ----
            # squash to [-1, 1] for stability
            rew_norm = float(np.tanh(rewards[best_idx]))
            desc = self._desc_from_adv(advs[best_idx])
            # z_idx is the best sample index; traits update will decay internally
            self.pi.update_user_profile(uid, z_idx=best_idx, reward=rew_norm, z_desc=desc)

        # log step stats
        step_stats.update({
            "reward_mean": float(np.mean(all_best)),
            "reward_std":  float(np.std(all_best)),
            "adv_mean":    float(np.mean(all_adv)),
            "kl_mean":     float(np.mean(all_kl)),
            })
        self._adapt_beta(step_stats["kl_mean"])
        self.history["e_rl2"].append(step_stats)

        if self.print_every and (len(self.history["e_rl2"]) % self.print_every == 0):
           i = len(self.history["e_rl2"])
           print(f"[E-RL^2 step {i}] baseline={sbar:.4f} reward_mean={step_stats['reward_mean']:.4f} "
              f"adv_mean={step_stats['adv_mean']:.4f} kl_mean={step_stats['kl_mean']:.4f}")
        self._global_step += 1


    def MCoTqstar_step(self,
                   batch_pairs: List[Tuple[int, List[int], int]],  # (uid, candidates, gt_item)
                   K_samples: int = 4,
                   beta: Optional[float] = None):
        """
        For each (uid, candidates, gt_item):
          A) RL on Z with reward = log p_theta(gt | Z, q) - beta * KL
             (Z sampled on-policy; use batch baseline; update latents)
          B) Optional PRM mixing: reward <- mix(tanh(base_reward), PRM(uid)).
          C) Optional decoder MLE step to improve scoring head.
        """

        if beta is None:
            beta = self.beta

        # decide whether this step should use traces at all
        use_trace = self.enable_traces and (self._global_step % self.trace_every == 0)

        print("A.1")
        # ---- A.1) Sample & score (BASE reward = log p(gt|Z,q) - beta*KL) ----
        groups = []  # (uid, candidates, gt_item, packs, base_rewards)
        for uid, candidates, gt_item in batch_pairs:
            packs = self.pi._sample_latent_pack(uid, K_samples=K_samples)
            base_rewards = []
            original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]

            for seq in packs:
                # Temporarily set latents to this seq
                for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                    t.mu = seq[k].astype(np.float32)

                logp, _, _, _ = self.item_logprob_given_Z(uid, candidates, gt_item)
                kl_pen = self.kl_to_ref_for_pack(uid, seq)
                kl_pen = min(kl_pen, 5.0)  # trust-region cap
                base = logp - beta * kl_pen
                base_rewards.append(base)

            # restore original latents
            for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                t.mu = original_latents[k]

            groups.append((uid, candidates, gt_item, packs, base_rewards))

        print("A.1.5")
        # ---- A.1.5) Optional PRM mixing in parallel (per user) ----
        # After this, we will have final 'rewards' for each seq.
        groups_with_rewards = []
        first_rewards = []

        if use_trace:
            # run PRM once per user in parallel
            uid_list = [uid for uid, _, _, _, _ in groups]
            uid_prm = self._batch_prm_for_uids(uid_list, mode="qstar")
        else:
            uid_prm = {}

        for uid, candidates, gt_item, packs, base_rewards in groups:
            if use_trace and (uid in uid_prm):
                prm_val = uid_prm[uid]  # in [-1,1]
                rewards = []
                for base in base_rewards:
                    base_scaled = float(np.tanh(base))
                    mixed_scaled = self.prm_mix * base_scaled + (1.0 - self.prm_mix) * prm_val
                    rewards.append(float(mixed_scaled))
            else:
                # either traces disabled or PRM failed → fall back to base
                rewards = [float(b) for b in base_rewards]

            groups_with_rewards.append((uid, candidates, gt_item, packs, rewards))
            first_rewards.append(rewards[0])

        groups = groups_with_rewards

        print("A.2")
        # ---- A.2) Baseline on FINAL rewards (as in original code) ----
        sbar = float(np.mean(first_rewards))
        denom = np.linalg.norm(np.array(first_rewards) - sbar) + 1e-8

        print("A.3")
        # ---- A.3) Update latents + collect stats ----
        step_stats = {"mode": "qstar", "n_users": len(batch_pairs), "baseline": sbar}
        all_best = []
        all_adv = []
        all_kl = []

        for uid, candidates, gt_item, packs, rewards in groups:
            advs = [(sk - sbar) / denom for sk in rewards]
            best_idx = int(np.argmax(advs))
            best_seq = packs[best_idx]
            prof = self.pi.user_profiles[uid]

            all_best.append(rewards[best_idx])
            all_adv.append(advs[best_idx])
            all_kl.append(self.kl_to_ref_for_user(uid))

            for k, tok in enumerate(prof["latents"]):
                delta = best_seq[k] - tok.mu
                step = np.clip(advs[best_idx] * delta, -self.clip, self.clip)
                tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)

                if max(advs) < 0:
                    tok.logvar = np.clip(tok.logvar - 0.03, -6.0, -0.2)
                else:
                    tok.logvar = np.clip(tok.logvar + 0.02, -6.0, -0.2)

            # ---- Personality update (q-STaR) ----
            rew_norm = float(np.tanh(rewards[best_idx]))   # ~[-1,1]
            adv_star = float(advs[best_idx])
            desc, pol_map = self.pi.gen_descriptor_via_llm(uid, adv_star)
            self.pi.update_user_profile_llm(uid, reward=rew_norm, descriptor=desc, judgments=pol_map)

        print("B")
        # ---- B) Optional decoder MLE (unchanged logic) ----
        if self.decoder is not None:
            for uid, candidates, gt_item, packs, rewards in groups:
                advs = [(sk - sbar) / denom for sk in rewards]
                best_idx = int(np.argmax(advs))
                best_seq = packs[best_idx]

                original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]
                for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                    t.mu = best_seq[k].astype(np.float32)

                prof = self.pi.user_profiles[uid]
                user_top = set(sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5])
                R = self.pi._latent_vector(uid)

                genre_terms, sim_terms = [], []
                for mid in candidates:
                    genres = self._get_item_genres(mid)
                    if not genres:
                        genre_terms.append(0.0)
                        sim_terms.append(0.0)
                        continue
                    
                    gm = self._genre_match(user_top, genres)
                    I = self._item_vec(genres)
                    sim = float(np.dot(R, I))
                    
                    genre_terms.append(gm)
                    sim_terms.append(sim)

                a, tau = self.decoder.get_alpha_tau()
                raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)
                probs = softmax_np(raw_scores, tau=tau)

                try:
                    gt_index = candidates.index(gt_item)
                except ValueError:
                    gt_index = -1
                if gt_index >= 0:
                    d_scores_d_alpha = np.array(genre_terms) - np.array(sim_terms)
                    self.decoder.mle_step(probs, gt_index, d_scores_d_alpha, d_scores_d_tau=None)

                for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                    t.mu = original_latents[k]

        print("B.1")
        # ---- Log stats ----
        step_stats.update({
            "reward_mean": float(np.mean(all_best)),
            "reward_std":  float(np.std(all_best)),
            "adv_mean":    float(np.mean(all_adv)),
            "kl_mean":     float(np.mean(all_kl)),
        })
        self._adapt_beta(step_stats["kl_mean"])
        self.history["qstar"].append(step_stats)

        if self.print_every and (len(self.history["qstar"]) % self.print_every == 0):
            i = len(self.history["qstar"])
            print(
                f"[q-STaR step {i}] baseline={sbar:.4f} "
                f"reward_mean={step_stats['reward_mean']:.4f} "
                f"adv_mean={step_stats['adv_mean']:.4f} kl_mean={step_stats['kl_mean']:.4f}",
                flush=True,
            )
        self._global_step += 1

    # -------------------------------------------------------------
    # Convenience high-level runners
    # -------------------------------------------------------------
    def train_e_rl2(self,
                    batches_of_uids: List[List[int]],
                    K_samples: int = 4,
                    beta: Optional[float] = None,
                    ema_every: int = 0):
        """
        Train using E-RL^2 batches of user ids.
        """
        for it, uids in enumerate(batches_of_uids, 1):
            self.rl_step_e_rl2(uids, K_samples=K_samples, beta=beta)
            if ema_every and (it % ema_every == 0):
                self.ema_update_ref()

    def MCoTtrain_qstar(self,
                    batches_of_pairs: List[List[Tuple[int, List[int], int]]],
                    K_samples: int = 4,
                    beta: Optional[float] = None,
                    ema_every: int = 0):
        """
        Train using q-STaR batches of (uid, candidates, gt_item).
        """
        for it, pairs in enumerate(batches_of_pairs, 1):
            self.MCoTqstar_step(pairs, K_samples=K_samples, beta=beta)
            if ema_every and (it % ema_every == 0):
                self.ema_update_ref()

    # def MToTqstar_step(self,
    #                batch_pairs: List[Tuple[int, List[int], int]],  # (uid, candidates, gt_item)
    #                K_samples: int = 4,
    #                beta: Optional[float] = None):
    #     """
    #     Meta-ToT training step.

    #     For each (uid, candidates, gt_item):
    #       A) RL on Z with reward = log p_theta(gt | Z, ToT) - beta * KL
    #          - Z are latent packs sampled on-policy.
    #          - For each sampled Z, we:
    #              * temporarily set user latents to Z,
    #              * run a Tree-of-Thought search over 'candidates'
    #                via self.pi.generate_tot_tree(...),
    #              * score the best leaf vs gt_item via self.pi.score_tree_leaf(...).
    #       B) Batch baseline over first-sample rewards; compute advantages.
    #       C) Update user latent tokens (mu/logvar) using finite-difference step.
    #       D) Update personality via LLM-based descriptor (gen_descriptor_via_llm).
    #       E) Optional decoder MLE step (TinyDecoder) stays unchanged.
    #     """

    #     if beta is None:
    #         beta = self.beta

    #     print("A.1 (Meta-ToT)")
    #     # ---- A.1) Sample & score with ToT reward ----
    #     groups = []  # (uid, candidates, gt_item, packs, base_rewards)
    #     for uid, candidates, gt_item in batch_pairs:
    #         packs = self.pi._sample_latent_pack(uid, K_samples=K_samples)
    #         base_rewards = []
    #         original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]

    #         for seq in packs:
    #             # Temporarily set latents to this seq (Z)
    #             for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
    #                 t.mu = seq[k].astype(np.float32)

    #             # ---- Tree-of-Thought over the candidates for this Z ----
    #             tree = self.pi.generate_tot_tree(uid, candidates)
    #             logp_tree = self.pi.score_tree_leaf(uid, tree, gt_item)
    #             # logp_tree ~ 0 if chosen_item == gt_item, ~ -5 otherwise

    #             # KL regularization w.r.t. reference latents
    #             kl_pen = self.kl_to_ref_for_pack(uid, seq)
    #             kl_pen = min(kl_pen, 5.0)  # trust-region cap

    #             base = logp_tree - beta * kl_pen
    #             base_rewards.append(base)

    #         # restore original latents
    #         for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
    #             t.mu = original_latents[k]

    #         groups.append((uid, candidates, gt_item, packs, base_rewards))

    #     # Note: For Meta-ToT we do NOT apply PRM mixing here.
    #     # Rewards = base_rewards directly.
    #     print("A.2 (Meta-ToT)")
    #     groups_with_rewards = []
    #     first_rewards = []

    #     for uid, candidates, gt_item, packs, base_rewards in groups:
    #         rewards = [float(b) for b in base_rewards]
    #         groups_with_rewards.append((uid, candidates, gt_item, packs, rewards))
    #         first_rewards.append(rewards[0])

    #     groups = groups_with_rewards

    #     # ---- A.2) Baseline on FINAL rewards ----
    #     sbar = float(np.mean(first_rewards))
    #     denom = np.linalg.norm(np.array(first_rewards) - sbar) + 1e-8

    #     print("A.3 (Meta-ToT)")
    #     # ---- A.3) Update latents + collect stats ----
    #     step_stats = {"mode": "meta_tot", "n_users": len(batch_pairs), "baseline": sbar}
    #     all_best = []
    #     all_adv = []
    #     all_kl = []

    #     for uid, candidates, gt_item, packs, rewards in groups:
    #         advs = [(sk - sbar) / denom for sk in rewards]
    #         best_idx = int(np.argmax(advs))
    #         best_seq = packs[best_idx]
    #         prof = self.pi.user_profiles[uid]

    #         all_best.append(rewards[best_idx])
    #         all_adv.append(advs[best_idx])
    #         all_kl.append(self.kl_to_ref_for_user(uid))

    #         # finite-diff style update on latent means
    #         for k, tok in enumerate(prof["latents"]):
    #             delta = best_seq[k] - tok.mu
    #             step = np.clip(advs[best_idx] * delta, -self.clip, self.clip)
    #             tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)

    #             # variance shaping based on whether any adv is positive
    #             if max(advs) < 0:
    #                 tok.logvar = np.clip(tok.logvar - 0.03, -6.0, -0.2)
    #             else:
    #                 tok.logvar = np.clip(tok.logvar + 0.02, -6.0, -0.2)

    #         # ---- Personality update (Meta-ToT) ----
    #         rew_norm = float(np.tanh(rewards[best_idx]))   # ~[-1,1]
    #         adv_star = float(advs[best_idx])
    #         desc, pol_map = self.pi.gen_descriptor_via_llm(uid, adv_star)
    #         self.pi.update_user_profile_llm(uid, reward=rew_norm, descriptor=desc, judgments=pol_map)

    #     print("B (Meta-ToT)")
    #     # ---- B) Optional decoder MLE (unchanged logic) ----
    #     if self.decoder is not None:
    #         for uid, candidates, gt_item, packs, rewards in groups:
    #             advs = [(sk - sbar) / denom for sk in rewards]
    #             best_idx = int(np.argmax(advs))
    #             best_seq = packs[best_idx]

    #             original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]
    #             for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
    #                 t.mu = best_seq[k].astype(np.float32)

    #             prof = self.pi.user_profiles[uid]
    #             user_top = set(sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5])
    #             R = self.pi._latent_vector(uid)

    #             genre_terms, sim_terms = [], []
    #             for mid in candidates:
    #                 genres = self._get_item_genres(mid)
                    
    #                 if not genres:
    #                     genre_terms.append(0.0)
    #                     sim_terms.append(0.0)
    #                     continue
                    
    #                 gm = self._genre_match(user_top, genres)
    #                 I = self._item_vec(genres)
    #                 sim = float(np.dot(R, I))
                    
    #                 genre_terms.append(gm)
    #                 sim_terms.append(sim)

    #             a, tau = self.decoder.get_alpha_tau()
    #             raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)
    #             probs = softmax_np(raw_scores, tau=tau)

    #             try:
    #                 gt_index = candidates.index(gt_item)
    #             except ValueError:
    #                 gt_index = -1
    #             if gt_index >= 0:
    #                 d_scores_d_alpha = np.array(genre_terms) - np.array(sim_terms)
    #                 self.decoder.mle_step(probs, gt_index, d_scores_d_alpha, d_scores_d_tau=None)

    #             for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
    #                 t.mu = original_latents[k]

    #     print("B.1 (Meta-ToT)")
    #     # ---- Log stats ----
    #     step_stats.update({
    #         "reward_mean": float(np.mean(all_best)),
    #         "reward_std":  float(np.std(all_best)),
    #         "adv_mean":    float(np.mean(all_adv)),
    #         "kl_mean":     float(np.mean(all_kl)),
    #     })
    #     self._adapt_beta(step_stats["kl_mean"])
    #     self.history["qstar"].append(step_stats)  # keep same key for compatibility

    #     if self.print_every and (len(self.history["qstar"]) % self.print_every == 0):
    #         i = len(self.history["qstar"])
    #         print(
    #             f"[Meta-ToT step {i}] baseline={sbar:.4f} "
    #             f"reward_mean={step_stats['reward_mean']:.4f} "
    #             f"adv_mean={step_stats['adv_mean']:.4f} kl_mean={step_stats['kl_mean']:.4f}",
    #             flush=True,
    #         )
    #     self._global_step += 1

    # def MToTtrain_qstar(self,
    #                 batches_of_pairs: List[List[Tuple[int, List[int], int]]],
    #                 K_samples: int = 4,
    #                 beta: Optional[float] = None,
    #                 ema_every: int = 0):
    #     """
    #     Train using Meta-ToT batches of (uid, candidates, gt_item).

    #     Each inner list 'pairs' is a batch of triples:
    #        (uid, candidates, gt_item)
    #     and is passed to qstar_step, which now implements Meta-ToT
    #     reward = log p(gt | Z, ToT) - beta * KL.
    #     """
    #     for it, pairs in enumerate(batches_of_pairs, 1):
    #         self.MToTqstar_step(pairs, K_samples=K_samples, beta=beta)
    #         if ema_every and (it % ema_every == 0):
    #             self.ema_update_ref()

    def MToTqstar_step(
        self,
        batch_pairs: List[Tuple[int, List[int], int]],
        K_samples: int = 4,
        beta: Optional[float] = None,
        max_depth: int = 2,
        branch_factor: int = 3,
        adv_clip: float = 5.0,
        denom_floor: float = 1e-2,
        lambda_tot: float = 0.6,
        lambda_z: float = 0.4,
        debug: bool = False,
    ):
        """
        Meta-ToT training step.

        Training-time purpose:
          Meta-ToT is used as an offline reasoning supervisor. For each sampled
          latent state Z, the model performs why-question ToT reasoning over the
          candidate set. A verifier-aware tree reward is computed against the
          ground-truth item. This reward updates the reusable user state:
            1. latent profile Z_u
            2. personality / behavioural traits
            3. optional decoder scoring head

        Test-time usage:
          Meta-ToT is NOT used at test time. The learned Z/personality/decoder are
          reused directly by candidate_probs_given_Z() or policy.predict_item().
        """

        if beta is None:
            beta = self.beta

        if len(batch_pairs) == 0:
            return {
                "mode": "meta_tot",
                "n_users": 0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "adv_mean": 0.0,
                "kl_mean": 0.0,
                "exact_gt_rate": 0.0,
            }
    
        # -------------------------------------------------------
        # A. Sample latent packs and score them with Meta-ToT
        # -------------------------------------------------------
        groups = []
        first_rewards = []

        exact_hits = []
        retained_hits = []
        tree_reward_values = []
        latent_logp_values = []
        kl_values = []
        kl_scaled_values = []
        combined_reward_values = []
    
        for uid, candidates, gt_item in batch_pairs:
            uid = int(uid)
            candidates = [int(x) for x in candidates]
            gt_item = int(gt_item)
    
            packs = self.pi._sample_latent_pack(uid, K_samples=K_samples)
            prof = self.pi.user_profiles[uid]
    
            original_latents = [t.mu.copy() for t in prof["latents"]]
    
            sample_records = []
    
            for sample_idx, seq in enumerate(packs):
                # Temporarily set user latent state to sampled Z
                for k, t in enumerate(prof["latents"]):
                    t.mu = seq[k].astype(np.float32)
    
                # Training-only Meta-ToT reasoning
                tree = self.pi.generate_meta_tot_tree(
                    uid=uid,
                    candidates=candidates,
                    max_depth=max_depth,
                    branch_factor=branch_factor,
                )
        
                # Verifier-aware reward from tree
                tree_reward = float(self.pi.score_meta_tree_leaf(uid, tree, gt_item))

                # # Optional compact outcome summary from tree
                # if hasattr(self.pi, "summarize_meta_tot_outcome"):
                #     outcome = self.pi.summarize_meta_tot_outcome(tree, gt_item)
                # else:
                #     outcome = self._fallback_tot_outcome(tree, gt_item)

                # # KL regularisation against reference latent state
                # kl_pen = float(self.kl_to_ref_for_pack(uid, seq))
                # kl_pen_capped = np.log1p(kl_pen)

                # # Final training reward
                # reward = float(tree_reward - beta * kl_pen_capped)
                
                # Latent scorer reward: This directly optimises the same scorer used at test time.
                logp_z, probs_z, raw_scores_z, gt_index_z = self.item_logprob_given_Z(
                    uid=uid,
                    candidates=candidates,
                    gt_item=gt_item
                )

                logp_z = float(logp_z)


                # KL penalty: Use log1p(KL) instead of hard cap.
                # This still penalises far samples but avoids reward collapse.

                kl_pen = float(self.kl_to_ref_for_pack(uid, seq))
                kl_pen_scaled = float(np.log1p(max(kl_pen, 0.0)))


                # Combined Meta-ToT reward
                # Meta-ToT improves reasoning, logp_z aligns the learned Z with
                # test-time candidate scoring.

                reward = float(
                    lambda_tot * tree_reward
                    + lambda_z * logp_z
                    - beta * kl_pen_scaled
                )

                # Optional compact outcome summary from tree
                if hasattr(self.pi, "summarize_meta_tot_outcome"):
                    outcome = self.pi.summarize_meta_tot_outcome(tree, gt_item)
                else:
                    outcome = self._fallback_tot_outcome(tree, gt_item)

                # Add scorer information to the outcome for debugging/personality update
                outcome["latent_logp_gt"] = logp_z
                outcome["gt_index"] = int(gt_index_z)
                outcome["combined_reward"] = reward
                outcome["tree_reward"] = tree_reward
                outcome["kl_raw"] = kl_pen
                outcome["kl_scaled"] = kl_pen_scaled
    
                sample_records.append({
                    "sample_idx": sample_idx,
                    "seq": [z.copy() for z in seq],
                    "tree": tree,
                    "outcome": outcome,
                    "tree_reward": tree_reward,
                    "latent_logp": logp_z,
                    "kl": kl_pen,
                    "kl_scaled": kl_pen_scaled,
                    "reward": reward,
                })

                tree_reward_values.append(tree_reward)
                latent_logp_values.append(logp_z)
                kl_values.append(kl_pen)
                kl_scaled_values.append(kl_pen_scaled)
                combined_reward_values.append(reward)
                exact_hits.append(1.0 if outcome.get("correct", False) else 0.0)

                chosen = outcome.get("chosen_item", None)
                retained = False
                if chosen == gt_item:
                    retained = True
                else:
                    nodes = tree.get("nodes", {}) if isinstance(tree, dict) else {}
                    if isinstance(nodes, dict):
                        for _, node in nodes.items():
                            try:
                                cands = [int(x) for x in node.get("candidate_ids", [])]
                                if gt_item in cands:
                                    retained = True
                                    break
                            except Exception:
                                pass
                retained_hits.append(1.0 if retained else 0.0)
    
            # Restore original latent state before actual update
            for k, t in enumerate(prof["latents"]):
                t.mu = original_latents[k]
    
            rewards = [r["reward"] for r in sample_records]
            groups.append((uid, candidates, gt_item, sample_records, rewards))
            first_rewards.append(float(rewards[0]))
    
        # -------------------------------------------------------
        # B. Stable baseline and advantage
        # -------------------------------------------------------
        sbar = float(np.mean(first_rewards))
        reward_std = float(np.std(first_rewards))
        denom = max(reward_std, denom_floor)

        all_best_rewards = []
        all_best_advs = []
        all_user_kls = []

        # -------------------------------------------------------
        # C. Update latent tokens and personality
        # -------------------------------------------------------
        for uid, candidates, gt_item, sample_records, rewards in groups:
            prof = self.pi.user_profiles[uid]

            advs = [
                float(np.clip((r - sbar) / denom, -adv_clip, adv_clip))
                for r in rewards
            ]
    
            # best_idx = int(np.argmax(advs))
            # Select exploratory sample, not the current mean
            if len(advs) > 1:
                best_idx = 1 + int(np.argmax(advs[1:]))
            else:
                best_idx = 0
            best_record = sample_records[best_idx]
            best_seq = best_record["seq"]

            best_reward = float(rewards[best_idx])
            mean_reward = float(rewards[0])
            best_adv = float(advs[best_idx])
            
            improvement = best_reward - mean_reward

            all_best_rewards.append(best_reward)
            all_best_advs.append(best_adv)
            all_user_kls.append(float(self.kl_to_ref_for_user(uid)))

            # Move only if sampled latent improves over current mean
            if improvement > 0:
                for k, tok in enumerate(prof["latents"]):
                    delta = best_seq[k] - tok.mu
                    step = np.clip(improvement * delta, -self.clip, self.clip)
                    tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)
            
            # best_reward = float(rewards[best_idx])
    
            # # Latent update: move current Z toward best sampled Z
            # for k, tok in enumerate(prof["latents"]):
            #     delta = best_seq[k] - tok.mu
            #     if best_adv > 0:
            #         step = np.clip(best_adv * delta, -self.clip, self.clip)
            #         tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)

                # # Direct latent alignment with GT item
                # # # This helps the reusable test-time scorer learn.
                # gt_genres = self._get_item_genres(gt_item)

                # if gt_genres:
                #     gt_vec = self._item_vec(gt_genres)

                #     token_sims = [
                #         float(np.dot(tok.mu, gt_vec))
                #         for tok in prof["latents"]
                #     ]
                    
                #     best_token_idx = int(np.argmax(token_sims))
                #     tok = prof["latents"][best_token_idx]

                #     # Use positive strength when the best reward is better than baseline
                #     align_strength = float(np.clip(best_adv, 0.0, 2.0))

                #     align_step = align_strength * gt_vec
                #     align_step = np.clip(align_step, -self.clip, self.clip)
                #     tok.mu = (tok.mu + 0.2 * self.pi.lr * align_step).astype(np.float32)
    
                # if max(advs) < 0:
                #     tok.logvar = np.clip(tok.logvar - 0.03, -6.0, -0.2)
                # else:
                #     tok.logvar = np.clip(tok.logvar + 0.02, -6.0, -0.2)
                # tok.logvar = tok.logvar  # keep variance fixed for stability in Meta-ToT
                
            # Direct inference-aware GT alignment
            self.direct_latent_alignment_step(
                uid=uid,
                candidates=candidates,
                gt_item=gt_item,
                strength=5.0,
                # adv=max(best_adv, 0.25),
            )

            # Personality update should be grounded in ToT outcome
            rew_norm = float(np.tanh(best_reward))
            outcome = best_record["outcome"]

            if hasattr(self.pi, "gen_descriptor_from_tot_outcome"):
                desc, pol_map = self.pi.gen_descriptor_from_tot_outcome(
                    uid=uid,
                    gt_item=gt_item,
                    chosen_item=outcome.get("chosen_item"),
                    reward=rew_norm,
                    advantage=best_adv,
                    tree_summary=outcome,
                )
            else:
                # fallback to your existing method
                desc, pol_map = self.pi.gen_descriptor_via_llm(uid, best_adv)
    
            self.pi.update_user_profile_llm(
                uid=uid,
                reward=rew_norm,
                descriptor=desc,
                judgments=pol_map,
            )

            if debug:
                print("\n[Meta-ToT Debug]")
                print("uid:", uid)
                print("gt_item:", gt_item)
                print("rewards:", rewards)
                print("advantages:", advs)
                print("best_idx:", best_idx)
                print("best_reward:", best_reward)
                print("best_adv:", best_adv)
                print("best_outcome:", outcome)

        
        # -------------------------------------------------------
        # D. Optional decoder MLE update
        # -------------------------------------------------------
        if self.decoder is not None:
            for uid, candidates, gt_item, sample_records, rewards in groups:
                prof = self.pi.user_profiles[uid]

                advs = [
                    float(np.clip((r - sbar) / denom, -adv_clip, adv_clip))
                    for r in rewards
                ]
    
                best_idx = int(np.argmax(advs))
                best_seq = sample_records[best_idx]["seq"]

                original_latents = [t.mu.copy() for t in prof["latents"]]

                for k, t in enumerate(prof["latents"]):
                    t.mu = best_seq[k].astype(np.float32)
    
                user_top = set(
                    sorted(
                        prof["preferences"],
                        key=prof["preferences"].get,
                        reverse=True
                    )[:5]
                )
    
                R = self.pi._latent_vector(uid)
    
                genre_terms, sim_terms = [], []
    
                for mid in candidates:
                    genres = self._get_item_genres(mid)

                    if not genres:
                        genre_terms.append(0.0)
                        sim_terms.append(0.0)
                        continue
    
                    gm = self._genre_match(user_top, genres)
                    I = self._item_vec(genres)
                    sim = float(np.dot(R, I))
    
                    genre_terms.append(gm)
                    sim_terms.append(sim)

                genre_terms = np.array(genre_terms, dtype=float)
                sim_terms = np.array(sim_terms, dtype=float)
    
                a, tau = self.decoder.get_alpha_tau()
                raw_scores = a * genre_terms + (1.0 - a) * sim_terms
                probs = softmax_np(raw_scores, tau=tau)
    
                try:
                    gt_index = candidates.index(gt_item)
                except ValueError:
                    gt_index = -1

                if gt_index >= 0:
                    d_scores_d_alpha = genre_terms - sim_terms
                    self.decoder.mle_step(
                        probs=probs,
                        gt_index=gt_index,
                        d_scores_d_alpha=d_scores_d_alpha,
                        d_scores_d_tau=None,
                    )
    
                # Restore latent state
                for k, t in enumerate(prof["latents"]):
                    t.mu = original_latents[k]
    
        # -------------------------------------------------------
        # E. Log training statistics
        # -------------------------------------------------------
        step_stats = {
            "mode": "meta_tot",
            "n_pairs": int(len(batch_pairs)),
            "baseline": float(sbar),

            # Final combined reward used for latent update
            "reward_mean": float(np.mean(all_best_rewards)) if all_best_rewards else 0.0,
            "reward_std": float(np.std(all_best_rewards)) if all_best_rewards else 0.0,

            # Components of the reward
            "tree_reward_mean": float(np.mean(tree_reward_values)) if tree_reward_values else 0.0,
            "latent_logp_mean": float(np.mean(latent_logp_values)) if latent_logp_values else 0.0,
            "combined_reward_mean": float(np.mean(combined_reward_values)) if combined_reward_values else 0.0,

            # Advantage stability
            "adv_mean": float(np.mean(all_best_advs)) if all_best_advs else 0.0,
            "adv_std": float(np.std(all_best_advs)) if all_best_advs else 0.0,

            # KL tracking
            "kl_mean": float(np.mean(all_user_kls)) if all_user_kls else 0.0,
            "sample_kl_mean": float(np.mean(kl_values)) if kl_values else 0.0,
            "sample_kl_scaled_mean": float(np.mean(kl_scaled_values)) if kl_scaled_values else 0.0,

            # ToT quality diagnostics
            "exact_gt_rate": float(np.mean(exact_hits)) if exact_hits else 0.0,
            "gt_retained_rate": float(np.mean(retained_hits)) if retained_hits else 0.0,

            # Hyperparameters
            "denom": float(denom),
            "beta": float(beta),
            "lambda_tot": float(lambda_tot),
            "lambda_z": float(lambda_z),
        }

        self._adapt_beta(step_stats["kl_mean"])

        self.history.setdefault("meta_tot", []).append(step_stats)
        self.history.setdefault("qstar", []).append(step_stats)
    
        if self.print_every and (len(self.history["meta_tot"]) % self.print_every == 0):
            i = len(self.history["meta_tot"])
            print(
                f"[Meta-ToT step {i}] "
                f"reward={step_stats['reward_mean']:.4f} "
                f"tree={step_stats['tree_reward_mean']:.4f} "
                f"logpZ={step_stats['latent_logp_mean']:.4f} "
                f"exact_gt={step_stats['exact_gt_rate']:.3f} "
                f"adv={step_stats['adv_mean']:.4f} "
                f"kl={step_stats['kl_mean']:.4f} "
                f"kl_sample_log={step_stats['sample_kl_scaled_mean']:.4f}",
                flush=True,
            )
            if step_stats["kl_mean"] > 10.0:
                self.pi.lr *= 0.7
    
        self._global_step += 1
    
        return step_stats

    def MToTtrain_qstar(
        self,
        batches_of_pairs: List[List[Tuple[int, List[int], int]]],
        K_samples: int = 4,
        beta: Optional[float] = None,
        ema_every: int = 0,
        max_depth: int = 2,
        branch_factor: int = 3,
        lambda_tot: float = 0.6,
        lambda_z: float = 0.4,
        debug: bool = False,
    ):
        """
        Train using Meta-ToT batches of (uid, candidates, gt_item).
    
        Meta-ToT is used only during training. The learned Z/personality/decoder
        are later reused during test-time prediction without ToT reasoning.
        """

        for it, pairs in enumerate(batches_of_pairs, 1):
            self.MToTqstar_step(
                batch_pairs=pairs,
                K_samples=K_samples,
                beta=beta,
                max_depth=max_depth,
                branch_factor=branch_factor,
                lambda_tot=lambda_tot,
                lambda_z=lambda_z,
                debug=debug,
            )

            if ema_every and (it % ema_every == 0):
                self.ema_update_ref()
    
    def candidate_probs_given_Z(self, uid: int, candidates: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test-time / evaluation scoring.
    
        IMPORTANT:
          This function does NOT call Meta-ToT or the LLM.
          It reuses the learned Z/personality/profile/decoder to score candidates.
        """
            
        prof = self.pi.user_profiles[uid]
        user_top = set(sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5])
        R = self.pi._latent_vector(uid)
    
        genre_terms, sim_terms = [], []
        for mid in candidates:
            genres = self._get_item_genres(mid)
                
            if not genres:
                genre_terms.append(0.0)
                sim_terms.append(-1e9)
                continue
            
            gm = self._genre_match(user_top, genres)
            I = self._item_vec(genres)
            sim = float(np.dot(R, I))
            
            genre_terms.append(gm)
            sim_terms.append(sim)
    
        genre_terms = np.array(genre_terms, dtype=float)
        sim_terms = np.array(sim_terms, dtype=float)

        if self.decoder is None:
            raw_scores = 0.7 * np.array(genre_terms) + 0.3 * np.array(sim_terms)
            tau = 0.5
        else:
            a, tau = self.decoder.get_alpha_tau()
            raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)
    
        probs = softmax_np(raw_scores, tau=tau)
        return probs, raw_scores

    def predict_page_meta_tot(self, uid: int, page_cands: List[int], K_branches: int = 4, agg: str = "mean") -> int:
        packs = self.pi._sample_latent_pack(uid, K_samples=K_branches)

        original = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]

        all_probs = []
        for seq in packs:
            for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                t.mu = seq[k].astype(np.float32)

            probs, _ = self.candidate_probs_given_Z(uid, page_cands)
            all_probs.append(np.asarray(probs))

        for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
            t.mu = original[k]

        P = np.vstack(all_probs)
        p_agg = P.mean(axis=0) if agg == "mean" else P.max(axis=0)

        return page_cands[int(np.argmax(p_agg))]
    
    def evaluate_latent_prediction(
        self,
        eval_pairs: List[Tuple[int, List[int], int]]
    ) -> Dict[str, float]:
        """
        Evaluate whether the learned latent parameters can identify the ground-truth item.
        Each pair is: (uid, candidates, gt_item).
        """

        hits = []
        ranks = []
        logps = []

        for uid, candidates, gt_item in eval_pairs:
            logp, probs, raw_scores, gt_index = self.item_logprob_given_Z(
                uid=uid,
                candidates=candidates,
                gt_item=gt_item
            )

            if gt_index < 0:
                continue

            pred_idx = int(np.argmax(probs))
            pred_item = candidates[pred_idx]

            hit = int(pred_item == gt_item)
            hits.append(hit)

            # Rank of GT item, 1 is best
            sorted_idx = np.argsort(-probs)
            rank = int(np.where(sorted_idx == gt_index)[0][0]) + 1
            ranks.append(rank)

            logps.append(logp)

        if len(hits) == 0:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "avg_rank": 0.0,
                "avg_logp": -20.0,
            }

        return {
            "hit_rate": float(np.mean(hits)),
            "mrr": float(np.mean([1.0 / r for r in ranks])),
            "avg_rank": float(np.mean(ranks)),
            "avg_logp": float(np.mean(logps)),
        }
    
    def latent_drift_summary(self, uids: List[int]) -> Dict[str, float]:
        """
        Measure how far current latent tokens moved from reference latents.
        """

        drifts = []
        norms = []
        kls = []

        for uid in uids:
            self._ensure_user_in_ref(uid)
    
            prof = self.pi.user_profiles[uid]

            for k, tok in enumerate(prof["latents"]):
                ref_mu = self.ref_profiles[uid][k]["mu"]

                drift = np.linalg.norm(tok.mu - ref_mu)
                norm = np.linalg.norm(tok.mu)

                drifts.append(float(drift))
                norms.append(float(norm))

            kls.append(self.kl_to_ref_for_user(uid))

        return {
            "mean_latent_drift": float(np.mean(drifts)),
            "mean_latent_norm": float(np.mean(norms)),
            "mean_kl": float(np.mean(kls)),
        }
    
    def direct_latent_alignment_step(
        self,
        uid: int,
        candidates: List[int],
        gt_item: int,
        strength: float = 1.0,
        # adv: float = 1.0,
    ):
        """
        Inference-aware contrastive latent update.

        Goal:
           Increase latent similarity to the GT item and decrease similarity
           to high-scoring negative candidates, because candidate_probs_given_Z()
           ranks items using latent similarity.
        """
        uid = int(uid)
        gt_item = int(gt_item)
        candidates = [int(x) for x in candidates]

        if gt_item not in candidates:
            return

        prof = self.pi.user_profiles[uid]

        gt_genres = self._get_item_genres(gt_item)

        if not gt_genres:
            return

        gt_vec = self._item_vec(gt_genres)

        # Current inference scores
        probs, raw_scores = self.candidate_probs_given_Z(uid, candidates)

        gt_idx = candidates.index(gt_item)

        # Use the highest-probability wrong item as hard negative
        neg_indices = [i for i, mid in enumerate(candidates) if mid != gt_item]
        if len(neg_indices) == 0:
            return
    
        # hard_neg_idx = max(neg_indices, key=lambda i: probs[i])
        # hard_neg_item = candidates[hard_neg_idx]
    
        # neg_genres = self._get_item_genres(hard_neg_item)
        # if not neg_genres:
        #     return
    
        # neg_vec = self._item_vec(neg_genres)

        # Use top-2 hard negatives, not only one
        hard_neg_indices = sorted(
            neg_indices,
            key=lambda i: probs[i],
            reverse=True
        )[:2]

        neg_vecs = []
        for idx in hard_neg_indices:
            neg_genres = self._get_item_genres(candidates[idx])
            if neg_genres:
                neg_vecs.append(self._item_vec(neg_genres))

        if not neg_vecs:
            return
        
        neg_vec = np.mean(neg_vecs, axis=0)
    
        # Gradient-like direction: pull toward GT, push away from hard negative
        direction = gt_vec - neg_vec

        # Normalize to avoid tiny genre-vector scale
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
    
        # Stronger update when GT probability is low
        gt_prob = float(probs[gt_idx])
        difficulty = 1.0 - gt_prob
    
        # # Use positive advantage if available, but do not block supervised alignment
        # adv_weight = float(np.clip(max(adv, 0.25), 0.25, 2.0))
        # Do not let update vanish
        # update_weight = max(float(adv), 0.25)
    
        # step = strength * self.pi.lr * adv_weight * difficulty * direction
        step = strength * self.pi.lr * difficulty * direction
        step = np.clip(step, -self.clip, self.clip)
    
        # Update all latent tokens lightly, not only one token
        # This changes the full user representation used by _latent_vector().
        for tok in prof["latents"]:
            tok.mu = (tok.mu + step).astype(np.float32)

    def train_latent_gt_alignment_epoch(
        self,
        train_pairs: List[Tuple[int, List[int], int]],
        repeats: int = 3,
        strength: float = 0.20,
    ):
        """
        Directly train user latent states using the same candidate ranking
        objective used at inference.

        This is the missing bridge between Meta-ToT training and test-time
        candidate_probs_given_Z().
        """

        for _ in range(repeats):
            for uid, candidates, gt_item in train_pairs:
                uid = int(uid)
                candidates = [int(x) for x in candidates]
                gt_item = int(gt_item)

                if gt_item not in candidates:
                    continue

                logp, probs, raw_scores, gt_idx = self.item_logprob_given_Z(
                    uid=uid,
                    candidates=candidates,
                    gt_item=gt_item,
                )

                if gt_idx < 0:
                    continue

                # Hardness-aware update: stronger when GT probability is low
                gt_prob = float(probs[gt_idx])
                adv = 1.0 - gt_prob

                self.direct_latent_alignment_step(
                    uid=uid,
                    candidates=candidates,
                    gt_item=gt_item,
                    strength=strength,
                    # adv=adv,
                )

    def train_decoder_supervised_epoch(
        self,
        train_pairs: List[Tuple[int, List[int], int]],
        repeats: int = 3,
    ):
        """
        Train TinyDecoder directly on train candidate sets.
        This improves test-time candidate_probs_given_Z().
        """

        if self.decoder is None:
            return

        for _ in range(repeats):
            for uid, candidates, gt_item in train_pairs:
                uid = int(uid)
                candidates = [int(x) for x in candidates]
                gt_item = int(gt_item)

                prof = self.pi.user_profiles[uid]

                user_top = set(
                    sorted(
                        prof["preferences"],
                        key=prof["preferences"].get,
                        reverse=True
                    )[:5]
                )

                R = self.pi._latent_vector(uid)

                genre_terms = []
                sim_terms = []

                for mid in candidates:
                    genres = self._get_item_genres(mid)

                    if not genres:
                        genre_terms.append(0.0)
                        sim_terms.append(0.0)
                        continue

                    gm = self._genre_match(user_top, genres)
                    I = self._item_vec(genres)
                    sim = float(np.dot(R, I))

                    genre_terms.append(gm)
                    sim_terms.append(sim)

                genre_terms = np.array(genre_terms, dtype=float)
                sim_terms = np.array(sim_terms, dtype=float)

                a, tau = self.decoder.get_alpha_tau()
                raw_scores = a * genre_terms + (1.0 - a) * sim_terms
                probs = softmax_np(raw_scores, tau=tau)

                try:
                    gt_index = candidates.index(gt_item)
                except ValueError:
                    continue

                d_scores_d_alpha = genre_terms - sim_terms

                self.decoder.mle_step(
                    probs=probs,
                    gt_index=gt_index,
                    d_scores_d_alpha=d_scores_d_alpha,
                    d_scores_d_tau=None,
                )
