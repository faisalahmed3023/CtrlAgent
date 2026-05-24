import json, math, time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

def safe_json_parse(txt: str) -> Dict[str, Any]:
    try:
        s, e = txt.find("{"), txt.rfind("}")
        return {} if s == -1 or e == -1 else json.loads(txt[s:e+1])
    except Exception:
        return {}

# ======================================================
# Latent token container (per-user parameters)
# Each token k has mean mu_k and logvar lv_k (diagonal Gaussian)
# ======================================================
@dataclass
class LatentToken:
    mu: np.ndarray        # (d,)
    logvar: np.ndarray    # (d,)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        sigma = np.exp(0.5 * self.logvar)
        eps = rng.normal(size=self.mu.shape)
        return self.mu + sigma * eps

# ======================================================
# Encoder to map genres -> latent target direction (warm-up)
# ======================================================
class GenreEncoder:
    def __init__(self, latent_dim: int, seed: int = 42):
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)
        self.emb: Dict[str, np.ndarray] = {}

    def _get_vec(self, g: str) -> np.ndarray:
        if g not in self.emb:
            v = self.rng.normal(scale=0.3, size=(self.latent_dim,))
            v /= (np.linalg.norm(v) + 1e-8)
            self.emb[g] = v
        return self.emb[g]

    def encode_genre_mixture(self, genre_weights: Dict[str, float]) -> np.ndarray:
        if not genre_weights:
            return np.zeros(self.latent_dim, dtype=np.float32)
        v = np.zeros(self.latent_dim, dtype=np.float32)
        total = 0.0
        for g, w in genre_weights.items():
            v += float(w) * self._get_vec(g)
            total += float(w)
        if total > 0:
            v /= total
        return v

# ======================================================
# LLMPolicy with LatentR3-style latent reasoning
# ======================================================
class LLMPolicy:
    """
    Latent reasoning policy that:
      - Maintains K compact latent tokens per user (continuous vectors).
      - Supports warm-up (SFT-like) to initialize latents from genre prefs.
      - Runs RL steps with perplexity-based rewards and batch-baseline advantage.
      - Inference uses latent tokens to re-rank candidates; no verbose CoT.

    This updates *only* user-specific latent parameters (mu/logvar), leaving the LLM frozen.
    """

    def __init__(
        self,
        data,
        model: str = MODEL_NAME,
        temperature: float = 0.3,
        max_retries: int = 3,
        lr: float = 1e-2,
        latent_dim: int = 32,
        K: int = 5,                 # latent reasoning length
        seed: int = 123,
        sigma_noise: float = 0.25,  # sampling strength for RL stage
        use_logprobs: bool = True,  # try to request token logprobs for PPL reward
    ):
        self.data = data
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.gamma = 0.9
        self.lr = lr
        self.latent_dim = latent_dim
        self.K = K
        self.sigma_noise = sigma_noise
        self.use_logprobs = use_logprobs

        self.user_profiles = defaultdict(self._init_profile)
        self.trait_decay = 0.95
        self.rng = np.random.RandomState(seed)
        self.encoder = GenreEncoder(latent_dim, seed=seed)

    # --------------------------
    # Profile init (keeps OCEAN)
    # --------------------------
    def _init_profile(self) -> Dict[str, Any]:
        latents = [LatentToken(
            mu=self.rng.normal(scale=0.1, size=(self.latent_dim,)).astype(np.float32),
            logvar=np.full((self.latent_dim,), -2.0, dtype=np.float32)  # small variance
        ) for _ in range(self.K)]

        return {
            "preferences": defaultdict(float),
            "traits": {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
            "memory": [],
            "latents": latents,  # NEW: K latent tokens per user
        }

    # ---------------------------------------
    # Persona helpers (unchanged from before)
    # ---------------------------------------
    def gamma_prompt_tag(self, gamma: float) -> str:
        if gamma <= 0.6:
            return "Preference: concise but reason explicitly on relevant factors."
        elif gamma <= 0.8:
            return "Preference: balanced reasoning; combine multiple viewpoints."
        else:
            return "Preference: deliberate reasoning; explore alternative perspectives."


    def persona_description(self, user_row: Dict[str, Any]) -> str:
        """
        Build persona description using dynamic user-column names from Data_Structure.
        """

        age_col = self.data.user_age_col
        status_col = self.data.user_status_col

        age = int(user_row[age_col])
        status = str(user_row[status_col]).lower()

        if age < 25:
            age_trait = "youthful and adaptive"
        elif age < 40:
            age_trait = "balanced and analytical"
        elif age < 60:
            age_trait = "experienced and thoughtful"
        else:
            age_trait = "wise and reflective"

        discipline_traits = {
            "writer": ("creative imagination", "values narrative depth and expression"),
            "engineer": ("structured logic", "prefers systematic reasoning and clarity"),
            "doctor": ("empathy and ethics", "values moral depth and human complexity"),
            "farmer": ("patience and realism", "appreciates perseverance and nature themes"),
            "scientist": ("curiosity and rationality", "seeks cause-effect explanations"),
            "lawyer": ("analytical debate", "enjoys justice, argument, and critical reasoning"),
            "artist": ("aesthetic sensitivity", "values beauty, emotion, and visual style"),
        }

        cog_trait, pref_trait = discipline_traits.get(
            status,
            ("open-minded curiosity", "adapts across genres and ideas")
        )

        return (
            f"The user is a {age_trait} {status}. "
            f"They exhibit {cog_trait} and {pref_trait}."
        )

    # ----------------------------------------------------
    # Ground-truth profile seed from user positives (≥4)
    # ----------------------------------------------------
    def extract_ground_truth_profile(self, uid: int):
        """
        Seed user preference profile from positive rated items.
        Uses dynamic item genre column from Data_Structure.
        """
        df = self.data.get_user_rated_items(uid, min_rating=4)
        profile = self.user_profiles[uid]

        if df.empty:
            return profile
        
        genre_col = self.data.item_genre_col

        genres = (
        df[genre_col]
        .fillna("")
        .astype(str)
        .str.replace(" | ", "|", regex=False)
        .str.split("|")
        .explode()
        .str.strip()
        )
        
        genres = genres[genres != ""]
        genre_dist = genres.value_counts(normalize=True)
        
        for g, w in genre_dist.items():
            profile["preferences"][g] += float(w)
        return profile

    # ==========================================
    # A) WARM-UP (SFT-like) for latent tokens
    # ==========================================
    def sft_warmup_step(self, uid: int, alpha: float = 0.5):
        """
        Cheap warm-up toward genre centroids (proxy for Lwarm).
        Moves latent means toward an encoded mixture of user-positive genres.
        """
        profile = self.user_profiles[uid]
        prefs = dict(profile["preferences"])
        if not prefs:
            # seed from inferred prefs if not initialized
            prefs = self.data.infer_user_genre_prefs(uid)
        target = self.encoder.encode_genre_mixture(prefs)  # (d,)
        if np.allclose(target, 0):
            return
        for tok in profile["latents"]:
            tok.mu = (1 - alpha) * tok.mu + alpha * target
            # keep variance modest
            tok.logvar = np.clip(tok.logvar, -4.0, -0.5)

    # ==========================================
    # B) RL utilities: sampling, reward, advantage
    # ==========================================
    def _sample_latent_pack(self, uid: int, K_samples: int = 4) -> List[List[np.ndarray]]:
        """
        Return a list of latent sequences; each sequence has K tokens.
        First sample is the deterministic means (k==0), followed by noisy samples.
        """
        profile = self.user_profiles[uid]
        base_seq = [tok.mu.copy() for tok in profile["latents"]]  # r1: means
        packs = [base_seq]
        for _ in range(K_samples - 1):
            seq = []
            for tok in profile["latents"]:
                # noise strength is controlled by tok.logvar plus global sigma
                sigma = np.exp(0.5 * tok.logvar)
                eps = self.rng.normal(size=sigma.shape)
                seq.append(tok.mu + (sigma + self.sigma_noise) * eps)
            packs.append(seq)
        return packs  # length = K_samples

    def _format_prompt_for_scoring(self, uid: int, latents: List[np.ndarray]) -> Tuple[str, str]:
        """
        Build a concise scoring prompt (x) and the target string (y) whose logprobs approximate PPL.
        Here we use a compact target as the next-preferred genre title name proxy.
        In your full stack, replace target with the true next-item textual label.
        """
        user = self.data.get_user(uid)
        persona = self.persona_description(user)
        prefs = self.data.infer_user_genre_prefs(uid)
        top_g = sorted(prefs, key=prefs.get, reverse=True)[:3] or ["General"]

        # Encode latents compactly to pass context (small payload)
        latent_summ = ";".join([f"t{k}:{float(np.linalg.norm(v)):.3f}" for k, v in enumerate(latents, 1)])

        x = (
            f"{persona}\n"
            f"Top genres: {', '.join(top_g)}.\n"
            f"Latent reasoning tokens: {latent_summ}\n"
            f"Task: predict the next liked item title from the candidate set."
        )

        # Proxy target: we use the held-out dominant genre token as target text when true labels unavailable
        # In your training loop, pass the TRUE next item/title here instead.
        y = top_g[0]
        return x, y

    def _ppl_reward(self, prompt: str, target: str) -> float:
        """
        Reward = -exp(- avg logprob ) ≈ negative perplexity (higher is better).
        Tries to request token-level logprobs; if unsupported, falls back to a heuristic proxy.
        """
        try:
            # Azure Chat Completions may support logprobs via 'logprobs' parameter on some deployments.
            # If unsupported, this call will ignore logprobs; we then use fallback.
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=0,          # we only want logprobs on prompt+target continuation
                logprobs=True,         # try requesting logprobs
                top_logprobs=0,
                messages=[
                    {"role": "system", "content": "Score target token logprobs only."},
                    {"role": "user", "content": prompt + "\nAnswer: " + target}
                ],
            )
            # Not all Azure stacks return token logprobs; if absent, fallback.
            choice = resp.choices[0]
            logprob_list = getattr(choice, "logprobs", None)
            if not logprob_list or not getattr(logprob_list, "content", None):
                raise RuntimeError("logprobs unavailable; using fallback.")

            # Aggregate average logprob over target tokens
            lp_vals = []
            for frag in logprob_list.content:
                if frag and hasattr(frag, "token") and hasattr(frag, "logprob"):
                    lp_vals.append(frag.logprob)
            if not lp_vals:
                raise RuntimeError("empty logprobs; using fallback.")

            avg_lp = float(np.mean(lp_vals))
            reward = -math.exp(-avg_lp)  # eq. (6): negative PPL proxy
            return float(reward)
        except Exception:
            # Fallback: simple string-model proxy = short LM call to judge match (low cost)
            # We map the model's confidence words to a soft score in [-1, 0].
            try:
                r = client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=16,
                    messages=[
                        {"role": "system", "content": "Respond with one word: High, Medium, Low."},
                        {"role": "user", "content": f"{prompt}\nIs '{target}' a highly likely next choice? Reply one word."},
                    ],
                )
                word = (r.choices[0].message.content or "").strip().lower()
                if "high" in word: return -math.exp(-(-0.1))   # closer to 0 (better)
                if "medium" in word: return -math.exp(-( -1.0))
                return -math.exp(-( -2.0))
            except Exception:
                # Worst-case static reward
                return -math.e
    
    def _batch_ppl_rewards(
        self,
        prompts_targets: List[Tuple[str, str]],
        max_workers: int = 4,
    ) -> List[float]:
        """
        Parallel wrapper around _ppl_reward using a ThreadPoolExecutor.

        Args:
            prompts_targets: list of (prompt, target) pairs.
            max_workers: max concurrent requests to AzureOpenAI.

        Returns:
            List of rewards in the same order as prompts_targets.
        """
        rewards = [None] * len(prompts_targets)

        def worker(idx: int, prompt: str, target: str) -> None:
            try:
                rewards[idx] = self._ppl_reward(prompt, target)
            except Exception:
                rewards[idx] = -math.e  # worst-case fallback

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i, (p, t) in enumerate(prompts_targets):
                futures.append(ex.submit(worker, i, p, t))
            # wait for completion (results go into rewards[])
            for f in as_completed(futures):
                pass

        # type: ignore
        return rewards  # List[float]

    # ==========================================
    # RL step: batch-level advantage + latent update
    # ==========================================
    def rl_step(
        self,
        batch: List[int],
        K_samples: int = 4,
        clip: float = 0.5,
        max_workers: int = 4,
    ):
        """
        One RL update over a batch of user IDs.

        Parallelized version:
        - Collect all (prompt, target) pairs for all users and samples.
        - Compute rewards in parallel via _batch_ppl_rewards.
        - Then do the same latent update logic as before.

        Args:
            batch: list of user IDs.
            K_samples: number of latent packs per user (first = means).
            clip: gradient clipping magnitude.
            max_workers: max concurrent AzureOpenAI calls.
        """
        # 1) For each uid, sample packs and build prompts
        all_requests: List[Tuple[str, str]] = []
        index_map: List[Tuple[int, int]] = []  # (user_idx_in_batch, sample_idx)
        user_packs: List[List[List[np.ndarray]]] = []  # per user: [packs]
        batch = list(batch)  # ensure indexable

        for u_idx, uid in enumerate(batch):
            packs = self._sample_latent_pack(uid, K_samples=K_samples)
            user_packs.append(packs)
            for s_idx, seq in enumerate(packs):
                prompt, target = self._format_prompt_for_scoring(uid, seq)
                all_requests.append((prompt, target))
                index_map.append((u_idx, s_idx))

        # 2) Parallel reward computation
        rewards_flat = self._batch_ppl_rewards(all_requests, max_workers=max_workers)

        # 3) Reshape rewards back into per-user structure
        user_rewards: List[List[float]] = [
            [0.0 for _ in range(K_samples)] for _ in range(len(batch))
        ]
        for (u_idx, s_idx), r in zip(index_map, rewards_flat):
            user_rewards[u_idx][s_idx] = float(r)

        # Build groups in same structure as before: (uid, packs, rewards)
        groups: List[Tuple[int, List[List[np.ndarray]], List[float]]] = []
        first_rewards = []
        for u_idx, uid in enumerate(batch):
            packs = user_packs[u_idx]
            rewards = user_rewards[u_idx]
            groups.append((uid, packs, rewards))
            first_rewards.append(rewards[0])

        # 4) Batch baseline (average of first-sample rewards)
        sbar = float(np.mean(first_rewards))
        denom = np.linalg.norm(np.array(first_rewards) - sbar) + 1e-8

        # 5) Update per user (only latent params) — same logic as before
        for uid, packs, rewards in groups:
            advs = [(sk - sbar) / denom for sk in rewards]  # batch-relative advantage
            profile = self.user_profiles[uid]

            # For stability: only use the top-adv sample among the noisy ones
            best_idx = int(np.argmax(advs))
            best_seq = packs[best_idx]

            for k, tok in enumerate(profile["latents"]):
                delta = best_seq[k] - tok.mu
                step = np.clip(advs[best_idx] * delta, -clip, clip)
                tok.mu = tok.mu + self.lr * step

                if max(advs) < 0:
                    tok.logvar = np.clip(tok.logvar - 0.05, -6.0, 1.0)
                else:
                    tok.logvar = np.clip(tok.logvar + 0.01, -6.0, 1.0)


    # ==========================================
    # Inference: latent-aware candidate scoring
    # ==========================================
    def _latent_vector(self, uid: int) -> np.ndarray:
        """Aggregate the K latent tokens into one vector for simple scoring."""
        profile = self.user_profiles[uid]
        vecs = [t.mu for t in profile["latents"]]
        v = np.mean(vecs, axis=0)
        nrm = np.linalg.norm(v) + 1e-8
        return v / nrm

    def _genre_vector(self, genres: List[str]) -> np.ndarray:
        weights = {g: 1.0 for g in genres}
        v = self.encoder.encode_genre_mixture(weights)
        nrm = np.linalg.norm(v) + 1e-8
        return v / nrm

    def predict_item(self, uid: int, candidates: List[int], data, p_latent: Optional[np.ndarray] = None) -> int:
        """
        Replaces genre-only matching with latent-guided scoring:
          score = 0.7 * genre_match + 0.3 * sim(latent, item_genre_vec)
        """

        if data is None:
            data = self.data
        profile = self.user_profiles[uid]
        user_top = set(sorted(profile["preferences"], key=profile["preferences"].get, reverse=True)[:5])
        R = self._latent_vector(uid)

        scores = []
        for mid in candidates:
            row = data.movies[data.movies[data.item_id_col] == int(mid)]
            if row.empty:
                scores.append(-1e9); continue
            
            r = row.iloc[0]
            genre_text = str(r[data.item_genre_col])
            
            genres = [
            g.strip()
            for g in genre_text.replace(" | ", "|").split("|")
            if g.strip()
            ]

            gset = set(genres)
            # Jaccard-like
            genre_match = len(user_top & gset) / max(len(user_top | gset), 1)
            # Latent similarity
            I = self._genre_vector(genres)
            sim = float(np.dot(R, I))

            score = 0.7 * genre_match + 0.3 * sim
            scores.append(score)

        return candidates[int(np.argmax(scores))]

    # ==========================================
    # Minimal reasoning JSON (still compact)
    # ==========================================
    def build_prompt(self, uid: int) -> str:
        """
        Build compact prompt for Meta-CoT trace generation.
        Uses dynamic column names from Data_Structure.
        """

        user = self.data.get_user(uid)
        persona_desc = self.persona_description(user)

        prefs = self.data.infer_user_genre_prefs(uid)
        top_g = self.data.top_genres(uid, n=3)

        traits = self.user_profiles[uid]["traits"]
        traits_str = ", ".join([f"{k}={v:.2f}" for k, v in traits.items()])
        budget_msg = self.gamma_prompt_tag(self.gamma)

        sample_df = self.data.movies.sample(
        n=min(30, len(self.data.movies)),
        random_state=0
        )

        catalog = []
        for _, r in sample_df.iterrows():
            item_id = int(r[self.data.item_id_col])
            title = str(r[self.data.item_title_col])

            if "genre_set" in r:
                genres = sorted(list(r["genre_set"]))
            else:
                genre_text = str(r[self.data.item_genre_col])
                genres = [
                    g.strip()
                    for g in genre_text.replace(" | ", "|").split("|")
                    if g.strip()
                ]

            catalog.append({
                "item_id": item_id,
                "title": title,
                "genres": genres
            })
        return (
            f"You are a movie recommendation reasoner using compact latent thinking.\n"
            f"{budget_msg}\n\n"
            f"PERSONA: {persona_desc}\n"
            f"OCEAN: {traits_str}\n"
            f"Top genres: {top_g if top_g else 'Unknown'}\n"
            f"Genre dist: {json.dumps(prefs, ensure_ascii=False)}\n"
            f"Catalog sample: {json.dumps(catalog, ensure_ascii=False)}"
            # append at the end of your prompt building
            f"IMPORTANT:\n"
            f"- Choose ONE item ONLY from the provided candidate IDs.\n"
            f"- Return compact JSON only.\n"
        )
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        LLM call using HuggingFace Router OpenAI-compatible client.
        """

        for _ in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=300,
                    messages=[
                        {
                            "role": "system",
                            "content": "Output compact JSON only."
                        },
                        {
                            "role": "user",
                            "content": (
                                prompt
                                + "\nReturn JSON with keys: "
                                + "'latent_thoughts' and 'reasoning_steps'. "
                                + "Each reasoning step should contain item_id, title, and reasoning."
                            )
                        }
                    ],
                )

                content = resp.choices[0].message.content or ""
                parsed = safe_json_parse(content)
    
                if parsed:
                    lt = parsed.get("latent_thoughts")
                    if isinstance(lt, dict):
                        parsed["latent_thoughts"] = [lt]

                    rs = parsed.get("reasoning_steps")
                    if isinstance(rs, dict):
                        parsed["reasoning_steps"] = [rs]

                    return parsed

            except Exception as e:
                print("LLM call failed:", e)
                time.sleep(1.5)

        return {
            "latent_thoughts": [
                {
                    "name": "intent",
                    "content": "general"
                }
            ],
            "reasoning_steps": []
        }

    # def call_llm(self, prompt: str) -> Dict[str, Any]:
    #     for _ in range(self.max_retries):
    #         try:
    #             resp = client.chat.completions.create(
    #                 model=self.model,
    #                 temperature=self.temperature,
    #                 max_tokens=200,
    #                 messages=[
    #                     {"role": "system", "content": "Output compact JSON only."},
    #                     {"role": "user", "content": prompt + "\nReturn JSON with 'latent_thoughts' (intent/novelty/mood) and 'reasoning_steps' (movie_id,title,reasoning)."}
    #                 ],
    #             )
    #             content = resp.choices[0].message.content or ""
    #             parsed = safe_json_parse(content)
    #             if parsed:
    #                lt = parsed.get("latent_thoughts")
    #                if isinstance(lt, dict): parsed["latent_thoughts"] = [lt]
    #                rs = parsed.get("reasoning_steps")
    #                if isinstance(rs, dict): parsed["reasoning_steps"] = [rs]
    #                return parsed
    #         except Exception as e:
    #             time.sleep(1.5)
    #     return {"latent_thoughts": [{"name": "intent", "content": "general"}], "reasoning_steps": []}

    def generate_trace(self, uid: int) -> Dict[str, Any]:
        # No explicit CoT; just lightweight JSON to keep compatibility with your logs.
        prompt = self.build_prompt(uid)
        return self.call_llm(prompt)

    # ==========================================
    # Personality update (unchanged semantics)
    # ==========================================
    def update_user_profile(self, uid: int, z_idx: int, reward: float, z_desc: str):
        α = self.lr
        profile = self.user_profiles[uid]
        reward = float(np.clip(reward, -1.0, 1.0))
        desc = z_desc.lower()

        if any(k in desc for k in ["curious", "new", "deep", "novel"]):
            profile["traits"]["openness"] += α * (reward)
        if any(k in desc for k in ["familiar", "resistant", "routine"]):
            profile["traits"]["openness"] -= α * abs(reward)

        if any(k in desc for k in ["goal", "organized", "feedback", "plan"]):
            profile["traits"]["conscientiousness"] += α * (reward)
        if any(k in desc for k in ["distracted", "lazy", "unfocused"]):
            profile["traits"]["conscientiousness"] -= α * abs(reward)

        if any(k in desc for k in ["active", "engage", "communicat"]):
            profile["traits"]["extraversion"] += α * (reward)
        if any(k in desc for k in ["avoid", "silent", "hesitant"]):
            profile["traits"]["extraversion"] -= α * abs(reward)

        if any(k in desc for k in ["empath", "cooperat", "polite", "trust"]):
            profile["traits"]["agreeableness"] += α * (reward)
        if any(k in desc for k in ["rude", "indifferent", "uncooperat"]):
            profile["traits"]["agreeableness"] -= α * abs(reward)

        if any(k in desc for k in ["emotional", "discouraged", "anxious"]):
            profile["traits"]["neuroticism"] += α * (reward)
        if any(k in desc for k in ["confident", "stable", "resilient"]):
            profile["traits"]["neuroticism"] -= α * abs(reward)

        # for k in profile["traits"]:
        #     profile["traits"][k] = float(np.clip(profile["traits"][k] * self.trait_decay, 0.0, 1.0))
        # AFTER (mean-reverts toward 0.5 instead)
        for k in profile["traits"]:
            # decay around 0.5: value ← 0.5 + τ * (value - 0.5)
            profile["traits"][k] = float(np.clip(0.5 + self.trait_decay * (profile["traits"][k] - 0.5), 0.0, 1.0))

        profile["memory"].append({"latent": z_desc, "reward": reward, "traits": dict(profile["traits"])})

    # ==========================================
    # Summary (unchanged)
    # ==========================================
    def summarize_user(self, uid: int) -> str:
        profile = self.user_profiles[uid]
        print("User Memory: ", profile["memory"])
        return json.dumps({
            "preferences": dict(sorted(profile["preferences"].items(), key=lambda x: -x[1])[:5]),
            "traits": profile["traits"],
            "memory_len": len(profile["memory"]),
            "latent_norms": [float(np.linalg.norm(t.mu)) for t in profile["latents"]],
        }, indent=2, ensure_ascii=False)

    # =========================
    # LLM-driven persona updates
    # =========================
    def _build_persona_update_prompt(self, uid: int, adv_val: float) -> str:
        """
        Build a compact rubric prompt for OCEAN judgments.
        Model must return strict JSON with fields:
        {
          "descriptor": "<short, keyword-rich phrase>",
          "judgments": [
             {"trait":"openness", "polarity":"positive"|"negative", "evidence":["..."]},
             {"trait":"conscientiousness", ...},
             {"trait":"extraversion", ...},
             {"trait":"agreeableness", ...},
             {"trait":"neuroticism", ...}
          ]
        }
        Notes:
          - We pass adv_val (batch-normalized advantage) as a weak signal of “good/bad” step.
          - Keep it short/cheap; we’ll parse with safe_json_parse.
        """
        user = self.data.get_user(uid)
        persona = self.persona_description(user)

        rubric = (
           "Openness:\n"
           "[Positive] Receptive to new content; Curious about new topics; Engage in deep conversation;\n"
           "[Negative] Prefer familiar content; Resistant to change; Lack of curiosity;\n\n"
           "Conscientiousness:\n"
           "[Positive] Goal-oriented; Organized and thoughtful; Provide useful feedback;\n"
           "[Negative] Lack of focus; Easily distracted; Little feedback;\n\n"
           "Extraversion:\n"
           "[Positive] Active participation; Enjoy engagement; Interested in communication;\n"
           "[Negative] Avoid interaction; Hesitant to express; Uninterested in socializing;\n\n"
           "Agreeableness:\n"
           "[Positive] Empathetic and caring; Cooperative and trusting; Polite and appreciative;\n"
           "[Negative] Indifferent to others; Uncooperative; Rude language;\n\n"
           "Neuroticism:\n"
           "[Positive] Emotional fluctuation; Lack of confidence; Easily discouraged;\n"
           "[Negative] Emotionally stable; Confident response; Handle challenges well;\n"
           )

        return (
           f"You rate a user’s transient behavioural signal using OCEAN.\n"
           f"Persona: {persona}\n"
           f"Signal: advantage={adv_val:.3f} (positive => desirable behaviour, negative => undesirable).\n\n"
           f"Rubric:\n{rubric}\n"
           "TASK: Based on the signal and rubric, output STRICT JSON only with keys: "
           "'descriptor' (≤ 8 words, lowercase, keyword-rich), and 'judgments' "
           "(an array of 5 objects, one per trait in this exact set: "
           "openness, conscientiousness, extraversion, agreeableness, neuroticism). "
           "Each object must be: {\"trait\": <name>, \"polarity\": \"positive\"|\"negative\", \"evidence\": [<1-3 short tokens>]}.\n"
           "Do NOT add explanations or extra keys. Output JSON only."
           )

    def gen_descriptor_via_llm(self, uid: int, adv_val: float) -> tuple[str, Dict[str, int]]:
        wanted = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

        def norm_trait(x: str) -> str:
            x = (x or "").strip().lower()
            aliases = {
               "open": "openness",
               "conscientious": "conscientiousness",
               "extra version": "extraversion",
               "extroversion": "extraversion",
               "agreeable": "agreeableness",
               "neurotic": "neuroticism",
            }
            return aliases.get(x, x)

        def norm_polarity(x: str) -> int:
            s = (x or "").strip().lower()
            pos = {"positive","pos","+","+1","increase","increasing","high","higher","more","present","active"}
            neg = {"negative","neg","-","-1","decrease","decreasing","low","lower","less","absent","stable","calm"}
            if s in pos: return +1
            if s in neg: return -1
            return 0

        def parse_payload(txt: str) -> tuple[str, Dict[str,int], bool]:
            parsed = safe_json_parse(txt)
            desc = str(parsed.get("descriptor", "")).strip().lower() if isinstance(parsed, dict) else ""
            judgments = parsed.get("judgments", []) if isinstance(parsed, dict) else []
            pol_map = {t: 0 for t in wanted}
            if isinstance(judgments, list):
               for j in judgments:
                   if not isinstance(j, dict): continue
                   t = norm_trait(j.get("trait", ""))
                   p = norm_polarity(j.get("polarity", ""))
                   if t in pol_map: pol_map[t] = p
            ok = bool(desc) and any(v != 0 for v in pol_map.values())
            return (desc if desc else "", pol_map, ok)

        prompt = self._build_persona_update_prompt(uid, adv_val)
        # print("Prompt: ", prompt)

        # --- Try 1: JSON Schema enforcement (preferred) ---
        try:
            resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=180,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ocean_judgment",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["descriptor", "judgments"],
                        "properties": {
                            "descriptor": {"type":"string", "minLength":1},
                            "judgments": {
                                "type":"array",
                                "minItems": 5,
                                "maxItems": 5,
                                "items": {
                                    "type":"object",
                                    "additionalProperties": False,
                                    "required": ["trait","polarity","evidence"],
                                    "properties": {
                                        "trait": {"type":"string"},
                                        "polarity": {"type":"string", "enum":["positive","negative"]},
                                        "evidence": {
                                            "type":"array",
                                            "minItems":1,
                                            "maxItems":3,
                                            "items":{"type":"string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            messages=[
                {"role": "system", "content": "Return STRICT JSON only that matches the schema."},
                {"role": "user", "content": prompt}
            ],
            )
            content = resp.choices[0].message.content or ""
            desc, pol_map, ok = parse_payload(content)
            if ok:
               # print("Try 1: ", desc, pol_map)
               return desc, pol_map
        except Exception:
            pass

        # --- Try 2: json_object (schema-less) ---
        try:
            resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=180,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return STRICT JSON only. No prose."},
                {"role": "user", "content": prompt}
            ],
            )
            content = resp.choices[0].message.content or ""
            desc, pol_map, ok = parse_payload(content)
            if ok:
               # print("Try 2: ", desc, pol_map)
               return desc, pol_map
        except Exception:
            pass

        # --- Try 3: Minimal prompt with tiny example (no response_format) ---
        try:
            mini = (
            "Output EXACTLY this JSON shape:\n"
            '{"descriptor":"<short>",'
            '"judgments":[{"trait":"openness","polarity":"positive","evidence":["curious"]},'
            '{"trait":"conscientiousness","polarity":"positive","evidence":["goal"]},'
            '{"trait":"extraversion","polarity":"positive","evidence":["engage"]},'
            '{"trait":"agreeableness","polarity":"positive","evidence":["polite"]},'
            '{"trait":"neuroticism","polarity":"negative","evidence":["confident"]}]}\n'
            "No extra text."
            )
            resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=160,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only. No prose."},
                {"role": "user", "content": mini}
            ],
            )
            content = resp.choices[0].message.content or ""
            desc, pol_map, ok = parse_payload(content)
            if ok:
               # print("Try 3: ", desc, pol_map)
               return desc, pol_map
        except Exception:
            pass

        # --- Final fallback: derive from advantage ---
        # (also provide a non-empty descriptor)
        # Print a compact debug once so you see why it fell back.
        # print("[gen_descriptor_via_llm] Using fallback mapping; model returned no usable JSON.")
        fallback_desc = self._fallback_descriptor_from_adv(adv_val)
        pol_map = {
            "openness":        +1 if adv_val >= 0 else -1,
            "conscientiousness":+1 if adv_val >= 0 else -1,
            "extraversion":    +1 if adv_val >= 0 else -1,
            "agreeableness":   +1 if adv_val >= 0 else -1,
            "neuroticism":     +1 if adv_val <  0 else -1,
        }
        # print("Fallback: ", desc, pol_map)
        return fallback_desc, pol_map


    def _fallback_descriptor_from_adv(self, adv_val: float) -> str:
        if adv_val >= 0.25:
           return "curious engaged cooperative confident"
        elif adv_val >= 0.0:
           return "curious balanced polite stable"
        elif adv_val >= -0.25:
           return "familiar hesitant indifferent distracted"
        else:
           return "resistant avoid uncooperative discouraged"

    def update_user_profile_llm(self, uid: int, reward: float,
                            descriptor: str,
                            judgments: Dict[str, int]):
        """
        Update OCEAN using per-trait polarity (+1/-1/0) from LLM, scaled by reward.
        Special handling for Neuroticism: +1 means *more* neurotic (worse), -1 means calmer/stable (better).
        We apply:
           delta = α * sign * f(reward)
        where sign = +1/-1, and f(reward) = reward for all traits except neuroticism where we invert.
        """
        α = self.lr
        profile = self.user_profiles[uid]
        reward = float(np.clip(reward, -1.0, 1.0))

        # Normalize keys
        trait_keys = {
           "openness": "openness",
           "conscientiousness": "conscientiousness",
           "extraversion": "extraversion",
           "agreeableness": "agreeableness",
           "neuroticism": "neuroticism",
         }

        for k_in, k_out in trait_keys.items():
            sign = int(judgments.get(k_in, 0))
            if sign == 0:
               continue

            if k_in == "neuroticism":
               # If LLM says "positive" neuroticism, increase neuroticism by +α*|reward|
               # If "negative", decrease neuroticism by α*|reward|
               delta = α * ( +abs(reward) if sign > 0 else -abs(reward) )
            else:
               # For other traits: "positive" increases with reward, "negative" decreases with |reward|
               delta = α * ( +reward if sign > 0 else -abs(reward) )

            profile["traits"][k_out] = float(np.clip(profile["traits"][k_out] + delta, 0.0, 1.0))

        # mean-revert around 0.5
        for k in profile["traits"]:
            profile["traits"][k] = float(np.clip(0.5 + self.trait_decay * (profile["traits"][k] - 0.5), 0.0, 1.0))

        # memory log
        profile["memory"].append({"latent": descriptor, "reward": reward, "traits": dict(profile["traits"])})
        # print("User ID: ", uid, "Personality Traits: ", profile["traits"])
    
    def build_meta_tot_prompt(
        self,
        uid: int,
        candidates: List[int],
        max_depth: int = 2,
        branch_factor: int = 3,
    ) -> str:
        """
        Build a compact Meta-ToT prompt.

        Meta-ToT is used only during training:
          - It asks why the user may prefer different candidate groups.
          - It creates preference-hypothesis branches.
          - It verifies the best branch.
          - The output is later scored against the ground-truth item to update Z/personality.
        """

        user = self.data.get_user(uid)
        persona_desc = self.persona_description(user)

        prefs = self.data.infer_user_genre_prefs(uid)
        top_g = self.data.top_genres(uid, n=3)

        traits = self.user_profiles[uid]["traits"]
        traits_str = ", ".join(f"{k}={v:.2f}" for k, v in traits.items())

        lat_norms = [
            float(np.linalg.norm(t.mu))
            for t in self.user_profiles[uid]["latents"]
        ]

        latent_summary = {
            "num_tokens": len(self.user_profiles[uid]["latents"]),
            "token_norms": lat_norms,
        }

        cand_meta = []

        for mid in candidates:
            row = self.data.movies[
                self.data.movies[self.data.item_id_col] == int(mid)
            ]

            if row.empty:
                continue

            r = row.iloc[0]
            title = str(r[self.data.item_title_col])

            if "genre_set" in r:
                genres = sorted(list(r["genre_set"]))
            else:
                genre_text = str(r[self.data.item_genre_col])
                genres = [
                    g.strip()
                    for g in genre_text.replace(" | ", "|").split("|")
                    if g.strip()
                ]

            desc = str(r[self.data.item_description_col])

            cand_meta.append({
                "item_id": int(mid),
                "title": title,
                "genres": genres,
                "description": desc[:180],
            })

        budget_msg = self.gamma_prompt_tag(self.gamma)

        return (
            "You are a Meta-ToT reasoning module for offline recommender-user simulation.\n"
            "This reasoning is used ONLY during training to produce a verifier reward.\n"
            "At test time, the learned latent profile Z and personality traits will be reused without ToT reasoning.\n\n"

            f"{budget_msg}\n\n"
            f"PERSONA: {persona_desc}\n"
            f"OCEAN traits: {traits_str}\n"
            f"Top user genres: {top_g if top_g else 'Unknown'}\n"
            f"Genre distribution: {json.dumps(prefs, ensure_ascii=False)}\n"
            f"Latent summary Z: {json.dumps(latent_summary, ensure_ascii=False)}\n\n"
            f"Candidate items: {json.dumps(cand_meta, ensure_ascii=False)}\n\n"

            "TASK:\n"
            f"- Build a compact Tree-of-Thought with at most {branch_factor} branches and depth <= {max_depth}.\n"
            "- Each branch must answer a WHY question about user preference.\n"
            "- Example why questions: why genre match, why novelty, why avoid mismatch, why personality fit.\n"
            "- Each branch must keep only candidate IDs from the fixed candidate list.\n"
            "- Each leaf must choose exactly one candidate item.\n"
            "- Add a verifier that checks whether the selected item is supported by persona, genre preference, and branch evidence.\n"
            "- Do not invent candidate IDs.\n\n"

            "OUTPUT STRICT JSON ONLY:\n"
            "{\n"
            '  "root": "n0",\n'
            '  "nodes": [\n'
            '    {\n'
            '      "id": "n0",\n'
            '      "depth": 0,\n'
            '      "why_question": "why would this user prefer some items over others?",\n'
            '      "thought": "short root reasoning",\n'
            '      "candidate_ids": [..],\n'
            '      "children": ["n1", "n2"]\n'
            '    },\n'
            '    {\n'
            '      "id": "n1",\n'
            '      "depth": 1,\n'
            '      "why_question": "why does this branch fit the user?",\n'
            '      "thought": "short branch reasoning",\n'
            '      "candidate_ids": [..],\n'
            '      "children": [],\n'
            '      "chosen_item": 123,\n'
            '      "branch_score": 0.0\n'
            '    }\n'
            "  ],\n"
            '  "best_leaf": "n1",\n'
            '  "verifier": {\n'
            '    "chosen_item": 123,\n'
            '    "supported_by_persona": true,\n'
            '    "supported_by_genre": true,\n'
            '    "supported_by_branch": true,\n'
            '    "confidence": 0.0,\n'
            '    "verification_reason": "short reason"\n'
            '  }\n'
            "}\n"
        )

    def call_llm_meta_tot(
        self,
        prompt: str,
        valid_candidates: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        ToT-specific LLM call.
        Normalizes:
          - nodes list -> dict
          - invalid candidate IDs are removed
          - invalid chosen_item is set to None
          - verifier is always present
        """

        valid_set = set(int(x) for x in valid_candidates) if valid_candidates is not None else None

        for _ in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": "Return STRICT JSON only. No prose."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                )

                content = resp.choices[0].message.content or ""
                tree = safe_json_parse(content)

                if not isinstance(tree, dict):
                    continue

                tree = self._normalize_meta_tot_tree(tree, valid_candidates=valid_candidates)

                if tree["nodes"]:
                    return tree

            except Exception:
                time.sleep(1.5)

        return self._fallback_meta_tot_tree(valid_candidates)

    def _fallback_meta_tot_tree(self, valid_candidates: Optional[List[int]] = None) -> Dict[str, Any]:
        cands = [int(x) for x in valid_candidates] if valid_candidates else []
        chosen = cands[0] if cands else None

        return {
            "root": "n0",
            "nodes": {
                "n0": {
                    "id": "n0",
                    "depth": 0,
                    "why_question": "fallback",
                    "thought": "fallback tree",
                    "candidate_ids": cands,
                    "children": [],
                    "chosen_item": chosen,
                    "branch_score": 0.0,
                }
            },
            "best_leaf": "n0",
            "verifier": {
                "chosen_item": chosen,
                "supported_by_persona": False,
                "supported_by_genre": False,
                "supported_by_branch": False,
                "confidence": 0.0,
                "verification_reason": "fallback verifier",
            }
        }


    def _normalize_meta_tot_tree(
        self,
        tree: Dict[str, Any],
        valid_candidates: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        valid_set = set(int(x) for x in valid_candidates) if valid_candidates is not None else None
    
        nodes = tree.get("nodes", {})

        if isinstance(nodes, list):
            nodes_dict = {}
            for i, n in enumerate(nodes):
                if not isinstance(n, dict):
                    continue
                nid = str(n.get("id", f"n{i}"))
                n["id"] = nid
                nodes_dict[nid] = n
            nodes = nodes_dict

        if not isinstance(nodes, dict):
            nodes = {}

        clean_nodes = {}

        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue

            node["id"] = str(node.get("id", nid))
            node["depth"] = int(node.get("depth", 0))

            cands = node.get("candidate_ids", [])

            if not isinstance(cands, list):
                cands = []

            clean_cands = []
            for x in cands:
                try:
                    x = int(x)
                    if valid_set is None or x in valid_set:
                        clean_cands.append(x)
                except Exception:
                    continue
    
            node["candidate_ids"] = list(dict.fromkeys(clean_cands))
    
            children = node.get("children", [])
            if not isinstance(children, list):
                children = []
            node["children"] = [str(c) for c in children]

            chosen = node.get("chosen_item", None)
            try:
                chosen = int(chosen) if chosen is not None else None
            except Exception:
                chosen = None

            if chosen is not None and valid_set is not None and chosen not in valid_set:
                chosen = None

            node["chosen_item"] = chosen

            try:
                node["branch_score"] = float(node.get("branch_score", 0.0))
            except Exception:
                node["branch_score"] = 0.0

            node["why_question"] = str(node.get("why_question", ""))
            node["thought"] = str(node.get("thought", ""))

            clean_nodes[node["id"]] = node

        tree["nodes"] = clean_nodes
    
        if "root" not in tree or str(tree["root"]) not in clean_nodes:
            keys = list(clean_nodes.keys())
            tree["root"] = keys[0] if keys else "n0"

        if "best_leaf" not in tree or str(tree["best_leaf"]) not in clean_nodes:
            tree["best_leaf"] = tree["root"]

        verifier = tree.get("verifier", {})
        if not isinstance(verifier, dict):
            verifier = {}

        chosen = verifier.get("chosen_item", None)
        try:
            chosen = int(chosen) if chosen is not None else None
        except Exception:
            chosen = None

        if chosen is not None and valid_set is not None and chosen not in valid_set:
            chosen = None
    
        verifier["chosen_item"] = chosen
        verifier["supported_by_persona"] = bool(verifier.get("supported_by_persona", False))
        verifier["supported_by_genre"] = bool(verifier.get("supported_by_genre", False))
        verifier["supported_by_branch"] = bool(verifier.get("supported_by_branch", False))
    
        try:
            verifier["confidence"] = float(verifier.get("confidence", 0.0))
        except Exception:
            verifier["confidence"] = 0.0
    
        verifier["confidence"] = float(np.clip(verifier["confidence"], 0.0, 1.0))
        verifier["verification_reason"] = str(verifier.get("verification_reason", ""))

        tree["verifier"] = verifier
    
        return tree

    def generate_meta_tot_tree(
        self,
        uid: int,
        candidates: List[int],
        max_depth: int = 2,
        branch_factor: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a normalized Meta-ToT tree.
        This is used only during training.
        """

        candidates = [int(x) for x in candidates]

        prompt = self.build_meta_tot_prompt(
            uid=uid,
            candidates=candidates,
            max_depth=max_depth,
            branch_factor=branch_factor,
        )

        tree = self.call_llm_meta_tot(
            prompt=prompt,
            valid_candidates=candidates,
        )

        tree = self._normalize_meta_tot_tree(
            tree,
            valid_candidates=candidates,
        )

        return tree
    
    def score_meta_tree_leaf(
        self,
        uid: int,
        tree: Dict[str, Any],
        gt_item: int,
    ) -> float:
        """
        Verifier-aware Meta-ToT reward.

        This function does NOT evaluate test-time prediction.
        It only produces a training reward for optimizing Z/personality.

        Reward design:
          0.0   exact GT selected by verifier or best leaf
         -1.0   GT retained in best leaf candidate set
         -2.0   GT appears somewhere in tree
         -5.0   GT absent or invalid tree

        Then a small verifier confidence bonus/penalty is added.
        """

        if not isinstance(tree, dict):
            return -5.0

        nodes = tree.get("nodes", {})
        if not isinstance(nodes, dict) or not nodes:
            return -5.0

        gt_item = int(gt_item)

        verifier = tree.get("verifier", {})
        if not isinstance(verifier, dict):
            verifier = {}

        verifier_chosen = verifier.get("chosen_item", None)

        try:
            verifier_chosen = int(verifier_chosen) if verifier_chosen is not None else None
        except Exception:
            verifier_chosen = None

        best_leaf_id = tree.get("best_leaf")
        if best_leaf_id not in nodes:
            best_leaf_id = tree.get("root")
    
        if best_leaf_id not in nodes:
            best_leaf_id = next(iter(nodes.keys()))
    
        leaf = nodes.get(best_leaf_id, {})
    
        leaf_chosen = leaf.get("chosen_item", None)
        try:
            leaf_chosen = int(leaf_chosen) if leaf_chosen is not None else None
        except Exception:
            leaf_chosen = None
    
        # Prefer verifier chosen item if it exists; otherwise use best leaf chosen item.
        chosen = verifier_chosen if verifier_chosen is not None else leaf_chosen
    
        confidence = float(verifier.get("confidence", 0.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
    
        supported_count = int(bool(verifier.get("supported_by_persona", False)))
        supported_count += int(bool(verifier.get("supported_by_genre", False)))
        supported_count += int(bool(verifier.get("supported_by_branch", False)))
    
        support_bonus = 0.15 * supported_count
        confidence_bonus = 0.25 * confidence
    
        # Case 1: exact GT selection
        if chosen == gt_item:
            return float(0.0 + support_bonus + confidence_bonus)
    
        # Case 2: GT appears in best leaf candidate set
        leaf_cands = leaf.get("candidate_ids", [])
        try:
            leaf_cands = [int(x) for x in leaf_cands]
        except Exception:
            leaf_cands = []

        if gt_item in leaf_cands:
            return float(-1.0 + support_bonus + 0.5 * confidence_bonus)

        # Case 3: GT appears somewhere in the tree
        for _, node in nodes.items():
            if not isinstance(node, dict):
                continue

            cands = node.get("candidate_ids", [])
            try:
                cands = [int(x) for x in cands]
            except Exception:
                cands = []

            if gt_item in cands:
                return float(-2.0 + 0.5 * support_bonus)

        # Case 4: GT absent
        return -5.0

    def summarize_meta_tot_outcome(
        self,
        tree: Dict[str, Any],
        gt_item: int,
    ) -> Dict[str, Any]:
        """
        Compact summary of Meta-ToT outcome.
        Useful for personality update and debugging.
        """

        nodes = tree.get("nodes", {}) if isinstance(tree, dict) else {}
        verifier = tree.get("verifier", {}) if isinstance(tree, dict) else {}

        best_leaf_id = tree.get("best_leaf") if isinstance(tree, dict) else None
        if isinstance(nodes, dict) and best_leaf_id not in nodes:
            best_leaf_id = tree.get("root")

        leaf = nodes.get(best_leaf_id, {}) if isinstance(nodes, dict) and best_leaf_id in nodes else {}

        chosen = verifier.get("chosen_item", None)
        if chosen is None:
            chosen = leaf.get("chosen_item", None)

        try:
            chosen = int(chosen) if chosen is not None else None
        except Exception:
            chosen = None

        gt_item = int(gt_item)

        return {
            "gt_item": gt_item,
            "chosen_item": chosen,
            "correct": chosen == gt_item,
            "best_leaf": best_leaf_id,
            "best_leaf_thought": str(leaf.get("thought", ""))[:200],
            "best_leaf_why": str(leaf.get("why_question", ""))[:120],
            "verifier_confidence": float(verifier.get("confidence", 0.0)) if isinstance(verifier, dict) else 0.0,
            "verification_reason": str(verifier.get("verification_reason", ""))[:200] if isinstance(verifier, dict) else "",
        }
