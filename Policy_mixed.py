# HFEmbeddingsAdapter using sentence-transformers
from typing import Optional
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import os, json

from Meta_CoT.MemoryModule import ShortTermMemory, LongTermMemory
from Meta_CoT.MemoryControl import MetaCoTController
# from openai import AzureOpenAI
from openai import OpenAI


# client = AzureOpenAI(
#     api_version=api_version,
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
# )
openai_api_key = ""  # or "sk-..." directly
deployment_name = "gpt-4o-mini"
client = OpenAI(api_key=openai_api_key)

# ===============================
# 4) Add task heads to LLMPolicy
# ===============================
class LLMPolicyWithMemoryMixin:
    def attach_memory_stack(self, uid: int, stm: ShortTermMemory, ltm: LongTermMemory, controller: MetaCoTController):
        self.user_profiles[uid]["STM"] = stm
        self.user_profiles[uid]["LTM"] = ltm
        self.user_profiles[uid]["MEM_CTRL"] = controller

    def memory_stack(self, uid: int):
        prof = self.user_profiles.get(uid, {})
        return prof.get("STM"), prof.get("LTM"), prof.get("MEM_CTRL")

    def _item_meta(self, mid: int) -> Dict[str, Any]:
        row = self.data.movies[self.data.movies["item_id"] == mid]
        if row.empty:
            return {"item_id": mid, "title": "Unknown", "genre": []}
        return {
            "item_id": int(row.iloc[0]["item_id"]),
            "title": str(row.iloc[0]["title"]),
            "genre": str(row.iloc[0]["genre"]).split(" | "),
        }

    def _memory_features(self, uid: int, task: str, mid: Optional[int] = None):
        ctrl: MetaCoTController = self.user_profiles[uid].get("MEM_CTRL")
        item_meta = self._item_meta(mid) if mid is not None else None
        ctx = ctrl.build_task_context(uid, task, item_meta=item_meta) if ctrl else {"memories": [], "persona":"", "top_prefs":{}}
        # cheap memory vector in the same latent genre space
        if ctx["memories"]:
            mem_vecs = []
            for text in ctx["memories"]:
                # detect genres appearing in memory text; fallback to item genres
                all_genres = self.data.movies["genre"].str.split(" | ").explode().dropna().unique()
                genres = [g for g in all_genres if g and g.lower() in text.lower()]
                if (not genres) and item_meta:
                    genres = item_meta["genre"]
                v = self.encoder.encode_genre_mixture({g:1.0 for g in genres}) if genres else np.zeros(self.latent_dim, dtype=np.float32)
                mem_vecs.append(v)
            if mem_vecs:
                M = np.mean(np.stack(mem_vecs, axis=0), axis=0)
                M = M / (np.linalg.norm(M) + 1e-8)
            else:
                M = np.zeros(self.latent_dim, dtype=np.float32)
        else:
            M = np.zeros(self.latent_dim, dtype=np.float32)
        return ctx, M

    def predict_rating(self, uid: int, mid: int) -> float:
        R = self._latent_vector(uid)
        meta = self._item_meta(mid)
        I = self._genre_vector(meta["genre"])
        base_sim = float(np.dot(R, I))  # [-1,1]
        ctx, M = self._memory_features(uid, task="rating_prediction", mid=mid)
        mem_align = float(np.dot(R, M)) if np.linalg.norm(M) > 0 else 0.0
        title_hit = any(meta["title"].lower() in m.lower() for m in ctx["memories"])
        score = 3.0 + 1.2*base_sim + 0.8*mem_align + (0.3 if title_hit else 0.0)
        return float(np.clip(score, 1.0, 5.0))

    def generate_feeling(self, uid: int, mid: int) -> str:
        meta = self._item_meta(mid)
        ctx, _ = self._memory_features(uid, task="feeling_generation", mid=mid)
        prompt = (
            "Write a single-sentence personal reaction (≤20 words) grounded in persona and memories.\n"
            f"Persona: {ctx['persona']}\n"
            f"Top prefs: {list(ctx['top_prefs'].keys())}\n"
            f"Memories: {ctx['memories'][:4]}\n"
            f"Book: {meta['title']} ({'|'.join(meta['genre'])})\n"
            "Tone: authentic, concise, no spoilers."
        )
        try:
            r = client.chat.completions.create(
                model=deployment_name,
                temperature=0.4,
                max_tokens=40,
                messages=[
                    {"role": "system", "content": "Return only the sentence, no quotes."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = (r.choices[0].message.content or "").strip()
        except Exception:
            text = "Feels aligned with my taste and current mood."
        ctrl: MetaCoTController = self.user_profiles[uid].get("MEM_CTRL")
        if ctrl:
            ctrl.feedback(uid, "feeling_generation", success=True, related_texts=ctx["memories"][:2])
        return text

    def analyze_sentiment(self, uid: int, text: str) -> str:
        ctx, _ = self._memory_features(uid, task="sentiment_analysis", mid=None)
        prompt = (
            "Classify sentiment (positive|neutral|negative). Consider persona and memories only if relevant.\n"
            f"Persona: {ctx['persona']}\n"
            f"Memories: {ctx['memories'][:3]}\n"
            f"Text: {text}\n"
            "Answer with one word only."
        )
        try:
            r = client.chat.completions.create(
                model=deployment_name,
                temperature=0.0,
                max_tokens=3,
                messages=[
                    {"role": "system", "content": "Return one of: positive, neutral, negative."},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = (r.choices[0].message.content or "").strip().lower()
            if "pos" in ans: return "positive"
            if "neg" in ans: return "negative"
            return "neutral"
        except Exception:
            return "neutral"

    # ---------- SAVE / LOAD PER-USER ----------
    def save_user_memory(self, uid: int, out_dir: str = "profiles/memory", include_ltm_vectors: bool = False):
        """
        Save STM/LTM/Controller for a single user to JSON: profiles/memory/user_<uid>.json
        Only pure data is stored (no client handles); re-inject clients via factories on load.
        """
        os.makedirs(out_dir, exist_ok=True)
        stm, ltm, ctrl = self.memory_stack(uid)
        payload = {
            "uid": int(uid),
            "stm": stm.state_dict() if hasattr(stm, "state_dict") else None,
            "ltm": ltm.state_dict(include_vectors=include_ltm_vectors) if hasattr(ltm, "state_dict") else None,
            "controller": ctrl.state_dict() if hasattr(ctrl, "state_dict") else None,
        }
        with open(os.path.join(out_dir, f"user_{uid}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_user_memory(self,
                         uid: int,
                         in_dir: str = "profiles/memory",
                         stm_factory=None,
                         ltm_factory=None,
                         controller_factory=None):
        """
        Restore STM/LTM/Controller for a single user. You MUST pass factories that
        build live objects with your clients/embedders, e.g.:
            stm_factory = lambda uid: ShortTermMemory(llm=client, embeddings_adapter=emb, capacity=10, enhance_threshold=3)
            ltm_factory = lambda uid: LongTermMemory(embeddings=emb, capacity=512)
            controller_factory = lambda uid: MetaCoTController(policy=self, embeddings=emb)
        """
        path = os.path.join(in_dir, f"user_{uid}.json")
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        stm = stm_factory(uid) if stm_factory else None
        ltm = ltm_factory(uid) if ltm_factory else None
        ctrl = controller_factory(uid) if controller_factory else None

        if stm and d.get("stm"): stm.load_state_dict(d["stm"])
        if ltm and d.get("ltm"): ltm.load_state_dict(d["ltm"])
        if ctrl and d.get("controller"): ctrl.load_state_dict(d["controller"])

        self.attach_memory_stack(uid, stm=stm, ltm=ltm, controller=ctrl)
        return True

    # ---------- SAVE / LOAD ALL USERS ----------
    def save_all_memory_stacks(self, out_dir: str = "profiles/memory", include_ltm_vectors: bool = False):
        """
        Save STM/LTM/Controller JSON for all users that have them attached.
        """
        os.makedirs(out_dir, exist_ok=True)
        for uid in self.user_profiles.keys():
            stm, ltm, ctrl = self.memory_stack(uid)
            if any([stm, ltm, ctrl]):
                self.save_user_memory(uid, out_dir=out_dir, include_ltm_vectors=include_ltm_vectors)

    def load_all_memory_stacks(self,
                               in_dir: str = "profiles/memory",
                               stm_factory=None,
                               ltm_factory=None,
                               controller_factory=None):
        """
        Load all per-user memory JSONs found under in_dir. Requires the same factories as load_user_memory().
        """
        if not os.path.isdir(in_dir):
            return 0
        count = 0
        for name in os.listdir(in_dir):
            if name.startswith("user_") and name.endswith(".json"):
                try:
                    uid = int(name[len("user_"):-len(".json")])
                except Exception:
                    continue
                ok = self.load_user_memory(uid, in_dir=in_dir,
                                           stm_factory=stm_factory,
                                           ltm_factory=ltm_factory,
                                           controller_factory=controller_factory)
                if ok: count += 1
        return count

    # ---------- OPTIONAL: add memory info to summarize_user ----------
    def summarize_user_with_memory(self, uid: int) -> str:
        prof = self.user_profiles[uid]
        stm, ltm, ctrl = self.memory_stack(uid)
        mem_log_len = len(prof.get("memory", []))
        stm_len = len(getattr(stm, "short_memories", [])) if stm else 0
        ltm_len = len(getattr(ltm, "store", getattr(ltm, "items", []))) if ltm else 0
        return json.dumps({
            "preferences": dict(sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]),
            "traits": prof["traits"],
            "latent_norms": [float(np.linalg.norm(t.mu)) for t in prof["latents"]],
            "logs_memory_len": mem_log_len,   # updates from update_user_profile(_llm)
            "stm_items": stm_len,
            "ltm_items": ltm_len,
            "has_controller": bool(ctrl is not None),
        }, indent=2, ensure_ascii=False)