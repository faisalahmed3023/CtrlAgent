# HFEmbeddingsAdapter using sentence-transformers
from typing import List, Union, Iterable, Optional
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Iterable, Optional, Tuple
import numpy as np
import time, json, math, random
import re
from sentence_transformers import SentenceTransformer

from openai import AzureOpenAI

# ======================================================
# Azure OpenAI connection (unchanged; consider env vars)
# ======================================================
endpoint = "https://genaisim.openai.azure.com/"
subscription_key = ""
deployment_name = "gpt-4o-mini"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


class HFEmbeddingsAdapter:
    """
    Hugging Face embeddings adapter (sentence-transformers).
    - Supports single string or list of strings
    - Batching + GPU/CPU selection
    - Optional L2 normalization
    - Optional instruction prefix (for E5/GTE/BGE-style models)
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fast
        device: Optional[str] = None,   # "cuda", "mps", or "cpu"
        normalize: bool = True,
        batch_size: int = 64,
        instruction: Optional[str] = None,  # e.g., "query: " for e5/gte/bge
    ):
        self.model_name  = model_name
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize   = bool(normalize)
        self.batch_size  = int(batch_size)
        self.instruction = instruction or ""

        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Dim can be inferred from a quick run if needed:
        self._dim = None

    def _maybe_norm(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return X
        X = X.astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return (X / norms).astype(np.float32)

    def embed(self, text_or_texts: Union[str, Iterable[str]]) -> Union[List[float], List[List[float]]]:
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else list(text_or_texts)

        # instruction prefix for models that expect it (e.g., e5/gte/bge)
        if self.instruction:
            texts = [f"{self.instruction}{t or ''}" for t in texts]

        # sentence-transformers handles batching internally via encode()
        with torch.inference_mode():
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we'll normalize ourselves for consistency
            )

        if self._dim is None:
            self._dim = embs.shape[1]

        embs = self._maybe_norm(embs)
        if is_single:
            return embs[0].tolist()
        return [row.tolist() for row in embs]
    
# ===============================
# Short-Term Memory (STM)
# ===============================

class ShortTermMemory:
    """
    Your STM with a small change: we inject an embeddings adapter instead of LangChain's class.
    """
    def __init__(self, llm, embeddings_adapter: HFEmbeddingsAdapter, capacity: int = 10, enhance_threshold: int = 3, verbose: bool = False):
        self.llm = llm
        self.emb = embeddings_adapter
        self.verbose = verbose

        self.capacity: int = capacity
        self.short_memories: List[str] = []
        self.short_embeddings: List[List[float]] = []
        self.memory_importance: List[float] = []

        self.enhance_cnt: List[int] = [0 for _ in range(self.capacity)]
        self.enhance_memories: List[List[str]] = [[] for _ in range(self.capacity)]
        self.enhance_threshold: int = enhance_threshold

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r"\n", (text or "").strip())
        lines = [line for line in lines if line.strip()]
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def get_short_term_insight(self, content: str) -> List[str]:
        """
        Ask the LLM for 1-sentence high-level insight from related memories.
        """
        prompt = (
            "There are some memories separated by semicolons (;):\n"
            f"{content}\n\n"
            "Infer a high-level insight about the person's character that is significantly different "
            "from the original memories. Respond in one sentence."
        )
        try:
            r = self.llm.chat.completions.create(
                model=deployment_name,
                temperature=0.2,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": "Return one sentence only."},
                    {"role": "user", "content": prompt},
                ],
            )
            sent = (r.choices[0].message.content or "").strip()
        except Exception:
            sent = "Shows stable preferences with moderate openness to novelty."
        return self._parse_list(sent)

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        a = np.asarray(embedding1, dtype=np.float32)
        b = np.asarray(embedding2, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b) / denom)

    def transfer_memories(self, observation: str):
        transfer_flag = False
        existing = [True for _ in range(len(self.short_memories))]
        memory_content, memory_importance, insight_content = [], [], []

        for idx, memory in enumerate(self.short_memories):
            if self.enhance_cnt[idx] >= self.enhance_threshold or existing[idx] is True:
                existing[idx] = False
                transfer_flag = True
                # combine related
                content = [memory] + self.enhance_memories[idx][:-1] + [observation]
                content = ';'.join(content)
                memory_content.append(memory)
                insight = self.get_short_term_insight(content)
                insight_content.append(insight)

        # remove transferred
        if transfer_flag:
            new_memories, new_embeddings, new_importance = [], [], []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if existing[idx]:
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_embeddings.append(self.short_embeddings[idx])
                    new_importance.append(self.memory_importance[idx])
            self.short_memories = new_memories
            self.short_embeddings = new_embeddings
            self.memory_importance = new_importance
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt

        if len(memory_content) > 0 and len(memory_importance) == 0:
            memory_importance = [0.5]

        return memory_content, memory_importance, insight_content

    def discard_memories(self) -> Optional[str]:
        if len(self.short_memories) > self.capacity:
            memory_dict = {
                self.short_memories[idx]: {
                    'enhance_count': self.enhance_cnt[idx],
                    'importance': self.memory_importance[idx]
                }
                for idx in range(len(self.short_memories) - 1)
            }
            sort_list = sorted(
                memory_dict.keys(),
                key=lambda x: (memory_dict[x]['importance'], memory_dict[x]['enhance_count'])
            )
            find_idx = self.short_memories.index(sort_list[0])
            self.enhance_cnt.pop(find_idx); self.enhance_cnt.append(0)
            self.enhance_memories.pop(find_idx); self.enhance_memories.append([])
            self.memory_importance.pop(find_idx)
            discard_memory = self.short_memories.pop(find_idx)
            self.short_embeddings.pop(find_idx)
            # remove cross-references
            for idx in range(len(self.short_memories)):
                if self.enhance_memories[idx].count(sort_list[0]) != 0:
                    self.enhance_memories[idx].remove(sort_list[0])
                    self.enhance_cnt[idx] -= 1
            return discard_memory
        return None

    def add_stm_memory(self, observation: str, importance: float, op: str):
        const = 0.1
        # embed observation
        observation_embedding = self.emb.embed(observation)

        # enhance matches
        for idx, mem_emb in enumerate(self.short_embeddings):
            similarity = self.cosine_similarity(observation_embedding, mem_emb)
            if idx + 1 == len(self.short_embeddings):
                similarity += const  # primacy
            prob = 1 / (1 + np.exp(-similarity))
            if prob >= 0.7 and random.random() <= prob:
                self.enhance_cnt[idx] += 1
                self.enhance_memories[idx].append(observation)

        mem_content, mem_importance, insight_content = self.transfer_memories(observation)

        if op == "add":
            self.short_memories.append(observation)
            self.memory_importance.append(float(importance))
            self.short_embeddings.append(observation_embedding)
            self.discard_memories()

        return mem_content, mem_importance, insight_content

    # =========================================================
    # 0) PATCH STM so add_task_episode always returns 4 items
    # =========================================================
    def add_task_episode(self, episode: dict, importance: float = 0.5):
        obs_text = json.dumps({
            "book": episode.get("book", {}),
            "predicted_rating": episode.get("predicted_rating"),
            "true_rating": episode.get("true_rating"),
            "trace": episode.get("trace", "")[:2000],
        })
        res = self.add_stm_memory(obs_text, importance, op="add")

        # old STM -> 3 items, new STM -> 4 items
        if len(res) == 3:
            mem_content, mem_importance, insight_content = res
            structured_eps = []
        else:
            mem_content, mem_importance, insight_content, structured_eps = res

        # make sure episodes array exists
        if not hasattr(self, "episodes"):
            self.episodes = [None for _ in range(len(self.short_memories))]
        # we just appended obs_text above, so overwrite last slot
        self.episodes[-1] = episode

        return mem_content, mem_importance, insight_content, structured_eps

    def state_dict(self) -> dict:
        return {
            "capacity": int(getattr(self, "capacity", 10)),
            "enhance_threshold": int(getattr(self, "enhance_threshold", 3)),
            "short_memories": list(getattr(self, "short_memories", [])),
            "memory_importance": [float(x) for x in getattr(self, "memory_importance", [])],
            "enhance_cnt": [int(x) for x in getattr(self, "enhance_cnt", [])],
            "enhance_memories": [list(x) for x in getattr(self, "enhance_memories", [[] for _ in range(self.capacity)])],
            # embeddings are recomputed at runtime; do NOT store vectors here
        }

    def load_state_dict(self, sd: dict):
        self.capacity = int(sd.get("capacity", self.capacity))
        self.enhance_threshold = int(sd.get("enhance_threshold", self.enhance_threshold))
        self.short_memories = list(sd.get("short_memories", []))
        self.memory_importance = [float(x) for x in sd.get("memory_importance", [])]
        self.enhance_cnt = [int(x) for x in sd.get("enhance_cnt", [])]
        self.enhance_memories = [list(x) for x in sd.get("enhance_memories", [[] for _ in range(self.capacity)])]
        # You can lazily rebuild embeddings on first use if needed.

# ===============================
# Long-Term Memory (LTM)
# ===============================

@dataclass
class LTMMemory:
    def __init__(self, content: str, embedding: List[float], importance: float, tags: List[str], ts: float):
        self.content = content
        self.embedding = embedding
        self.importance = importance
        self.tags = tags
        self.ts = ts
        # OPTIONAL: parsed rule if content is JSON
        self.parsed: Optional[Dict[str, Any]] = None
        try:
            obj = json.loads(content)
            # we only treat it as a rule if it has condition/effect
            if isinstance(obj, dict) and "condition" in obj and "effect" in obj:
                self.parsed = obj
        except Exception:
            pass

class LongTermMemory:
    def __init__(self, embeddings: HFEmbeddingsAdapter, capacity: int = 512, decay: float = 0.995):
        self.emb = embeddings
        self.capacity = int(capacity)
        self.decay = float(decay)
        self.store: List[LTMMemory] = []

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # =====================================================
    # plain text add_or_update (your current one)
    # =====================================================
    def add_or_update(
        self,
        text: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        sim_threshold: float = 0.9,
    ):
        tags = tags or []
        new_emb = self.emb.embed(text)

        best_idx = None
        best_sim = -1.0

        for i, m in enumerate(self.store):
            if tags and not (set(tags) & set(m.tags)):
                continue
            sim = self._cos(new_emb, m.embedding)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None and best_sim >= sim_threshold:
            ex = self.store[best_idx]
            ex.content = text
            ex.embedding = new_emb
            ex.importance = max(ex.importance, float(importance))
            ex.tags = list(set(ex.tags) | set(tags))
            ex.ts = time.time()
            # try to parse rule
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "condition" in obj and "effect" in obj:
                    ex.parsed = obj
            except Exception:
                pass
        else:
            mem = LTMMemory(text, new_emb, float(importance), tags, time.time())
            self.store.append(mem)
            self._prune_if_needed()

    # =====================================================
    # NEW: add_or_refine_rule (robust to string conditions)
    # =====================================================
    def add_or_refine_rule(
        self,
        rule: Dict[str, Any],
        importance: float = 0.6,
        tags: Optional[List[str]] = None,
    ):
        """
        rule format (flexible now):
        {
          "condition": {...}  OR  "condition": "some text the LLM wrote",
          "effect": {...},
          "rationale": "...",
          "source_task": "..."
        }
        We only add this rule if there is no MORE SPECIFIC overlapping rule.
        Specificity = number of keys in condition.
        """
        tags = (tags or []) + ["rule/book_rating"]

        # --- normalize condition ---
        raw_condition = rule.get("condition", {})
        if not isinstance(raw_condition, dict):
            # LLM gave us a sentence instead of a dict → wrap it
            raw_condition = {"note": str(raw_condition)}
            rule["condition"] = raw_condition

        note_text = raw_condition.get("note")
        if note_text:
            # ----- GATE 2: dedupe by note text -----
            for m in self.store:
                if "rule/movie_rating" not in m.tags:
                    continue
                # m.content is JSON string
                try:
                    obj = json.loads(m.content)
                except Exception:
                    continue
                cond = obj.get("condition", {})
                if isinstance(cond, dict) and cond.get("note") == note_text:
                    # same rule already exists → just bump importance and stop
                    m.importance = max(m.importance, float(importance))
                    return

        condition = raw_condition
        specificity_new = len(condition)

        overlapping_idx = []
        for i, m in enumerate(self.store):
            # only compare to items that were parsed (rules)
            if not getattr(m, "parsed", None):
                continue
            if not (set(tags) & set(m.tags)):
                continue

            # normalize existing condition too (in case old ones were strings)
            old_cond = m.parsed.get("condition", {})
            if not isinstance(old_cond, dict):
                old_cond = {"note": str(old_cond)}
                m.parsed["condition"] = old_cond

            if self._conditions_overlap(old_cond, condition):
                overlapping_idx.append(i)

        # if any overlapping rule is already as specific or more -> just bump importance
        for i in overlapping_idx:
            existing = self.store[i]
            cond_old = existing.parsed.get("condition", {})
            if not isinstance(cond_old, dict):
                cond_old = {"note": str(cond_old)}
                existing.parsed["condition"] = cond_old

            if len(cond_old) >= specificity_new:
                existing.importance = max(existing.importance, float(importance))
                return

        # otherwise add as new rule
        text = json.dumps(rule, ensure_ascii=False)
        emb = self.emb.embed(text)
        mem = LTMMemory(text, emb, float(importance), tags, time.time())
        mem.parsed = rule
        self.store.append(mem)
        self._prune_if_needed()

    # =====================================================
    # helper: overlap between two condition dicts (robust)
    # =====================================================
    @staticmethod
    def _conditions_overlap(c_old: Dict[str, Any], c_new: Dict[str, Any]) -> bool:
        """
        basic overlap: share at least one key with compatible value

        now robust to accidental strings from LLM
        """
        if not isinstance(c_old, dict):
            c_old = {"note": str(c_old)}
        if not isinstance(c_new, dict):
            c_new = {"note": str(c_new)}

        for k, v in c_old.items():
            if k in c_new:
                v2 = c_new[k]
                # list/list or list/str compatibility
                if isinstance(v, list):
                    if isinstance(v2, list):
                        if any(x in v for x in v2):
                            return True
                    else:
                        if v2 in v:
                            return True
                else:
                    if v == v2:
                        return True
        return False

    # original add kept for raw append
    def add(self, text: str, importance: float = 0.5, tags: Optional[List[str]] = None):
        e = self.emb.embed(text)
        self.store.append(LTMMemory(text, e, float(importance), tags or [], time.time()))
        self._prune_if_needed()

    def bulk_add(self, texts: List[str], importance: float = 0.5, tags: Optional[List[str]] = None):
        for t in texts:
            self.add(t, importance=importance, tags=tags)

    # =====================================================
    # RAG-style retrieve: text query
    # =====================================================
    def retrieve(self, query: str, top_k: int = 5, min_sim: float = 0.15) -> List[LTMMemory]:
        if not self.store:
            return []
        q = self.emb.embed(query)
        scored = [(self._cos(q, m.embedding) * (0.5 + 0.5*m.importance), m) for m in self.store]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored[:top_k] if s >= min_sim]

    # =====================================================
    # NEW: retrieve_for_item -> match conditions to movie features
    # =====================================================
    def _normalize_genres(self, raw) -> str:
        """
        Turn a genre field into a clean, pipe-separated string.
        Handles list or 'Adventure|Animation|Film-Noir' style strings.
        """
        if raw is None:
            return ""
        if isinstance(raw, str):
            parts = raw.split("|")
        else:
            parts = list(raw)
        parts = [str(g).strip() for g in parts if str(g).strip()]
        return "|".join(sorted(set(p.lower() for p in parts)))


    def retrieve_for_item(self,
                      movie: Dict[str, Any],
                      min_score: float = 0.20,
                      top_k: int = None,
                      uid: int = None,   # Optional user isolation
                      ) -> List[LTMMemory]:
        """
        General-purpose high-precision item-aware memory retrieval
        for ALL users.

        Signals used:
        - Semantic similarity (embedding)
        - Memory importance
        - Memory success reliability
        - Genre overlap
        - Sentiment polarity alignment
        - Optional user filtering
        """

        if not self.store:
            return []

        # -------------------------------
        # 1) Build item representation
        # -------------------------------
        title = movie.get("title", "") or movie.get("movie_title", "")
        genres = self._normalize_genres(
            movie.get("genre") or movie.get("genres")
        )
        desc = movie.get("description", "") or movie.get("plot", "")

        movie_text = f"title={title}; genres={genres}; desc={desc}"
        q_emb = self.emb.embed(movie_text)

        movie_genres = set(genres or [])

        scored: List[tuple[float, LTMMemory]] = []

        # -------------------------------
        # 2) Memory scoring loop
        # -------------------------------
        for m in self.store:

            # OPTIONAL: user-specific isolation
            if uid is not None and hasattr(m, "uid"):
                if m.uid != uid:
                    continue

            if m.embedding is None:
                continue

            # --- A) Semantic similarity ---
            base_sim = self._cos(q_emb, m.embedding)  # [-1, 1]

            # --- B) Importance weighting ---
            importance = getattr(m, "importance", 0.5)  # [0,1]
            importance_weight = 0.5 + 0.5 * importance

            # --- C) Success reliability ---
            success = getattr(m, "success", 0)
            success_norm = min(success / 10.0, 1.0)  # clipped [0,1]

            # --- D) Genre overlap ---
            mem_genres = set(getattr(m, "genres", []) or [])
            if movie_genres and mem_genres:
                genre_sim = len(movie_genres & mem_genres) / max(len(movie_genres), 1)
            else:
                genre_sim = 0.0

            # --- E) Sentiment polarity from memory text ---
            mem_text = getattr(m, "content", "").lower()
 
            polarity_boost = 0.0
            if any(w in mem_text for w in ["love", "enjoy", "like", "prefer"]):
                polarity_boost += 0.05
            if any(w in mem_text for w in ["hate", "dislike", "boring", "avoid"]):
                polarity_boost -= 0.05

            # --- F) Tag bonus (your original logic) ---
            tag_bonus = 0.0
            tags = getattr(m, "tags", []) or []
            if "insight" in tags:
                tag_bonus += 0.05
            if "training" in tags:
                tag_bonus += 0.05

            # -------------------------------
            # 3) Final fused score
            # -------------------------------
            score = (
                0.40 * base_sim +
                0.20 * importance_weight +
                0.20 * success_norm +
                0.15 * genre_sim +
                polarity_boost +
                tag_bonus
            )

            if score >= min_score:
                scored.append((score, m))

        # -------------------------------
        # 4) Rank and limit
        # -------------------------------
        scored.sort(key=lambda x: x[0], reverse=True)

        memories = [m for s, m in scored]

        if top_k is not None:
            memories = memories[:top_k]

        return memories


    def decay_and_prune(self, keep: int = 512):
        for m in self.store:
            m.importance *= self.decay
        self.store.sort(key=lambda m: (m.importance, m.ts), reverse=True)
        self.store = self.store[:keep]

    def _prune_if_needed(self):
        if len(self.store) > self.capacity:
            self.decay_and_prune(keep=self.capacity)

    # =====================================================
    # helper to see what's inside
    # =====================================================
    def print_ltm_contents(self, top_k: int = 50):
        items = sorted(self.store, key=lambda m: m.importance, reverse=True)[:top_k]
        for i, m in enumerate(items, 1):
            print(f"[{i}] imp={m.importance:.3f} ts={m.ts:.0f} tags={m.tags}")
            print(f"     {m.content}")

    # =====================================================
    # OPTIONAL: consolidation
    # =====================================================
    def consolidate(self, tag_prefix: Optional[str] = None, sim_threshold: float = 0.95):
        new_store = []
        for m in self.store:
            if tag_prefix and not any(t.startswith(tag_prefix) for t in m.tags):
                new_store.append(m)
                continue

            merged = False
            for n in new_store:
                if tag_prefix and not any(t.startswith(tag_prefix) for t in n.tags):
                    continue
                if set(m.tags) & set(n.tags):
                    sim = self._cos(m.embedding, n.embedding)
                    if sim >= sim_threshold:
                        if m.importance > n.importance:
                            n.content = m.content
                            n.embedding = m.embedding
                            n.importance = m.importance
                            n.parsed = m.parsed
                        n.ts = max(n.ts, m.ts)
                        merged = True
                        break
            if not merged:
                new_store.append(m)
        self.store = new_store
        self._prune_if_needed()

    def state_dict(self, include_vectors: bool = False, precision: int = 4) -> dict:
        items = []
        for m in self.store:
            if include_vectors:
                v = [round(float(x), precision) for x in m.embedding]
            else:
                v = None
            items.append({
                "content": m.content,
                "embedding": v,
                "importance": m.importance,
                "tags": m.tags,
                "ts": m.ts,
            })
        return {
            "capacity": self.capacity,
            "items": items,
        }

    def load_state_dict(self, sd: dict):
        self.capacity = int(sd.get("capacity", self.capacity))
        self.store = []
        for e in sd.get("items", []):
            emb = e.get("embedding")
            if emb is None and "content" in e:
                emb = self.emb.embed(e["content"])
            mem = LTMMemory(
                content=e.get("content", ""),
                embedding=emb,
                importance=float(e.get("importance", 0.5)),
                tags=e.get("tags", []),
                ts=float(e.get("ts", time.time())),
            )
            # try to parse rule back
            try:
                obj = json.loads(mem.content)
                if isinstance(obj, dict) and "condition" in obj and "effect" in obj:
                    mem.parsed = obj
            except Exception:
                pass
            self.store.append(mem)

    def bump_importance(self, mems: list, factor: float = 1.15, max_imp: float = 1.0):
        for m in mems:
            m.importance = min(max_imp, m.importance * factor)

    def consolidate_movie_rating_rules(self, max_rules: int = 15):
        """Keep only the top-N most important movie-rating rules, drop the rest."""
        # pick rules
        rules = [m for m in self.store if "rule/book_rating" in m.tags]
        # sort by importance desc
        rules.sort(key=lambda m: m.importance, reverse=True)
        # keep top-N
        keep = set(rules[:max_rules])
        new_store = []
        for m in self.store:
            if m in keep or "rule/book_rating" not in m.tags:
               new_store.append(m)
        self.store = new_store
