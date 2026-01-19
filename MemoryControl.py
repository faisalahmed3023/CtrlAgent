# HFEmbeddingsAdapter using sentence-transformers
from typing import List, Union, Iterable, Optional
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Iterable, Optional, Tuple
import numpy as np
import time, json, math, random

import json
import time
from typing import Dict, Any, Optional, List

# you already have these imported in your file:
from Meta_CoT.MemoryModule import ShortTermMemory, LongTermMemory, HFEmbeddingsAdapter

# ---------------------------------------------------------------------
# 1) helper prompt for generative IE -> delta rule
# ---------------------------------------------------------------------
DELTA_RULE_PROMPT = """You are a controller that edits user memory for book rating.

You will receive:
1. USER PROFILE
2. BOOK (the item the agent rated)
3. AGENT TRACE (the reasoning the agent used)
4. PREDICTED and TRUE rating

Your job:
- If the predicted rating already matches the true rating, return a rule with "skip": true.
- Otherwise, create the SMALLEST and MOST SPECIFIC condition that explains why the true rating is correct.
- Then specify the effect on rating.

Return ONLY JSON with this shape:
{{
  "skip": false,
  "condition": { ... },
  "effect": { ... },
  "rationale": "...",
  "source_task": "..."
}}

Examples of condition keys you can use:
- "genre_in": ["Comedy","Romance"]
- "needs_narrative_depth": true
- "tone": "emotional"

Examples of effect:
- "rating_bias": +1
- "cap_at": 3
- "prefer": "emotional"

Now fill it.

USER PROFILE:
{user_profile}

BOOK:
{book}

AGENT TRACE:
{trace}

PREDICTED RATING: {pred_rating}
TRUE RATING: {true_rating}
SOURCE TASK: {source_task}
"""

import json, re, time
from typing import Dict, Any, Optional, List

# put this inside the same file, above the class or inside the class as @staticmethod
def _safe_json_from_llm(text: str):
    """
    Try to turn LLM text into JSON.
    1) direct json.loads
    2) regex: first {...}
    3) else return None
    """
    text = (text or "").strip()
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) grab first {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 3) give up
    return None

# ---------------------------------------------------------------------
# 2) controller
# ---------------------------------------------------------------------
class MetaCoTController:
    """
    Controls STM→LTM flow and now ALSO does task-level, non-destructive
    memory refinement when a single interaction fails.
    """
    def __init__(self, policy, embeddings: HFEmbeddingsAdapter, client, deployment_name: str):
        self.pi = policy
        self.emb = embeddings
        self.client = client
        self.deployment_name = deployment_name

        # policies: you can tune these
        self.mem_policies = {
            "perfect":  {"importance": 0.97, "decay_tag": "decay/slow"},
            "good":     {"importance": 0.8,  "decay_tag": "decay/normal"},
            "noisy":    {"importance": 0.45, "decay_tag": "decay/fast"},
        }

    # ---------------- attach ----------------
    def attach_to_user(self, uid: int, stm: 'ShortTermMemory', ltm: 'LongTermMemory'):
        prof = self.pi.user_profiles[uid]
        prof["STM"] = stm
        prof["LTM"] = ltm

    # ============================================================
    # STM observe (no LTM writes here)
    # NOTE: updated to accept the new STM return shape
    # ============================================================
    def observe(self, uid: int, text: str, importance: float = 0.5, op: str = "add") -> dict:
        """
        Send raw observation to STM and get back STM's promoted memories/insights.
        We DO NOT store them in LTM here — that will be done after we see reward
        or after we see the whole episode.
        """
        prof = self.pi.user_profiles[uid]
        stm: 'ShortTermMemory' = prof.get("STM")
        if stm is None:
            return {"stm_memories": [], "stm_importance": [], "stm_insights": [], "stm_episodes": []}

        # your modified STM should now return 4 things
        mem_content, mem_importance, insight_content = stm.add_stm_memory(text, importance, op)

        return {
            "stm_memories": mem_content,
            "stm_importance": mem_importance,
            "stm_insights": insight_content,  # list of lists
            # "stm_episodes": structured_episodes,  # NEW: episodes promoted by STM
        }

    # ============================================================
    # helper: write to LTM safely (text memories)
    # ============================================================
    def _ltm_add_or_update(self, ltm: 'LongTermMemory', text: str, importance: float, tags: List[str], sim_threshold: float = 0.9):
        if hasattr(ltm, "add_or_update"):
            ltm.add_or_update(text=text, importance=importance, tags=tags, sim_threshold=sim_threshold)
        else:
            ltm.add(text=text, importance=importance, tags=tags)

    def _episode_to_delta_rule(self, episode: Dict[str, Any], source_task: str) -> Dict[str, Any]:
        """
            Robust version:
            - build prompt without .format on JSON
            - try to parse LLM output
            - if LLM fails, synthesize a minimal local rule
        """
        user_profile = episode.get("user_profile", "")
        movie = episode.get("book", {}) or {}
        movie_json = json.dumps(movie, ensure_ascii=False)
        trace = (episode.get("trace", "") or "")[:2000]
        pred_rating = episode.get("predicted_rating")
        true_rating = episode.get("true_rating")

        prompt = (
                "You are a controller that edits user memory for book rating.\n\n"
                "You get: USER PROFILE, BOOK, AGENT TRACE, PREDICTED RATING, TRUE RATING.\n"
                "If predicted == true, return JSON: {\"skip\": true}.\n"
                "Else, produce the smallest, most specific condition that would fix the rating next time.\n"
                "Return ONLY JSON.\n\n"
                f"USER PROFILE:\n{user_profile}\n\n"
                f"BOOK:\n{movie_json}\n\n"
                f"AGENT TRACE:\n{trace}\n\n"
                f"PREDICTED RATING: {pred_rating}\n"
                f"TRUE RATING: {true_rating}\n"
                f"SOURCE TASK: {source_task}\n"
        )

        resp = self.client.chat.completions.create(
            model=self.deployment_name,
            temperature=0.2,
            max_tokens=250,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""

        # try to parse the LLM's answer
        data = _safe_json_from_llm(raw)

        if data is None:
            # ---- fallback: build a local minimal rule ----
            genres = movie.get("genres", [])
            bias = 0
            if pred_rating is not None and true_rating is not None:
                bias = true_rating - pred_rating  # negative => model overpredicted
            data = {
                "skip": False,
                "condition": {},
                "effect": {},
                "rationale": "auto-fallback: LLM returned invalid JSON, rule synthesized locally.",
                "source_task": source_task,
            }
            if genres:
                data["condition"]["genre_in"] = genres
            if bias != 0:
                data["effect"]["rating_bias"] = bias
            else:
                # at least make the model not go above true next time
                if true_rating is not None:
                    data["effect"]["cap_at"] = true_rating

        # normalize fields
        data.setdefault("skip", False)
        data.setdefault("condition", {})
        data.setdefault("effect", {})
        data.setdefault("rationale", "controller-added rule")
        data.setdefault("source_task", source_task)
        return data

    def _is_rule_like(self, mem_content):
        """
        Return True if the existing LTM memory content looks like a JSON rule
        (dict with 'condition' and 'effect'), False otherwise.
        Works for both dict objects and JSON strings.
        """
        if isinstance(mem_content, dict):
           return "condition" in mem_content and "effect" in mem_content
        if isinstance(mem_content, str):
           mem_content = mem_content.strip()
           if mem_content.startswith("{") and mem_content.endswith("}"):
                try:
                   obj = json.loads(mem_content)
                   return "condition" in obj and "effect" in obj
                except Exception:
                   return False
        return False


    # ============================================================
    # NEW: handle promoted STM episodes
    # ============================================================
    def process_stm_transfers(self, uid: int, stm_result: dict, loss: Optional[float] = None):
        """
        After calling observe(...) you can call this if it returned structured episodes.
        For each episode, if prediction was wrong, we create a delta rule and
        put it into LTM with add_or_refine_rule so old logic stays.
        """
        prof = self.pi.user_profiles[uid]
        ltm: 'LongTermMemory' = prof["LTM"]
        policy = self.reward_to_policy(loss if loss is not None else 1.5)  # if no loss -> assume noisy
        imp = policy["importance"]

        episodes = stm_result.get("stm_episodes", []) or []
        for ep in episodes:
            pred = ep.get("predicted_rating")
            true = ep.get("true_rating")
            # we only care about supervised episodes
            if true is None:
                continue

            if pred == true:
                # good episode → we can optionally store an insight
                continue

            # bad episode → build delta rule
            source_task = f"task_{int(time.time())}"
            delta_rule = self._episode_to_delta_rule(ep, source_task=source_task)
            # print(delta_rule)
            # if delta_rule.get("skip"):
            #     continue

            # # ===== decide if LTM is rule-friendly =====
            # ltm_has_rule_like = False
            # for m in getattr(ltm, "store", []):
            #     if self._is_rule_like(getattr(m, "content", m)):
            #        ltm_has_rule_like = True
            #        break

            # if ltm_has_rule_like and hasattr(ltm, "add_or_refine_rule"):
            effect = delta_rule.get("effect", {})
            if not effect:       # empty dict → no actual change
               continue
            # safe to use your rule API
            ltm.add_or_refine_rule(
                rule=delta_rule,
                importance=imp,
                tags=["llm-update", policy['decay_tag']],
                )
            # else:
            #     # fallback: store as text
            #     self._ltm_add_or_update(
            #         ltm,
            #         text=json.dumps(delta_rule, ensure_ascii=False),
            #         importance=imp,
            #         tags=["llm-update", policy['decay_tag']],
            #     )

        # also keep your existing decay
        ltm.decay_and_prune()

    # ============================================================
    # RL-style memory update (kept, but slightly simplified)
    # ============================================================
    def rl_memory_update(
        self,
        uid: int,
        observation_text: str,
        discipline_views: dict,
        stm_result: dict,
        reward: float,
    ):
        """
        Your previous logic: store discipline views, STM insights, and observation
        when reward is good. We keep it because it's orthogonal to rule storage.
        """
        prof = self.pi.user_profiles[uid]
        ltm: 'LongTermMemory' = prof["LTM"]
        pol = self.reward_to_policy(reward)
        imp = pol["importance"]
        decay_tag = pol["decay_tag"]

        # # 1) store discipline views as aggregated memories
        # for view_name, data in discipline_views.items():
        #     if not isinstance(data, dict):
        #         continue
        #     pos_list = data.get("positive", []) or []
        #     neg_list = data.get("negative", []) or []

        #     if pos_list:
        #         txt = f"[POS-{view_name}] user tends to like: {', '.join(sorted(set(pos_list)))}"
        #         self._ltm_add_or_update(
        #             ltm,
        #             text=txt,
        #             importance=imp,
        #             tags=[f"view/{view_name}", "pos", "llm-derived", decay_tag],
        #         )
        #     if neg_list:
        #         txt = f"[NEG-{view_name}] user tends to dislike: {', '.join(sorted(set(neg_list)))}"
        #         self._ltm_add_or_update(
        #             ltm,
        #             text=txt,
        #             importance=imp,
        #             tags=[f"view/{view_name}", "neg", "llm-derived", decay_tag],
        #         )

        # 2) store STM insights (NOT raw stm_memories) if reward is ok
        if reward > -1.0:
            for insight_group in stm_result.get("stm_insights", []):
                for ins in insight_group:
                    self._ltm_add_or_update(
                        ltm,
                        text=ins,
                        importance=max(0.6, imp),
                        tags=["insight", decay_tag],
                    )

        # 3) optionally store the observation itself ONLY for good reward
        if reward > -0.5 and observation_text:
            self._ltm_add_or_update(
                ltm,
                text=observation_text,
                importance=max(0.5, imp),
                tags=["obs", decay_tag],
            )

        ltm.decay_and_prune()

    def _llm_refine_memory(self, bad_mem_list, item_meta: dict,
                                pred_rating: int, true_rating: int) -> list[str]:
        """
        bad_mem_list: list of memory strings that guided a wrong prediction.
        Returns:
                A list of refined memory strings (may be same length or different),
                guaranteed to be a Python list even if the LLM output is messy.
        """
        title = item_meta.get("title", "")
        genres = item_meta.get("genres", [])

        # Format original memories nicely for the prompt
        mem_block = "\n".join(f"- {m}" for m in bad_mem_list)

        prompt = (
                "You are rewriting user preference memories for a book rating agent.\n"
                "I will give you:\n"
                "1) the current memories about the user that previously helped rating prediction\n"
                "2) the new book (title + genres)\n"
                "3) the predicted rating and the true rating\n\n"
                "Your task:\n"
                "- Refine, merge, or slightly adjust these memories so that they better capture "
                "  the user's stable preferences and behaviors (both positive and negative),\n"
                "- in a way that should improve future rating predictions on similar books.\n"
                "- Do NOT add totally new facts about the user that are unsupported by the given memories.\n\n"
                "Output format (IMPORTANT):\n"
                "- Return ONLY a valid JSON array of strings (no extra text, no comments).\n"
                "- Example: [\"Memory sentence 1\", \"Memory sentence 2\"]\n\n"
                f"ORIGINAL_MEMORIES:\n{mem_block}\n\n"
                f"BOOK:\nTitle: {title}\nGenres: {', '.join(genres)}\n\n"
                f"Predicted rating: {pred_rating}\nTrue rating: {true_rating}\n"
            )

        resp = self.client.chat.completions.create(
                model=self.deployment_name,
                 temperature=0.3,
         max_tokens=512,
         messages=[
             {
                 "role": "system",
                 "content": (
                     "You rewrite user preference memories for a book-rating agent. "
                     "Always respond with a VALID JSON array of strings and NOTHING else."
                 ),
             },
             {"role": "user", "content": prompt},
         ],
         )

        raw = (resp.choices[0].message.content or "").strip()

        # ---- robust parsing to list[str] ----
        try:
            refined = json.loads(raw)
            if isinstance(refined, str):
               refined = [refined]
            elif not isinstance(refined, list):
               refined = [str(refined)]
        except json.JSONDecodeError:
            # fallback: treat whole output as one memory string
            refined = [raw]

        # ensure all are strings
        refined = [str(m).strip() for m in refined if str(m).strip()]

        return refined

    def _rank_memories_for_item(self, ltm, item_meta: dict, top_k: int = 5):
        """
        Return the LTM memories most relevant to this movie/item.
        We make a query from title + genres and do a normal retrieve.
        """
        title = item_meta.get("title", "")
        genres = item_meta.get("genres", [])
        q = f"rating_prediction; title={title}; genres={','.join(genres)}"
        top = ltm.retrieve(q, top_k=top_k)
        return top


    def apply_rating_feedback(
        self,
        uid: int,
        item_meta: dict,
        pred_rating: int,
        true_rating: int,
        ):
        """
        - find LTM memories that were most relevant for this item
        - if rating matched: reward those memories (bump importance)
        - if rating mismatched: refine ONLY those memories (1–2) and update LTM
        """
        prof = self.pi.user_profiles[uid]
        ltm: LongTermMemory = prof["LTM"]

        # 1) get top memories for this item
        top_mems = self._rank_memories_for_item(ltm, item_meta, top_k=5)

        # perfect or near-perfect prediction → reinforce
        if pred_rating == true_rating:
            # bump importance of exactly the ones that matched this item
            ltm.bump_importance(top_mems, factor=1.12, max_imp=1.0)
            return

        # mismatch → refine 1–2 most relevant ones
        # we pick the top one; you can pick 2
        to_refine = top_mems[:2]

        for m in to_refine:
            refined_text = self._llm_refine_memory(
                bad_mem_text=m.content,
                item_meta=item_meta,
                pred_rating=pred_rating,
                true_rating=true_rating,
            )
            # don't store empty / too generic improvements
            if len(refined_text) < 15:
                continue

            # update the old one instead of adding a new one
            if hasattr(ltm, "add_or_update"):
                ltm.add_or_update(
                    text=refined_text,
                    importance=max(0.7, m.importance),
                    tags=list(set(m.tags) | {"llm-update", "decay/normal"}),
                    sim_threshold=0.88,
                )
            else:
                ltm.add(
                    text=refined_text,
                    importance=max(0.7, m.importance),
                    tags=list(set(m.tags) | {"llm-update", "decay/normal"}),
                )


    # ============================================================
    # reward → policy
    # ============================================================
    def reward_to_policy(self, loss: float) -> dict:
        # you used loss, so we keep that semantics
        if loss == 0.0:
            return self.mem_policies["perfect"]
        elif loss <= 1.0:
            return self.mem_policies["good"]
        else:
            return self.mem_policies["noisy"]

    # ============================================================
    # build task context using rule-aware LTM
    # ============================================================
    def build_task_context(self,
                           uid: int,
                           task: str,
                           item_meta: Optional[Dict[str, Any]] = None,
                           ) -> Dict[str, Any]:
        """
        Build the context for a rating / choice task.

        - Persona: from user row.
        - Top prefs: from profile.
        - Memories: ONLY item-aware LTM memories, filtered by relevance
            via retrieve_for_item (no generic retrieve fallback).
        """
        prof = self.pi.user_profiles[uid]
        ltm: 'LongTermMemory' = prof.get("LTM")

        # persona string from your existing helper
        user_row = self.pi.data.get_user(uid)
        persona = self.pi.persona_description(user_row)

        # top-5 genre preferences (or whatever your prefs dict encodes)
        prefs = dict(sorted(prof.get("preferences", {}).items(),
               key=lambda x: -x[1])[:5])

        memories: List[str] = []

        if ltm is not None and item_meta is not None:
            # Normalize movie dict to the fields retrieve_for_item expects
            movie = {
                "title": item_meta.get("title", ""),
                "genre": item_meta.get("genre") or item_meta.get("genres"),
                "description": item_meta.get("description", ""),
            }
            matched = ltm.retrieve_for_item(movie, min_score=0.54, uid=uid)
            memories = [m.content for m in matched]

        return {
            "task": task,
            "persona": persona,
            "top_prefs": prefs,
            "memories": memories,
        }


    # ============================================================
    # housekeeping (kept)
    # ============================================================
    def housekeeping(self, uid: int):
        prof = self.pi.user_profiles[uid]
        ltm: 'LongTermMemory' = prof.get("LTM")
        if ltm:
            cons = prof["traits"].get("conscientiousness", 0.5)
            old_decay = ltm.decay
            ltm.decay = 0.99 - 0.04*(1.0 - cons)   # [0.95, 0.99]
            ltm.decay_and_prune()
            ltm.decay = old_decay

    # ============================================================
    # store_updates
    # ============================================================
    def store_updates(self, uid: int, updates: List[str], policy: dict):
        prof = self.pi.user_profiles[uid]
        ltm: 'LongTermMemory' = prof["LTM"]
        imp = policy["importance"]
        decay_tag = policy["decay_tag"]

        for upd in updates:
            if hasattr(ltm, "add_or_update"):
                ltm.add_or_update(
                    text=upd,
                    importance=imp,
                    tags=["llm-update", decay_tag],
                    sim_threshold=0.9,
                )
            else:
                ltm.add(
                    text=upd,
                    importance=imp,
                    tags=["llm-update", decay_tag],
                )
        ltm.decay_and_prune()

    # ============================================================
    # state save/load
    # ============================================================
    def state_dict(self) -> dict:
        return {
            "version": 3,
            "routing": {
                "mem_policies": self.mem_policies,
            },
        }

    def load_state_dict(self, sd: dict):
        rt = sd.get("routing", {})
        if "mem_policies" in rt:
            self.mem_policies = rt["mem_policies"]

    def llm_predict_rating(self, memory_text: str, title: str, genres: str) -> dict:
        TEXT_RATER_PROMPT = """
        You are an agent and your task is to rate a movie based on your memory.
        You receive:
        - MEMORY: user persona and user behavioral characteristics
        - MOVIE: title and genres
        
        Infer what rating (1-5) this user would give to this movie.
        Return ONLY valid JSON: {{"rating": <int 1-5>, "explanation": "<short reason>"}}.
        
        MEMORY:
        {memory}
        
        MOVIE:
        title: {title}
        genres: {genres}
        """

        prompt = TEXT_RATER_PROMPT.format(
            memory=memory_text,
            title=title,
            genres=genres,
        )

        resp = self.client.chat.completions.create(
            model=self.deployment_name,
            temperature=0.2,
            max_tokens=200,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        txt = resp.choices[0].message.content.strip()
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            txt = txt[s:e+1]

        data = json.loads(txt)
        # print("data: ", data)

        # clamp rating
        data["rating"] = int(max(1, min(5, data.get("rating", 3))))
   
        if "explanation" not in data:
            data["explanation"] = "LLM provided no explanation."

        return data
