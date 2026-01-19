# ============================
# Save/Load: user profiles & policy meta
# ============================
import os, json, time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict
import math
import pandas as pd

from Meta_CoT.LLM_Policy import LatentToken
from Meta_CoT.Data import Data_Structure
from Meta_CoT.LLM_Policy import LLMPolicy
from Meta_CoT.Policy_mixed import LLMPolicyWithMemoryMixin
from Meta_CoT.Trainer import MetaCoTTrainer
from Meta_CoT.MemoryControl import MetaCoTController
from Meta_CoT.MemoryModule import HFEmbeddingsAdapter, ShortTermMemory, LongTermMemory, LTMMemory
import pickle

os.makedirs("checkpoints", exist_ok=True)

# -------- helpers to (de)serialize numpy safely --------
def _tolist(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [ _tolist(v) for v in x ]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x

def _to_ndarray(x, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype)
    return arr

# -------- memory extraction (optional/lightweight) --------
def _export_stm(stm) -> Optional[Dict[str, Any]]:
    if stm is None:
        return None
    # NOTE: we do NOT persist embeddings (can be recomputed) to keep files small
    out = {
        "capacity": int(getattr(stm, "capacity", 10)),
        "enhance_threshold": int(getattr(stm, "enhance_threshold", 3)),
        "short_memories": list(getattr(stm, "short_memories", [])),
        "memory_importance": [float(x) for x in getattr(stm, "memory_importance", [])],
        "enhance_cnt": [int(x) for x in getattr(stm, "enhance_cnt", [])],
        "enhance_memories": list(getattr(stm, "enhance_memories", [])),
    }
    return out

def _export_ltm(ltm, include_vectors: bool = False, float_precision: int = 4) -> Optional[Dict[str, Any]]:
    if ltm is None:
        return None
    # Expecting ltm to have .items like [{"text":..., "vector":[...]}]
    items = []
    store = getattr(ltm, "store", None) or getattr(ltm, "items", None) or []
    for e in store:
        text = e.get("text", "")
        vec = e.get("vector", None)
        if include_vectors and vec is not None:
            # quantize a bit to reduce size
            v = [ round(float(x), float_precision) for x in vec ]
        else:
            v = None
        items.append({"text": text, "vector": v})
    out = {
        "capacity": int(getattr(ltm, "capacity", 512)),
        "items": items
    }
    return out

def _import_stm(stm_dict, stm_obj):
    if not stm_dict or stm_obj is None:
        return
    stm_obj.capacity = int(stm_dict.get("capacity", stm_obj.capacity))
    stm_obj.enhance_threshold = int(stm_dict.get("enhance_threshold", stm_obj.enhance_threshold))
    stm_obj.short_memories = list(stm_dict.get("short_memories", []))
    stm_obj.memory_importance = [float(x) for x in stm_dict.get("memory_importance", [])]
    stm_obj.enhance_cnt = [int(x) for x in stm_dict.get("enhance_cnt", [])]
    stm_obj.enhance_memories = list(stm_dict.get("enhance_memories", [[] for _ in range(stm_obj.capacity)]))

def _import_ltm(ltm_dict, ltm_obj):
    if not ltm_dict or ltm_obj is None:
        return
    ltm_obj.capacity = int(ltm_dict.get("capacity", ltm_obj.capacity))
    items = []
    for e in ltm_dict.get("items", []):
        items.append({"text": e.get("text",""), "vector": e.get("vector", None)})
    # accept either .store or .items
    if hasattr(ltm_obj, "store"):
        ltm_obj.store = items
    elif hasattr(ltm_obj, "items"):
        ltm_obj.items = items

# -------- user profile save/load --------
def profile_to_dict(policy, uid: int, include_memory: bool = True, include_vectors: bool = False) -> Dict[str, Any]:
    prof = policy.user_profiles[uid]
    latents = []
    for t in prof["latents"]:
        latents.append({
            "mu": _tolist(t.mu),
            "logvar": _tolist(t.logvar)
        })
    out = {
        "uid": int(uid),
        "preferences": {k: float(v) for k, v in prof["preferences"].items()},
        "traits": {k: float(v) for k, v in prof["traits"].items()},
        "latents": latents,
        "memory_log": list(prof.get("memory", []))[:200],  # cap to avoid huge files
    }

    if include_memory:
        # if you attached memory stacks via policy.attach_memory_stack(uid, stm, ltm, controller)
        mem_stack = getattr(policy, "_memory_stacks", {}).get(uid, None)
        if mem_stack:
            out["stm"] = _export_stm(mem_stack.get("stm"))
            out["ltm"] = _export_ltm(mem_stack.get("ltm"), include_vectors=include_vectors)
    return out

def dict_to_profile(policy, d: Dict[str, Any]):
    uid = int(d["uid"])
    # build LatentTokens fresh
    rebuilt_latents = []
    for z in d.get("latents", []):
        mu = _to_ndarray(z["mu"], dtype=np.float32)
        lv = _to_ndarray(z["logvar"], dtype=np.float32)
        rebuilt_latents.append(policy.__class__.__mro__[1].__dict__.get('LatentToken', None) or None)  # placeholder
    # simpler: instantiate LatentToken from your existing class
    from dataclasses import dataclass
    # Access the LatentToken class directly from your implementation:
    # If LatentToken is globally imported:
    rebuilt_latents = []
    for z in d.get("latents", []):
        mu = _to_ndarray(z["mu"], dtype=np.float32)
        lv = _to_ndarray(z["logvar"], dtype=np.float32)
        rebuilt_latents.append(LatentToken(mu=mu, logvar=lv))

    prof = policy.user_profiles[uid]  # ensures a slot exists
    prof["preferences"].clear()
    for k, v in d.get("preferences", {}).items():
        prof["preferences"][k] = float(v)
    for k in prof["traits"]:
        if k in d.get("traits", {}):
            prof["traits"][k] = float(d["traits"][k])
    prof["latents"] = rebuilt_latents
    prof["memory"]  = list(d.get("memory_log", []))


# -------------------------------------------------------------
# Optional helpers for batching (simple examples)
# -------------------------------------------------------------
def sample_users_uniform(data, batch_size: int, rng: np.random.RandomState) -> List[int]:
    user_ids = list(data.users["user_id"].astype(int).values)  # expects data.users DataFrame
    return list(rng.choice(user_ids, size=min(batch_size, len(user_ids)), replace=False))

def sample_candidates_including_gt(data, uid: int, k: int, rng: np.random.RandomState) -> Tuple[List[int], int]:
    """
    Pick a ground-truth positive for uid (rating >= 4) and fill the rest with random items.
    Returns (candidates, gt_item). If no positive exists, falls back to random gt.
    """
    df_pos = data.get_user_rated_items(uid, min_rating=4)
    if not df_pos.empty:
        gt_item = int(df_pos.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])
    else:
        # fallback to random
        gt_item = int(data.movies.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])

    candidates = {gt_item}
    # fill
    while len(candidates) < k:
        mid = int(data.movies.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])
        candidates.add(mid)
    return list(candidates), gt_item


def save_metacot_checkpoint(policy, trainer, path: str = "checkpoints/metacot_qstar.pkl"):
    """
    Save the *minimal* state needed for test-time inference:
      - per-user latent tokens (mu, logvar)
      - user preferences & traits
      - encoder genre embeddings
      - TinyDecoder params (if used)
    """
    serializable_profiles = {}

    for uid, prof in policy.user_profiles.items():
        serializable_profiles[int(uid)] = {
            "preferences": dict(prof["preferences"]),
            "traits": dict(prof["traits"]),
            "memory": list(prof["memory"]),  # already JSON-like
            "latents": [
                {
                    "mu": t.mu.astype(np.float32),
                    "logvar": t.logvar.astype(np.float32),
                }
                for t in prof["latents"]
            ],
        }

    encoder_emb = {
        g: v.astype(np.float32) for g, v in policy.encoder.emb.items()
    }

    if trainer.decoder is not None:
        decoder_state = {
            "raw_alpha": float(trainer.decoder.raw_alpha),
            "raw_tau": float(trainer.decoder.raw_tau),
        }
    else:
        decoder_state = None

    state = {
        "latent_dim": policy.latent_dim,
        "K": policy.K,
        "user_profiles": serializable_profiles,
        "encoder_emb": encoder_emb,
        "decoder": decoder_state,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"Saved Meta-CoT checkpoint to {path}")

def update_memory_success(memory_store, task_ctx, true_rating, pred_rating, ltm):
    """
    Update an ID-indexed memory_store when prediction is correct.

    memory_store format:
        {
          1: {"memory": "<long text A>", "success": 3, "importance": 0.75},
          2: {"memory": "<long text B>", "success": 1, "importance": 0.42},
          ...
        }

    task_ctx["memories"] is a list of long memory texts (strings).

    ltm : LongTermMemory
        Used to read LTMMemory.importance by matching LTMMemory.content == mem_text.

    If true_rating == pred_rating:
      - For each memory in task_ctx["memories"]:
          * If EXACT same text already exists in memory_store["memory"],
                - increment its "success"
                - refresh its "importance" from LTM if found
          * Otherwise, create a new entry with:
                - "memory": mem_text
                - "success": 1
                - "importance": taken from LTM if found, else 0.0
    """

    # Only update when prediction is correct
    if true_rating != pred_rating:
        return memory_store

    memories = task_ctx.get("memories", [])
    if not memories:
        return memory_store

    for mem in memories:
        # ensure it's a string
        mem_text = mem if isinstance(mem, str) else str(mem)

        # ---- look up importance from LTM (by exact content match) ----
        ltm_importance = None
        if ltm is not None:
            for m in ltm.store:
                if m.content == mem_text:
                    ltm_importance = float(m.importance)
                    break

        # 1) search if this exact memory already exists in memory_store
        existing_id = None
        for mid, rec in memory_store.items():
            if rec["memory"] == mem_text:
                existing_id = mid
                break

        if existing_id is not None:
            # already seen: increment success
            memory_store[existing_id]["success"] += 1
            # refresh importance from LTM if we found it
            if ltm_importance is not None:
                memory_store[existing_id]["importance"] = ltm_importance
        else:
            # new memory: assign next integer ID
            new_id = max(memory_store.keys(), default=0) + 1
            memory_store[new_id] = {
                "memory": mem_text,
                "success": 1,
                "importance": ltm_importance if ltm_importance is not None else 0.0,
            }

    return memory_store

# ------------------------------
# 1) Helper: load checkpoint
# ------------------------------
def load_metacot_checkpoint(
    checkpoint_path: str,
    users_path: str,
    movies_path: str,
    ratings_path: str,
    seed: int = 2025,
):
    # Rebuild data
    mdata = Data_Structure(users_path, movies_path, ratings_path)

    # Recreate policy + trainer skeleton
    class MemoryLLMPolicy(LLMPolicy, LLMPolicyWithMemoryMixin):
        pass

    base_policy = MemoryLLMPolicy(
        data=mdata,
        latent_dim=32,     # must match training
        K=5,               # must match training
        lr=1e-2,
        seed=seed,
        use_logprobs=False,
    )

    trainer = MetaCoTTrainer(
        policy=base_policy,
        beta=0.02,
        clip=0.5,
        ema_tau=0.05,
        use_decoder_head=True,
        dec_lr=5e-3,
        enable_traces=False,    # no need to trace during pure eval
        trace_every=9999,
        prm_mix=0.5,
    )

    # Load saved state
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)

    # ---- Restore encoder embeddings ----
    base_policy.encoder.emb = {
        g: np.asarray(v, dtype=np.float32)
        for g, v in state["encoder_emb"].items()
    }

    # ---- Restore user profiles ----
    restored_profiles = {}
    for uid, prof in state["user_profiles"].items():
        latents = []
        for lt in prof["latents"]:
            mu = np.asarray(lt["mu"], dtype=np.float32)
            logvar = np.asarray(lt["logvar"], dtype=np.float32)
            from Meta_CoT.LLM_Policy import LatentToken
            latents.append(LatentToken(mu=mu, logvar=logvar))

        restored_profiles[int(uid)] = {
            "preferences": dict(prof["preferences"]),
            "traits": dict(prof["traits"]),
            "memory": list(prof["memory"]),
            "latents": latents,
        }

    base_policy.user_profiles = restored_profiles

    # ---- Restore decoder, if any ----
    if state.get("decoder") is not None and trainer.decoder is not None:
        dec_state = state["decoder"]
        trainer.decoder.raw_alpha = float(dec_state["raw_alpha"])
        trainer.decoder.raw_tau = float(dec_state["raw_tau"])

    # Rebuild reference profiles for KL (if you ever resume training)
    trainer.ref_profiles = trainer._snapshot_ref_profiles()

    return mdata, base_policy, trainer


# ------------------------------
# 2) Evaluation: simple Hit@k on test
# ------------------------------
def evaluate_hit_at_k(
    policy,
    mdata,
    k: int = 10,
    n_users: int = 100,
    seed: int = 123,
) -> float:
    """
    Simple offline eval:
      - switch data mode to 'test'
      - for a subset of users, sample candidates + GT
      - compute Hit@k where policy.predict_item selects a single item
    """
    rng = np.random.RandomState(seed)
    mdata.set_mode("test")   # <-- use test split

    user_ids = mdata.all_user_ids()
    # rng.shuffle(user_ids)
    user_ids = user_ids[:n_users]

    hits = 0
    total = 0

    for uid in user_ids:
        # sample from test interactions
        cands, gt = sample_candidates_including_gt(mdata, uid, k=k, rng=rng)
        # print("cands:", cands, "gt:", gt)
        if gt is None or len(cands) == 0:
            continue

        pred = policy.predict_item(uid, cands, mdata)
        # print("pred:", pred)
        hits += int(pred == gt)
        total += 1

    hit_rate = hits / max(total, 1)
    print(f"Hit@{k} over {total} evaluated users: {hit_rate:.4f}")
    return hit_rate

def get_movie_dict(mdata, mid: int):
    """
    Convert a movie_id into a rich movie dict with a normalised 'genres' field.
    Assumes mdata.movies has columns: movie_id, title, genre/genres/genre_set, description.
    """
    m = mdata.movies[mdata.movies["item_id"] == mid].iloc[0].to_dict()

    # normalise genres
    if "genres" in m and isinstance(m["genres"], str):
        g = [x.strip() for x in m["genres"].split(" | ") if x.strip()]
        m["genres"] = g
    elif "genre" in m and isinstance(m["genre"], str):
        g = [x.strip() for x in m["genre"].split(" | ") if x.strip()]
        m["genres"] = g
    elif "genre_set" in m:
        m["genres"] = list(m["genre_set"])
    else:
        m["genres"] = []

    return m

def run_memory_training_for_user(
    policy,
    user_id: int,
    mdata,
    train_df,
    mc: MetaCoTController,
    EPOCHS: int = 5,
    memory_store: Optional[dict] = None,
):
    if memory_store is None:
        memory_store = {}

    history = []

    # restrict to this user's rows
    user_train_df = train_df[train_df["user_id"] == user_id].copy()
    if user_train_df.empty:
        print(f"[memory-train] User {user_id} has no train rows, skipping.")
        return memory_store, history

    # pull the attached STM/LTM from the profile
    prof = policy.user_profiles[user_id]
    stm: ShortTermMemory = prof["STM"]
    ltm: LongTermMemory = prof["LTM"]

    for ep in range(EPOCHS):
        train_loss = 0.0
        train_n = 0

        # shuffle user-specific training rows each epoch
        user_train_df = user_train_df.sample(
            frac=1.0, random_state=42 + ep
        ).reset_index(drop=True)

        for _, row in user_train_df.iterrows():
            mid = int(row["item_id"])
            true_rating = int(row["rating"])

            # 1) fetch user + movie meta
            if hasattr(mdata, "get_user"):
                user_row = mdata.get_user(user_id)
            else:
                user_row = mdata.users[
                    mdata.users["user_id"] == user_id
                ].iloc[0].to_dict()

            movie = get_movie_dict(mdata, mid)

            # 2) controller: build task context (RAG from LTM)
            item_meta = {
                "item_id": movie.get("item_id", ""),
                "title": movie.get("title", ""),
                "genres": movie.get("genres", []),
                "description": movie.get("description", ""),
            }
            task_ctx = mc.build_task_context(
                user_id,
                task="rating_prediction",
                item_meta=item_meta,
            )

            # 3) build memory text for rating
            memory_text = task_ctx["persona"] + "\n" + "\n".join(task_ctx["memories"])

            # 4) predict rating via controller's LLM rater
            pred_obj = mc.llm_predict_rating(
                memory_text=memory_text,
                title=movie.get("title", ""),
                genres=",".join(movie.get("genres", [])),
            )

            pred_rating = int(pred_obj["rating"])
            trace_text = pred_obj.get("explanation", "no-explanation")

            # 4a) refine memories when the prediction is wrong
            if task_ctx["memories"] and pred_rating != true_rating:
                refined_list = mc._llm_refine_memory(
                    task_ctx["memories"],
                    item_meta,
                    pred_rating,
                    true_rating,
                )
                for txt in refined_list:
                    ltm.add_or_update(txt, importance=0.8, sim_threshold=0.65)

            # 5) compute loss (L1)
            loss = abs(pred_rating - true_rating)
            train_loss += loss
            train_n += 1
            print(f"[user {user_id}] loss={loss} pred={pred_rating} true={true_rating}")

            # 6) update success statistics
            # assumes you already implemented update_memory_success(...)
            memory_store = update_memory_success(
                memory_store, task_ctx, true_rating, pred_rating, ltm
            )

            # 7) build an episode to feed STM
            episode = {
                "user_profile": mc.pi.persona_description(user_row),
                "movie": {
                    "item_id": mid,
                    "title": movie.get("title", ""),
                    "genres": movie.get("genres", []),
                },
                "trace": trace_text,
                "predicted_rating": pred_rating,
                "true_rating": true_rating,
            }

            (
                stm_mem_content,
                stm_mem_importance,
                stm_insights,
                stm_episodes,
            ) = stm.add_task_episode(episode, importance=0.5)

            # 8) RL-style memory update (your existing controller logic)
            obs_txt = (
                f"user={user_id} movie={movie.get('title','')} "
                f"pred={pred_rating} true={true_rating}"
            )
            mc.rl_memory_update(
                uid=user_id,
                observation_text=obs_txt,
                discipline_views=prof.get("discipline_views", {}) or {},
                stm_result={"stm_insights": stm_insights},
                reward=loss,  # interpreted as loss by controller
            )

            # keep LTM clean (optional: drop transient "obs" memories)
            ltm.store = [m for m in ltm.store if "obs" not in m.tags]

        # end of epoch for this user
        avg_train_loss = train_loss / max(1, train_n)
        print(
            f"[memory-train] User {user_id} Epoch {ep+1}/{EPOCHS} "
            f"| train L1={avg_train_loss:.3f}"
        )

        # housekeeping so LTM doesn’t explode
        mc.housekeeping(user_id)

        history.append(
            {
                "epoch": ep + 1,
                "train_loss": float(avg_train_loss),
                "ltm_size": len(ltm.store),
            }
        )

        # optional: debug print of LTM
        # ltm.print_ltm_contents()

    return memory_store, history

def rebuild_ltm_from_memory_store(
    ltm,
    memory_store,
    success_thr=50,
    top_k=40,
):
    """
    Rebuild LTM for a single user from their memory_store dict.

    memory_store: dict[mid] -> {"memory": str, "success": int, "importance": float, ...}

    - keep only entries with success > success_thr
    - sort by success desc
    - keep at most top_k memories
    """
    # 1) filter
    filtered = {
        mid: rec
        for mid, rec in memory_store.items()
        if rec.get("success", 0) > success_thr
    }

    if not filtered:
        print("[rebuild_ltm_from_memory_store] no memories above threshold; LTM left unchanged")
        return

    # 2) sort by success desc and keep top_k
    items = sorted(filtered.items(), key=lambda x: x[1]["success"], reverse=True)
    items = items[:top_k]

    max_success = max(rec["success"] for _, rec in items)

    # 3) clear existing LTM
    ltm.store = []

    # 4) rebuild
    for mid, rec in items:
        text = rec["memory"]
        base_imp = float(rec.get("importance", 0.5))
        succ = float(rec.get("success", 0.0))

        if max_success > 0:
            success_factor = succ / max_success
        else:
            success_factor = 1.0

        # importance scaled into [0.3, 1.0]-ish
        importance = float(0.3 + 0.7 * base_imp * success_factor)

        emb = ltm.emb.embed(text)
        tags = ["training", "decay/slow"]

        mem = LTMMemory(
            content=text,
            embedding=emb,
            importance=importance,
            tags=tags,
            ts=time.time(),
        )
        ltm.store.append(mem)

    print(f"[rebuild_ltm_from_memory_store] LTM rebuilt with {len(ltm.store)} memories")

def rebuild_all_users_ltm_from_memory_store(
    policy,
    memory_store_by_user,
    success_thr=50,
    top_k=40,
):
    """
    For each user in memory_store_by_user:
      - Rebuild policy.user_profiles[uid]['LTM'] from that user's memory_store.
      - Limit to top_k memories with success > success_thr.
    """
    for uid, user_store in memory_store_by_user.items():
        # keys might be strings in the pickle; make them int if needed
        try:
            int_uid = int(uid)
        except Exception:
            int_uid = uid

        prof = policy.user_profiles.get(int_uid)
        if prof is None:
            print(f"[rebuild_all_users_ltm] uid={int_uid} not in policy.user_profiles, skipping")
            continue

        ltm = prof.get("LTM")
        if ltm is None:
            print(f"[rebuild_all_users_ltm] uid={int_uid} has no LTM, skipping")
            continue

        print(f"[rebuild_all_users_ltm] rebuilding LTM for uid={int_uid}")
        rebuild_ltm_from_memory_store(
            ltm=ltm,
            memory_store=user_store,
            success_thr=success_thr,
            top_k=top_k,
        )

def inspect_user_ltm(policy, uid: int, max_items: int = 10):
    """
    Print a readable view of one user's LTM memories.
    """
    if uid not in policy.user_profiles:
        print(f"[inspect_user_ltm] user {uid} not in policy.user_profiles")
        return

    prof = policy.user_profiles[uid]
    ltm = prof.get("LTM")
    if ltm is None:
        print(f"[inspect_user_ltm] user {uid} has no LTM attached")
        return

    store = getattr(ltm, "store", None)
    if store is None:
        print(f"[inspect_user_ltm] user {uid} LTM has no 'store'")
        return

    print(f"=== LTM for user {uid} ===")
    print(f"total memories: {len(store)}\n")

    for i, m in enumerate(store[:max_items]):
        text = getattr(m, "content", getattr(m, "text", ""))  # content or text
        imp = getattr(m, "importance", None)
        tags = getattr(m, "tags", [])
        print(f"[{i:02d}] importance={imp:.3f}  tags={list(tags)}")
        print("     ", text[:200].replace("\n", " "))
        if len(text) > 200:
            print("      ...")
        print()

def format_movie_text(item_meta: dict) -> str:
    """
    Turn a movie row (meta dict) into a single text string
    for the embedding model.
    """
    title = item_meta.get("title", "")
    genres = item_meta.get("genre") or item_meta.get("genres") or ""
    desc = item_meta.get("description", "")
    return f"title={title}; genres={genres}; desc={desc}"

def safe_json_parse(txt: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the LLM response.
    Looks for the first '{' and last '}' and parses what's inside.
    Returns {} on failure.
    """
    try:
        s = txt.find("{")
        e = txt.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return {}
        return json.loads(txt[s:e+1])
    except Exception:
        return {}
    
# def evaluate_policy_paged_ranking(
#     policy,
#     trainer,
#     mdata,
#     recsys,
#     test_df: pd.DataFrame,
#     k: int = 10,
#     page_size: int = 4,
#     n_pages: int = 10,
#     max_users: int = None,
# ):
#     """
#     Evaluate a policy with paged recommendation using MF-based candidate rankings.

#     For each user:
#       - Candidates = MF ranking over all test items: recsys.get_rankings_over_all_test_items(...)
#       - Limit to page_size * n_pages items.
#       - Show 10 pages (by default), each with `page_size` items.
#       - On each page, call:
#             pred = policy.predict_item(uid, cands, mdata)
#         where `cands` is the list of items on that page.
#       - Remove the picked item from the remaining pool; next page is based on
#         the remaining ranking (i.e., no repeats).
#       - Collect the sequence of chosen items (one per page => up to n_pages picks).
#       - Compute Precision@K, Recall@K, F1@K, Hit@K, NDCG@K based on these picks
#         vs ground-truth items from `test_df`.

#     Parameters
#     ----------
#     policy : object
#         Must implement:
#             predict_item(uid: int, cands: List[int], mdata) -> int
#         returning one of the items in `cands`.
#     mdata : MovieData-like object
#         Used by policy.predict_item. If it has set_mode("test"), we call it.
#     recsys : Recommender
#         Trained MF+BPR Recommender with method:
#             get_rankings_over_all_test_items(return_original_ids=True)
#     test_df : pd.DataFrame
#         Test interactions with ORIGINAL IDs. Must have columns:
#             ['user_id', 'item_id', 'rating'].
#     k : int
#         Cutoff for @K metrics. Should be <= n_pages (since 1 pick per page).
#     page_size : int
#         Number of items per page (e.g., 4).
#     n_pages : int
#         Number of pages user can explore (e.g., 10).
#     max_users : int or None
#         If given, evaluate on at most this many users (first in sorted order).

#     Returns
#     -------
#     metrics : dict
#         {
#           "precision_at_k": float,
#           "recall_at_k": float,
#           "f1_at_k": float,
#           "hit_at_k": float,
#           "ndcg_at_k": float,
#           "num_users": int,
#         }
#     """
#     # --- 1) Put mdata in test mode if possible ---
#     if hasattr(mdata, "set_mode"):
#         mdata.set_mode("test")

#     # --- 2) MF-based rankings over global test pool (ORIGINAL IDs) ---
#     # rankings_by_user_orig: {orig_user_id: np.array([item_id1, item_id2, ...])}
#     rankings_by_user_orig = recsys.get_rankings_over_all_test_items(
#         return_original_ids=True
#     )

#     # --- 3) Ground-truth test items per user (ORIGINAL IDs) ---
#     gt_items_by_user = defaultdict(set)
#     for row in test_df.itertuples(index=False):
#         u = int(row.user_id)
#         i = int(row.item_id)
#         gt_items_by_user[u].add(i)

#     # --- 4) Users that have both rankings and GT ---
#     users = sorted(
#         u for u in rankings_by_user_orig.keys() if u in gt_items_by_user
#     )
#     if max_users is not None and max_users < len(users):
#         users = users[:max_users]

#     # --- 5) Metric accumulators ---
#     sum_prec = 0.0
#     sum_recall = 0.0
#     sum_f1 = 0.0
#     sum_hit = 0.0
#     sum_ndcg = 0.0
#     num_eval_users = 0

#     def dcg_at_k(rels, k):
#         dcg = 0.0
#         for rank, rel in enumerate(rels[:k], start=1):
#             if rel > 0:
#                 dcg += rel / math.log2(rank + 1)
#         return dcg

#     for uid in users:
#         gt_items = gt_items_by_user[uid]
#         if not gt_items:
#             continue

#         ranking = rankings_by_user_orig[uid]
#         if ranking.size == 0:
#             continue

#         # Limit total candidate pool to what user can see across pages
#         max_items = page_size * n_pages
#         candidate_pool = ranking[:max_items].tolist()  # list of item_ids

#         picked_items = []  # sequence of items chosen by policy

#         remaining = list(candidate_pool)

#         for page_idx in range(n_pages):
#             if not remaining:
#                 break

#             # current page candidates: first `page_size` of remaining
#             page_cands = remaining[:page_size]
#             check_gt = list(set(page_cands) & set(gt_items))
#             print(uid, ":", check_gt)

#             logp, probs, raw_scores, gt_index = trainer.item_logprob_given_Z(uid, page_cands, check_gt)
            
#             # Rank items by predicted probability
#             probs = np.asarray(probs)
#             order = np.argsort(-probs)   # descending
#             pred = order[0]
#             # topk_items = {candidates[i] for i in topk_idx}

#             # policy chooses one item from this page
#             # pred = policy.predict_item(uid, page_cands, mdata)
#             pred = int(pred)

#             # safety: if policy returns something outside page_cands, fall back
#             if pred not in page_cands:
#                 # fall back to top item on this page
#                 pred = page_cands[0]

#             picked_items.append(pred)

#             # remove chosen item from remaining candidate pool
#             remaining = [i for i in remaining if i != pred]
#         print(uid, ":", picked_items)

#         # --- compute metrics for this user ---
#         if not picked_items:
#             continue

#         k_eff = min(k, len(picked_items))
#         topk = picked_items[:k_eff]
#         topk_set = set(topk)

#         hits = len(topk_set & gt_items)

#         precision = hits / float(k_eff)
#         recall = hits / float(len(gt_items))
#         if precision + recall > 0:
#             f1 = 2 * precision * recall / (precision + recall)
#         else:
#             f1 = 0.0
#         hit_flag = 1.0 if hits > 0 else 0.0

#         # NDCG@k based on picked_items order
#         rels = [1 if item in gt_items else 0 for item in topk]
#         dcg = dcg_at_k(rels, k_eff)
#         ideal_rels = [1] * min(k_eff, len(gt_items))
#         idcg = dcg_at_k(ideal_rels, k_eff)
#         ndcg = dcg / idcg if idcg > 0 else 0.0

#         sum_prec += precision
#         sum_recall += recall
#         sum_f1 += f1
#         sum_hit += hit_flag
#         sum_ndcg += ndcg
#         num_eval_users += 1

#     if num_eval_users == 0:
#         print("No users with both rankings and ground-truth test items.")
#         return {
#             "precision_at_k": 0.0,
#             "recall_at_k": 0.0,
#             "f1_at_k": 0.0,
#             "hit_at_k": 0.0,
#             "ndcg_at_k": 0.0,
#             "num_users": 0,
#         }

#     metrics = {
#         "precision_at_k": sum_prec / num_eval_users,
#         "recall_at_k": sum_recall / num_eval_users,
#         "f1_at_k": sum_f1 / num_eval_users,
#         "hit_at_k": sum_hit / num_eval_users,
#         "ndcg_at_k": sum_ndcg / num_eval_users,
#         "num_users": num_eval_users,
#     }

#     print(
#         f"[Paged Eval @ K={k}, pages={n_pages}, page_size={page_size}] "
#         f"Precision: {metrics['precision_at_k']:.4f}, "
#         f"Recall: {metrics['recall_at_k']:.4f}, "
#         f"F1: {metrics['f1_at_k']:.4f}, "
#         f"Hit: {metrics['hit_at_k']:.4f}, "
#         f"NDCG: {metrics['ndcg_at_k']:.4f}, "
#         f"Users: {metrics['num_users']}"
#     )

#     return metrics

def evaluate_policy_paged_ranking(
    policy,
    trainer,
    mdata,
    recsys,
    test_df: pd.DataFrame,
    k: int = 10,
    page_size: int = 4,
    n_pages: int = 10,
    max_users: int = None,
):
    """
    Evaluate a policy with paged recommendation using MF-based candidate rankings.

    NEW behavior:
      - The user flips pages until they land on a page that contains at least
        one ground-truth item.
      - Only on pages where page_cands ∩ GT != ∅ do we perform a prediction.
      - Pages with no GT are treated as pure exploration: items are shown once,
        then skipped and never reappear.
    """
    # --- 1) Put mdata in test mode if possible ---
    if hasattr(mdata, "set_mode"):
        mdata.set_mode("test")

    # --- 2) MF-based rankings over global test pool (ORIGINAL IDs) ---
    rankings_by_user_orig = recsys.get_rankings_over_all_test_items(
        return_original_ids=True
    )

    # --- 3) Ground-truth test items per user (ORIGINAL IDs) ---
    gt_items_by_user = defaultdict(set)
    for row in test_df.itertuples(index=False):
        u = int(row.user_id)
        i = int(row.item_id)
        gt_items_by_user[u].add(i)

    # --- 4) Users that have both rankings and GT ---
    users = sorted(
        u for u in rankings_by_user_orig.keys() if u in gt_items_by_user
    )
    if max_users is not None and max_users < len(users):
        users = users[:max_users]

    # --- 5) Metric accumulators ---
    sum_prec = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_hit = 0.0
    sum_ndcg = 0.0
    num_eval_users = 0

    def dcg_at_k(rels, k):
        dcg = 0.0
        for rank, rel in enumerate(rels[:k], start=1):
            if rel > 0:
                dcg += rel / math.log2(rank + 1)
        return dcg

    for uid in users:
        gt_items = gt_items_by_user[uid]
        if not gt_items:
            continue

        ranking = rankings_by_user_orig[uid]
        if ranking.size == 0:
            continue

        # Limit total candidate pool to what user can see across pages
        max_items = page_size * n_pages
        candidate_pool = ranking[:max_items].tolist()  # list of item_ids

        picked_items = []  # sequence of items chosen by policy
        remaining = list(candidate_pool)

        for page_idx in range(n_pages):
            if not remaining:
                break

            # current page candidates: first `page_size` of remaining
            page_cands = remaining[:page_size]
            check_gt = list(set(page_cands) & gt_items)
            # print(uid, "check_gt:", check_gt)

            # --- NEW: if this page has no GT, user just explores and moves on ---
            if not check_gt:
                # user saw these items but they were all irrelevant → skip the page
                remaining = remaining[page_size:]
                continue

            # at least one GT item on this page → we go for prediction
            # if multiple GT items on the page, just pick one as the "gt_for_page"
            gt_for_page = check_gt[0]

            # # Meta-CoT prediction: GT-free, uses branches
            # probs, raw_scores = trainer.candidate_probs_given_Z(
            #     uid, page_cands
            # )

            # # Rank items by predicted probability
            # probs = np.asarray(probs)
            # order = np.argsort(-probs)   # descending
            # pred_idx = int(order[0])
            # pred = page_cands[pred_idx]  # convert index -> item_id

            # Meta-ToT prediction: GT-free, uses branches
            pred = trainer.predict_page_meta_tot(uid, page_cands, K_branches=4, agg="mean")

            # safety: if something goes weird, fall back to first item
            if pred not in page_cands:
                pred = page_cands[0]

            picked_items.append(pred)

            # remove chosen item from remaining candidate pool
            # (other items on this page stay in pool, same as your original code)
            remaining = [i for i in remaining if i != pred]

        # print(uid, "picked_items:", picked_items)

        # --- compute metrics for this user ---
        if not picked_items:
            continue

        k_eff = min(k, len(picked_items))
        topk = picked_items[:k_eff]
        topk_set = set(topk)

        hits = len(topk_set & gt_items)

        precision = hits / float(k_eff)
        recall = hits / float(len(gt_items))
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        hit_flag = 1.0 if hits > 0 else 0.0

        # NDCG@k based on picked_items order
        rels = [1 if item in gt_items else 0 for item in topk]
        dcg = dcg_at_k(rels, k_eff)
        ideal_rels = [1] * min(k_eff, len(gt_items))
        idcg = dcg_at_k(ideal_rels, k_eff)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        sum_prec += precision
        sum_recall += recall
        sum_f1 += f1
        sum_hit += hit_flag
        sum_ndcg += ndcg
        num_eval_users += 1

    if num_eval_users == 0:
        print("No users with both rankings and ground-truth test items.")
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "hit_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "num_users": 0,
        }

    metrics = {
        "precision_at_k": sum_prec / num_eval_users,
        "recall_at_k": sum_recall / num_eval_users,
        "f1_at_k": sum_f1 / num_eval_users,
        "hit_at_k": sum_hit / num_eval_users,
        "ndcg_at_k": sum_ndcg / num_eval_users,
        "num_users": num_eval_users,
    }

    print(
        f"[Paged Eval @ K={k}, pages={n_pages}, page_size={page_size}] "
        f"Precision: {metrics['precision_at_k']:.4f}, "
        f"Recall: {metrics['recall_at_k']:.4f}, "
        f"F1: {metrics['f1_at_k']:.4f}, "
        f"Hit: {metrics['hit_at_k']:.4f}, "
        f"NDCG: {metrics['ndcg_at_k']:.4f}, "
        f"Users: {metrics['num_users']}"
    )

    return metrics
