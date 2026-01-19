import Meta_CoT.utlis as utlis
from typing import Any, Dict, Set, List, Tuple, Iterable
import math
from collections import Counter, defaultdict
import difflib

def calculate_correctness(true_ratings, pred_ratings):
    """
    Computes correctness using the formula:
    
        Correctness = (1/N) * Σ I(true == pred)
    
    where I(true == pred) = 1 if equal, else 0.

    Parameters
    ----------
    true_ratings : list or array of ints
        Ground-truth ratings.
    pred_ratings : list or array of ints
        Predicted ratings.

    Returns
    -------
    correctness : float
        Proportion of exact matches.
    """
    if len(true_ratings) != len(pred_ratings):
        raise ValueError("true_ratings and pred_ratings must have the same length")

    total = len(true_ratings)
    correct = sum(int(t == p) for t, p in zip(true_ratings, pred_ratings))

    return correct / total if total > 0 else 0.0


def f1_memo_retrieval(all_gold_ids, all_pred_ids):
    """
    Memory Retrieval F1.
    The model must pick one or more correct contents option memory IDs.

    Args
    ----
    Gold: which memory entries should be retrieved for a given item query (indices annotate).
    Pred: which entries your retrieval component actually returns from predicted the result.
    all_gold_ids, all_pred_ids: list[list[int] or set[int]]
      - gold_ids[k]: indices of memories that WERE used for item k
                           (from task_ctx["memories"])
      - pred_ids[k]: indices of memories that COULD be used for item k
                           (from LLM prompt)

    Returns: (precision, recall, f1)
    """
    tp = fp = fn = 0

    for gold, pred in zip(all_gold_ids, all_pred_ids):
        gold_set = set(gold)
        pred_set = set(pred)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def llm_select_candidate_memories_for_item(
    client,
    ltm,
    item_meta: Dict[str, Any],
    true_rating: int,
    model_name: str = "your-deployment-name-here",
) -> List[int]:
    """
    Ask the LLM which memories from ltm.store *could* be used to explain/predict
    the given true_rating for this item.

    Returns a list of memory indices (w.r.t. ltm.store).
    """

    # 1) Build a compact indexed memory list
    indexed_mems = []
    for idx, mem_obj in enumerate(ltm.store):
        text = getattr(mem_obj, "content", "")
        if not text:
            continue
        # keep it short-ish; you can truncate or summarize if needed
        indexed_mems.append({"id": idx, "text": text})

    # 2) Build a prompt
    title = item_meta.get("title", "")
    genres = item_meta.get("genres", []) or item_meta.get("genre", [])
    desc = item_meta.get("description", "")

    # Make genres a simple string
    if isinstance(genres, list):
        genres_str = ", ".join(map(str, genres))
    else:
        genres_str = str(genres)

    mem_str_lines = []
    for m in indexed_mems:
        mem_str_lines.append(f"- id={m['id']}: {m['text']}")
    mem_block = "\n".join(mem_str_lines)

    user_prompt = f"""
You are helping to choose which user long-term memories are most relevant
for explaining a movie rating decision perfectly.

You are given:
1) A movie description (title, genres, description).
2) The TRUE rating that the user gave to this movie (on a 1-5 scale).
3) A list of user memories, each with an integer id and a short text.

Your job:
- Select the IDs of the memories that are genuinely helpful for predicting
  or explaining WHY the user gave this TRUE rating to this movie.
- Prefer memories about similar genres, themes, or behavioral patterns that
  match this rating.
- Ignore memories that are off-topic or irrelevant.

Return ONLY valid JSON of the form:
{{
  "selected_ids": [<int>, <int>, ...]
}}

Movie:
- title: {title}
- genres: {genres_str}
- description: {desc}

True rating: {true_rating}

Candidate memories:
{mem_block}
    """.strip()

    # 3) Call the LLM (adapt this to your Azure/OpenAI client)
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=2096,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise JSON generator. "
                    "Always return valid JSON with a 'selected_ids' array of integers."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    parsed = utlis.safe_json_parse(raw)

    if not parsed or "selected_ids" not in parsed:
        return []

    ids = parsed.get("selected_ids", [])
    # Filter to ints only
    candidate_ids = []
    for x in ids:
        try:
            candidate_ids.append(int(x))
        except Exception:
            continue
    return candidate_ids


def build_memory_item_sets(
    all_used_ids: List[List[int]],
    all_candidate_ids: List[List[int]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    For each memory index m, build:
      - gold_items_for_mem[m]: set of example indices where m WAS used (gold)
      - pred_items_for_mem[m]: set of example indices where m was predicted as usable (candidate)

    all_used_ids[k]      : list of memory indices actually used for example k
    all_candidate_ids[k] : list of memory indices LLM says could be used for example k
    """
    gold_items_for_mem: Dict[int, Set[int]] = defaultdict(set)
    pred_items_for_mem: Dict[int, Set[int]] = defaultdict(set)

    for ex_idx, (used, cand) in enumerate(zip(all_used_ids, all_candidate_ids)):
        for mid in used:
            gold_items_for_mem[mid].add(ex_idx)
        for mid in cand:
            pred_items_for_mem[mid].add(ex_idx)

    return gold_items_for_mem, pred_items_for_mem

def memory_writing_f1_per_memory(
    gold_items_for_mem: Dict[int, Set[int]],
    pred_items_for_mem: Dict[int, Set[int]],
):
    """
    Compute Memory-Writing-style F1 per memory index.
    How well does this memory’s conceptual content (its topic) line up with the actual region of item-meta space (movies & ratings) where it was helpful?

    High-F1 memories = clean, well-aligned item-meta concepts.
    Low-F1 memories = noisy or misaligned memories (maybe too generic or inconsistent).

    Returns:
      mem_scores: dict[mid] = {"precision": P, "recall": R, "f1": F1}
      macro_f1  : average F1 over memories with at least one gold or pred item
      micro_f1  : TP/FP/FN aggregated over all memories
    """
    mem_scores: Dict[int, Dict[str, float]] = {}
    all_mids = set(gold_items_for_mem.keys()) | set(pred_items_for_mem.keys())

    total_tp = total_fp = total_fn = 0
    f1_values = []

    for mid in all_mids:
        gold_items = gold_items_for_mem.get(mid, set())
        pred_items = pred_items_for_mem.get(mid, set())

        tp = len(gold_items & pred_items)
        fp = len(pred_items - gold_items)
        fn = len(gold_items - pred_items)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        mem_scores[mid] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # include only memories that had some activity
        if (tp + fp + fn) > 0:
            f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    # micro-F1 across memories
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1        = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0 else 0.0
    )

    return mem_scores, macro_f1, micro_f1

def recall_at_k_for_example(
    gold_ids: List[int],
    pred_ids: List[int],
    k: int
) -> float:
    """
    LOCOMO-style: 1 if any of top-k pred_ids hits a gold_id, else 0.
    If no gold_ids, returns NaN (you can choose to treat differently).
    """
    if not gold_ids:
        return float("nan")

    gold_set = set(gold_ids)
    topk = pred_ids[:k]
    return 1.0 if any(m in gold_set for m in topk) else 0.0


def recall_at_k_over_lists(
    all_gold: List[List[int]],
    all_pred: List[List[int]],
    ks: Iterable[int] = (1, 3, 5, 10, 20)
) -> Dict[int, float]:
    """
    Mean R@k over all examples (ignoring NaNs).
    """
    results = {}
    for k in ks:
        vals = []
        for gold_ids, pred_ids in zip(all_gold, all_pred):
            r = recall_at_k_for_example(gold_ids, pred_ids, k)
            if not math.isnan(r):
                vals.append(r)
        results[k] = sum(vals) / len(vals) if len(vals) > 0 else float("nan")
    return results

# ---------- Rating label mapping (for F1) ----------
def rating_to_label(r: int) -> str:
    """
    Map 1–5 rating -> class label for F1.
    Adjust if you want a different mapping.
    """
    if r <= 2:
        return "neg"
    elif r == 3:
        return "neu"
    else:
        return "pos"


def precision_recall_f1(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """
    Micro-averaged precision/recall/F1 over multi-class labels.
    """
    assert len(y_true) == len(y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for t, p in zip(y_true, y_pred):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def rating_regression_metrics(
    true_ratings: List[int],
    pred_ratings: List[int]
) -> Dict[str, float]:
    """
    MAE, RMSE, Hit@exact, Hit@±1
    """
    assert len(true_ratings) == len(pred_ratings)
    n = len(true_ratings)
    if n == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "hit_exact": float("nan"),
            "hit_pm1": float("nan"),
        }

    abs_errors = []
    sq_errors = []
    hits_exact = 0
    hits_pm1 = 0

    for t, p in zip(true_ratings, pred_ratings):
        err = p - t
        abs_errors.append(abs(err))
        sq_errors.append(err * err)
        if p == t:
            hits_exact += 1
        if abs(err) <= 1:
            hits_pm1 += 1

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    hit_exact = hits_exact / n
    hit_pm1 = hits_pm1 / n

    return {
        "mae": mae,
        "rmse": rmse,
        "hit_exact": hit_exact,
        "hit_pm1": hit_pm1,
    }


def text_similarity(a: str, b: str) -> float:
    """
    Simple similarity between two strings: 0..1.
    Uses SequenceMatcher; you can replace with something fancier later.
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def dedupe_memories_by_similarity(
    used_texts,
    used_indices,
    ltm,
    sim_threshold: float = 0.8,
):
    """
    Given:
      - used_texts: list[str]   from task_ctx["memories"]
      - used_indices: list[int] indices into ltm.store (same order as used_texts)

    Returns:
      - new_used_texts, new_used_indices
        where near-duplicates (similarity >= sim_threshold) are removed,
        keeping the more important memory (by .importance or .success).
    """
    n = len(used_texts)
    keep = [True] * n

    # Precompute importance scores for each memory
    importance = []
    for idx in used_indices:
        mem_obj = ltm.store[idx]
        # Try importance, then success, else 0.0
        imp = getattr(mem_obj, "importance", None)
        if imp is None:
            imp = getattr(mem_obj, "success", 0.0)
        importance.append(float(imp))

    # Pairwise compare and mark less important duplicates
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            sim = text_similarity(used_texts[i], used_texts[j])
            if sim >= sim_threshold:
                # remove the less important one
                if importance[i] >= importance[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is removed; no need to compare it further

    new_used_texts = [t for t, k in zip(used_texts, keep) if k]
    new_used_indices = [ix for ix, k in zip(used_indices, keep) if k]
    return new_used_texts, new_used_indices

def recall_at_ks(all_gold_ids, all_pred_ids, ks=(1, 3, 5, 10)):
    """
    Returns a dict: {k: recall@k}
    """
    assert len(all_gold_ids) == len(all_pred_ids)

    # Prepare containers
    recalls_per_k = {k: [] for k in ks}

    for gold, pred in zip(all_gold_ids, all_pred_ids):
        gold_set = set(gold)
        if not gold_set:
            continue

        for k in ks:
            topk_pred = set(pred[:k])
            hit = len(gold_set & topk_pred)
            recalls_per_k[k].append(hit / len(gold_set))

    # Average over all samples
    avg_recalls = {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in recalls_per_k.items()
    }
    return avg_recalls

def extract_memory_context_from_indexed_ltm(candidate_ids, ltm):
    """
    candidate_ids: list[int] -> indices over ltm.store
    ltm.store: list[LTMMemory]

    Returns:
        memory_context (str)
        used_memories (list[LTMMemory])
    """

    selected = []

    n = len(ltm.store)

    for idx in candidate_ids:
        if isinstance(idx, int) and 0 <= idx < n:
            selected.append(ltm.store[idx])

    # ---- Build prompt memory block ----
    lines = []
    for i, m in enumerate(selected):
        content = getattr(m, "content", "")
        importance = getattr(m, "importance", 0.0)
        success = getattr(m, "success", None)

        if success is not None:
            lines.append(
                f"[{candidate_ids[i]}] {content} "
                f"(importance={importance:.3f}, success={success})"
            )
        else:
            lines.append(
                f"[{candidate_ids[i]}] {content} "
                f"(importance={importance:.3f})"
            )

    memory_context = "\n".join(lines)

    return memory_context, selected

