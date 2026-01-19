from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional

class Data_Structure:
    def __init__(self, users_path: str = "",
                       movies_path: str = "",
                       ratings_path: str = "",
                       like_threshold: int = 4):
        self.like_threshold = like_threshold

        # Load
        self.users   = pd.read_csv(users_path)
        self.movies  = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        # Normalize column names
        self.users.columns   = [c.strip().lower() for c in self.users.columns]
        self.movies.columns  = [c.strip().lower() for c in self.movies.columns]
        self.ratings.columns = [c.strip().lower() for c in self.ratings.columns]

        # Basic type safety
        for col in ("user_id",):
            if col in self.users:   self.users[col]   = self.users[col].astype(int)
            if col in self.ratings: self.ratings[col] = self.ratings[col].astype(int)
        if "item_id" in self.movies:
            self.movies["item_id"] = self.movies["item_id"].astype(int)
        if "item_id" in self.ratings:
            self.ratings["item_id"] = self.ratings["item_id"].astype(int)
        if "rating" in self.ratings:
            self.ratings["rating"] = self.ratings["rating"].astype(int)

        # Precompute genre sets
        self.movies["genre"] = self.movies["genre"].fillna("")
        self.movies["genre_set"] = self.movies["genre"].apply(
            lambda s: set(g.strip() for g in str(s).split(" | ") if g.strip())
        )

        # Fast lookups
        self.movie_by_id: Dict[int, pd.Series] = {
            int(r.item_id): r for _, r in self.movies.iterrows()
        }
        self.user_by_id: Dict[int, pd.Series] = {
            int(r.user_id): r for _, r in self.users.iterrows()
        }

        # ---- NEW: split-related state ----
        self.train_ratings: Optional[pd.DataFrame] = None
        self.val_ratings:   Optional[pd.DataFrame] = None
        self.test_ratings:  Optional[pd.DataFrame] = None

        # mode can be: "full" (use self.ratings), "train", "val", "test"
        self.mode: str = "full"

    # ---- NEW: helper to get active ratings according to mode ----
    def _get_active_ratings(self) -> pd.DataFrame:
        if self.mode == "full" or self.train_ratings is None:
            return self.ratings
        if self.mode == "train":
            if self.train_ratings is None:
                raise RuntimeError("Train split not initialized. Call dataset_split() first.")
            return self.train_ratings
        if self.mode == "val":
            if self.val_ratings is None:
                raise RuntimeError("Val split not initialized. Call dataset_split() first.")
            return self.val_ratings
        if self.mode == "test":
            if self.test_ratings is None:
                raise RuntimeError("Test split not initialized. Call dataset_split() first.")
            return self.test_ratings
        raise ValueError(f"Unknown mode: {self.mode}")

    # ---- NEW: public API to change mode ----
    def set_mode(self, mode: str) -> None:
        """
        Set the active mode for rating-based operations.

        mode ∈ {"full", "train", "val", "test"}.
        - "full": use all ratings (self.ratings)
        - "train"/"val"/"test": use the corresponding split (requires dataset_split() called)
        """
        mode = mode.lower()
        if mode not in {"full", "train", "val", "test"}:
            raise ValueError(f"Invalid mode {mode}. Use 'full', 'train', 'val', or 'test'.")
        # If user picks a split before splitting, fail loudly
        if mode in {"train", "val", "test"} and self.train_ratings is None:
            raise RuntimeError("You must call dataset_split() before using train/val/test mode.")
        self.mode = mode

    # Convenience accessors
    def all_user_ids(self) -> List[int]:
        return self.users["user_id"].astype(int).tolist()

    def all_movie_ids(self) -> List[int]:
        return self.movies["item_id"].astype(int).tolist()

    def get_user(self, uid: int) -> Optional[pd.Series]:
        return self.user_by_id.get(int(uid))
    
    def get_genres_by_id(self, item_ids):
        """
        Get genre of items by item id.
        """
        # return [self.items[item_id]["genre"] for item_id in item_ids]
        return [
            genre
            for item_id in item_ids
            for genre in self.movies["item_id"][item_id]["genre"].split('|')
        ]

    def get_movie(self, mid: int) -> Optional[pd.Series]:
        return self.movie_by_id.get(int(mid))

    def movie_genres(self, mid: int) -> Set[str]:
        row = self.get_movie(mid)
        return set() if row is None else set(row["genre_set"])

    # Positives / preferences (NOW mode-aware)
    def user_positive_items(self, uid: int, thr: Optional[int] = None) -> List[int]:
        thr = self.like_threshold if thr is None else thr
        ratings = self._get_active_ratings()
        df = ratings.query("user_id == @uid and rating >= @thr")
        return df["item_id"].astype(int).tolist()

    def user_pos_df(self, uid: int, thr: Optional[int] = None) -> pd.DataFrame:
        thr = self.like_threshold if thr is None else thr
        ratings = self._get_active_ratings()
        return ratings.query("user_id == @uid and rating >= @thr").copy()

    def infer_user_genre_prefs(self, uid: int, thr: Optional[int] = None) -> Dict[str, float]:
        """Return normalized genre distribution for a user's positives (on current mode)."""
        thr = self.like_threshold if thr is None else thr
        pos_df = self.user_pos_df(uid, thr=thr)
        counts: Dict[str, int] = {}
        for mid in pos_df["item_id"].astype(int).tolist():
            for g in self.movie_genres(mid):
                counts[g] = counts.get(g, 0) + 1
        total = sum(counts.values())
        return {} if total == 0 else {g: c / total for g, c in counts.items()}

    def top_genres(self, uid: int, n: int = 3, thr: Optional[int] = None) -> List[str]:
        prefs = self.infer_user_genre_prefs(uid, thr=thr)
        return [g for g, _ in sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:n]]

    # Retrieve all rated items (NOW mode-aware)
    def get_user_rated_items(self, uid: int, min_rating: Optional[int] = None) -> pd.DataFrame:
        """
        Return all movies rated by the user on the *current mode*'s ratings,
        optionally filtered by minimum rating.

        Args:
            uid (int): user ID.
            min_rating (int, optional): minimum rating threshold (e.g., 4).
                                        If None, returns all rated items.

        Returns:
            pd.DataFrame: merged DataFrame of (item_id, title, genre, rating)
        """
        ratings = self._get_active_ratings()
        df = ratings.query("user_id == @uid").copy()
        if min_rating is not None:
            df = df[df["rating"] >= min_rating]

        merged = df.merge(
            self.movies[["item_id", "title", "genre", "genre_set"]],
            on="item_id", how="left"
        )
        return merged.reset_index(drop=True)

    # ---- UPDATED: dataset_split stores splits and is mode-aware ----
    def dataset_split(
        self,
        df: Optional[pd.DataFrame] = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train/val/test such that:
          - Items do NOT overlap between splits.
          - Row counts roughly follow the given ratios.
          - Best-effort: each user with >= 3 distinct items gets at least
            one interaction in each split.

        Also updates:
            self.train_ratings, self.val_ratings, self.test_ratings
        and leaves mode unchanged (you can call set_mode() after this).

        Args:
            df: Interactions DataFrame. If None, uses self.ratings.
            user_col: Name of the user id column.
            item_col: Name of the item/movie id column.
            train_ratio, val_ratio, test_ratio: Fractions that must sum to 1.
            random_state: Seed for reproducibility.

        Returns:
            (train_df, val_df, test_df)
        """
        # Use full ratings as default interactions
        if df is None:
            df = self.ratings

        splits = ["train", "val", "test"]

        # ---- 1) Check ratios ----
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # ---- 2) Per-item row counts ----
        item_counts = df[item_col].value_counts().to_dict()
        items = list(item_counts.keys())

        rng = np.random.default_rng(random_state)
        rng.shuffle(items)

        total_rows = len(df)
        targets = {
            "train": total_rows * train_ratio,
            "val":   total_rows * val_ratio,
            "test":  total_rows * test_ratio,
        }
        split_rows = {s: 0 for s in splits}
        item_to_split: Dict[int, str] = {}

        # ---- 3) Initial greedy assignment (row-based) ----
        for it in items:
            remaining = {s: targets[s] - split_rows[s] for s in splits}
            best_split = max(remaining, key=remaining.get)
            item_to_split[it] = best_split
            split_rows[best_split] += item_counts[it]

        # ---- 4) Ensure each split has at least one item (if possible) ----
        for s in splits:
            has_item = any(item_to_split[it] == s for it in items)
            if not has_item:
                # move a small item from the largest split
                donor = max(splits, key=lambda x: split_rows[x])
                donor_items = [it for it in items if item_to_split[it] == donor]
                move_item = min(donor_items, key=lambda it: item_counts[it])
                item_to_split[move_item] = s
                split_rows[donor] -= item_counts[move_item]
                split_rows[s]     += item_counts[move_item]

        # ---- 5) Best-effort per-user coverage ----
        n_splits = len(splits)

        user_items_map = (
            df.groupby(user_col)[item_col]
            .apply(lambda x: x.unique().tolist())
            .to_dict()
        )

        for uid, u_items in user_items_map.items():
            # If user has fewer items than splits, cannot cover all splits
            if len(u_items) < n_splits:
                continue

            # Count this user's items per split
            user_split_counts = {s: 0 for s in splits}
            for it in u_items:
                s = item_to_split[it]
                user_split_counts[s] += 1

            missing_splits = [s for s in splits if user_split_counts[s] == 0]
            if not missing_splits:
                continue

            for s_missing in missing_splits:
                # we can only move from splits where user has >1 items
                candidate_items = [
                    it for it in u_items
                    if user_split_counts[item_to_split[it]] > 1
                ]
                if not candidate_items:
                    break  # can't fix this user further

                # move the smallest item by count
                cand = min(candidate_items, key=lambda it: item_counts[it])
                old_split = item_to_split[cand]

                item_to_split[cand] = s_missing
                split_rows[old_split] -= item_counts[cand]
                split_rows[s_missing] += item_counts[cand]

                user_split_counts[old_split] -= 1
                user_split_counts[s_missing] += 1

        # ---- 6) Build final DataFrames ----
        item_split_series = df[item_col].map(item_to_split)

        train_df = df[item_split_series == "train"].reset_index(drop=True)
        val_df   = df[item_split_series == "val"].reset_index(drop=True)
        test_df  = df[item_split_series == "test"].reset_index(drop=True)

        # ---- store splits on the object ----
        self.train_ratings = train_df
        self.val_ratings   = val_df
        self.test_ratings  = test_df

        return train_df, val_df, test_df
    
    def get_user_num(self):
        """
        Return the number of users.
        """
        return len(self.all_user_ids())

    def get_item_num(self):
        """
        Return the number of items.
        """
        return len(self.all_movie_ids())
