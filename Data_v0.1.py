from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional, Any
from contextlib import contextmanager

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

        # Check structure and bind real column names
        self._check_structure_and_bind_columns()

        # Precompute genre sets using the actual genre column name
        self.movies["genre_set"] = self.movies[self.item_genre_col].fillna("").apply(
            lambda s: set(
                g.strip()
                for g in str(s).replace("|", " | ").split(" | ")
                if g.strip()
            )
        )

        # Fast lookups
        self.movie_by_id: Dict[int, pd.Series] = {
            int(r[self.item_id_col]): r for _, r in self.movies.iterrows()
        }

        self.user_by_id: Dict[int, pd.Series] = {
            int(r[self.user_id_col]): r for _, r in self.users.iterrows()
        }

        # ---- NEW: split-related state ----
        self.train_ratings: Optional[pd.DataFrame] = None
        self.val_ratings:   Optional[pd.DataFrame] = None
        self.test_ratings:  Optional[pd.DataFrame] = None

        # mode can be: "full" (use self.ratings), "train", "val", "test"
        self.mode: str = "full"

    def _check_structure_and_bind_columns(self) -> None:
        """
        Check the structure of users, movies/items, and ratings files.

        The class does NOT rename columns.
        Instead, it stores the real column names based on expected order.

        Expected structures:
          users   : user_id, name, gender, age, status
          movies  : item_id, title, genre, description
          ratings : user_id, item_id, rating
        """

        # Clean only extra spaces from column names, but keep original meaning
        self.users.columns = [str(c).strip() for c in self.users.columns]
        self.movies.columns = [str(c).strip() for c in self.movies.columns]
        self.ratings.columns = [str(c).strip() for c in self.ratings.columns]

        # Required number of columns
        if self.users.shape[1] < 5:
            raise ValueError(
                f"User file must contain at least 5 columns in this order: "
                f"user_id, name, gender, age, status. "
                f"Found columns: {list(self.users.columns)}"
            )

        if self.movies.shape[1] < 4:
            raise ValueError(
                f"Item/movie file must contain at least 4 columns in this order: "
                f"item_id, title, genre, description. "
                f"Found columns: {list(self.movies.columns)}"
            )

        if self.ratings.shape[1] < 3:
            raise ValueError(
                f"Rating file must contain at least 3 columns in this order: "
                f"user_id, item_id, rating. "
                f"Found columns: {list(self.ratings.columns)}"
            )

        # Bind column roles based on position, not column name
        self.user_id_col = self.users.columns[0]
        self.user_name_col = self.users.columns[1]
        self.user_gender_col = self.users.columns[2]
        self.user_age_col = self.users.columns[3]
        self.user_status_col = self.users.columns[4]

        self.item_id_col = self.movies.columns[0]
        self.item_title_col = self.movies.columns[1]
        self.item_genre_col = self.movies.columns[2]
        self.item_description_col = self.movies.columns[3]

        self.rating_user_id_col = self.ratings.columns[0]
        self.rating_item_id_col = self.ratings.columns[1]
        self.rating_col = self.ratings.columns[2]

        # Type conversion using actual column names
        self.users[self.user_id_col] = self.users[self.user_id_col].astype(int)
        self.movies[self.item_id_col] = self.movies[self.item_id_col].astype(int)

        self.ratings[self.rating_user_id_col] = self.ratings[self.rating_user_id_col].astype(int)
        self.ratings[self.rating_item_id_col] = self.ratings[self.rating_item_id_col].astype(int)
        self.ratings[self.rating_col] = pd.to_numeric(
            self.ratings[self.rating_col], errors="coerce"
        )

        self.ratings = self.ratings.dropna(subset=[self.rating_col])
        self.ratings[self.rating_col] = self.ratings[self.rating_col].astype(float)

        # Remove duplicates using actual column names
        self.users = self.users.drop_duplicates(
            subset=[self.user_id_col]
        ).reset_index(drop=True)

        self.movies = self.movies.drop_duplicates(
            subset=[self.item_id_col]
        ).reset_index(drop=True)

        self.ratings = self.ratings.drop_duplicates(
            subset=[self.rating_user_id_col, self.rating_item_id_col]
        ).reset_index(drop=True)

        # Keep only ratings whose users/items exist
        valid_users = set(self.users[self.user_id_col])
        valid_items = set(self.movies[self.item_id_col])

        self.ratings = self.ratings[
            self.ratings[self.rating_user_id_col].isin(valid_users)
            & self.ratings[self.rating_item_id_col].isin(valid_items)
        ].reset_index(drop=True)

        print("Structure checking completed.")
        print("User column mapping:")
        print({
            "user_id": self.user_id_col,
            "name": self.user_name_col,
            "gender": self.user_gender_col,
            "age": self.user_age_col,
            "status": self.user_status_col,
        })

        print("Item column mapping:")
        print({
            "item_id": self.item_id_col,
            "title": self.item_title_col,
            "genre": self.item_genre_col,
            "description": self.item_description_col,
        })

        print("Rating column mapping:")
        print({
            "user_id": self.rating_user_id_col,
            "item_id": self.rating_item_id_col,
            "rating": self.rating_col,
        })

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
        return self.users[self.user_id_col].astype(int).tolist()

    def all_movie_ids(self) -> List[int]:
        return self.movies[self.item_id_col].astype(int).tolist()

    def get_user(self, uid: int) -> Optional[pd.Series]:
        return self.user_by_id.get(int(uid))
    
    def get_genres_by_id(self, item_ids):
        """
        Get genre of items by item id.
        """
        genres = []

        for item_id in item_ids:
            row = self.get_movie(int(item_id))

            if row is None:
                continue

            genres.extend(list(row["genre_set"]))

        return genres

    def get_movie(self, mid: int) -> Optional[pd.Series]:
        return self.movie_by_id.get(int(mid))

    def movie_genres(self, mid: int) -> Set[str]:
        row = self.get_movie(mid)
        return set() if row is None else set(row["genre_set"])

    # Positives / preferences (NOW mode-aware)
    def user_positive_items(self, uid: int, thr: Optional[int] = None) -> List[int]:
        thr = self.like_threshold if thr is None else thr
        ratings = self._get_active_ratings()

        df = ratings[
            (ratings[self.rating_user_id_col] == uid)
            & (ratings[self.rating_col] >= thr)
        ]

        return df[self.rating_item_id_col].astype(int).tolist()


    def user_pos_df(self, uid: int, thr: Optional[int] = None) -> pd.DataFrame:
        thr = self.like_threshold if thr is None else thr
        ratings = self._get_active_ratings()

        return ratings[
            (ratings[self.rating_user_id_col] == uid)
            & (ratings[self.rating_col] >= thr)
        ].copy()


    def infer_user_genre_prefs(self, uid: int, thr: Optional[int] = None) -> Dict[str, float]:
        """
        Return normalized genre distribution for a user's positive items.
        """
        thr = self.like_threshold if thr is None else thr
        pos_df = self.user_pos_df(uid, thr=thr)

        counts: Dict[str, int] = {}

        for mid in pos_df[self.rating_item_id_col].astype(int).tolist():
            for g in self.movie_genres(mid):
                counts[g] = counts.get(g, 0) + 1

        total = sum(counts.values())

        return {} if total == 0 else {g: c / total for g, c in counts.items()}


    def top_genres(self, uid: int, n: int = 3, thr: Optional[int] = None) -> List[str]:
        prefs = self.infer_user_genre_prefs(uid, thr=thr)

        return [
            g for g, _ in sorted(
                prefs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]
        ]

    # Retrieve all rated items (NOW mode-aware)
    def get_user_rated_items(
        self,
        uid: int,
        min_rating: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Return all items rated by the user using the original column names.
        """

        ratings = self._get_active_ratings()

        df = ratings[
            ratings[self.rating_user_id_col] == uid
        ].copy()

        if min_rating is not None:
            df = df[df[self.rating_col] >= min_rating]

        merged = df.merge(
            self.movies[
                [
                    self.item_id_col,
                    self.item_title_col,
                    self.item_genre_col,
                    self.item_description_col,
                    "genre_set",
                ]
            ],
            left_on=self.rating_item_id_col,
            right_on=self.item_id_col,
            how="left"
        )

        return merged.reset_index(drop=True)

    # ---- UPDATED: dataset_split stores splits and is mode-aware ----
    def dataset_split(
        self,
        df: Optional[pd.DataFrame] = None,
        user_col: Optional[str] = None,
        item_col: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if df is None:
            df = self.ratings

        if user_col is None:
            user_col = self.rating_user_id_col

        if item_col is None:
            item_col = self.rating_item_id_col

        splits = ["train", "val", "test"]

        total_ratio = train_ratio + val_ratio + test_ratio

        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        item_counts = df[item_col].value_counts().to_dict()
        items = list(item_counts.keys())

        rng = np.random.default_rng(random_state)
        rng.shuffle(items)

        total_rows = len(df)

        targets = {
            "train": total_rows * train_ratio,
            "val": total_rows * val_ratio,
            "test": total_rows * test_ratio,
        }

        split_rows = {s: 0 for s in splits}
        item_to_split: Dict[int, str] = {}

        for it in items:
            remaining = {s: targets[s] - split_rows[s] for s in splits}
            best_split = max(remaining, key=remaining.get)

            item_to_split[it] = best_split
            split_rows[best_split] += item_counts[it]

        for s in splits:
            has_item = any(item_to_split[it] == s for it in items)

            if not has_item:
                donor = max(splits, key=lambda x: split_rows[x])
                donor_items = [it for it in items if item_to_split[it] == donor]

                move_item = min(donor_items, key=lambda it: item_counts[it])

                item_to_split[move_item] = s
                split_rows[donor] -= item_counts[move_item]
                split_rows[s] += item_counts[move_item]

        n_splits = len(splits)

        user_items_map = (
            df.groupby(user_col)[item_col]
            .apply(lambda x: x.unique().tolist())
            .to_dict()
        )

        for uid, u_items in user_items_map.items():
            if len(u_items) < n_splits:
                continue

            user_split_counts = {s: 0 for s in splits}

            for it in u_items:
                s = item_to_split[it]
                user_split_counts[s] += 1

            missing_splits = [s for s in splits if user_split_counts[s] == 0]

            if not missing_splits:
                continue

            for s_missing in missing_splits:
                candidate_items = [
                    it for it in u_items
                    if user_split_counts[item_to_split[it]] > 1
                ]

                if not candidate_items:
                    break

                cand = min(candidate_items, key=lambda it: item_counts[it])
                old_split = item_to_split[cand]

                item_to_split[cand] = s_missing

                split_rows[old_split] -= item_counts[cand]
                split_rows[s_missing] += item_counts[cand]

                user_split_counts[old_split] -= 1
                user_split_counts[s_missing] += 1

        item_split_series = df[item_col].map(item_to_split)

        train_df = df[item_split_series == "train"].reset_index(drop=True)
        val_df = df[item_split_series == "val"].reset_index(drop=True)
        test_df = df[item_split_series == "test"].reset_index(drop=True)

        self.train_ratings = train_df
        self.val_ratings = val_df
        self.test_ratings = test_df

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
    
    def get_user_all_item_metadata(self, uid: int) -> pd.DataFrame:
        """
        Return all item metadata for all items rated/interacted by a given user.
        Uses the actual column names detected by _check_structure_and_bind_columns().
        """

        ratings = self._get_active_ratings()
    
        # Get this user's rating records
        user_ratings = ratings[
            ratings[self.rating_user_id_col] == int(uid)
        ].copy()
    
        if user_ratings.empty:
            print(f"No rated items found for user {uid}")
            return pd.DataFrame()

        # Merge with item/movie metadata
        merged = user_ratings.merge(
            self.movies[
                [
                    self.item_id_col,
                    self.item_title_col,
                    self.item_genre_col,
                    self.item_description_col,
                    "genre_set",
                ]
            ],
            left_on=self.rating_item_id_col,
            right_on=self.item_id_col,
            how="left"
        )

        return merged.reset_index(drop=True)
    
    def get_user_info(self, uid: int) -> Optional[pd.Series]:
        """
        Return the corresponding user information for a given user ID.
        Uses the actual user-id column detected by _check_structure_and_bind_columns().
        """

        uid = int(uid)

        user_row = self.users[
            self.users[self.user_id_col] == uid
        ]

        if user_row.empty:
            print(f"No user information found for user {uid}")
            return None

        return user_row.iloc[0]
    
    def schema_summary(self) -> Dict[str, Any]:
        """
        Return the detected column mapping and dataset sizes.
        Useful for debugging before training.
        """

        return {
            "users": {
                "shape": self.users.shape,
                "user_id_col": self.user_id_col,
                "name_col": self.user_name_col,
                "gender_col": self.user_gender_col,
                "age_col": self.user_age_col,
                "status_col": self.user_status_col,
                "columns": list(self.users.columns),
            },
            "items": {
                "shape": self.movies.shape,
                "item_id_col": self.item_id_col,
                "title_col": self.item_title_col,
                "genre_col": self.item_genre_col,
                "description_col": self.item_description_col,
                "columns": list(self.movies.columns),
            },
            "ratings": {
                "shape": self.ratings.shape,
                "user_id_col": self.rating_user_id_col,
                "item_id_col": self.rating_item_id_col,
                "rating_col": self.rating_col,
                "columns": list(self.ratings.columns),
            },
            "mode": self.mode,
            "has_split": self.train_ratings is not None,
        }

    def get_item_metadata(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Return item metadata as a clean dictionary using dynamic column names.
        """

        row = self.get_movie(int(item_id))

        if row is None:
            return None

        return {
            "item_id": int(row[self.item_id_col]),
            "title": str(row[self.item_title_col]),
            "genre": str(row[self.item_genre_col]),
            "description": str(row[self.item_description_col]),
            "genre_set": set(row["genre_set"]) if "genre_set" in row else set(),
        }

    def user_split_summary(self, uid: int) -> Dict[str, Any]:
        """
        Show how many interactions/positives a user has in full/train/val/test.
        Useful for checking whether one-user debugging is meaningful.
        """

        uid = int(uid)

        def summarize(df: Optional[pd.DataFrame], name: str) -> Dict[str, Any]:
            if df is None:
                return {
                    "split": name,
                    "num_interactions": 0,
                    "num_positives": 0,
                    "positive_items": [],
                }

            user_df = df[df[self.rating_user_id_col] == uid].copy()
            pos_df = user_df[user_df[self.rating_col] >= self.like_threshold].copy()

            return {
                "split": name,
                "num_interactions": int(len(user_df)),
                "num_positives": int(len(pos_df)),
                "positive_items": pos_df[self.rating_item_id_col].astype(int).tolist(),
            }

        return {
            "uid": uid,
            "full": summarize(self.ratings, "full"),
            "train": summarize(self.train_ratings, "train"),
            "val": summarize(self.val_ratings, "val"),
            "test": summarize(self.test_ratings, "test"),
        }

    def check_split_integrity(self) -> Dict[str, Any]:
        """
        Check overlap between train/val/test interactions and items.
        """

        if self.train_ratings is None or self.val_ratings is None or self.test_ratings is None:
            raise RuntimeError("Call dataset_split() first.")

        item_col = self.rating_item_id_col

        train_items = set(self.train_ratings[item_col].astype(int))
        val_items = set(self.val_ratings[item_col].astype(int))
        test_items = set(self.test_ratings[item_col].astype(int))

        return {
            "train_size": int(len(self.train_ratings)),
            "val_size": int(len(self.val_ratings)),
            "test_size": int(len(self.test_ratings)),
            "train_val_item_overlap": len(train_items & val_items),
            "train_test_item_overlap": len(train_items & test_items),
            "val_test_item_overlap": len(val_items & test_items),
        }

    @contextmanager
    def use_mode(self, mode: str):
        """
        Temporarily switch active rating mode.
        Example:
            with mdata.use_mode("train"):
                policy.extract_ground_truth_profile(uid)
        """

        old_mode = self.mode
        self.set_mode(mode)

        try:
            yield
        finally:
            self.set_mode(old_mode)

    def get_user_positive_metadata(
        self,
        uid: int,
        split: str = "train",
        thr: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Return positive item metadata for a user from a selected split.
        split ∈ {"full", "train", "val", "test"}.
        """

        thr = self.like_threshold if thr is None else thr

        old_mode = self.mode
        self.set_mode(split)

        try:
            df = self.get_user_rated_items(uid, min_rating=thr)
        finally:
            self.set_mode(old_mode)

        return df
