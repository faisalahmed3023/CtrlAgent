import numpy as np
from typing import List, Tuple
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split



class EmbeddingRatingHeadClassifier:
    """
    5-class classifier: rating ∈ {1,2,3,4,5}
    from (memory_text, movie_text) using HFEmbeddingsAdapter.
    """
    def __init__(self, emb_adapter, C: float = 1.0):
        self.emb = emb_adapter
        self.model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            class_weight="balanced",  # important if rating distribution is skewed
        )
        self._fitted = False

    def _encode_pair(self, memory_text: str, movie_text: str) -> np.ndarray:
        mem_vec = np.array(self.emb.embed(memory_text), dtype=np.float32)
        mov_vec = np.array(self.emb.embed(movie_text), dtype=np.float32)

        cos_sim = float(np.dot(mem_vec, mov_vec))

        feat = np.concatenate([
            mem_vec,
            mov_vec,
            mem_vec * mov_vec,
            np.array([cos_sim], dtype=np.float32),
        ])
        # Simpler feature vector: [u, v, cos_sim]
        # feat = np.concatenate([mem_vec,
        #                    mov_vec,
        #                    np.array([cos_sim], dtype=np.float32)])
        return feat

    def fit(self,
            pairs: List[Tuple[str, str]],
            ratings: List[float]):
        X = np.stack([self._encode_pair(m, i) for (m, i) in pairs])
        y = np.array(ratings, dtype=int)  # ratings must be 1..5 ints
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, memory_text: str, movie_text: str) -> dict:
        assert self._fitted, "Call fit() first."
        x = self._encode_pair(memory_text, movie_text).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]   # shape (n_classes,)
        classes = self.model.classes_           # e.g., array([1,2,3,4,5])
        return {int(c): float(p) for c, p in zip(classes, probs)}

    def predict_rating(self, memory_text: str, movie_text: str) -> int:
        probs = self.predict_proba(memory_text, movie_text)
        # choose argmax class
        return max(probs.items(), key=lambda kv: kv[1])[0]
    

class EmbeddingRatingRegressorHead:
    """
    Regression head: rating ∈ [1,5] (continuous)
    from (memory_text, movie_text) using HFEmbeddingsAdapter.
    """
    def __init__(self, emb_adapter):
        self.emb = emb_adapter
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
        self._fitted = False

    def _encode_pair(self, memory_text: str, movie_text: str) -> np.ndarray:
        mem_vec = np.array(self.emb.embed(memory_text), dtype=np.float32)
        mov_vec = np.array(self.emb.embed(movie_text), dtype=np.float32)
        cos_sim = float(np.dot(mem_vec, mov_vec))

        # richer non-linear-friendly features
        feat = np.concatenate([
            mem_vec,
            mov_vec,
            mem_vec * mov_vec,
            np.array([cos_sim], dtype=np.float32),
        ])
        return feat

    def fit(self, pairs, ratings):
        X = np.stack([self._encode_pair(m, i) for (m, i) in pairs])
        y = np.array(ratings, dtype=float)
        self.model.fit(X, y)
        self._fitted = True

    def predict_score(self, memory_text: str, movie_text: str) -> float:
        assert self._fitted, "Call fit() first."
        x = self._encode_pair(memory_text, movie_text).reshape(1, -1)
        score = float(self.model.predict(x)[0])
        return float(np.clip(score, 1.0, 5.0))

    def predict_rating(self, memory_text: str, movie_text: str) -> int:
        score = self.predict_score(memory_text, movie_text)
        return int(np.clip(round(score), 1, 5))
    

class EmbeddingXGBRatingHead:
    """
    XGBoost regression head: rating ∈ [1,5] (continuous),
    predicted from (memory_text, movie_text) using HFEmbeddingsAdapter.
    """

    def __init__(self, emb_adapter):
        self.emb = emb_adapter
        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        self._fitted = False

    def _encode_pair(self, memory_text: str, movie_text: str) -> np.ndarray:
        """
        Feature vector:
          - normalized memory embedding
          - normalized movie embedding
          - elementwise product
          - absolute difference
          - cosine similarity scalar
        """
        mem_vec = np.array(self.emb.embed(memory_text), dtype=np.float32)
        mov_vec = np.array(self.emb.embed(movie_text), dtype=np.float32)

        # L2 normalize to stabilize
        mem_norm = np.linalg.norm(mem_vec) + 1e-8
        mov_norm = np.linalg.norm(mov_vec) + 1e-8
        mem_vec = mem_vec / mem_norm
        mov_vec = mov_vec / mov_norm

        cos_sim = float(np.dot(mem_vec, mov_vec))
        prod_vec = mem_vec * mov_vec
        diff_vec = np.abs(mem_vec - mov_vec)

        feat = np.concatenate(
            [
                mem_vec,
                mov_vec,
                prod_vec,
                diff_vec,
                np.array([cos_sim], dtype=np.float32),
            ]
        )
        return feat

    def fit(self,
            pairs: List[Tuple[str, str]],
            ratings: List[float],
            use_class_balance: bool = True):
        """
        Train the XGBRegressor on (memory_text, movie_text) → rating.
        """
        X = np.stack([self._encode_pair(m, i) for (m, i) in pairs])
        y = np.array(ratings, dtype=float)

        sample_weight = None
        if use_class_balance:
            # Inverse frequency weighting by rating value
            counts = Counter(y)
            sample_weight = np.array([1.0 / counts[r] for r in y], dtype=float)

        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        self._fitted = True

    def predict_score(self, memory_text: str, movie_text: str) -> float:
        """
        Predict a continuous rating in [1,5].
        """
        assert self._fitted, "Call fit() first."
        x = self._encode_pair(memory_text, movie_text).reshape(1, -1)
        score = float(self.model.predict(x)[0])
        # clamp to rating range
        return float(np.clip(score, 1.0, 5.0))

    def predict_rating(self, memory_text: str, movie_text: str) -> int:
        """
        Predict a discrete rating 1–5 by rounding the score.
        """
        score = self.predict_score(memory_text, movie_text)
        return int(np.clip(round(score), 1, 5))


class EmbeddingLGBMOrdinalHead:
    """
    LightGBM Ordinal Regression for ratings 1–5.
    Predicts 4 ordered thresholds:
        y>=2, y>=3, y>=4, y>=5
    """

    def __init__(self, emb_adapter):
        self.emb = emb_adapter
        self.models = []  # 4 binary models

    def _encode_pair(self, memory_text: str, movie_text: str) -> np.ndarray:
        mem_vec = np.array(self.emb.embed(memory_text), dtype=float)
        mov_vec = np.array(self.emb.embed(movie_text), dtype=float)

        # normalize
        mem_vec = mem_vec / (np.linalg.norm(mem_vec) + 1e-9)
        mov_vec = mov_vec / (np.linalg.norm(mov_vec) + 1e-9)

        prod_vec = mem_vec * mov_vec
        diff_vec = np.abs(mem_vec - mov_vec)
        cos_sim = np.dot(mem_vec, mov_vec)

        return np.concatenate([mem_vec, mov_vec, prod_vec, diff_vec,
                               np.array([cos_sim], dtype=float)])

    def fit(self, pairs, ratings, learning_rate=0.05):
        X = np.stack([self._encode_pair(m,i) for (m,i) in pairs])
        y = np.array(ratings, dtype=int)

        thresholds = [2,3,4,5]
        self.models = []

        for t in thresholds:
            # create binary label y >= t
            y_bin = (y >= t).astype(int)

            train_data = lgb.Dataset(X, label=y_bin)

            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "learning_rate": learning_rate,
                "num_leaves": 63,
                "max_depth": -1,
                "min_data_in_leaf": 30,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 3,
                "metric": "binary_logloss",
                "verbose": -1,
            }

            model = lgb.train(params, train_data, num_boost_round=300)
            self.models.append(model)

    def predict_rating(self, memory_text, movie_text):
        x = self._encode_pair(memory_text, movie_text).reshape(1, -1)

        th_preds = []
        for model in self.models:
            p = model.predict(x)[0]
            th_preds.append(p > 0.5)   # threshold probability

        # rating = 1 + number of thresholds passed
        rating = 1 + sum(th_preds)
        return int(np.clip(rating, 1, 5))


class EmbeddingLGBMOrdinalHeadE:
    """
    LightGBM Ordinal Regression for ratings 1–5.
    Trains 4 binary models for thresholds: y>=2, y>=3, y>=4, y>=5,
    with strong regularisation + early stopping.
    """

    def __init__(self, emb_adapter):
        self.emb = emb_adapter
        self.models = []   # one lgb.Booster per threshold
        self.thresholds = [2, 3, 4, 5]

    def _encode_pair(self, memory_text: str, movie_text: str) -> np.ndarray:
        mem_vec = np.array(self.emb.embed(memory_text), dtype=float)
        mov_vec = np.array(self.emb.embed(movie_text), dtype=float)

        # L2-normalise
        mem_vec = mem_vec / (np.linalg.norm(mem_vec) + 1e-9)
        mov_vec = mov_vec / (np.linalg.norm(mov_vec) + 1e-9)

        prod_vec = mem_vec * mov_vec
        diff_vec = np.abs(mem_vec - mov_vec)
        cos_sim = float(np.dot(mem_vec, mov_vec))

        feat = np.concatenate([
            mem_vec,
            mov_vec,
            prod_vec,
            diff_vec,
            np.array([cos_sim], dtype=float),
        ])
        return feat

    def fit(self,
            pairs,
            ratings,
            valid_frac: float = 0.2,
            random_state: int = 42):
        """
        Train 4 binary LGBM models with internal train/val split
        + early stopping for each threshold.
        """
        X = np.stack([self._encode_pair(m, i) for (m, i) in pairs])
        y = np.array(ratings, dtype=int)

        self.models = []

        for t in self.thresholds:
            # binary target: y >= t
            y_bin = (y >= t).astype(int)

            # stratified-ish split for each threshold
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_bin,
                test_size=valid_frac,
                random_state=random_state,
                stratify=y_bin if len(np.unique(y_bin)) > 1 else None,
            )

            # inverse-frequency sample weights to help rare positives
            counts = Counter(y_tr)
            w_tr = np.array([1.0 / counts[yy] for yy in y_tr], dtype=float)

            train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,           # smaller tree
                "max_depth": 6,             # shallow tree
                "min_data_in_leaf": 50,     # require more data per leaf
                "feature_fraction": 0.8,    # col subsampling
                "bagging_fraction": 0.8,    # row subsampling
                "bagging_freq": 3,
                "lambda_l1": 0.5,           # L1 regularisation
                "lambda_l2": 1.0,           # L2 regularisation
                "min_gain_to_split": 0.01,  # don't split on tiny gains
                "metric": "binary_logloss",
                "verbose": -1,
            }
            callbacks = [
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                ]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=800,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=callbacks,      # <-- old LightGBM uses callbacks instead
            )

            self.models.append(model)

    def predict_rating(self, memory_text: str, movie_text: str) -> int:
        x = self._encode_pair(memory_text, movie_text).reshape(1, -1)

        # sequential threshold predictions
        th_preds = []
        for model in self.models:
            p = float(model.predict(x)[0])
            th_preds.append(p > 0.5)  # you can also tune this threshold

        rating = 1 + sum(th_preds)
        return int(np.clip(rating, 1, 5))
    

