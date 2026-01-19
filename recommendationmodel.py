from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Iterable
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

import math
import random
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from Meta_CoT.Data import Data_Structure

def calculate_entropy(movie_types):
    type_freq = {}
    for movie_type in movie_types:
        if movie_type in type_freq:
            type_freq[movie_type] += 1
        else:
            type_freq[movie_type] = 1

    total_movies = len(movie_types)

    entropy = 0
    for key in type_freq:
        prob = type_freq[key] / total_movies
        entropy -= prob * math.log2(prob)

    return entropy


def get_entropy(inters, data):
    genres = Data_Structure.get_genres_by_id(inters)
    entropy = calculate_entropy(genres)
    return entropy


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, config, n_users, n_items):
        self.config = config
        self.items = None

    def get_full_sort_items(self, user_id, *args, **kwargs):
        """Get a list of sorted items for a given user."""
        raise NotImplementedError

    def _sort_full_items(self, user_id, *args, **kwargs):
        """Sort a list of items for a given user."""
        raise NotImplementedError
    
class MF(BaseModel, nn.Module):
    def __init__(self, config, n_users, n_items):
        BaseModel.__init__(self, config, n_users, n_items)
        nn.Module.__init__(self)
        self.config = config
        self.embedding_size = config["embedding_size"]
        self.n_users = n_users
        self.n_items = n_items
        torch.manual_seed(config['seed'])
        
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Biases
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    # def forward(self, user, item):
    #     """Predicts the rating of a user for an item."""
    #     user_embed = self.user_embedding(user)
    #     item_embed = self.item_embedding(item)

    #     # Dot product between user and item embeddings to predict rating
    #     predicted_rating = (user_embed * item_embed).sum(1)

    #     return predicted_rating
    def forward(self, user, item):
        """
        user: LongTensor of shape (B,)
        item: LongTensor of shape (B,)
        returns: predicted rating, shape (B,)
        """
        user_embed = self.user_embedding(user)          # (B, D)
        item_embed = self.item_embedding(item)          # (B, D)

        dot = (user_embed * item_embed).sum(dim=1, keepdim=True)  # (B, 1)
        ub = self.user_bias(user)                                  # (B, 1)
        ib = self.item_bias(item)                                  # (B, 1)

        out = dot + ub + ib + self.global_bias                     # (B, 1)
        return out.squeeze(-1)                                     # (B,)

    # def get_full_sort_items(self, user, items):
    #     """Get a list of sorted items for a given user."""
    #     predicted_ratings = self.forward(user, items)
    #     sorted_items = self._sort_full_items(user, predicted_ratings, items)
    #     return sorted_items.tolist()
    def get_full_sort_items(self, user, items):
        """
        Get sorted items for one user given a 1D tensor of item indices.
        user: scalar tensor or int
        items: LongTensor of shape (N,)
        """
        if not torch.is_tensor(user):
            user = torch.tensor(user, dtype=torch.long, device=items.device)

        # repeat user index for all items
        user_vec = user.expand_as(items)
        predicted_ratings = self.forward(user_vec, items)
        sorted_items = self._sort_full_items(user, predicted_ratings, items)
        return sorted_items.tolist()

    def _sort_full_items(self, user, predicted_ratings, items):
        """Sort items based on their predicted ratings."""
        # Sort items based on ratings in descending order and return item indices
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices]
    
class LightGCN(BaseModel, nn.Module):
    """
    LightGCN for recommendation.
    Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    """

    def __init__(
        self,
        config,
        n_users: int,
        n_items: int,
        interactions: Iterable[Tuple[int, int]],
        n_layers: int = 3,
        embedding_dim: int = 64,
        device: Optional[torch.device] = None,
    ):
        """
        interactions: iterable of (user_id, item_id) pairs (duplicates okay, handled as multi-edges of weight 1)
        """
        BaseModel.__init__(self, config, n_users, n_items)
        nn.Module.__init__(self)

        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.get("seed", 2023))

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Build normalized adjacency once
        self.Graph = self._build_normalized_adj(interactions).to(self.device)

        self.to(self.device)

    # ---------- Graph utilities ----------
    def _build_normalized_adj(self, interactions: Iterable[Tuple[int, int]]) -> torch.sparse.FloatTensor:
        """
        Build the symmetrically normalized adjacency matrix:
            A_hat = D^{-1/2} * A * D^{-1/2}
        for the bipartite user-item graph where nodes = users + items
        Size: (n_nodes, n_nodes) with n_nodes = n_users + n_items
        """
        n_nodes = self.n_users + self.n_items

        # Collect COO edges (user <-> item, undirected)
        rows = []
        cols = []
        vals = []

        for u, i, *rest in interactions:
            # Allow (u,i) or (u,i,rating) — rating ignored here (implicit 1)
            if u < 0 or u >= self.n_users or i < 0 or i >= self.n_items:
                continue
            i_offset = self.n_users + i  # item node index in unified graph
            # u -> i
            rows.append(u); cols.append(i_offset); vals.append(1.0)
            # i -> u
            rows.append(i_offset); cols.append(u); vals.append(1.0)

        if len(rows) == 0:
            # Empty graph fallback (identity to avoid NaNs)
            indices = torch.arange(n_nodes, dtype=torch.long)
            indices = torch.stack([indices, indices], dim=0)
            values = torch.ones(n_nodes, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        A = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

        # Degree vector d = sum of rows
        deg = torch.sparse.sum(A, dim=1).to_dense()  # (n_nodes,)
        # Avoid divide-by-zero
        deg = torch.clamp(deg, min=1e-12)
        d_inv_sqrt = torch.pow(deg, -0.5)

        # Normalize values: for each edge (i,j), val *= d^-1/2[i] * d^-1/2[j]
        row, col = A.indices()
        norm_vals = A.values() * d_inv_sqrt[row] * d_inv_sqrt[col]

        A_hat = torch.sparse_coo_tensor(A.indices(), norm_vals, A.size()).coalesce()
        return A_hat

    # ---------- Embedding propagation ----------
    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform K-layer LightGCN propagation and return final user & item embeddings.
        E^(0) = concat([U, I])  -> shape (n_users + n_items, d)
        E_final = 1/(K+1) * sum_{k=0..K} E^(k)
        """
        E_u0 = self.user_embedding.weight
        E_i0 = self.item_embedding.weight
        E0 = torch.cat([E_u0, E_i0], dim=0)  # (n_users + n_items, d)

        all_layers = [E0]
        x = E0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.Graph, x)  # LightGCN propagation (no weights, no nonlinearity)
            all_layers.append(x)

        E = torch.stack(all_layers, dim=0).mean(dim=0)  # layer-wise average
        Eu, Ei = torch.split(E, [self.n_users, self.n_items], dim=0)
        return Eu, Ei

    # ---------- Scoring ----------
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for (users, items). Supports:
          - users: (B,), items: (B,)  -> elementwise scores
          - users: (B,), items: (N,)  -> broadcast users for all N items (returns (B,N))
        """
        users = users.to(self.device)
        items = items.to(self.device)

        Eu, Ei = self.propagate()

        if users.dim() == 1 and items.dim() == 1 and users.shape[0] == items.shape[0]:
            u_emb = Eu[users]                   # (B, d)
            i_emb = Ei[items]                   # (B, d)
            return (u_emb * i_emb).sum(dim=1)   # (B,)

        # Broadcast to (B, N)
        u_emb = Eu[users]                       # (B, d)
        i_emb = Ei[items]                       # (N, d)
        scores = u_emb @ i_emb.t()              # (B, N)
        return scores

    def predict(self, user_ids: List[int], item_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Convenience method to get scores as numpy.
        - If item_ids is None: scores for all items for each user -> shape (B, n_items)
        - Else: scores for user_ids x item_ids -> (B, len(item_ids))
        """
        self.eval()
        with torch.no_grad():
            u = torch.tensor(user_ids, dtype=torch.long, device=self.device)
            if item_ids is None:
                items = torch.arange(self.n_items, dtype=torch.long, device=self.device)
                scores = self.forward(u, items)        # (B, n_items)
            else:
                items = torch.tensor(item_ids, dtype=torch.long, device=self.device)
                scores = self.forward(u, items)        # (B, len(item_ids))
        return scores.detach().cpu().numpy()

    # ---------- Ranking API (BaseModel) ----------
    def get_full_sort_items(self, user_id: int, seen_items: Optional[Iterable[int]] = None, top_k: Optional[int] = None) -> List[int]:
        """
        Rank all items for a given user (descending by predicted score).
        Optionally drop previously seen items.
        """
        scores = self.predict([user_id], None).ravel()  # (n_items,)
        if seen_items is not None:
            # push seen items to bottom
            seen_items = [i for i in seen_items if 0 <= i < self.n_items]
            scores[np.array(seen_items, dtype=np.int64)] = -np.inf

        order = np.argsort(-scores)  # descending
        if top_k is not None:
            order = order[:top_k]
        return order.tolist()

    def _sort_full_items(self, user_id: int, predicted_ratings: torch.Tensor, items: torch.Tensor):
        # Not used here; keeping for BaseModel compatibility
        _, idx = torch.sort(predicted_ratings, descending=True)
        return items[idx]


class BPRLoss(nn.Module):
    """
    Pairwise Bayesian Personalized Ranking loss with L2 regularization on embeddings.
    """
    def __init__(self, reg: float = 1e-4):
        super().__init__()
        self.reg = reg

    def forward(self, u_emb, pos_emb, neg_emb):
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / u_emb.shape[0]
        return loss + self.reg * reg
    
# ------------------------------
# Utility: set all seeds
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sasrec_pointwise_step(model, batch, device, logit_clip=20.0):
    users, seqs, pos_items, neg_items, mask = batch
    seqs, pos_items, neg_items = seqs.to(device), pos_items.to(device), neg_items.to(device)
    mask = mask.to(device).bool()

    seq_out = model(seqs)
    item_emb = model.item_embedding.weight

    pos_logits = torch.sum(seq_out * item_emb[pos_items], dim=-1).clamp(-logit_clip, logit_clip)
    neg_logits = torch.sum(seq_out * item_emb[neg_items], dim=-1).clamp(-logit_clip, logit_clip)

    valid_mask = (pos_items > 0) & mask
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_pos = F.binary_cross_entropy_with_logits(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
    loss_neg = F.binary_cross_entropy_with_logits(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))

    loss = loss_pos + loss_neg

    # L2 reg, skip padding embedding
    if model.l2_emb > 0:
        loss += model.l2_emb * (item_emb[1:].norm(p=2) ** 2) / 2

    return loss

# ------------------------------
# Dataset & Collate
# ------------------------------
class SASRecDataset(Dataset):
    """Builds sequences for SASRec training.

    Args:
        user2items: dict mapping user_id -> list of interacted item ids (in time order)
        n_items: total number of items (max ID)
        max_seq_len: truncate/pad sequences to this length
        min_seq_len: smallest effective length (>=2 to have a next item)
    """
    def __init__(self,
                 user2items: Dict[int, List[int]],
                 n_items: int,
                 max_seq_len: int = 50,
                 min_seq_len: int = 2):
        self.user2items = user2items
        self.users = [u for u, seq in user2items.items() if len(seq) >= min_seq_len]
        self.n_items = n_items
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        full = self.user2items[user]
        # Truncate to latest max_seq_len
        seq = full[-self.max_seq_len:]
        return user, seq


def _pad_sequence(seq: list, max_len: int) -> list:
    """Left-pad sequence with 0 to max_len."""
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

def _build_pos_items(seq: list) -> list:
    """Next-item targets, last position padded with 0."""
    return seq[1:] + [0]


def _sample_negatives(seq: list, n_items: int) -> list:
    """Sample negatives for each position, avoiding seq items."""
    user_set = set(seq)
    negatives = []
    for _ in range(len(seq)):
        neg = random.randint(1, n_items - 1)  # avoid 0
        while neg in user_set:
            neg = random.randint(1, n_items - 1)
        negatives.append(neg)
    return negatives


def sasrec_collate(batch, n_items: int, max_seq_len: int):
    users, seqs = zip(*batch)
    seqs = [_pad_sequence(s, max_seq_len) for s in seqs]

    pos_items = [_build_pos_items(s) for s in seqs]
    neg_items = [_sample_negatives(s, n_items) for s in seqs]
    mask = [[1 if x != 0 else 0 for x in s] for s in seqs]

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float),
    )


def build_user2items(train_data):
    user2items = defaultdict(list)
    for u, i, r in sorted(train_data, key=lambda x: (x[0], x[2])):
        user2items[u].append(i)
    return user2items


# ------------------------------
# Model
# ------------------------------
class SASRec(nn.Module, BaseModel):
    def __init__(self, config, n_users: int, n_items: int):
        nn.Module.__init__(self)  # initialize nn.Module
        BaseModel.__init__(self, config, n_users, n_items)  # keep BaseModel logic

        self.n_items = n_items
        self.hidden_units = int(config.get("hidden_units", 128))
        self.max_seq_len = int(config.get("max_seq_len", 50))
        self.num_heads = int(config.get("num_heads", 2))
        self.num_blocks = int(config.get("num_blocks", 2))
        self.dropout_rate = float(config.get("dropout_rate", 0.2))
        self.l2_emb = float(config.get("l2_emb", 0.0))

        # Embeddings
        self.item_embedding = nn.Embedding(n_items + 1, self.hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_units)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_units,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_units * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=True,
            ) for _ in range(self.num_blocks)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_units)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def _causal_mask(self, L: int, device=None):
        # True means masked in PyTorch
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence safely, with padding & causal masks.
        Args:
            item_seq: (B, L) padded with 0 on left
        Returns:
            seq_out: (B, L, H)
        """
        B, L = item_seq.shape
        device = item_seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # Embeddings
        x = self.item_embedding(item_seq) + self.position_embedding(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        key_padding_mask = (item_seq == 0)  # True = pad
        attn_mask = self._causal_mask(L, device=device)  # True = masked

        for blk in self.blocks:
            x = blk(
            x,
            src_mask=attn_mask,
            src_key_padding_mask=key_padding_mask
            )
            # Safety: replace NaN/inf
            if torch.isnan(x).any() or torch.isinf(x).any():
               x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        return x

    @torch.no_grad()
    def get_full_sort_items(self, user_id, user_seq: torch.Tensor):
        """Return sorted item IDs by score for a single user.
        Args:
            user_seq: (1, L)
        Returns:
            torch.Tensor of sorted item ids (desc)
        """
        self.eval()
        seq_out = self.forward(user_seq)[:, -1, :]  # (1, H)
        all_item_emb = self.item_embedding.weight  # (n_items+1, H)
        scores = torch.matmul(seq_out, all_item_emb.t()).squeeze(0)  # (n_items+1,)
        # Avoid recommending padding id 0
        scores[0] = -1e9
        return torch.argsort(scores, descending=True)

    def _sort_full_items(self, user_id, *args, **kwargs):
        raise NotImplementedError("Use get_full_sort_items(user_id, user_seq)")
    

class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config):
        self.config = config
        self.data_struct = Data_Structure("/opt/home/s4065511/Meta_CoT/Dataset/AmazonBook/user_profiles.csv", "/opt/home/s4065511/Meta_CoT/Dataset/AmazonBook/amazon_book_item.csv", "/opt/home/s4065511/Meta_CoT/Dataset/AmazonBook/amazon_book_interaction_rating.csv")

        self.page_size = self.config["page_size"]
        self.items_per_page = self.config["items_per_page"]
        self.random_k = self.config["rec_random_k"]
        self.train_data = []
        self.n_layers = 3
        self.embedding_dim = 4
        if self.config["rec_model"] == "MF":
           self.model = MF(self.config, self.data_struct.get_user_num(), self.data_struct.get_item_num())
        elif self.config["rec_model"] == "LightGCN":
           self.model = LightGCN(self.config, self.data_struct.get_user_num(), self.data_struct.get_item_num(), self.train_data, n_layers=self.n_layers, embedding_dim=self.embedding_dim)
        elif self.config["rec_model"] == "SASRec":
           self.model = SASRec(self.config, self.data_struct.get_user_num(), self.data_struct.get_item_num())
        else:
           raise ValueError(f"Unknown model: {self.config['rec_model']}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.epoch_num = self.config["epoch_num"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.record = {}
        self.round_record = {}
        self.positive = {}
        self.interaction_dict = {}
        self.inter_df = None
        self.inter_num = 0
        for user in self.data_struct.all_user_ids():
            self.record[user] = []
            self.positive[user] = []
            self.round_record[user] = []
        self.user_data = {
            "user": [],
            "N_expose": [],
            "N_view": [],
            "N_like": [],
            "N_exit": [],
            "S_sat": []
            }
        self.rating_feeling = {
            "User": [],
            "Rating": [],
            "Feelings": []
        }
        
        # placeholders for reindex maps (filled in create_train_data)
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def sample_bpr_triples(self,
                           user_pos_items: List[List[int]],
                           n_items: int,
                           batch_size: int,
                           device: torch.device,
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        user_pos_items: list of lists; for each user u, a list of items they've interacted with (positive)
        returns (users, pos_items, neg_items) tensors of length batch_size
        """
        users = []
        pos = []
        neg = []
        for _ in range(batch_size):
           # sample a user with at least one positive
           while True:
               u = np.random.randint(0, len(user_pos_items))
               if len(user_pos_items[u]) > 0:
                  break

           i = np.random.choice(user_pos_items[u])
           # sample a negative item

           while True:
               j = np.random.randint(0, n_items)
               if j not in user_pos_items[u]:
                  break

           users.append(u)
           pos.append(i)
           neg.append(j)

        return (
        torch.tensor(users, dtype=torch.long, device=self.device),
        torch.tensor(pos, dtype=torch.long, device=self.device),
        torch.tensor(neg, dtype=torch.long, device=self.device),
        )

    def train_lightgcn_bpr(self,
        reg: float = 1e-4,
        log: bool = True):
        """
        Train LightGCN with BPR loss.
        - train_interactions/val_interactions can be (u,i) or (u,i,rating>0) tuples.
        - If ckpt_path is provided, saves the best (by simple val recall proxy) state dict.
        """

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = BPRLoss(reg=reg)

        # Build user -> positives list
        user_pos = [[] for _ in range(self.model.n_users)]
        for u, i, *rest in self.train_data:
            if 0 <= u < self.model.n_users and 0 <= i < self.model.n_items:
               user_pos[u].append(i)

        # Basic validation proxy: count of positives ranked in top-10 (cheap & optional)
        def quick_val_topk_hits(k: int = 10) -> float:
            if self.val_data is None:
               return -1.0
            # build val positives per user
            val_pos = [[] for _ in range(self.model.n_users)]
            for u, i, *rest in self.val_data:
               if 0 <= u < self.model.n_users and 0 <= i < self.model.n_items:
                   val_pos[u].append(i)

            hits = 0
            total = 0

            for u in range(self.model.n_users):
               if not val_pos[u]:
                  continue
               recs = self.model.get_full_sort_items(u, seen_items=set(user_pos[u]), top_k=k)
               s = set(val_pos[u])
               hits += len([r for r in recs if r in s])
               total += min(k, len(s))
            return hits / total if total > 0 else -1.0

        best_metric = -math.inf

        # Build full checkpoint file path
        ckpt_file = os.path.join(self.config['checkpoint_path'], "best_lightGCN_model.pth")
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)

        for epoch in range(1, self.epoch_num + 1):
            self.model.train()

            # One epoch of mini-batch BPR
            n_steps = max(1, sum(len(v) for v in user_pos) // max(1, self.config['batch_size']))
            losses = []
            for _ in range(n_steps):
                users, pos_items, neg_items = self.sample_bpr_triples(user_pos, self.model.n_items, self.config['batch_size'], self.device)
                Eu, Ei = self.model.propagate()
                u_emb = Eu[users]
                p_emb = Ei[pos_items]
                n_emb = Ei[neg_items]
                loss = self.criterion(u_emb, p_emb, n_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            # quick validation metric
            metric = quick_val_topk_hits(k=10)
            if log:
               print(f"[Epoch {epoch:3d}] BPR Loss: {np.mean(losses):.4f} | Val@10: {metric:.4f}")

            # Save checkpoint if validation improves
            if metric > best_metric:
               best_metric = metric
               torch.save({
                   "epoch": self.epoch_num + 1,
                   "model_state_dict": self.model.state_dict(),
                   "n_users": self.data_struct.get_user_num(),
                   "n_items": self.data_struct.get_item_num(),
                   "n_layers": self.n_layers,
                   "embedding_dim": self.embedding_dim,
                   "metric": metric,
                   }, ckpt_file)
               print(f"Best model updated at epoch {epoch+1}, saved to {ckpt_file}")

        # Load best (optional)
        if ckpt_file is not None and best_metric > -math.inf:
           # At the end, reload the best weights for inference
           checkpoint = torch.load(ckpt_file)
           self.model.load_state_dict(checkpoint['model_state_dict'])

    def _build_user_pos_items(self):
        """
        Build mapping from user_idx -> set of positive item_idx from train_data.
        Must be called AFTER create_train_data().
        """
        self.user_pos_items = defaultdict(set)
        for u, i, r in self.train_data:
            # treat any interaction in train_data as positive
            self.user_pos_items[u].add(i)

        # all candidate items (reindexed space)
        self._all_item_indices = np.arange(self.model.n_items, dtype=np.int64)

    def _sample_negative_item(self, user, rng: np.random.Generator):
        """
        Sample a negative item_idx for given user_idx (not in user's positives).
        """
        pos_items = self.user_pos_items[user]
        n_items = self.model.n_items

        while True:
            j = int(rng.integers(0, n_items))
            if j not in pos_items:
                return j
            
    def _sample_bpr_batch(self, batch_size: int, rng: np.random.Generator):
        """
        Sample a BPR batch of (user, pos_item, neg_item) indices.

        Returns:
            user_t, pos_t, neg_t  (LongTensors on self.device)
        """
        if not hasattr(self, "user_pos_items"):
            raise RuntimeError(
                "user_pos_items not built. Call _build_user_pos_items() first "
                "or call train_mf_bpr(), which does it automatically."
            )

        users_list = list(self.user_pos_items.keys())
        u_batch, i_batch, j_batch = [], [], []

        for _ in range(batch_size):
            # pick a random user with at least one positive item
            u = int(users_list[rng.integers(0, len(users_list))])
            pos_items = list(self.user_pos_items[u])
            if not pos_items:
                continue

            i_pos = pos_items[rng.integers(0, len(pos_items))]
            j_neg = self._sample_negative_item(u, rng)

            u_batch.append(u)
            i_batch.append(i_pos)
            j_batch.append(j_neg)

        if not u_batch:
            # fallback in weird case (shouldn't really happen)
            return None, None, None

        user_t = torch.tensor(u_batch, dtype=torch.long, device=self.device)
        pos_t  = torch.tensor(i_batch, dtype=torch.long, device=self.device)
        neg_t  = torch.tensor(j_batch, dtype=torch.long, device=self.device)

        return user_t, pos_t, neg_t
    
    def _bpr_loss(self, pos_scores, neg_scores, reg_lambda: float = 0.0):
        """
        BPR loss: -E[ log sigma(s_pos - s_neg) ] + L2 regularization (optional).
        """
        # numeric-stable: log(sigmoid(x)) = -softplus(-x)
        x = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(x) + 1e-8).mean()

        if reg_lambda > 0.0:
            reg_term = 0.0
            # simple L2 regularization over embeddings used in this batch
            # (approximate; for full regularization you'd sum all params)
            # You can omit this or refine it.
            return loss + reg_lambda * reg_term

        return loss



    def bce_sampled_loss(self, seq_out: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        item_embedding: nn.Embedding,
        mask: torch.Tensor,
        l2_emb: float = 0.0) -> torch.Tensor:
        """Binary cross-entropy on sampled positives/negatives per position.
        Args:
            seq_out: (B, L, H)
            pos_items: (B, L) next-item ids (0 where no target)
            neg_items: (B, L) sampled negatives (0 where no target)
            item_embedding: embedding module (to fetch item vectors)
            mask: (B, L) boolean, True where a target exists (i.e., pos_items > 0)
            l2_emb: weight decay on item embeddings (regularizes pos/neg lookups)
        """
        B, L, H = seq_out.shape

        pos_vecs = item_embedding(pos_items)  # (B, L, H)
        neg_vecs = item_embedding(neg_items)  # (B, L, H)

        pos_logits = (seq_out * pos_vecs).sum(-1)  # (B, L)
        neg_logits = (seq_out * neg_vecs).sum(-1)  # (B, L)

        # Targets: pos -> 1, neg -> 0
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits[mask], torch.ones_like(pos_logits[mask]))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
        loss = pos_loss + neg_loss

        if l2_emb > 0:
           reg = (pos_vecs[mask].pow(2).sum() + neg_vecs[mask].pow(2).sum()) / mask.sum().clamp_min(1)
           loss = loss + l2_emb * reg
        return loss

    def train_sasrec(self, grad_clip=1.0, logit_clip=20.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build user->items dict and dataset
        user2items = build_user2items(self.train_data)
        n_items_global = int(self.data_struct.get_item_num())
        max_seq_len = self.model.max_seq_len

        train_dataset = SASRecDataset(user2items, n_items=n_items_global, max_seq_len=max_seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: sasrec_collate(batch, n_items=n_items_global, max_seq_len=max_seq_len)
            )

        # Validation loader
        val_user2items = build_user2items(self.val_data)
        val_dataset = SASRecDataset(val_user2items, n_items=n_items_global, max_seq_len=max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: sasrec_collate(batch, n_items=n_items_global, max_seq_len=max_seq_len)
            )

        # Checkpoint setup
        ckpt_file = os.path.join(self.config['checkpoint_path'], "best_SASRec_model.pth")
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        best_metric = -float("inf")

        for epoch in range(1, self.epoch_num + 1):
           self.model.train()
           running = 0.0
           n_steps = 0

           for users, seqs, pos, neg, umask in train_loader:
              users, seqs, pos, neg, umask = (
                users.to(self.device),
                seqs.to(self.device),
                pos.to(self.device),
                neg.to(self.device),
                umask.to(self.device).bool(),  # ensure bool type
                )

              # Compute mask dynamically
              mask = pos > 0
              mask = mask.bool()
              if mask.sum() == 0:
                 continue  # skip batch with no valid positions

              # Compute stable loss
              loss = sasrec_pointwise_step(self.model, (users, seqs, pos, neg, mask), device=self.device, logit_clip=logit_clip)

              self.optimizer.zero_grad(set_to_none=True)
              loss.backward()
              if grad_clip is not None:
                 nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
              self.optimizer.step()

              running += loss.item()
              n_steps += 1

           avg_loss = running / max(1, n_steps)
           print(f"Epoch {epoch}/{self.epoch_num} - train loss: {avg_loss:.4f}")

           # Validation
           if val_loader is not None:
              self.model.eval()
              val_loss = 0.0
              n_val_steps = 0
              with torch.no_grad():
                  for users, seqs, pos, neg, umask in val_loader:
                    #  print("Users:", users)
                    #  print("Pos min/max:", pos.min().item(), pos.max().item())
                    #  print("Neg min/max:", neg.min().item(), neg.max().item())
                     users, seqs, pos, neg, umask = (
                         users.to(self.device),
                         seqs.to(self.device),
                         pos.to(self.device),
                         neg.to(self.device),
                         umask.to(self.device).bool(),)

                     mask = pos > 0
                     if mask.sum() == 0:
                        # print("Skipped empty batch")
                        continue

                     loss = sasrec_pointwise_step(self.model, (users, seqs, pos, neg, mask), device=self.device, logit_clip=logit_clip)
                    #  print("Batch loss:", loss.item())
                     val_loss += loss.item()
                     n_val_steps += 1
              val_loss /= max(1, n_val_steps)
              metric = -val_loss
              print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
           else:
              metric = -avg_loss

           # Save best model
           if metric > best_metric:
              best_metric = metric
              torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metric": metric,
              }, ckpt_file)
              print(f"Best model updated at epoch {epoch}, saved to {ckpt_file}")

        # Load best model after training
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def swap_items(self, lst, page_size, random_k):
        total_pages = len(lst) // page_size
        lst = lst[: total_pages * page_size]
        for page in range(1, total_pages // 2 + 1):
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size - 1
            symmetric_start_idx = (total_pages - page) * page_size
            symmetric_end_idx = symmetric_start_idx + page_size

            for k in range(1, random_k + 1):
                lst[end_idx - k], lst[symmetric_end_idx - k] = (
                    lst[symmetric_end_idx - k],
                    lst[end_idx - k],
                )

        return lst

    def add_random_items(self, user, item_ids):
        item_ids = self.swap_items(item_ids, self.page_size, self.random_k)
        return item_ids


    def get_inter_num(self):
        return self.inter_num


    def save_interaction(self):
        """
        Save the interaction history to a csv file.
        """
        inters = []
        users = self.data_struct.all_user_ids()
        for user in users:
            for item in self.positive[user]:
                new_row = {"user_id": user, "item_id": item, "rating": 1}
                inters.append(new_row)

            for item in self.record[user]:
                if item in self.positive[user]:
                    continue
                new_row = {"user_id": user, "item_id": item, "rating": 0}
                inters.append(new_row)

        df = pd.DataFrame(inters)
        df.to_csv(
            self.config["interaction_path"],
            index=False,
        )

        self.inter_df = df

    def add_train_data(self, user, item, label):
        self.train_data.append((user, item, label))

    def clear_train_data(self):
        self.train_data = []

    def get_entropy(
        self,
    ):
        tot_entropy = 0
        for user in self.record.keys():
            inters = self.record[user]
            genres = self.data_struct.get_genres_by_id(inters)
            entropy = calculate_entropy(genres)
            tot_entropy += entropy

        return tot_entropy / len(self.record.keys())

    def check_train_data(self):
        """
        Print or inspect the training data.
        """
        print("Training Data:")
        for user, item, label in self.train_data:
            print(f"User: {user}, Item: {item}, Label: {label}")

    def create_train_data(self):
        """
        Create a training dataset with random samples.

        Args:
            num_samples (int): Number of samples to generate.
        """
        self.clear_train_data()  # Clear existing training data

        self.train_data, self.val_data, self.test_data = self.data_struct.dataset_split()

        if isinstance(self.train_data, pd.DataFrame):
            self.train_data = list(
                self.train_data[["user_id", "item_id", "rating"]]
                .itertuples(index=False, name=None)
            )
        
        if isinstance(self.val_data, pd.DataFrame):
            self.val_data = list(
                self.val_data[["user_id", "item_id", "rating"]]
                .itertuples(index=False, name=None)
            )

        if isinstance(self.test_data, pd.DataFrame):
            self.test_data = list(
                self.test_data[["user_id", "item_id", "rating"]]
                .itertuples(index=False, name=None)
            )

        train_users = max([u for u, i, r in self.train_data]) + 1
        train_items = max([i for u, i, r in self.train_data]) + 1

        val_users = max([u for u, i, r in self.val_data]) + 1
        val_items = max([i for u, i, r in self.val_data]) + 1

        test_users = max([u for u, i, r in self.test_data]) + 1
        test_items = max([i for u, i, r in self.test_data]) + 1

        n_items_global = int(self.data_struct.get_item_num())

        # Initialize user-item matrix
        self.train_matrix = np.zeros((train_users, n_items_global), dtype=np.float32)
        self.val_matrix = np.zeros((val_users, n_items_global), dtype=np.float32)
        self.test_matrix = np.zeros((test_users, n_items_global), dtype=np.float32)

        # Fill interactions safely
        for u, i, r in self.train_data:
           if i >= n_items_global: continue  # skip bad indices
           self.train_matrix[u, i] = r

        for u, i, r in self.val_data:
           if i >= n_items_global: continue
           self.val_matrix[u, i] = r

        for u, i, r in self.test_data:
           if i >= n_items_global: continue
           self.test_matrix[u, i] = r

        # 4) build reindex maps over ALL splits (train+val+test)
        all_triples = self.train_data + self.val_data + self.test_data

        user_ids = sorted({u for (u, i, r) in all_triples})
        item_ids = sorted({i for (u, i, r) in all_triples})

        self.user2idx = {u: idx for idx, u in enumerate(user_ids)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}

        self.item2idx = {i: idx for idx, i in enumerate(item_ids)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        # 5) reindex triples to 0..N-1
        self.train_data = [
            (self.user2idx[u], self.item2idx[i], float(r))
            for (u, i, r) in self.train_data
        ]
        self.val_data = [
            (self.user2idx[u], self.item2idx[i], float(r))
            for (u, i, r) in self.val_data
        ]
        self.test_data = [
            (self.user2idx[u], self.item2idx[i], float(r))
            for (u, i, r) in self.test_data
        ]

        # 6) re-init MF with correct sizes (only for MF)
        if self.config["rec_model"] == "MF":
            n_users = len(self.user2idx)
            n_items = len(self.item2idx)
            print(f"[DEBUG] Rebuilding MF with n_users={n_users}, n_items={n_items}")

            self.model = MF(self.config, n_users, n_items).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config["lr"], weight_decay=1e-5
            )
            # criterion already defined in __init__

            max_u = max(u for (u, i, r) in self.train_data)
            max_i = max(i for (u, i, r) in self.train_data)
            print(f"[DEBUG] max train user_idx={max_u}, max train item_idx={max_i}")
            print(
                f"[DEBUG] user_embedding.num_embeddings={self.model.user_embedding.num_embeddings}, "
                f"item_embedding.num_embeddings={self.model.item_embedding.num_embeddings}"
            )

    # def evaluate(self, dataset):
    #     self.model.eval()
    #     users = torch.tensor([x[0] for x in dataset])
    #     items = torch.tensor([x[1] for x in dataset])
    #     labels = torch.tensor([x[2] for x in dataset]).float()

    #     with torch.no_grad():
    #          outputs = self.model(users, items)
    #          loss = self.criterion(outputs, labels)
    #     return loss.item()
    def evaluate(self, dataset):
        """
        Compute mean MSE loss on `dataset`.

        `dataset` can be:
          - list of (user, item, rating), or
          - pandas DataFrame with columns ['user_id', 'item_id', 'rating'].
        """

        self.model.eval()

        if isinstance(dataset, pd.DataFrame):
            raise ValueError("evaluate() expects reindexed list-of-triples, not DataFrame.")

        if len(dataset) == 0:
            return float("nan")

        users = [int(u) for (u, i, r) in dataset]
        items = [int(i) for (u, i, r) in dataset]
        labels = [float(r) for (u, i, r) in dataset]

        users_t  = torch.tensor(users,  dtype=torch.long,    device=self.device)
        items_t  = torch.tensor(items,  dtype=torch.long,    device=self.device)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.model(users_t, items_t).view(-1)
            loss = self.criterion(outputs, labels_t)

        self.model.train()  # back to train mode for next epoch
        return loss.item()


    def load_checkpoint(self, path="best_model.pth", resume_training=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if resume_training:
           # Load optimizer state to resume training exactly where it left off
           self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
           start_epoch = checkpoint["epoch"]
           return start_epoch
        else:
           self.model.eval()  # set to eval mode for inference

           
    def evaluate_recall_at_k(self, k: int = 10):
        """
        Compute Recall@k over the global test item pool for each user,
        using get_rankings_over_all_test_items() and self.test_data.

        Returns
        -------
        recall_by_user : dict
            {user_idx: recall_at_k}
        mean_recall : float
            Average Recall@k over users with at least one test item.
        """
        if self.config["rec_model"] != "MF":
            raise ValueError("evaluate_recall_at_k is implemented only for MF.")

        # 1) rankings over global test pool (reindexed item IDs)
        rankings_by_user = self.get_rankings_over_all_test_items(
            return_original_ids=False
        )

        # 2) ground-truth test items per user (reindexed)
        test_items_by_user = defaultdict(set)
        for u, i, r in self.test_data:
            test_items_by_user[u].add(i)

        recall_by_user = {}

        for u, ranking in rankings_by_user.items():
            gt_items = test_items_by_user.get(u, set())
            if not gt_items:
                continue

            top_k = set(ranking[:k])
            hits = len(gt_items & top_k)
            recall = hits / len(gt_items)
            recall_by_user[u] = recall

        if not recall_by_user:
            return {}, float("nan")

        mean_recall = sum(recall_by_user.values()) / len(recall_by_user)
        return recall_by_user, mean_recall
        
    # ---------- training loop for MF ----------
    def train_mf_bpr(
        self,
        epochs: int = None,
        batch_size: int = None,
        steps_per_epoch: int = None,
        reg_lambda: float = None,
        k_for_recall: int = 10,
    ):
        """
        Train MF using BPR with negative sampling.

        - Treats each (user_idx, item_idx, rating) in self.train_data as a positive.
        - Samples a negative item per positive via _sample_bpr_batch.
        - Optimizes BPR loss: -log sigma(s(u, i_pos) - s(u, i_neg)).
        - **Checkpointing is done using Recall@k (higher is better).**
        """
        if len(self.train_data) == 0:
            print("No training data! Did you call create_train_data() first?")
            return

        if epochs is None:
            epochs = self.epoch_num
        if batch_size is None:
            batch_size = self.config["batch_size"]
        if reg_lambda is None:
            reg_lambda = self.config.get("bpr_reg", 0.0)

        # build user_pos_items / item universe for negative sampling
        self._build_user_pos_items()

        rng = np.random.default_rng(self.config.get("seed", 42))

        if steps_per_epoch is None:
            steps_per_epoch = max(1, len(self.train_data) // batch_size)

        # ----- best checkpoint tracked by Recall@k (maximize) -----
        best_recall = -float("inf")
        best_epoch = -1

        ckpt_file = os.path.join(
            self.config["checkpoint_path"], "best_MF_model_bpr.pth"
        )
        os.makedirs(self.config["checkpoint_path"], exist_ok=True)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_steps = 0

            for step in range(steps_per_epoch):
                user_t, pos_t, neg_t = self._sample_bpr_batch(batch_size, rng)
                if user_t is None:
                    continue

                pos_scores = self.model(user_t, pos_t)
                neg_scores = self.model(user_t, neg_t)

                loss = self._bpr_loss(pos_scores, neg_scores, reg_lambda=reg_lambda)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

            avg_train_loss = total_loss / max(1, total_steps)

            # ----- ranking metric: Recall@k -----
            _, mean_recall = self.evaluate_recall_at_k(k=k_for_recall)

            # (optional) still compute MSE just for logging/debugging
            val_mse = self.evaluate(self.val_data)

            print(
                f"[BPR] Epoch {epoch+1}/{epochs}, "
                f"Train BPR Loss: {avg_train_loss:.4f}, "
                f"Mean Recall@{k_for_recall}: {mean_recall:.4f}, "
                f"Val MSE: {val_mse:.4f}"
            )

            # save checkpoint if Recall@k improves
            if mean_recall > best_recall:
                best_recall = mean_recall
                best_epoch = epoch + 1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "mean_recall": mean_recall,
                        "val_mse": val_mse,
                    },
                    ckpt_file,
                )
                print(
                    f"[BPR] Best model (by Recall@{k_for_recall}) "
                    f"updated at epoch {epoch+1}, saved to {ckpt_file}"
                )

        # reload best weights for inference
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[BPR] Loaded best model from epoch {checkpoint['epoch']} "
            f"with Recall@{k_for_recall} = {checkpoint['mean_recall']:.4f}"
        )


    # def train_mf(self):
    #     if len(self.train_data) == 0:
    #         print("No training data! Did you call create_train_data() first?")
    #         return

    #     # unpack triples
    #     users = [u for (u, i, r) in self.train_data]
    #     items = [i for (u, i, r) in self.train_data]
    #     labels = [r for (u, i, r) in self.train_data]

    #     users_t = torch.tensor(users, dtype=torch.long)
    #     items_t = torch.tensor(items, dtype=torch.long)
    #     labels_t = torch.tensor(labels, dtype=torch.float32)

    #     dataset = TensorDataset(users_t, items_t, labels_t)

    #     train_loader = DataLoader(
    #         dataset,
    #         batch_size=self.config["batch_size"],
    #         shuffle=True,
    #     )

    #     self.model.train()

    #     best_val_loss = float("inf")

    #     ckpt_file = os.path.join(self.config["checkpoint_path"], "best_MF_model.pth")
    #     os.makedirs(self.config["checkpoint_path"], exist_ok=True)

    #     for epoch in range(self.epoch_num):
    #         total_train_loss = 0.0
    #         total_train_count = 0

    #         for user, item, label in train_loader:
    #             user = user.to(self.device)
    #             item = item.to(self.device)
    #             label = label.to(self.device)

    #             self.optimizer.zero_grad()
    #             outputs = self.model(user, item).view(-1)
    #             loss = self.criterion(outputs, label)
    #             loss.backward()
    #             self.optimizer.step()

    #             batch_size = label.size(0)
    #             total_train_loss += loss.item() * batch_size
    #             total_train_count += batch_size

    #         train_loss = total_train_loss / total_train_count

    #         # validation loss recomputed every epoch
    #         val_loss = self.evaluate(self.val_data)

    #         print(
    #             f"Epoch {epoch+1}/{self.epoch_num}, "
    #             f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    #         )

    #         # Save checkpoint if validation improves
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             torch.save(
    #                 {
    #                     "epoch": epoch + 1,
    #                     "model_state_dict": self.model.state_dict(),
    #                     "optimizer_state_dict": self.optimizer.state_dict(),
    #                     "val_loss": val_loss,
    #                 },
    #                 ckpt_file,
    #             )
    #             print(f"Best model updated at epoch {epoch+1}, saved to {ckpt_file}")

    #     # reload best weights for inference
    #     checkpoint = torch.load(ckpt_file, map_location=self.device)
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     self.model.to(self.device)
    #     self.model.eval()

    # def train_mf(self):
    #     if len(self.train_data) == 0:
    #         print("No training data!")
    #         return

    #     users = [x[0] for x in self.train_data]
    #     items = [x[1] for x in self.train_data]
    #     labels = [x[2] for x in self.train_data]


    #     dataset = torch.utils.data.TensorDataset(
    #     torch.tensor(users), torch.tensor(items), torch.tensor(labels))

    #     train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=self.config["batch_size"], shuffle=True)

    #     self.model.train()

    #     best_val_loss = float("inf")

    #     # Build full checkpoint file path
    #     ckpt_file = os.path.join(self.config['checkpoint_path'], "best_MF_model.pth")
    #     os.makedirs(self.config['checkpoint_path'], exist_ok=True)

    #     for epoch in range(self.epoch_num):
    #         epoch_loss = 0.0

    #         for user, item, label in train_loader:

    #             self.optimizer.zero_grad()
    #             outputs = self.model(user, item)
    #             loss = self.criterion(outputs, label.float())
    #             loss.backward()
    #             self.optimizer.step()
    #             epoch_loss += loss.item()

    #         val_loss = self.evaluate(self.val_data)  # Evaluate on validation set

    #         print(
    #         f"Epoch {epoch+1}/{self.epoch_num}, Train Loss: {epoch_loss/len(train_loader):.4f}, "
    #         f"Val Loss: {val_loss:.4f}")

    #         # Save checkpoint if validation improves
    #         if val_loss < best_val_loss:
    #            best_val_loss = val_loss
    #            torch.save({
    #             "epoch": epoch + 1,
    #             "model_state_dict": self.model.state_dict(),
    #             "optimizer_state_dict": self.optimizer.state_dict(),
    #             "val_loss": val_loss,
    #             }, ckpt_file)
    #            print(f"Best model updated at epoch {epoch+1}, saved to {ckpt_file}")

    #     # At the end, reload the best weights for inference
    #     checkpoint = torch.load(ckpt_file)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])

    def load_best_model(self):
        if self.config['rec_model'] == 'MF':
           ckpt_file = os.path.join(self.config['checkpoint_path'], "best_MF_model_bpr.pth")
        elif self.config['rec_model'] == 'LightGCN':
           ckpt_file = os.path.join(self.config['checkpoint_path'], "best_lightGCN_model.pth")
        elif self.config['rec_model'] == 'SASRec':
           ckpt_file = os.path.join(self.config['checkpoint_path'], "best_SASRec_model.pth")
        else:
           raise ValueError(f"Unknown model type: {self.config['rec_model']}")

        # Build full checkpoint file path
        checkpoint = torch.load(ckpt_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def get_full_rankings(self, use_test=False, batch_size=512):
        """
        Compute full rankings for all users in self.data.
        - training items are pushed to the end
        - optionally, ground-truth test items can be put on top
        """
        if self.config['rec_model'] == 'MF':
           
           self.model.eval()
           # --- sizes from the MF model / reindexed mappings ---
           # After create_train_data(), we re-initialised MF with correct sizes:
           #   n_users = len(self.user2idx), n_items = len(self.item2idx)
           n_users = self.model.n_users
           n_items = self.model.n_items
           
           # --- build quick lookup: train items per user, test items per user ---
           train_items_by_user = defaultdict(list)
           for u, i, r in self.train_data:
               train_items_by_user[u].append(i)

           test_items_by_user = defaultdict(list)
           for u, i, r in self.test_data:
               test_items_by_user[u].append(i)

           # allocate result: ranking of all item indices for each user
           self.full_rankings = np.zeros((n_users, n_items), dtype=np.int64)

           # pre-create item index tensor once
           item_indices = torch.arange(n_items, dtype=torch.long, device=self.device)

           with torch.no_grad():
               for u in range(n_users):
                   # repeat user index for all candidate items
                   user_vec = torch.full(
                       (n_items,),
                       u,
                       dtype=torch.long,
                       device=self.device,
                   )

                   # scores using full MF (including biases)
                   scores = self.model(user_vec, item_indices)  # (n_items,)
                   scores = scores.detach().cpu().numpy()
 
                   # push training items to the end by giving them very low score
                   train_items = train_items_by_user.get(u, [])
                   for i in train_items:
                       if 0 <= i < n_items:
                           scores[i] = -1e9  # or -np.inf

                   # base ranking: descending scores
                   ranking = np.argsort(-scores)  # highest score first

                   if use_test:
                       raw_test_items = test_items_by_user.get(u, [])
                       # Move ground-truth test items (if any) to the front
                       test_items = [
                           i for i in dict.fromkeys(raw_test_items)  # unique, keep order
                           if 0 <= i < n_items
                       ]
                       if test_items:
                           # Remove test items from current ranking, then prepend them
                           mask = np.ones_like(ranking, dtype=bool)
                           for ti in test_items:
                               idx = np.where(ranking == ti)[0]
                               if idx.size > 0:
                                   mask[idx[0]] = False
                           others = ranking[mask]
                           ranking = np.array(test_items + others.tolist(), dtype=np.int64)

                   # now ranking length must equal n_items
                   assert ranking.shape[0] == n_items, (
                        f"Ranking length mismatch for user {u}: "
                        f"{ranking.shape[0]} vs n_items={n_items}"
                        )
                   self.full_rankings[u] = ranking

           return self.full_rankings
        #    n_users = self.data_struct.get_user_num()
        #    n_items = self.data_struct.get_item_num()

        #    item_embed = self.model.item_embedding.weight[:n_items, :]

        #    self.full_rankings = np.zeros((n_users, n_items), dtype=int)

        #    for user in range(n_users):
        #       user_tensor = torch.tensor([user])
        #       user_embed = self.model.user_embedding(user_tensor)

        #       scores = torch.matmul(user_embed, item_embed.T).squeeze(0).detach().numpy()

        #       # Only consider valid item indices
        #       test_items = [x[1] for x in self.test_data if x[0] == user and x[1] < n_items]
        #       scores[test_items] = -np.inf

        #       self.full_rankings[user] = np.argsort(-scores)

        #       # # Optionally move ground-truth test items on top
        #       if use_test:
        #          test_items = [x[1] for x in self.test_data if x[0] == user and x[1] < n_items]
        #          for idx, item in enumerate(test_items):
        #             if item in self.full_rankings[user]:
        #                current_pos = np.where(self.full_rankings[user] == item)[0][0]
        #                self.full_rankings[user][idx], self.full_rankings[user][current_pos] = (
        #                  self.full_rankings[user][current_pos],
        #                  self.full_rankings[user][idx])

        elif self.config['rec_model'] == 'LightGCN':
            n_users = self.data_struct.get_user_num()
            n_items = self.data_struct.get_item_num()

            # === 1. Get all user/item embeddings from LightGCN ===
            self.model.eval()
            with torch.no_grad():
                all_user_emb, all_item_emb = self.model.propagate()
                # shapes: (n_users, embed_dim), (n_items, embed_dim)

            self.full_rankings = np.zeros((n_users, n_items), dtype=int)

            for user in range(n_users):
               # Get user embedding
               user_embed = all_user_emb[user].unsqueeze(0)   # (1, embed_dim)

               # Compute scores for all items (dot product)
               scores = torch.matmul(user_embed, all_item_emb.T).squeeze(0).cpu().numpy()

               # Push training items to -inf
               test_items = [x[1] for x in self.test_data if x[0] == user and x[1] < n_items]
               scores[test_items] = -np.inf

               # Sort descending
               self.full_rankings[user] = np.argsort(-scores)

               # # Optionally move ground-truth test items on top
               if use_test:
                  test_items = [x[1] for x in self.test_data if x[0] == user and x[1] < n_items]
                  for idx, item in enumerate(test_items):
                     if item in self.full_rankings[user]:
                        current_pos = np.where(self.full_rankings[user] == item)[0][0]
                        self.full_rankings[user][idx], self.full_rankings[user][current_pos] = (
                           self.full_rankings[user][current_pos],
                           self.full_rankings[user][idx])


        elif self.config['rec_model'] == 'SASRec':
            n_users = self.data_struct.get_user_num()
            n_items = self.data_struct.get_item_num()

            self.full_rankings = np.zeros((n_users, n_items), dtype=int)
            self.model.eval()
            device = next(self.model.parameters()).device

            with torch.no_grad():
               for start in range(0, n_users, batch_size):
                  end = min(start + batch_size, n_users)
                  batch_users = list(range(start, end))

                  # Build input sequences for batch
                  batch_seqs = []
                  for u in batch_users:
                     # Get user interaction sequence from train_data
                     user_items = [x[1] for x in self.test_data if x[0] == u]
                     padded_seq = _pad_sequence(user_items, self.model.max_seq_len)
                     batch_seqs.append(padded_seq)

                  batch_seqs = torch.tensor(batch_seqs, dtype=torch.long, device=device)

                  # Forward pass: get sequence embeddings
                  seq_out = self.model(batch_seqs)  # (B, L, H)
                  seq_out_last = seq_out[:, -1, :]  # use last position (B, H)

                  # All item embeddings
                  all_item_emb = self.model.item_embedding.weight[:n_items, :]  # (n_items, H)

                  # Compute scores
                  scores = torch.matmul(seq_out_last, all_item_emb.T)  # (B, n_items)
                  scores = scores.cpu().numpy()

                  # Mask training items
                  for i, u in enumerate(batch_users):
                     test_items = [x[1] for x in self.test_data if x[0] == u and x[1] < n_items]
                     scores[i, test_items] = -np.inf  # push train items to the end

                     ranking = np.argsort(-scores[i])  # full ranking by score (highest first)
                     if use_test:
                        test_items = [x[1] for x in self.test_data if x[0] == u and x[1] < n_items]
                        # Keep only test items that appear in ranking
                        test_items_in_ranking = [item for item in ranking if item in test_items]
                        # Take at most 5 test items
                        top_test_items = test_items_in_ranking[:5]
                        # Remaining items (exclude the ones we forced to the top)
                        other_items = [item for item in ranking if item not in top_test_items]

                        # New ranking: top test items first, then the rest in score order
                        ranking = np.array(top_test_items + other_items)
                     # Store final ranking
                     self.full_rankings[u] = ranking

    def get_rankings_over_all_test_items(self, return_original_ids: bool = False):
        """
        For each user, rank *all items that appear in the test set*.

        - Global candidate pool = union of item_idx's from self.test_data.
        - For each user_idx u (0..n_users-1), we compute scores for all
          these test items and sort them in descending order.
        - If `return_original_ids`:
            returns {orig_user_id: np.array(orig_item_ids_sorted)}
          else:
            returns {user_idx: np.array(item_idx_sorted)}.

        Assumes:
          - rec_model == "MF"
          - self.test_data is a list of (user_idx, item_idx, rating)
            after reindexing in create_train_data().
          - self.model is MF, trained.
          - self.user2idx / self.idx2user / self.item2idx / self.idx2item exist.
        """
        if self.config["rec_model"] != "MF":
            raise ValueError("get_rankings_over_all_test_items is implemented only for MF.")

        self.model.eval()

        # 1) Build global candidate item set from *all* test entries
        test_item_set = sorted({i for (u, i, r) in self.test_data})
        if not test_item_set:
            return {}

        # Deduplicated list of candidate item_idx's
        candidate_items = np.array(test_item_set, dtype=np.int64)
        n_candidates = candidate_items.shape[0]

        # 2) Basic sizes
        n_users = self.model.n_users  # reindexed users

        # 3) Prepare tensors on the correct device
        item_tensor = torch.tensor(candidate_items, dtype=torch.long, device=self.device)

        rankings_by_user = {}

        with torch.no_grad():
            for u in range(n_users):
                # user_idx repeated for all candidate items
                user_tensor = torch.full(
                    (n_candidates,),
                    u,
                    dtype=torch.long,
                    device=self.device,
                )

                # MF scores (includes biases)
                scores = self.model(user_tensor, item_tensor)  # (n_candidates,)
                scores = scores.detach().cpu().numpy()

                # sort candidate_items by scores desc
                order = np.argsort(-scores)
                ranked_items = candidate_items[order]  # still in reindexed item space

                if return_original_ids:
                    orig_user = self.idx2user[u]
                    orig_items = np.array(
                        [self.idx2item[i] for i in ranked_items],
                        dtype=np.int64,
                    )
                    rankings_by_user[orig_user] = orig_items
                else:
                    rankings_by_user[u] = ranked_items

        return rankings_by_user
    def debug_user_item_scores(self, num_users: int = 3, num_items: int = 5):
        """
        Print MF scores for a few users and items to see if they differ.
        Uses the global test item pool.
        """
        if self.config["rec_model"] != "MF":
            print("Only for MF.")
            return

        self.model.eval()

        # global candidate set of test items (reindexed)
        test_item_set = sorted({i for (u, i, r) in self.test_data})
        if not test_item_set:
            print("No test items.")
            return

        candidate_items = np.array(test_item_set, dtype=np.int64)
        n_candidates = candidate_items.shape[0]
        k_items = min(num_items, n_candidates)

        # pick first k_items from candidate pool
        item_subset = candidate_items[:k_items]
        print("Item subset (reindexed):", item_subset)

        item_tensor = torch.tensor(item_subset, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for u in range(num_users):
                user_tensor = torch.full(
                    (k_items,),
                    u,
                    dtype=torch.long,
                    device=self.device,
                )
                scores = self.model(user_tensor, item_tensor).detach().cpu().numpy()
                print(f"user {u} scores:", scores)

