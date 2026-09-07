import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pykeops.torch import LazyTensor

class SparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        kq_dim: int,
        val_dim: int,
        num_heads: int,
        k: int,
        head_agg: str,
        random_attention: bool = False,
        random_fraction: float = 0.1,
    ):
        """
        Args:
            dim: the dimension of the input
            kq_dim: the dimension of the key and query space
            val_dim: the dimension of the value space
            num_heads: the number of heads, so also the number of different graphs that are generated
            k: for each query, the number of nearest keys to consider
            head_agg: how to aggregate the heads. Options are "Linear" and "None"
            random_attention: if True, first select `random_fraction` keys randomly, then select the k nearest keys
            random_fraction: the fraction of keys to select randomly
        """
        super().__init__()

        self.dim = dim
        self.kq_dim = kq_dim
        self.val_dim = val_dim
        self.num_heads = num_heads
        self.k = k
        self.random_attention = random_attention
        self.random_fraction = random_fraction

        self.MQs = nn.Linear(dim, kq_dim * num_heads)  # query transformations
        self.MKs = nn.Linear(dim, kq_dim * num_heads)  # key transformations
        self.MVs = nn.Linear(dim, val_dim * num_heads)  # value transformations

        self.head_agg = head_agg
        if head_agg == "Linear":
            self.MO = nn.Linear(val_dim * num_heads, dim)  # output transformations
        elif head_agg in ("none", "None", None):
            assert dim == val_dim * num_heads
            pass
        else:
            raise ValueError(f"Unknown head_agg: {head_agg}, type: {type(head_agg)}")

    def reset_parameters(self):
        self.MQs.reset_parameters()
        self.MKs.reset_parameters()
        self.MVs.reset_parameters()
        self.MO.reset_parameters()

    def nearest_k_keys(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute the nearest k keys for each query. Return their indices in a [**, num_heads, Nq, k] tensor.

        Args:
            queries: [**, num_heads, Nq, kq_dim]
            keys: [**, num_heads, Nk, kq_dim]

        Returns:
            [**, num_heads, Nq, k]
        """

        # Raise an error if the memory has almost run out
        free_memory, _ = torch.cuda.mem_get_info()
        if free_memory < 1024 * 1024 * 50: # 50 MB
            raise RuntimeError("Memory almost full - aborting PyKeOps operation")


        # --- Compute the nearest k keys: [**, num_heads, N, k] ---

        # [**, num_heads, N, 1, kq_dim]
        queries_extended: LazyTensor = LazyTensor(queries[..., :, :, None, :])
        #  [**, num_heads, 1, N, kq_dim]
        keys_extended: LazyTensor = LazyTensor(keys[..., :, None, :, :])
        # [**, num_heads, N, N]
        full_attention_weights: LazyTensor = (queries_extended * keys_extended).sum(
            -1
        ) / math.sqrt(self.kq_dim)

        # [**, num_heads, N, k]
        ndims = len(full_attention_weights.shape)
        assert ndims in [3,4]
        nearest_key_indices = (-full_attention_weights).argKmin(self.k, dim=ndims-1)

        return nearest_key_indices

    def sparse_self_attention(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [**, num_heads, Nq, kq_dim]
            keys: [**, num_heads, Nk, kq_dim]
            values: [**, num_heads, Nk, val_dim]

        Returns:
            [**, num_heads, Nq, val_dim]
        """

        trailing_dimensions = queries.shape[:-3]
        Nq = queries.shape[-2]
        Nk = keys.shape[-2]
        assert queries.shape == (*trailing_dimensions, self.num_heads, Nq, self.kq_dim)
        assert keys.shape == (*trailing_dimensions, self.num_heads, Nk, self.kq_dim)
        assert values.shape == (*trailing_dimensions, self.num_heads, Nk, self.val_dim)
        # assert queries.is_contiguous()

        # --- Compute the attention weights: [**, num_heads, Nq, k] ---

        # [**, num_heads, Nq, k]
        nearest_key_indices = self.nearest_k_keys(queries, keys)
        assert nearest_key_indices.shape == (
            *trailing_dimensions,
            self.num_heads,
            Nq,
            self.k,
        )
        # assert nearest_key_indices.is_contiguous()

        # the k keys nearest to each query
        # [**, num_heads, Nq, k, kq_dim]
        nearest_keys = torch.gather(
            input=keys.unsqueeze(-2).expand(
                *keys.shape[:-1], self.k, self.kq_dim
            ),  # [**, num_heads, Nk, k, kq_dim]
            dim=-3,
            index=nearest_key_indices.unsqueeze(-1).expand(
                *nearest_key_indices.shape, self.kq_dim
            ),  # [**, num_heads, Nq, k, kq_dim]
            # sparse_grad=True,
        )
        assert nearest_keys.shape == (
            *trailing_dimensions,
            self.num_heads,
            Nq,
            self.k,
            self.kq_dim,
        )
        # assert nearest_keys.is_contiguous()

        # the values corresponding to those keys
        # [**, num_heads, Nq, k, val_dim]
        nearest_values = torch.gather(
            input=values.unsqueeze(-2).expand(
                *keys.shape[:-1], self.k, self.val_dim
            ),  # [**, num_heads, Nk, k, val_dim]
            dim=-3,
            index=nearest_key_indices.unsqueeze(-1).expand(
                *nearest_key_indices.shape, self.val_dim
            ),  # [**, num_heads, N, k, val_dim]
            # sparse_grad=True,
        )
        assert nearest_values.shape == (
            *trailing_dimensions,
            self.num_heads,
            Nq,
            self.k,
            self.val_dim,
        )
        # assert nearest_values.is_contiguous()

        # [**, num_heads, Nq, k, kq_dim]
        queries_extended = queries.unsqueeze(-2).expand(
            *queries.shape[:-1], self.k, self.kq_dim
        )
        assert queries_extended.shape == (
            *trailing_dimensions,
            self.num_heads,
            Nq,
            self.k,
            self.kq_dim,
        )
        # assert queries_extended.is_contiguous()

        # [**, num_heads, Nq, k]
        largest_attention_weights = (queries_extended * nearest_keys).sum(
            -1
        ) / math.sqrt(self.kq_dim)
        largest_attention_weights = F.softmax(largest_attention_weights, dim=-1)
        assert largest_attention_weights.shape == (
            *trailing_dimensions,
            self.num_heads,
            Nq,
            self.k,
        )
        # assert largest_attention_weights.is_contiguous()

        # Apply the attention weights to the values
        # [**, num_heads, Nq, val_dim]
        out = (largest_attention_weights.unsqueeze(-1) * nearest_values).sum(
            dim=-2
        )  # sum over k nearest keys for each query
        assert out.shape == (*trailing_dimensions, self.num_heads, Nq, self.val_dim)
        # assert out.is_contiguous()

        return out

    def split_heads(self, x: torch.Tensor, small_dim: int):
        """
        Rearrange the input from [**, N, num_heads*small_dim] to [**, num_heads, N, small_dim]

        The important cases are small_dim = kq_dim and small_dim = val_dim
        """
        assert x.shape[-1] == self.num_heads * small_dim
        assert x.ndim >= 2
        x = (
            x.reshape(*x.shape[:-1], self.num_heads, small_dim)
            .transpose(-3, -2)
            .contiguous()
        )
        assert x.is_contiguous()  # more efficient for pykeops
        return x

    def combine_heads(self, x: torch.Tensor, small_dim: int):
        """
        Rearrange the input from [**, num_heads, N, small_dim] to [**, N, num_heads * small_dim]
        """
        assert x.shape[-3] == self.num_heads
        assert x.shape[-1] == small_dim
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], self.num_heads * small_dim)
        assert x.is_contiguous()  # more efficient for pykeops
        return x
    
    def select_random(self, keys: torch.Tensor, values: torch.Tensor):
        """
        select a random fraction of the keys, plus their corresponding values

        Args:
            keys: [**, num_heads, Nk, kq_dim]
            values: [**, num_heads, Nk, val_dim]
        """

        first_dimensions = keys.shape[:-2]
        Nk = keys.shape[-2]

        num_random = int(Nk * self.random_fraction)

        # defines the keys and values that will be selected
        # [**, num_heads, Nk*random_fraction]
        node_indices = torch.empty((np.prod(first_dimensions), num_random), dtype=torch.int64, device=keys.device)
        for i in range(np.prod(first_dimensions)):
            node_indices[i] = torch.randperm(Nk, device=keys.device)[:num_random]
        node_indices = node_indices.view(*first_dimensions, num_random)

        # [**, num_heads, Nk*random_fraction, kq_dim]
        keys_random = torch.gather(
            input=keys,
            dim=-2,
            index=node_indices.unsqueeze(-1).expand(*node_indices.shape, keys.shape[-1])    # [**, num_heads, Nk*random_fraction, kq_dim]
        )
        values_random = torch.gather(
            input=values,
            dim=-2,
            index=node_indices.unsqueeze(-1).expand(*node_indices.shape, values.shape[-1])    # [**, num_heads, Nk*random_fraction, val_dim]
        )

        return keys_random, values_random


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [**, N, dim]

        Returns:
            [**, N, dim]

        TODO: implement masking to ignore padding nodes
        TODO: try attention dropout
        """

        # [**, num_heads, N, kq_dim]
        queries = self.split_heads(self.MQs(x), small_dim=self.kq_dim)
        keys = self.split_heads(self.MKs(x), small_dim=self.kq_dim)
        # [**, num_heads, N, val_dim]
        values = self.split_heads(self.MVs(x), small_dim=self.val_dim)

        if self.random_attention:
            # [**, num_heads, N*random_fraction, kq_dim]
            # [**, num_heads, N*random_fraction, val_dim]
            keys, values = self.select_random(keys, values)
        
        # [**, num_heads, N, val_dim]
        x = self.sparse_self_attention(queries, keys, values)
        # [**, N, num_heads * val_dim]
        x = self.combine_heads(x, small_dim=self.val_dim)

        if self.head_agg == "Linear":
            # [**, N, dim]
            x = self.MO(x)
        elif self.head_agg in ("none", "None", None):
            pass
        else:
            raise ValueError(f"Unknown head_agg: {self.head_agg}")

        return x