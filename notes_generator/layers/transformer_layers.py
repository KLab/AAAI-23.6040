# https://github.com/novdov/music-transformer/blob/master/music_transformer/modules/attention.py
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Apply multi-head attention to input data

    Parameters
    ----------
    num_heads : int
        Number of parallel attention heads.
    d_model : int
        Number of expected features in the encoder/decoder of the model.
    dropout : float
        Rate of Dropout layer after computing attention.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.depth = d_model // num_heads

        projection_inout = (self.d_model, self.d_model)
        self.query_projection = nn.Linear(*projection_inout)
        self.key_projection = nn.Linear(*projection_inout)
        self.value_projection = nn.Linear(*projection_inout)
        self.attention_projection = nn.Linear(*projection_inout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        query : torch.Tensor
            Tensor of shape (batch, query_len, d_model).
        key : torch.Tensor
            Tensor of shape (batch, memory_len, d_model).
        value : torch.Tensor
            Tensor of shape (batch, memory_len, d_model).
        mask : Optional[torch.Tensor]
            Mask tensor of shape (memory_len, memory_len)
            containing boolean-like values.
            Attention is prevent for certain position of memory that
            corresponding value of the mask tensor is zero.

        Returns
        -------
        output : torch.Tensor
            Tensor of shape (batch, query_len, d_model)
        weights : torch.Tensor
            Tensor of shape (batch, query_len, memory_len)

        """
        queries = self.split_heads(self.query_projection(query))
        keys = self.split_heads(self.key_projection(key))
        values = self.split_heads(self.value_projection(value))

        logits = torch.matmul(queries, keys.transpose(-2, -1))
        logits = logits / math.sqrt(self.depth)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        weights = self.dropout(F.softmax(logits, dim=-1))
        output = torch.matmul(weights, values)
        output = output.permute(0, 2, 1, 3).contiguous().view(output.size(0), -1, self.d_model)
        return self.attention_projection(output), weights

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert the shape of tensors for multi-head attention.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, seq_len, num_heads, depth),
            where `depth = d_model // num_heads`.

        """
        batch_size, _, _ = tensor.size()
        return tensor.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)


class RelativeGlobalAttention(MultiheadAttention):
    def __init__(
        self,
        num_heads: int,
        max_relative_position: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__(num_heads, d_model, dropout)
        self.max_relative_position = max_relative_position
        length = max(max_relative_position, self.depth)
        range_vec = torch.arange(length)
        relative_mat = torch.clamp(
            range_vec[None, :] - range_vec[:, None], -max_relative_position, +max_relative_position
        )
        relative_mat = relative_mat + max_relative_position
        self.relative_embedding = relative_mat[:max_relative_position, : self.depth].float()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.split_heads(self.query_projection(query))
        keys = self.split_heads(self.query_projection(key))
        values = self.split_heads(self.query_projection(value))

        length_q = queries.shape[2]
        length_k = keys.shape[2]

        logits = torch.matmul(queries, keys.transpose(-2, -1))
        key_relative_embedding = self._get_relative_embedding_left(length_q).to(query.device)
        rel_logits = torch.einsum("bhld,md->bhlm", [queries, key_relative_embedding])
        rel_logits = self._skew(rel_logits, length_k)
        logits = (logits + rel_logits) / math.sqrt(self.depth)
        # logits = logits / math.sqrt(self.depth)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        weights = self.dropout(F.softmax(logits, dim=-1))
        output = torch.matmul(weights, values)
        output = output.permute(0, 2, 1, 3).contiguous().view(output.size(0), -1, self.d_model)
        return self.attention_projection(output), weights

    def _get_relative_embedding_left(self, length: int) -> torch.Tensor:
        starting_point = max(0, self.max_relative_position - length)
        return self.relative_embedding[starting_point:, :]

    @staticmethod
    def _skew(rel_logits: torch.Tensor, length_key) -> torch.Tensor:
        batch_size, num_heads, length_q, _ = rel_logits.size()
        assert rel_logits.shape[-2] == rel_logits.shape[-1]
        # (B, H, L, L) -> (B, H, L, 1 + L)
        rel_logits = F.pad(rel_logits, [1, 0, 0, 0])
        # (B, H, L, 1 + L) -> (B, H, 1 + L, L)
        rel_logits = rel_logits.reshape(batch_size, num_heads, 1 + length_q, length_q)
        # (B, H, 1 + L, L) -> (B, H, L, L)
        rel_logits = rel_logits[:, :, 1:, :]
        if length_key > length_q:  # M > L
            # (B, H, L, L) -> (B, H, L, M)
            rel_logits = F.pad(rel_logits, [0, length_key - length_q, 0, 0])
        elif length_key < length_q:
            # (B, H, L, L) -> (B, H, L, M)
            rel_logits = rel_logits[:, :, :, :length_key]
        return rel_logits


class RelLearnbaleAttention(MultiheadAttention):
    """Attention layer for TransformerXL model.

    Parameters
    ----------
    num_heads : int
        Number of parallel attention heads.
    d_model : int
        Number of expected features in the encoder/decoder of the model.
    dropout : float
        Rate of Dropout layer after computing attention.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__(num_heads, d_model, dropout)
        projection_inout = (self.d_model, self.d_model)
        self.pos_proejction = nn.Linear(*projection_inout)

    # noinspection PyMethodOverriding
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_: torch.Tensor,
        u: nn.Parameter,
        v: nn.Parameter,
        mem: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        query : torch.Tensor
            Tensor of shape (batch, query_len, d_model).
        key : torch.Tensor
            Tensor of shape (batch, memory_len, d_model).
        value : torch.Tensor
            Tensor of shape (batch, memory_len, d_model).
        pos_ : torch.Tensor
            Tensor of shape (memory_len + query_len, d_model)
            containing positional embedding.
        u : nn.Parameter
            Learnable parameter of shape (n_head, d_model // n_head)
            containing learnable global content bias.
        v : nn.Parameter
            Learnable parameter of shape (n_head, d_model // n_head)
            containing learnable global positional bias.
        mem : torch.Tensor
            Tensor of shape (batch, mem_len, d_model)
            containing memory from previous sentence.
        mask : Optional[torch.Tensor]
            Mask tensor of shape (query_len, memory_len + query_len)
            containing boolean-like values.
            Attention is prevent for certain position of memory that
            corresponding value of the mask tensor is zero.

        Returns
        -------
        output : torch.Tensor
            Tensor of shape (batch, query_len, d_model)
        weights : torch.Tensor
            Tensor of shape (batch, n_head, query_len, memory_len)

        """
        batch = query.size(0)
        # query: (B, L, E)
        # key, value: (B, M, E)
        if mem.size(1) > 0:
            key = torch.cat((mem, key), dim=1)
            value = torch.cat((mem, value), dim=1)
        # (B, L, H, D)
        queries = self.query_projection(query).view(batch, -1, self.num_heads, self.depth)
        # (B, M, H, D)
        keys = self.key_projection(key).view(batch, -1, self.num_heads, self.depth)
        values = self.value_projection(value).view(batch, -1, self.num_heads, self.depth)
        # (M, H, D)
        pos_ = self.pos_proejction(pos_).view(-1, self.num_heads, self.depth)

        # term (a) (c)
        content_attention = torch.einsum("blhd,bmhd->bhlm", [(queries + u), keys])
        # term (b) (d)
        pos_attention = torch.einsum("blhd,mhd->bhlm", [(queries + v), pos_])
        pos_attention = self._skew(pos_attention)
        logits = content_attention + pos_attention
        logits = logits / math.sqrt(self.depth)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        weights = self.dropout(F.softmax(logits, dim=-1))
        # (B, H, L, M), (B, H, M, D) -> (B, L, H, D)
        output = torch.einsum("bhlm,bmhd->blhd", [weights, values])
        # (B, L, H x D)
        output = output.contiguous().view(output.size(0), -1, self.d_model)
        return self.attention_projection(output), weights

    @staticmethod
    def _skew(rel_logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, length_q, length_m = rel_logits.size()
        # (B, H, L, M) -> (B, H, L, 1 + M)
        rel_logits = F.pad(rel_logits, [1, 0, 0, 0])
        # (B, H, L, 1 + M) -> (B, H, 1 + M, L)
        rel_logits = rel_logits.view(batch_size, num_heads, 1 + length_m, length_q)
        # (B, H, 1 + M, L) -> (B, H, L, M)
        rel_logits = rel_logits[:, :, 1:, :].view(batch_size, num_heads, length_q, length_m)
        return rel_logits


class LocalRNN(nn.Module):
    def __init__(self, output_dim, ksize):
        super(LocalRNN, self).__init__()
        self.ksize = ksize
        self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())
        idx = [
            i
            for j in range(self.ksize - 1, 10000, 1)
            for i in range(j - (self.ksize - 1), j + 1, 1)
        ]
        self.select_index = torch.LongTensor(idx)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x = self.get_k(x)  # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
        return h.view(batch, l, d_model)

    def get_k(self, x):
        batch_size, l, d_model = x.shape
        x = F.pad(x, [0, 0, self.ksize - 1, 0])
        key = torch.index_select(x, 1, self.select_index[: self.ksize * l].to(x.device))
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
    def __init__(self, output_dim, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(output_dim, ksize)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.local_rnn(x)))


# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        is_r_transformer: bool = False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.self_attn = self.build_attention()
        self.is_r_transformer = is_r_transformer
        if is_r_transformer:
            self.local_rnn = LocalRNNLayer(d_model, ksize=7, dropout=0.1)

    def build_attention(self):
        return MultiheadAttention(self.nhead, self.d_model, self.dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        subsequent_mask = merge_mask(src_mask, src_key_padding_mask)
        if self.is_r_transformer:
            src = self.local_rnn(src)
        src2 = self.self_attn(src, src, src, mask=subsequent_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RGATransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_relative_position: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        self.max_relative_position = max_relative_position
        super(RGATransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout)

    def build_attention(self):
        return RelativeGlobalAttention(
            self.nhead, self.max_relative_position, self.d_model, self.dropout
        )


class TransformerXLEncoderLayer(TransformerEncoderLayer):
    def build_attention(self):
        return RelLearnbaleAttention(self.nhead, self.d_model, self.dropout)

    def forward(self, src, pos, u, v, mem, src_mask=None, src_key_padding_mask=None):
        subsequent_mask = merge_mask(src_mask, src_key_padding_mask)
        if self.is_r_transformer:
            src = self.local_rnn(src)
        src2 = self.self_attn(src, src, src, pos, u, v, mem, mask=subsequent_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        # Implementation of Feedforward model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.self_attn, self.multihead_attn = self.build_attention()

    def build_attention(self):
        self_attn = MultiheadAttention(self.nhead, self.d_model, dropout=self.dropout)
        multihead_attn = MultiheadAttention(self.nhead, self.d_model, dropout=self.dropout)
        return self_attn, multihead_attn

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        subsequent_mask = merge_mask(tgt_mask, tgt_key_padding_mask)
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=subsequent_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class RGATransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_relative_position: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        self.max_relative_position = max_relative_position
        super(RGATransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout)

    def build_attention(self):
        self_attn = RelativeGlobalAttention(
            self.nhead, self.max_relative_position, self.self.d_model, dropout=self.dropout
        )
        multihead_attn = RelativeGlobalAttention(
            self.nhead, self.max_relative_position, self.d_model, dropout=self.dropout
        )
        return self_attn, multihead_attn


class TransformerXL(nn.Module):
    """A TransformerXL model.

    Features below are introduced to vanilla Transformer:
        * Recurrent mechanism
        * Relative positional encoding

    Parameters
    ----------
    d_model : int
        Number of expected features in the encoder/decoder of the model.
    nhead : int
        Number of parallel attention heads.
    num_layers : int
        Number of sub-encoder-layers in the encoder. default: `6`
    dim_feedforward : int
        Size of hidden layer of feed forward network in encoder.
        default: `2048`
    max_mem_length : int
        Maximum length of memory that attention is applied to.
        default: `100`
    dropout : float
        Rate of Dropout layer after computing attention.
        default: `0.1`
    has_mask : bool
        If `True`, source mask will be applied so as to prevent an attention
        to future information.
        default: `True`
    is_r_transformer : bool
        If `True`, insert Local RNN structure before computing attention
        in each encoder layers.
        default: `False`
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_mem_length: int = 100,
        dropout: float = 0.1,
        has_mask: bool = True,
        is_r_transformer: bool = False,
    ):

        assert d_model % nhead == 0, f"d_model: {d_model}, nhead: {nhead}"
        super(TransformerXL, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
                TransformerXLEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, is_r_transformer
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = num_layers
        self.max_mem_length = max_mem_length
        self.has_mask = has_mask
        # content bias
        self.u = nn.Parameter(torch.zeros(nhead, d_model // nhead), requires_grad=True)
        nn.init.xavier_normal_(self.u)
        # pos bias
        self.v = nn.Parameter(torch.zeros(nhead, d_model // nhead), requires_grad=True)
        nn.init.xavier_normal_(self.v)
        self.pos_enb = PositionalEncoding(d_model, apply_positional_encoding="add")

    def create_mask(self, q_len: int, m_len: int) -> torch.Tensor:
        """Create an attention mask tensor

        Parameters
        ----------
        q_len : int
            Size of query sequence.
        m_len : int
            Size of memory sequence.

        Returns
        -------
        torch.Tensor
            Tensor of shape (q_len, q_len + m_len) containing boolean values,
            where `True` intends that place to be masked.

        """
        return torch.triu(torch.ones(q_len, q_len + m_len), diagonal=m_len + 1) == 0

    def init_mem(self, batch_size: int, device: str) -> List[torch.Tensor]:
        """Initialize memory sequence.

        Parameters
        ----------
        batch_size : int
            Size of minibatch.
        device : str
            The desired device in which the computation is performed.
            choices: [`cpu`, `cuda`].

        Returns
        -------
        List[torch.Tensor]
            List of length `num_layers` containing initial memory tensors
            whose shapes are (batch, 0, d_model).

        """
        param = next(self.parameters())
        return [
            torch.zeros(batch_size, 0, self.d_model, dtype=param.dtype, device=device)
            for _ in range(self.n_layers)
        ]

    def _update_mems(
        self, memory: List[torch.Tensor], outputs: List[torch.Tensor], mlen: int, qlen: int
    ) -> List[torch.Tensor]:
        """Update memory with new output

        Parameters
        ----------
        memory : torch.Tensor
            Tensor of shape (num_layers, mlen)
             containing previous memory.
        outputs : torch.Tensor
            Tensor of shape (num_layers, )
             containing outputs from encoder layers.
        mlen : int
            Length of memory.
        qlen : int
            Length of query.

        Returns
        -------
        new_mems : List[torch.Tensor]
            List of length `num_layers` containing tensors of shape (num_layers, mem_length)
            containing updated memory, where mem_length is calculated as below:
                `mem_length = min(mlen + qlen, max_mem_length)`

        """
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            start_idx = max(0, end_idx - self.max_mem_length)
            for mem, output in zip(memory, outputs):
                cat = torch.cat([mem, output], dim=1)
                new_mems.append(cat[:, start_idx:end_idx].detach())
        return new_mems

    def forward(self, tgt: torch.Tensor, memory: List[torch.Tensor] = None):
        """

        Parameters
        ----------
        tgt : torch.Tensor
            Tensor of shape (batch, query_len, d_model)
        memory : List[torch.Tensor]
            List of length `num_layers` containing tensors of shape (num_layers, mem_length)
            containing memory, where mem_length is calculated as below:
                `mem_length = min(mlen + qlen, max_mem_length)`

        Returns
        -------
        output : torch.Tensor
            Tensor of shape (batch, query_len, d_model)
        new_mems : List[torch.Tensor]
            List of length `num_layers` containing tensors of shape (num_layers, mem_length)
            containing updated memory, where mem_length is calculated as below:
                `mem_length = min(mlen + qlen, max_mem_length)`

        """
        if memory is None:
            memory = self.init_mem(tgt.size(0), tgt.device)
        mlen = memory[0].size(1)
        qlen = tgt.size(1)
        pos = torch.zeros(1, mlen + qlen, self.d_model).to(tgt.device)
        pos = self.pos_enb(pos).squeeze(0) / math.sqrt(self.d_model)
        output = tgt
        mask = None
        if self.has_mask:
            mask = self.create_mask(qlen, mlen).to(tgt.device)
        new_mems = []
        for mem, layer in zip(memory, self.encoder_layers):
            new_mems.append(output.detach())
            output = layer(output, pos, self.u, self.v, mem, mask)
        new_mems = self._update_mems(memory, new_mems, mlen, qlen)
        assert len(new_mems) == self.n_layers
        return output, new_mems


class ScaledEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    POSSIBLE_METHODS = ("add", "concat")

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        apply_positional_encoding: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_positional_encoding = apply_positional_encoding

        self.positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_positional_encoding not in self.POSSIBLE_METHODS:
            raise ValueError(
                f"{self.apply_positional_encoding} should be one of {self.POSSIBLE_METHODS}s"
            )

        self.positional_encoding = self.positional_encoding.to(x.device)
        if self.apply_positional_encoding == "add":
            x = x + self.positional_encoding[:, : x.size(1)]
        else:
            batch_size, seq_len, _ = x.size()
            x = torch.cat(
                [x, self.positional_encoding[:, :seq_len].repeat(batch_size, 1, 1)], dim=-1
            )
        return self.dropout(x)


def create_subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions."""
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def merge_mask(mask, key_padding_mask):
    if key_padding_mask is not None:
        key_padding_mask = ~key_padding_mask[:, None, None, :]
        subsequent_mask = key_padding_mask & mask
    else:
        subsequent_mask = mask
    return subsequent_mask
