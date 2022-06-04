import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """
    BERT4REC Layernorm.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    """
    Gelu implementation. BERT4REC mentions its usage
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class PositionwiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that applies to
    the last dimension, which means the same dense layers are used for each position item in the sequence, so called position-wise.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# -----------------Attention modules
class Attention(nn.Module):
    """
    Compute Scaled Dot Product Attention
    """

    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[float] = None,
    ):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            _MASKING_VALUE = (
                -1e9 if query.dtype == torch.float32 else -1e4
            )  # 32 & 16bit support
            scores = scores.masked_fill(mask == 0, _MASKING_VALUE)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = torch.nn.functional.dropout(
                p_attn,
                p=dropout,
                training=self.training,
            )  # Change dropout to functional for torchscript compatibility
            # p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Module for attention mechanisms which runs through an attention mechanism several times in parallel.
    """

    def __init__(self, h, d_model, dropout=0.1, n_layers=3):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = dropout  # nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# ----------------Embedding


class PositionalEmbedding(nn.Module):
    """
    Computes positional embedding following "Attention is all you need"
    """

    def __init__(self, max_length, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_length, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_length, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )  # TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_length=max_length, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)
        x += self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)


# -------------Transformer block


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden,
        attn_heads,
        feed_forward_hidden,
        dropout,
        n_attention_layers=3,
    ):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads,
            d_model=hidden,
            dropout=dropout,
            n_layers=n_attention_layers,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.input_norm = LayerNorm(hidden)
        self.input_dropout = nn.Dropout(p=dropout)

        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_norm = LayerNorm(hidden)
        self.ooutput_dropout = nn.Dropout(p=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Input sublayer
        x_b = self.input_norm(x)
        x_b = self.attention(x_b, x_b, x_b, mask=mask)
        x_b = self.input_dropout(x_b)
        x = x + x_b

        # Output sublayer
        x_b = self.output_norm(x)
        x_b = self.feed_forward(x_b)
        x_b = self.ooutput_dropout(x_b)
        x = x + x_b

        # x = x + self.input_dropout(sublayer(self.norm(x)))

        # With lambdas - Is incompatible with torchscript so i moved it to this forward fn
        # x = self.input_sublayer(
        #     x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        # )
        # x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
