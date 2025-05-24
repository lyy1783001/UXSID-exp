import torch
from torch import nn
from torch import Tensor, Size
from torch.nn.functional import linear # _in_projection_packed
import copy
from typing import Optional, Any, Union, Callable, List, Tuple
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.pytorch.layers.attentions import ScaledDotProductAttention

import numbers


__all__ = [
    "AbridgedTransformerDecoderLayer",
    "NoParamLayer",
    "Encoder",
    "MultiHeadTargetAttention",
]

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AbridgedTransformerDecoderLayer(nn.Module):
    r"""AbridgedTransformerDecoderLayer is made up of multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = AbridgedTransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = AbridgedTransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ['batch_first', 'norm_first']

    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5, 
        batch_first: bool = True, 
        norm_first: bool = False,
        device=None, 
        dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)  # the dropout is to zero attn_weights
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        # self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        # self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)


    def forward(
            self, 
            tgt: Tensor, 
            memory: Tensor,
            memory_mask: Optional[Tensor] = None, 
            memory_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional). 
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            tmp_x, weights = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + tmp_x
            x = x + self._ff_block(self.norm3(x))
        else:
            tmp_x, weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + tmp_x)
            x = self.norm3(x + self._ff_block(x))

        return x, weights

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, weights = self.multihead_attn(x, mem, mem,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=True)
        return self.dropout2(x), weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class NoParamLayer(nn.Module):

    def forward(self, query, key_value):
        attn_score = torch.bmm(query, key_value.transpose(1, 2))   # [B, L_q, D] x [B, L_kv, D] -> [B, L_q, L_kv]
        attn_score = torch.softmax(attn_score, dim=-1)
        output = attn_score @ key_value
        return output, attn_score



class Encoder(nn.Module):
    r"""A stack of N encoder layers

    Args:
        encoder_layer: an instance of the AbridgedTransformerDecoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = AbridgedTransformerDecoderLayer(d_model=512, nhead=8)
        >>> encoder = Encoder(encoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = encoder(tgt, memory)
    """
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, query, key_value, key_padding_mask=None, return_last_projected_value=False):
        output = query
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                last_query = output
            output, weights = layer(output, key_value, memory_key_padding_mask=key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        if return_last_projected_value:
            last_projected_value = self.get_last_projected_value(last_query, key_value)
            return output, weights, last_projected_value
        else:
            return output, weights
 
    
    def get_last_projected_value(self, query, key_value):
        '''
        self.layers[-1].multihead_attn.in_proj_weight: (3D, D)
        self.layers[-1].multihead_attn.in_proj_bias: (3D)
        '''
        q = query.transpose(0, 1)       # (K, B, D)
        k = key_value.transpose(0, 1)    # (L, B, D)
        v = k
        projected_q, projected_k, projected_v = _in_projection_packed(
                                                    q, k, v,
                                                    self.layers[-1].multihead_attn.in_proj_weight,
                                                    self.layers[-1].multihead_attn.in_proj_bias)
        return projected_v.transpose(0, 1)



def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

  

class MultiHeadTargetAttention(nn.Module):
    def __init__(self,
                 q_dim=64,
                 kv_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True):
        super(MultiHeadTargetAttention, self).__init__()
        if not use_qkvo:
            attention_dim = q_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(q_dim, attention_dim, bias=False)
            self.W_k = nn.Linear(kv_dim, attention_dim, bias=False)
            self.W_v = nn.Linear(kv_dim, attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, q_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key = self.W_k(history_sequence)
            value = self.W_v(history_sequence)
        else:
            query, key, value = target_item, history_sequence, history_sequence

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        return output
    

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        初始化 RMSNorm 模块。

        参数:
        d_model: 输入张量的最后一个维度的大小。
        eps: 防止除零错误的一个小常数。
        """
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.eps = eps

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x: 输入张量，形状为 [batch_size, seq_len, d_model]。

        返回:
        归一化后的张量，形状同输入张量。
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算均方根
        x_norm = x / rms  # 标准化
        return self.gamma * x_norm  # 缩放
