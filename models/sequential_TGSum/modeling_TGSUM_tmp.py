
"""
TGSUM second version --- section incorporation

"""

# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model."""
import math
import os
import random
import re
from typing import Optional, Tuple, Union, List, Any
from urllib.error import HTTPError

from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# from graph_data_prepartion.preprocess_utils.conceptnet import merged_relations
from data_processor.others.vocab_wrapper import VocabWrapper
from models.encoder import TransformerEncoder, LEDEncoderSection
from models.generation_utils import GenerationMixin
from models.networks.decoding_network import DecoderNetwork
from models.topic import TopicModel
from transformers.activations import ACT2FN
from transformers.modeling_utils import no_init_weights, get_checkpoint_shard_files, load_state_dict, \
    _load_state_dict_into_model
# from .generation_utils import GenerationMixin
# from .model_outputs import BaseModelOutput
# from ..grease_model import modeling_gnn
# from ..grease_model.utils import layers
from transformers import LEDModel, LEDConfig, LEDForConditionalGeneration, RobertaConfig
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel, BartDecoder

from transformers import PretrainedConfig
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)
from transformers.models.led.modeling_led import LEDEncoder, LEDEncoderBaseModelOutput, \
    LEDDecoder, LEDLearnedPositionalEmbedding, _expand_mask, _make_causal_mask, LEDDecoderAttention
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers.utils import logging, WEIGHTS_INDEX_NAME, FLAX_WEIGHTS_NAME, RepositoryNotFoundError, \
    RevisionNotFoundError, EntryNotFoundError, has_file, ContextManagers, HUGGINGFACE_CO_RESOLVE_ENDPOINT, ModelOutput
from transformers.utils.versions import require_version_core

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


# def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
#     """
#     Make causal mask used for bi-directional self-attention.
#     create # [bsz * num_summary, seq_len] -> [bsz * num_summary, 1, tgt_seq_len, src_seq_len]
#
#     """
#     bsz, n_summary, tgt_len = input_ids_shape
#     mask = torch.full((tgt_len, tgt_len), float("-inf"))
#     mask_cond = torch.arange(mask.size(-1))
#     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#     mask = mask.to(dtype)
#
#     if past_key_values_length > 0:
#         mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
#     return mask[None, None, :, :].expand(bsz*n_summary, 1, tgt_len, tgt_len + past_key_values_length)

def _to_tensors(array):
    return torch.tensor(array).cuda()

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int, labels):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    try:
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    except:
        import pdb;pdb.set_trace()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Seq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    loss_topic: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class LEDSeq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    topic_info: Any = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None


class LEDDecoderAttentionTopicAware(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        topic_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        k: int= 5,
        init_k: int= 5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topic_dim = topic_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.init_k = init_k

        self.k = None
        if self.is_decoder:
            self.k = k if k > init_k else None

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.inc_topic = False
        if k > init_k:
            self.inc_topic = True
            self.topic_k_proj = nn.Linear(topic_dim, embed_dim, bias=bias)
            self.topic_v_proj = nn.Linear(topic_dim, embed_dim, bias=bias)
            self.linear_topic_w = nn.Linear(self.num_heads * self.head_dim * 3, self.num_heads)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_states_topical: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_value_topic: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if not self.training:
            # import pdb;pdb.set_trace()
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None and past_key_value_topic is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]

            if self.inc_topic:
                key_states_topic = past_key_value[2]
                value_states_topic = past_key_value[3]

        elif is_cross_attention:
            # cross_attentions
            try:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            except:
                import pdb;pdb.set_trace()

            if self.inc_topic:
                key_states_topic = self._shape(self.topic_k_proj(key_value_states_topical), -1, bsz)
                value_states_topic = self._shape(self.topic_v_proj(key_value_states_topical), -1, bsz)


        elif past_key_value is not None and past_key_value_topic is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

            if self.inc_topic:
                key_states_topic = self._shape(self.topic_k_proj(hidden_states), -1, bsz)
                value_states_topic = self._shape(self.topic_v_proj(hidden_states), -1, bsz)
                key_states_topic = torch.cat([past_key_value[2], key_states_topic], dim=2)
                value_states_topic = torch.cat([past_key_value[3], value_states_topic], dim=2)

        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)


        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            if self.inc_topic:
                past_key_value = (key_states, value_states, key_states_topic, value_states_topic)
            else:
                past_key_value = (key_states, value_states)


        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        # query is the same for both token and topic...
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)

        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        if self.inc_topic:
            key_states_topic = key_states_topic.view(*proj_shape)
            value_states_topic = value_states_topic.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if self.inc_topic:
            attn_weights_topic = torch.bmm(query_states, key_states_topic.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if self.inc_topic:
                attn_weights_topic = attn_weights_topic.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                attn_weights_topic = attn_weights_topic.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if self.inc_topic:
            attn_weights_topic = nn.functional.softmax(attn_weights_topic, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if self.inc_topic:
                attn_weights_topic = layer_head_mask.view(1, -1, 1, 1) * attn_weights_topic.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights_topic = attn_weights_topic.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)

            if self.inc_topic:
                attn_weights_topic_reshaped = attn_weights_topic.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights_topic = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)


        else:
            attn_weights_reshaped = None
            attn_weights_topic_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if self.inc_topic:
            attn_probs_topic = nn.functional.dropout(attn_weights_topic, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(bsz, -1, self.num_heads * self.head_dim)

        if self.inc_topic:
            attn_output_topic = torch.bmm(attn_probs_topic, value_states_topic)

            attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            query_states = query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
            attn_output_topic = attn_output_topic.view(bsz, self.num_heads, tgt_len, self.head_dim)

            # import pdb;pdb.set_trace()
            # p_vec = torch.cat([attn_output, attn_output_topic, query_states], -1).transpose(1, 2).contiguous().view(bsz*self.num_heads, -1, self.num_heads * self.head_dim * 3)
            # p_vec = torch.cat([attn_output.view(bsz, self.num_heads, tgt_len, -1), attn_output_topic, query_states], -1).transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim * 3)
            p_vec = torch.cat([attn_output, attn_output_topic, query_states], -1).transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim * 3)
            topic_p = torch.sigmoid(self.linear_topic_w(p_vec).transpose(1, 2)).unsqueeze(-1)
            # attn_probs = topic_p.reshape(bsz*self.num_heads, -1, 1) * attn_probs + (1 - topic_p).reshape(bsz*self.num_heads, -1, 1) * attn_probs_topic
            attn_probs = topic_p * attn_probs.view(bsz, self.num_heads, tgt_len, -1) + (1 - topic_p) * attn_probs_topic.view(bsz, self.num_heads, tgt_len, -1)
            # attn_probs = topic_p * attn_probs + (1 - topic_p) * attn_probs_topic
            attn_output = torch.bmm(attn_probs.reshape(bsz * self.num_heads, tgt_len, src_len), value_states)


        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = (
            attn_output
            # attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            # .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class LEDTopicDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        topic_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(topic_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(topic_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                # import pdb;pdb.set_trace()
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class TGSumDecoderLayer(nn.Module):
    def __init__(self, config: LEDConfig, use_topic=True, k=0):
        super().__init__()
        self.embed_dim = config.d_model
        self.use_topic = use_topic

        self.self_attn = LEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # self.encoder_attn = LEDDecoderAttention(
        #     self.embed_dim,
        #     config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )

        # if self.use_topic:

        # self.encoder_attn = LEDDecoderAttentionTopicAware(
        #     self.embed_dim,
        #     200,
        #     config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        #     init_k=-1,
        #     k=k,
        # )
        # self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

            # self.fc3 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.encoder_attn = LEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn_section = LEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm_section = nn.LayerNorm(self.embed_dim)


        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        section_encoder_outputs=None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask_section: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape *(seq_len, batch, embed_dim)*
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(decoder_attention_heads,)*.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size *(decoder_attention_heads,)*.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`): Whether the base model outputs attentions.
                This requires the attentions tensor to be reshaped in this function.
        """
        residual = hidden_states

        # Self-Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        cross_attn_present_key_value_word = None
        cross_attn_weights_word = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple

            if self.use_topic:
                cross_attn_past_key_value_word = past_key_value[2:4] if past_key_value is not None else None
                # cross_attn_past_key_value_section = past_key_value[4:] if past_key_value is not None else None
                # try:


                hidden_states, cross_attn_weights_word, cross_attn_present_key_value_word = \
                    self.encoder_attn(
                    hidden_states=hidden_states,  # decoder query
                    key_value_states=encoder_hidden_states.repeat(hidden_states.size(0), 1, 1) if self.training else encoder_hidden_states, # encoder memory
                    # key_value_states_topical=topic_vec_ge.repeat(hidden_states.size(0), 1, 1) if self.training else topic_vec_ge,  # topic memory
                    # key_value_states=topic_vec_ge.repeat(hidden_states.size(0), 1, 1) if self.training else topic_vec_ge[None, :, :],  # topic memory #5
                    attention_mask=encoder_attention_mask,
                    # attention_mask=encoder_attention_mask[:, :, :, 0][:, :, :, None],
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value_word,
                    output_attentions=output_attentions,
                )
                hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                hidden_states = residual + hidden_states
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

                # try:
                #     # import pdb;pdb.set_trace()
                #
                #     hidden_states, cross_attn_weights_section, cross_attn_present_key_value_section = \
                #     self.encoder_attn_section(
                #         hidden_states=hidden_states,  # decoder query
                #         key_value_states=section_encoder_outputs.repeat(hidden_states.size(0), 1, 1) if self.training else encoder_hidden_states,
                #         attention_mask=encoder_attention_mask_section,
                #         layer_head_mask=cross_attn_layer_head_mask,
                #         past_key_value=cross_attn_past_key_value_section,
                #         output_attentions=output_attentions,
                #     )
                #     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                #     hidden_states = residual + hidden_states
                #     hidden_states = self.encoder_attn_layer_norm_section(hidden_states)
                #     # import pdb;pdb.set_trace()
                #
                # except:
                #     import pdb;pdb.set_trace()


            # cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None

            # hidden_states, cross_attn_weights, cross_attn_present_key_value = \
            # self.encoder_attn(
            #     hidden_states=hidden_states, # decoder query
            #     key_value_states=encoder_hidden_states.repeat(hidden_states.size(0), 1, 1) if self.training else encoder_hidden_states, # encoder memory
            #     attention_mask=encoder_attention_mask,
            #     layer_head_mask=cross_attn_layer_head_mask,
            #     past_key_value=cross_attn_past_key_value,
            #     output_attentions=output_attentions,
            # )
            # import pdb;pdb.set_trace()


            # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # hidden_states = residual + hidden_states
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # present_key_value = present_key_value + cross_attn_present_key_value + cross_attn_present_key_value_word
            # present_key_value = present_key_value + cross_attn_present_key_value_word + cross_attn_present_key_value_section
            present_key_value = present_key_value + cross_attn_present_key_value_word
            # import pdb;pdb.set_trace()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, cross_attn_weights_word)

        if use_cache:
            outputs += (present_key_value,)

        return outputs




class TGSumDecoder(LEDDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`LEDDecoderLayer`]
    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None, use_topic=False):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_decoder_position_embeddings

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList([TGSumDecoderLayer(config, use_topic=use_topic) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        section_encoder_outputs=None,
        attention_mask=None,
        summ_attn_mask=None,
        global_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_attention_mask_section=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`LEDTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            global_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to decide the attention given on each token, local attention or global attention. Tokens with
                global attention attends to all other tokens, and all other tokens attend to them. This is important
                for task-specific finetuning because it makes the model more flexible at representing the task. For
                example, for classification, the <s> token should be given global attention. For QA, all question
                tokens should also have global attention. Please refer to the [Longformer
                paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:
                - 0 for local attention (a sliding window attention),
                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all ``decoder_input_ids``` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor`
                of shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            try:
                input_shape = input_ids.size()
            except:
                import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            input_ids = input_ids.view(-1, input_shape[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
        #     import pdb;
        #     pdb.set_trace()

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

            if summ_attn_mask is not None: # batch > 1
                encoder_attention_mask = ~(summ_attn_mask[:, None, :, None].expand_as(_expand_mask(encoder_attention_mask.repeat(input_shape[0], 1),
                                                                                                   inputs_embeds.dtype, tgt_len=input_shape[-1])).bool())
                encoder_attention_mask = encoder_attention_mask.int()


                encoder_attention_mask_section = ~(summ_attn_mask[:, None, :, None].expand_as(_expand_mask(encoder_attention_mask_section.repeat(input_shape[0], 1),
                                                                                                   inputs_embeds.dtype, tgt_len=input_shape[-1])).bool())
                encoder_attention_mask_section = encoder_attention_mask_section.int()


            else:
                if not self.training:
                    encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
                    encoder_attention_mask_section = _expand_mask(encoder_attention_mask_section, inputs_embeds.dtype, tgt_len=input_shape[-1])
                else:
                    encoder_attention_mask = _expand_mask(encoder_attention_mask.repeat(input_shape[0], 1), inputs_embeds.dtype, tgt_len=input_shape[-1])
                    # if input_shape[-1] == 144:
                    #     import pdb;pdb.set_trace()
                    encoder_attention_mask_section = _expand_mask(encoder_attention_mask_section.repeat(input_shape[0], 1), inputs_embeds.dtype, tgt_len=input_shape[-1])


        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    section_encoder_outputs,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_attention_mask_section,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    # topic_vec_ge=topic_vec_ge,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # if torch.isnan(layer_outputs[0][0][0][0]):
            #     import pdb;pdb.set_trace()

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class TGSumModel(LEDModel):
    def __init__(self, config: LEDConfig, use_topic=False):

        # config.prefix = 'led'
        # self.base_model_prefix = 'led'
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        # self.encoder_section = LEDEncoderSection(d_model=config.d_model, d_ff=config.encoder_ffn_dim, heads=4, dropout=0.2, d_topic=200, num_inter_layers=2)
        roberta_config = RobertaConfig.from_pretrained("roberta-large")
        roberta_config.num_hidden_layers = 4
        self.encoder_section = RobertaEncoder(roberta_config)
        self.decoder = TGSumDecoder(config, self.shared, use_topic=use_topic)

        # Initialize weights and apply final processing
        self.post_init()

        self.use_topic = use_topic

        if use_topic:
            #### TOPIC MODELING
            topic_emb_size = 200
            # self.loss_lambda = 0.001
            # self.voc_wrapper = VocabWrapper("word2vec")
            # self.voc_wrapper.load_emb("/disk1/sajad/w2v_embeds/w2v_mup.emb")
            # voc_emb = torch.tensor(self.voc_wrapper.get_emb())
            # self.hidden_size = voc_emb


            # self.topic_model = TopicModel(voc_emb.size(0), voc_emb.size(-1), topic_num=50, noise_rate=0.5,
            #                               embeddings=voc_emb)

            self.n_components = 20
            self.topic_model = DecoderNetwork(input_size=3000, bert_size=config.d_model, infnet='combined', n_components=self.n_components, model_type='prodLDA',
                                        hidden_sizes=(topic_emb_size, topic_emb_size), activation='softplus',
                                        dropout=0.2, learn_priors=True, label_size=0)



            # self.split_noise = split_noise
            #
            # if self.split_noise:
            #     self.topic_gate_linear_summ = nn.Linear(config.d_model + topic_emb_size, topic_emb_size)
            #     self.topic_emb_linear_summ = nn.Linear(config.d_model, topic_emb_size)
            #     self.topic_gate_linear_noise = nn.Linear(config.d_model + topic_emb_size, topic_emb_size)
            #     self.topic_emb_linear_noise = nn.Linear(config.d_model, topic_emb_size)
            # else:
            self.topic_gate_linear = nn.Linear(config.d_model + topic_emb_size, config.d_model)
            self.topic_emb_linear = nn.Linear(topic_emb_size, config.d_model)

            # if self.use_topic:
                # if self.split_noise:
                #     for p in self.topic_gate_linear_summ.parameters():
                #         self._set_parameter_linear(p)
                #     for p in self.topic_emb_linear_summ.parameters():
                #         self._set_parameter_linear(p)
                #     for p in self.topic_gate_linear_noise.parameters():
                #         self._set_parameter_linear(p)
                #     for p in self.topic_emb_linear_noise.parameters():
                #         self._set_parameter_linear(p)
                # else:
                # for p in self.topic_gate_linear.parameters():
                #     self._set_parameter_linear(p)
                # for p in self.topic_emb_linear.parameters():
                #     self._set_parameter_linear(p)

    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    # def _topic_vec_pn(self, batch, topic_info):
    #
    #     src, ex_segs = batch.src, batch.ex_segs
    #     bsz, max_len = len(ex_segs), max(ex_segs)
    #     topic_vec_all, topic_vec_cust, topic_vec_agent = topic_info
    #     customer = (topic_vec_cust is not None)
    #     agent = (topic_vec_agent is not None)
    #
    #     if customer:
    #         cust_mask = torch.split(src[:, 1].eq(self.customer_token), ex_segs)
    #         cust_mask = pad_sequence(cust_mask, batch_first=True, padding_value=0).float()
    #     if agent:
    #         agent_mask = torch.split(src[:, 1].eq(self.agent_token), ex_segs)
    #         agent_mask = pad_sequence(agent_mask, batch_first=True, padding_value=0).float()
    #
    #     if agent and customer:
    #         if self.args.split_noise:
    #             topic_vec_agent_summ, topic_vec_agent_noise = topic_vec_agent
    #             topic_vec_cust_summ, topic_vec_cust_noise = topic_vec_cust
    #             topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
    #             topic_vec_summ = torch.cat([topic_vec_agent_summ.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                         topic_vec_cust_summ.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                         topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec_noise = torch.cat([topic_vec_agent_noise.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                          topic_vec_cust_noise.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                          topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec = (topic_vec_summ, topic_vec_noise)
    #         else:
    #             topic_vec = torch.cat([topic_vec_agent.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                    topic_vec_cust.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                    topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #     elif agent:
    #         if self.args.split_noise:
    #             topic_vec_agent_summ, topic_vec_agent_noise = topic_vec_agent
    #             topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
    #             topic_vec_summ = torch.cat([topic_vec_agent_summ.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                         topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec_noise = torch.cat([topic_vec_agent_noise.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                          topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec = (topic_vec_summ, topic_vec_noise)
    #         else:
    #             topic_vec = torch.cat([topic_vec_agent.unsqueeze(1).expand(bsz, max_len, -1) * agent_mask.unsqueeze(-1),
    #                                    topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #     elif customer:
    #         if self.args.split_noise:
    #             topic_vec_cust_summ, topic_vec_cust_noise = topic_vec_cust
    #             topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
    #             topic_vec_summ = torch.cat([topic_vec_cust_summ.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                         topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec_noise = torch.cat([topic_vec_cust_noise.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                         topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #             topic_vec = (topic_vec_summ, topic_vec_noise)
    #         else:
    #             topic_vec = torch.cat([topic_vec_cust.unsqueeze(1).expand(bsz, max_len, -1) * cust_mask.unsqueeze(-1),
    #                                    topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)], -1)
    #     else:
    #         if self.args.split_noise:
    #             topic_vec_all_summ, topic_vec_all_noise = topic_vec_all
    #             topic_vec_summ = topic_vec_all_summ.unsqueeze(1).expand(bsz, max_len, -1)
    #             topic_vec_noise = topic_vec_all_noise.unsqueeze(1).expand(bsz, max_len, -1)
    #             topic_vec = (topic_vec_summ, topic_vec_noise)
    #         else:
    #             topic_vec = topic_vec_all.unsqueeze(1).expand(bsz, max_len, -1)
    #
    #     return topic_vec

    def _topic_vec_ge(self, topic_vec_all, max_len, vec):
        # if self.training:
        #     bsz = 1
        # else:
        #     bsz = topic_vec_all.size(0)
        # import pdb;pdb.set_trace()

        # topic_vec = topic_vec_all.unsqueeze(0).expand(bsz, max_len, -1)
        topic_vec = topic_vec_all
        mapped_vec = vec
        gate = torch.sigmoid(self.topic_gate_linear(torch.cat([mapped_vec.squeeze(0), topic_vec], dim=-1)))
        # fused_vec = (1 - gate) * topic_vec + gate * self.topic_emb_linear(mapped_vec)
        fused_vec = (1 - gate) * self.topic_emb_linear(topic_vec) + gate * mapped_vec

        return fused_vec


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_encoder_section(self):
        return self.encoder_section

    def get_topic_model(self):
        return self.topic_model

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        source_bow=None,
        summ_bow=None,
        doc_ids=None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        section_token_index=None,
        topic_model_outputs=None,
        encoder_outputs_section=None,
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # import pdb;pdb.set_trace()
        # Using this like Bart, as LED is derived from it. So far
        # No checkpoint on the hub exists that uses that in practice.
        # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # import pdb;
            # pdb.set_trace()
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        bsz = 1
        if decoder_input_ids is not None:
            n_summary = len(decoder_input_ids[0])
            # summ_bow = pad_sequence(summ_bow[0], batch_first=True, padding_value=0).unsqueeze(0)
        else:
            n_summary = 1
        # only bsz=1 is supported...

        summ_attn_mask = None
        if bsz * n_summary > 1:
            summ_attn_mask = (pad_sequence(decoder_input_ids[0], batch_first=True, padding_value=-1) != -1).to(attention_mask.dtype)

        if self.training: # in validation and test we will generate only one summary
            decoder_input_ids = pad_sequence(decoder_input_ids[0], batch_first=True, padding_value=self.config.pad_token_id).unsqueeze(0).view(bsz*n_summary, -1)

        # pad sequence [bsz, n_summary, dim]
        section_encodings = None
        if self.use_topic:
            if input_ids is not None:
                section_token_indices = ((input_ids[0] == 0).nonzero(as_tuple=True)[0])
            else:
                section_token_indices = section_token_index.cuda()
            section_pre_encodings = torch.index_select(encoder_outputs[0], 1, section_token_indices)

            if topic_model_outputs is None:
                if not self.training:
                    reshp = source_bow.shape
                    prior_mean, prior_variance, posterior_mean, posterior_variance,  posterior_log_variance, word_dists, estimated_labels, topic_emb = self.topic_model(source_bow.view(-1, 3000),section_pre_encodings.view(-1, 1024), labels=None)
                    reshaped = [posterior_mean, posterior_variance,  posterior_log_variance, word_dists, topic_emb]
                    posterior_mean, posterior_variance,  posterior_log_variance, word_dists, topic_emb = [v.view(reshp[0], reshp[1], -1) for v in reshaped]

                else:
                    # import pdb;pdb.set_trace()
                    prior_mean, prior_variance, posterior_mean, posterior_variance, \
                    posterior_log_variance, word_dists, estimated_labels, topic_emb = self.topic_model(source_bow.squeeze(0),
                                                                                                    section_pre_encodings.squeeze(
                                                                                                           0), labels=None)
            else:
                prior_mean, prior_variance, posterior_mean, posterior_variance, \
                posterior_log_variance, word_dists, estimated_labels, topic_emb = topic_model_outputs

            # if not self.training:


            if encoder_outputs_section is None:



                section_mask = torch.BoolTensor([True]*section_pre_encodings.size(1)).unsqueeze(0).cuda()
                section_encodings = self.encoder_section(hidden_states=section_pre_encodings, attention_mask=section_mask)[0]
                # import pdb;pdb.set_trace()

                # section_encodings = self._topic_vec_ge(topic_emb, section_pre_encodings.size(1), section_encodings)

            else:
                section_encodings = encoder_outputs_section
                # section_encodings = self._topic_vec_ge(topic_emb, section_pre_encodings.size(1), section_encodings)
            if not self.training:
                import pdb;pdb.set_trace()

            # print(topic_emb.sum())
            # encoder_outputs[0]
            # topic_loss, topic_info = self.topic_model(source_bow, summ_bow)
            # self.topic_loss = self.loss_lambda * topic_loss
            # topic_vec_ge = self._topic_vec_ge(topic_info, attention_mask.size(1), encoder_outputs[0], summ_bow.size(1) if self.training else 1)
            # prior_mean, prior_variance, posterior_mean, posterior_variance,  posterior_log_variance, word_dists, estimated_labels, topic_emb = self.topic_model(source_bow, encoder_outputs[0][:, 0], labels=None) # X_contexual is the source embedding
            # self.topic_model(source_bow, section_encodings, labels=None)

            # import pdb;pdb.set_trace()

            # topic_vec_ge = self._topic_vec_ge(topic_emb, attention_mask.size(1), encoder_outputs[0])
            # if self.training:
            #     topic_vec_ge = (topic_vec_ge.view(1, attention_mask.size(1), -1))
                # import pdb;pdb.set_trace()
                # topic_emb = (topic_emb[:, :, None].view(1, attention_mask.size(1), -1))

        # import pdb;pdb.set_trace()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            # topic_vec_ge=topic_vec_ge,
            section_encoder_outputs=section_encodings,
            # topic_vec_ge=topic_emb,
            attention_mask=decoder_attention_mask,
            summ_attn_mask=summ_attn_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            encoder_attention_mask_section=section_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            topic_info=(source_bow, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance) if self.use_topic else None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )


class TGSumForConditionalGeneration(LEDForConditionalGeneration, GenerationMixin):
    # _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    def __init__(self, config: LEDConfig, use_topic=True):
        super().__init__(config)
        self.led = TGSumModel(config, use_topic=use_topic)
        #
        global do_topic
        do_topic = use_topic
        self.use_topic = use_topic
        # Initialize weights and apply final processing
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.post_init()

        #

    def set_split_noise(self, split_noise):
        self.led.split_noise = split_noise

    def get_encoder(self):
        return self.led.get_encoder()

    def get_section_encoder(self):
        return self.led.get_encoder_section()

    def get_topic_model(self):
        return self.led.get_topic_model()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


    def _topic_loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division =   prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * ( var_division + diff_term - self.n_components + logvar_det_division).mean()

        # Reconstruction term
        # import pdb;pdb.set_trace()
        RL = -(torch.sum(inputs * torch.log(word_dists + 1e-10), dim=2).sum())

        #loss = self.weights["beta"]*KL + RL

        return KL, RL

    def forward(
            self,
            input_ids=None,
            src_bow=None,
            summ_bow=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            doc_ids=None,
            section_token_index=None,
            split_noise=False,
            topic_model_outputs=None,
            encoder_outputs_section=None,
            global_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        # self.set_split_noise(split_noise)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # return_dict = True

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.led(
            input_ids,
            source_bow=src_bow,
            summ_bow=summ_bow,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            global_attention_mask=global_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            doc_ids=doc_ids,
            section_token_index=section_token_index,
            topic_model_outputs=topic_model_outputs,
            encoder_outputs_section=encoder_outputs_section,
        )

        # import pdb;
        # pdb.set_trace()

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        topic_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = pad_sequence(labels[0], batch_first=True, padding_value=-100).unsqueeze(0).cuda()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # if (torch.isnan(masked_lm_loss)):
            #     import pdb;pdb.set_trace()
            if self.use_topic:
                source_bow, word_dists, prior_mean, prior_variance, posterior_mean, \
                posterior_variance, posterior_log_variance = outputs.topic_info

                # backward pass
                self.n_components = self.led.n_components

                # import pdb;pdb.set_trace()
                kl_loss, rl_loss = self._topic_loss(
                    source_bow, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance,
                    posterior_log_variance
                )
                # this is gonna be added to the whole loss...
                topic_loss = (kl_loss + rl_loss).sum()

                # masked_lm_loss += (0.8) * masked_lm_loss + (0.2) *

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            loss_topic=topic_loss if self.use_topic else None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            src_bow=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "src_bow": src_bow,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # labels is a list of tensors
        ret = []
        for batch_labels in labels:
            ret_summaries = []
            for lbl in batch_labels:
                ret_summaries.append(shift_tokens_right(lbl.unsqueeze(0), self.config.pad_token_id, self.config.decoder_start_token_id, labels).squeeze(0))

            ret.append(ret_summaries)

        return ret

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""
            override from_pretrained
        """


        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", True)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        use_topic = kwargs.pop("use_topic", True)

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            # low_cpu_mem_usage requires PyTorch >= 1.9 to have the meta device.
            require_version_core("torch>=1.9")

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                        "weights."
                    )
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                        f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    # import pdb;pdb.set_trace()
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(  pretrained_model_name_or_path, filename=filename, revision=revision, mirror=mirror )
                archive_file_roberta = hf_bucket_url( "roberta-large", filename=filename, revision=revision, mirror=mirror )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

                resolved_archive_file_roberta = cached_path(
                    archive_file_roberta,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == WEIGHTS_NAME:
                    try:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        archive_file = hf_bucket_url(
                            pretrained_model_name_or_path,
                            filename=WEIGHTS_INDEX_NAME,
                            revision=revision,
                            mirror=mirror,
                        )
                        resolved_archive_file = cached_path(
                            archive_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            user_agent=user_agent,
                        )
                        is_sharded = True
                    except EntryNotFoundError:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "mirror": mirror,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {WEIGHTS_NAME} but there is a file for TensorFlow weights. Use `from_tf=True` to"
                                " load this model from those weights."
                            )
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {WEIGHTS_NAME} but there is a file for Flax weights. Use `from_flax=True` to load"
                                " this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME},"
                                f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                            )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                    f" {FLAX_WEIGHTS_NAME}.\nCheckout your internet connection or see how to run the library in"
                    " offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                mirror=mirror,
            )

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)
                state_dict_roberta = load_state_dict(resolved_archive_file_roberta)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = next(iter(state_dict.values())).dtype
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file)
                            torch_dtype = next(iter(one_state_dict.values())).dtype
                            del one_state_dict  # free CPU memory
                    else:
                        raise ValueError(
                            f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = [k for k in state_dict.keys()]
            if low_cpu_mem_usage:
                state_dict = None

        # state_dict = {n.replace('led.', ''): p for n, p in state_dict.items()}
        # state_dict = {"model." + n: p for n, p in state_dict.items() if 'encoder.' in n or 'decoder.' in n}

        loaded_state_dict_keys = state_dict.keys()
        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        with ContextManagers(init_contexts):
            model = cls(config, use_topic=use_topic, *model_args, **model_kwargs)

        # import pdb;pdb.set_trace()
        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)


        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()

        # add encoder_attn_topic keys to state_dict...
        # initialize it with encocer_attn keys...

        added_params = {}
        for n, p in state_dict.items():
            if 'encoder_attn.' in n:
                # import pdb;pdb.set_trace()
                added_params[n.replace('encoder_attn', 'encoder_attn_section')] = p
            if 'encoder_attn_layer_norm.' in n:
                # import pdb;pdb.set_trace()
                added_params[n.replace('encoder_attn_layer_norm', 'encoder_attn_layer_norm_section')] = p
        # now update state_dict keys...

        for n, p in state_dict_roberta.items():
            try:
                layer_num = int(n.split('roberta.encoder.layer.')[1].split('.')[0])
                if layer_num >= 20:
                    added_params[n.replace(f'roberta.encoder.layer.{layer_num}', f'led.encoder_section.layer.{layer_num-20}')] = p
            except:
                pass

        state_dict.update(added_params)

        # import pdb;pdb.set_trace()

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=False,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            # make sure all params are requrires_grad True in the model!

            # for n, param in model.named_parameters():
            #     param.requires_grad = True

            return model, loading_info

        return model
