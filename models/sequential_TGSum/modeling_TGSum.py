
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
from models.encoder_sect import TransformerEncoder
from models.generation_utils import GenerationMixin
from models.networks.decoding_network import DecoderNetwork
from models.sequential_TGSum.roberta_tgsum import RobertaEncoder
from models.tokenization_TGSum import TGSumTokenizer
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
    sect_loss: Optional[torch.FloatTensor] = None
    sent_loss: Optional[torch.FloatTensor] = None
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
    sect_scores: Any = None
    sent_scores: Any = None
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
        # self.topic_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_states_topical: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]

            if self.inc_topic:
                key_states_topic = past_key_value[2]
                value_states_topic = past_key_value[3]

        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

            if self.inc_topic:
                key_states_topic = self._shape(self.topic_k_proj(key_value_states_topical), -1, bsz)
                value_states_topic = self._shape(self.topic_v_proj(key_value_states_topical), -1, bsz)


        elif past_key_value is not None:
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

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if self.inc_topic:
            attn_probs_topic = nn.functional.dropout(attn_weights_topic, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        if self.inc_topic:
            attn_output_topic = torch.bmm(attn_probs_topic, value_states_topic)

        # if attn_probs.size(0) == 32:
        #     import pdb;pdb.set_trace()
        # combine attention vectors...
            p_vec = torch.cat([attn_output, attn_output_topic, query_states], -1).transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim * 3)
            topic_p = torch.sigmoid(self.linear_topic_w(p_vec).transpose(1, 2)).unsqueeze(-1)
            try:
                attn_probs = topic_p.reshape(bsz*self.num_heads, -1, 1) * attn_probs + (1 - topic_p).reshape(bsz*self.num_heads, -1, 1) * attn_probs_topic
            except:
                import pdb;pdb.set_trace()
            # attn_probs_topic = nn.functional.dropout(attn_weights_topic, p=self.dropout, training=self.training)
            try:
                attn_output = torch.bmm(attn_probs, value_states)
            except:
                import pdb;pdb.set_trace()

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
    def __init__(self, config: LEDConfig, use_topic=True, k=0, init_k=5):
        super().__init__()
        self.embed_dim = config.d_model
        self.use_topic = use_topic
        self.k = k
        self.init_k = init_k

        self.self_attn = LEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

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

        # if self.use_topic:

            # self.encoder_attn = LEDDecoderAttentionTopicAware(
            #     self.embed_dim,
            #     topic_dim=100,
            #     num_heads=config.decoder_attention_heads,
            #     dropout=config.attention_dropout,
            #     is_decoder=True,
            #     k=k,
            #     init_k=init_k,
            # )
            # self.fc3 = nn.Linear(self.embed_dim * 2, self.embed_dim)



        # self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        section_encoder_outputs: Optional[torch.Tensor] = None,
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

        if encoder_hidden_states is not None:
            residual = hidden_states

            cross_attn_past_key_value_word = past_key_value[2:4] if past_key_value is not None else None
            cross_attn_past_key_value_section = past_key_value[4:] if past_key_value is not None else None

            hidden_states, cross_attn_weights_section, cross_attn_present_key_value_section = \
                self.encoder_attn_section(
                    hidden_states=hidden_states,  # decoder query
                    key_value_states=section_encoder_outputs.repeat(hidden_states.size(0), 1, 1) if self.training else section_encoder_outputs,
                    attention_mask=encoder_attention_mask_section,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value_section,
                    output_attentions=output_attentions,
                )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm_section(hidden_states)

            hidden_states, cross_attn_weights_word, cross_attn_present_key_value_word = \
                self.encoder_attn(
                    hidden_states=hidden_states,  # decoder query
                    # key_value_states=encoder_hidden_states.repeat(hidden_states.size(0), 1, 1) if self.training else encoder_hidden_states.unsqueeze(0).repeat(encoder_attention_mask.size(0), 1, 1),
                    key_value_states=encoder_hidden_states.repeat(hidden_states.size(0), 1, 1) if self.training else encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value_word,
                    output_attentions=output_attentions,
                )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)



            present_key_value = present_key_value + cross_attn_present_key_value_word + cross_attn_present_key_value_section
            # present_key_value = present_key_value + cross_attn_present_key_value_word

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
            outputs += (self_attn_weights, cross_attn_weights_word, cross_attn_weights_section)

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
        self.layers = nn.ModuleList([TGSumDecoderLayer(config, use_topic=use_topic, k=idx) for idx in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        section_encodings=None,
        attention_mask=None,
        encoder_attention_mask_section=None,
        summ_attn_mask=None,
        global_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
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

        # if not self.training:
        #     import pdb;pdb.set_trace()
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
                encoder_attention_mask = ~(summ_attn_mask[:, None, :, None].expand_as(_expand_mask(encoder_attention_mask.repeat(input_shape[0], 1), inputs_embeds.dtype, tgt_len=input_shape[-1])).bool())
                encoder_attention_mask = encoder_attention_mask.int()

                encoder_attention_mask_section = ~(summ_attn_mask[:, None, :, None].expand_as(
                    _expand_mask(encoder_attention_mask_section.repeat(input_shape[0], 1),
                                 inputs_embeds.dtype, tgt_len=input_shape[-1])).bool())
                encoder_attention_mask_section = encoder_attention_mask_section.int()

            else:
                if not self.training:
                    encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
                    # import pdb;pdb.set_trace()
                    encoder_attention_mask_section = _expand_mask(encoder_attention_mask_section, inputs_embeds.dtype, tgt_len=input_shape[-1])
                    # encoder_attention_mask_section = _expand_mask(encoder_attention_mask_section, inputs_embeds.dtype, tgt_len=input_shape[-1])

                else:
                    try:
                        encoder_attention_mask = _expand_mask(encoder_attention_mask.repeat(input_shape[0], 1), inputs_embeds.dtype, tgt_len=input_shape[-1])
                    except:
                        import pdb;pdb.set_trace()
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
                # import pdb;pdb.set_trace()

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    section_encodings,
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
                    section_encoder_outputs=section_encodings,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_section=encoder_attention_mask_section,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

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

    def __init__(self, config: LEDConfig, use_topic=False, sampling_from='all'):

        # config.prefix = 'led'
        # self.base_model_prefix = 'led'
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.sent_scorer = nn.Linear(config.d_model, 1)
        self.sect_scorer = nn.Linear(config.d_model, 1)

        self.encoder = LEDEncoder(config, self.shared)
        self.encoder.gradient_checkpointing = True

        self.decoder = TGSumDecoder(config, self.shared, use_topic=use_topic)

        # Initialize weights and apply final processing
        self.post_init()

        self.use_topic = use_topic

        topic_emb_size = 300
        self.top_words_dim = 3001

        self.is_hier = True
        if self.is_hier:
            self.hier_encoder = TransformerEncoder(config.d_model, d_ff=2048, heads=4,
                                                   dropout=0.2, num_inter_layers=2)
                # roberta_config = RobertaConfig.from_pretrained('roberta-large')
                # roberta_config.num_hidden_layers = 4
                # self.hier_encoder = RobertaEncoder(roberta_config)

        self.SAMPLING_FROM = sampling_from
        self.num_topics = 100
        self.topic_model = DecoderNetwork(input_size=self.top_words_dim, bert_size=config.d_model,
                                          infnet='combined', n_components=self.num_topics, model_type='prodLDA',
                                          hidden_sizes=(100, 100), activation='softplus',
                                          dropout=0.2, learn_priors=True, label_size=0)

        self.topic_gate_linear = nn.Linear(config.d_model + topic_emb_size, config.d_model)
        self.topic_emb_linear = nn.Linear(topic_emb_size, config.d_model)

        for p in self.topic_gate_linear.parameters():
            self._set_parameter_linear(p)
        for p in self.topic_emb_linear.parameters():
            self._set_parameter_linear(p)


    def _set_parameter_linear(self, p):
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()

    def _set_parameter_tf(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _topic_vec_ge(self, topic_emb, max_len, vec):
        if self.training:
            bsz = 1
        else:
            bsz = topic_emb.size(0)

        topic_vec = topic_emb.unsqueeze(1).expand(bsz, max_len, -1)
        mapped_vec = vec

        try:
            gate = torch.sigmoid(self.topic_gate_linear(torch.cat([mapped_vec, topic_vec], dim=-1)))
                # gate = torch.sigmoid(self.topic_gate_linear(torch.cat([mapped_vec.unsqueeze(0), topic_vec], dim=-1)))
        except:
            gate = torch.sigmoid(self.topic_gate_linear(torch.cat([mapped_vec.unsqueeze(0), topic_vec], dim=-1)))
            # import pdb;pdb.set_trace()
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

    def is_hier(self):
        return self.is_hier

    def get_hier_encoder(self):
        return self.hier_encoder

    def get_sent_scorer(self):
        return self.sent_scorer

    def get_sect_scorer(self):
        return self.sect_scorer

    def get_topic_model(self):
        return self.topic_model

    def get_topic_fuse(self):
        return self._topic_vec_ge

    def get_decoder(self):
        return self.decoder

    def get_repr_from_index(self, from_tensor, index):
        return from_tensor[:,index, :]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        src_bow_global=None,
        src_bow_section=None,
        doc_ids=None,
        section_len=None,
        sent_len=None,
        step=None,
        sections_sentence_encoding=None,
        selected_sent_embeddings=None,
        topic_model_outputs=None,
        section_token_index=None,
        encoder_section_outputs=None,
        topic_model_global_outputs=None,
        topic_model_section_outputs=None,
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
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # import pdb;
            # pdb.set_trace()
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            global_attention_mask = torch.zeros_like(attention_mask)
            sent_ids = ((input_ids[0] == 0).nonzero(as_tuple=True)[0])
            global_attention_mask[:, sent_ids] = 1

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
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        if topic_model_global_outputs is None:
            prior_mean, prior_variance, posterior_mean, posterior_variance, \
            posterior_log_variance, word_dists, estimated_labels, topic_emb = self.topic_model(src_bow_global,
                                                                                               encoder_outputs[0].mean(dim=1),
                                                                                               labels=None)  # X_contexual is the source embedding
        else:
            prior_mean, prior_variance, posterior_mean, posterior_variance, \
            posterior_log_variance, word_dists, estimated_labels, topic_emb = topic_model_global_outputs

        encoder_outputs.last_hidden_state = self._topic_vec_ge(topic_emb, attention_mask.size(1), encoder_outputs.last_hidden_state)

            # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False





        """"""""""""""""""""""""""""""""""""""""""""""""""""
                    ext labels and section scores
        """""""""""""""""""""""""""""""""""""""""""""""""""



        sent_scores = None
        sect_scores = None
        if sections_sentence_encoding is None:
            sent_repr = self.get_repr_from_index(encoder_outputs[0],
                                                     index=((input_ids[0] == 0).nonzero(as_tuple=True)[0]))

            section_repr = self.get_repr_from_index(encoder_outputs[0],
                                                    index=((input_ids[0] == input_ids[0][0]).nonzero(as_tuple=True)[0]))

            sent_scores = torch.sigmoid(
                self.sent_scorer(sent_repr)
            ).squeeze(-1)

            sect_scores = torch.nn.functional.softmax(self.sect_scorer(section_repr), dim=-2).squeeze(-1)


            LIMIT = 4096  # tokens

            if self.SAMPLING_FROM=='section':
                sample_sect_dist = torch.round(torch.tensor([LIMIT] * (section_repr.size(1))).unsqueeze(0).cuda() * sect_scores.squeeze(-1))

                sent_real_ids = (input_ids[0] == 0).nonzero(as_tuple=True)[0]
                end_pre_ids = (input_ids[0] == 2).nonzero(as_tuple=True)[0]

                try:
                    sent_len = ((end_pre_ids - sent_real_ids) + 1)[None, :]
                except:
                    print('found noncons')
                    # import pdb;pdb.set_trace()

                try:
                    sects_batch_sent_scores = pad_sequence(torch.split(sent_scores[0], section_len[0].tolist()), batch_first=True, padding_value=-1)[None, :, :].repeat(input_ids.size(0), 1, 1)
                except Exception as ee:
                    print(ee)
                    print(doc_ids)

                try:
                    sects_batch_sent_lens = pad_sequence(torch.split(sent_len[0], section_len[0].tolist()), batch_first=True, padding_value=0)[None, :, :].repeat(input_ids.size(0), 1, 1)
                except:
                    import pdb;pdb.set_trace()

                # sects_batch_sent_lens.sum(dim=2)
                # updating sample_sect_dist based on the total number of tokens in each section.

                updated_sect_dist = (sample_sect_dist - sects_batch_sent_lens.sum(dim=2))
                fulfilled_sects = updated_sect_dist > 0

                if LIMIT < input_ids.size(-1) and ((~fulfilled_sects).float().sum()) > 0 and fulfilled_sects.float().sum() != fulfilled_sects.size(1):
                    remaining_toks = (updated_sect_dist * fulfilled_sects).sum()
                    revisit_section_dist = (sect_scores * ~fulfilled_sects) / (sect_scores * ~fulfilled_sects).sum()
                    sample_sect_dist_v2 = torch.round(revisit_section_dist * remaining_toks)
                    sample_sect_dist += sample_sect_dist_v2
                    sample_sect_dist = torch.where(fulfilled_sects, sample_sect_dist-updated_sect_dist, sample_sect_dist)

                sect_sent_mask = (sects_batch_sent_lens > 0).float()

                top_sents_idxs = torch.argsort(sects_batch_sent_scores, descending=True)

                top_sects_batch_sent_lens = torch.zeros_like(top_sents_idxs).cuda()
                top_sects_batch_sent_lens.scatter_(2, top_sents_idxs, sects_batch_sent_lens)

                sect_sent_n_selects = (~((torch.cumsum(top_sects_batch_sent_lens, dim=-1)) > sample_sect_dist[:, :, None].expand_as(top_sects_batch_sent_lens)) * sect_sent_mask).sum(dim=-1)
                top_sents_mask = ((pad_sequence([torch.arange(xx) for x in sect_sent_n_selects for xx in x], padding_value=-1).t()) != -1).cuda()
                pre_idx = top_sents_idxs[:, :, :int(sect_sent_n_selects.max().item())] * top_sents_mask

                section_len_cum = torch.cumsum(section_len, dim=-1)
                shifted_section_len = torch.cat((torch.tensor([0])[None, :].cuda(), section_len_cum), dim=-1)
                # pre_idx_hyp = torch.cat((pre_idx, torch.zeros_like(pre_idx)[:, :, :1]), dim=-1)
                shifted_section_len = shifted_section_len[:, :, None].repeat(1, 1, pre_idx.size(2))[:, :-1, :]
                pre_idx = (((pre_idx) + shifted_section_len) * top_sents_mask)

                pre_idx = ((pre_idx+1) * top_sents_mask) - 1

                pre_idx = torch.where(pre_idx > -1, sent_real_ids[pre_idx], 10000000)
                end_pre_ids = (pre_idx + top_sects_batch_sent_lens[:,:,:pre_idx.size(-1)])

                # sort ids
                pre_idx = pre_idx.sort()[0]
                end_pre_ids = end_pre_ids.sort()[0]

                sections_sentence_encoding = []
                selected_sent_embeddings = []

                for l in range(pre_idx.size(1)):
                    # import pdb;
                    # pdb.set_trace()
                    if top_sents_mask[l].float().sum() == 0:
                        continue
                    section_sent_encoding = []
                    start_idxs = pre_idx[0, l]
                    end_idxs = end_pre_ids[0, l]

                    for ll in range(start_idxs.size(-1)):
                        # import pdb;pdb.set_trace()

                        if start_idxs[ll] > 10000000 - 1:
                            continue
                        try:
                            sent_encoding = encoder_outputs[0][:, start_idxs[ll]:end_idxs[ll]+1]
                        except:
                            sent_encoding = encoder_outputs[0][:, start_idxs[ll]:end_idxs[ll]]

                        selected_sent_embeddings.append(encoder_outputs[0][:, start_idxs[ll]].unsqueeze(0))
                        section_sent_encoding.append(sent_encoding)
                    try:
                        sections_sentence_encoding.append(torch.cat(section_sent_encoding, dim=1))
                    except:
                        import pdb;pdb.set_trace()

                sections_sentence_encoding = torch.cat(sections_sentence_encoding, dim=1).cuda()
                selected_sent_embeddings = torch.cat(selected_sent_embeddings, dim=1).cuda()

                if self.is_hier:
                    selected_sent_embeddings = self.hier_encoder(selected_sent_embeddings,
                                                             # torch.ones(selected_sent_embeddings.size(0), selected_sent_embeddings.size(1)).cuda()).last_hidden_state
                                                             torch.ones(selected_sent_embeddings.size(0), selected_sent_embeddings.size(1)).bool().cuda()
                                                                 )


            else:

                sent_real_ids = (input_ids[0] == 0).nonzero(as_tuple=True)[0]
                end_pre_ids = (input_ids[0] == 2).nonzero(as_tuple=True)[0]

                sent_len = ((end_pre_ids - sent_real_ids) + 1)[None, :]

                top_sents_ids = torch.argsort(sent_scores, descending=True)
                # top_sents_len = torch.zeros_like(sent_len)

                # top_sents_len.scatter_(1, top_sents_ids, sent_len)
                top_sents_len = torch.index_select(sent_len, 1, top_sents_ids[0])
                top_sents_included = (~(torch.cumsum(top_sents_len, dim=-1) > LIMIT)).sum()
                top_sents_ids = top_sents_ids[:, :top_sents_included]


                top_sents_start_ids = sent_real_ids[top_sents_ids.sort(dim=1)[0]]
                sent_len = torch.index_select(sent_len, 1, top_sents_ids.sort(dim=1)[0].squeeze(0))
                top_sens_end_ids = top_sents_start_ids + sent_len

                masked_top_sents_input = torch.zeros_like(input_ids)
                num_of_masked= 0
                selected_sent_embeddings = []
                for start_idx, end_idx in zip(top_sents_start_ids[0], top_sens_end_ids[0]):
                    masked_top_sents_input[:, start_idx:end_idx] = 1
                    num_of_masked += len(masked_top_sents_input[0, start_idx:end_idx])
                    selected_sent_embeddings.append(encoder_outputs[0][:, start_idx])

                try:
                    selected_sent_embeddings = torch.cat(selected_sent_embeddings, dim=0)
                except:
                    print(doc_ids)
                    import pdb;pdb.set_trace()

                selected_sent_embeddings = selected_sent_embeddings.unsqueeze(0)
                # torch.where(masked_top_sents_input > -1, input_ids,)
                mask = masked_top_sents_input.unsqueeze(-1).expand_as(encoder_outputs[0]).bool()
                sections_sentence_encoding = torch.masked_select(encoder_outputs[0], mask).view(1, num_of_masked, -1)
                if self.is_hier:
                    selected_sent_embeddings = self.hier_encoder(selected_sent_embeddings, torch.ones(selected_sent_embeddings.size(0), selected_sent_embeddings.size(1)).bool().cuda())

            """""
            
            """

        bsz = 1
        if decoder_input_ids is not None:
            n_summary = len(decoder_input_ids[0])
        else: # in validation/test we will generate one summary
            n_summary = 1


        # creating summary attention mask; no need for summ_attn_mask in val/test
        summ_attn_mask = None
        if bsz * n_summary > 1 and self.training:
            summ_attn_mask = (pad_sequence(decoder_input_ids[0], batch_first=True, padding_value=-1) != -1).to(attention_mask.dtype)

        if self.training: # in validation and test we will generate only one summary + if batching is needed in training
            decoder_input_ids = pad_sequence(decoder_input_ids[0], batch_first=True, padding_value=self.config.pad_token_id).unsqueeze(0).view(bsz*n_summary, -1)



        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """"""""""""""""""""" Section encoding """"""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


        # if encoder_section_outputs is None:
        #     section_token_indices = ((input_ids[0] == 0).nonzero(as_tuple=True)[0])
        #     section_pre_encodings = torch.index_select(encoder_outputs[0], 1, section_token_indices)
        #
        # if encoder_section_outputs is None:
        #     section_mask = torch.BoolTensor([True] * section_pre_encodings.size(1)).unsqueeze(0).cuda()
        ##     encoder_section_outputs = self.encoder_section(hidden_states=section_pre_encodings, topic_emb=topic_emb_sct.unsqueeze(0), attention_mask=section_mask)[0]
        #
        # else:
        ##     encoder_section_outputs is not none.
            # section_mask = torch.BoolTensor([True] * encoder_section_outputs.size(1)).unsqueeze(0).repeat(encoder_section_outputs.size(0), 1).cuda()

        # if not self.training:
        #     import pdb;pdb.set_trace()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            section_encodings=selected_sent_embeddings.cuda(),
            attention_mask=decoder_attention_mask,
            encoder_attention_mask_section=torch.ones(selected_sent_embeddings.size(0), selected_sent_embeddings.size(1)).cuda(),
            summ_attn_mask=summ_attn_mask,
            encoder_hidden_states=sections_sentence_encoding.cuda(),
            encoder_attention_mask=torch.ones(sections_sentence_encoding.size(0), sections_sentence_encoding.size(1)).cuda(),
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

        # import pdb;pdb.set_trace()
        return LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            topic_info=(src_bow_global, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance) if self.use_topic else None,
            past_key_values=decoder_outputs.past_key_values,
            sent_scores=sent_scores if sent_scores is not None else None,
            sect_scores=sect_scores if sect_scores is not None else None,
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
    def __init__(self, config: LEDConfig, use_topic=True, task='all'):
        super().__init__(config)
        self.SAMPLING_FROM = 'all'

        self.led = TGSumModel(config, use_topic=use_topic, sampling_from=self.SAMPLING_FROM)
        self.task = task

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

    def is_hier(self):
        return self.led.is_hier

    def get_sent_scorer(self):
        return self.led.get_sent_scorer()

    def get_sect_scorer(self):
        return self.led.get_sect_scorer()

    def get_hier_encoder(self):
        return self.led.get_hier_encoder()

    def get_topic_model(self):
        return self.led.get_topic_model()

    def get_topic_fuse(self):
        return self.led.get_topic_fuse()

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
              posterior_mean, posterior_variance, posterior_log_variance, is_sct=False):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms

        if not is_sct:
            KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)
            RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        else:
            KL = (0.5 * (var_division + diff_term - self.n_components + logvar_det_division)).mean()
            RL = -(torch.sum(inputs * torch.log(word_dists + 1e-10), dim=2).mean())


        return KL, RL

    def _convert_2d_labels_to_tensor(self, ext_labels, section_scores):

        # removing batch
        try:
            ext_labels_tensor = [[] for _ in range(len(ext_labels[0][0]))]
        except:
            import pdb;pdb.set_trace()

        for ext_labels_batch in ext_labels:
            for sect_labels in ext_labels_batch:
                for j, sum_sec_label in enumerate(sect_labels):
                    ext_labels_tensor[j].extend(sum_sec_label.cpu().numpy().tolist())


        ext_labels_tensor_ret = [torch.Tensor(ext_label_tensor).cuda() for ext_label_tensor in ext_labels_tensor]

        section_scores_tensor = [[] for _ in range(len(section_scores[0][0]))]

        for section_scores_batch in section_scores:
            for sect_scores in section_scores_batch:
                for j, sum_sec_label in enumerate(sect_scores):
                    section_scores_tensor[j].append(sum_sec_label.item())

        section_scores_tensor_ret = [torch.Tensor(x).cuda() for x in section_scores_tensor]

        # section_scores_tensor = torch.Tensor(section_scores_tensor).cuda()
        # import pdb;pdb.set_trace()
        # section_scores_tensor = [torch.Tensor(section_score_tensor).cuda() for section_score_tensor_batch in section_scores_tensor]

        return ext_labels_tensor_ret, section_scores_tensor_ret

    def forward(
            self,
            input_ids=None,
            src_bow_global=None,
            # src_bow_section=None,
            ext_labels=None,
            section_scores=None,
            section_len=None,
            sent_len=None,
            sections_sentence_encoding=None,
            selected_sent_embeddings=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            encoder_section_outputs=None,
            topic_model_global_outputs=None,
            topic_model_section_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            doc_ids=None,
            step=None, # for debugging
            split_noise=False,
            global_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        ext_labels, section_scores = self._convert_2d_labels_to_tensor(ext_labels, section_scores)

        # import pdb;pdb.set_trace()

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
            src_bow_global=src_bow_global,
            # src_bow_section=src_bow_section,
            attention_mask=attention_mask,
            section_len=section_len,
            sent_len=sent_len,
            decoder_input_ids=decoder_input_ids,
            sections_sentence_encoding=sections_sentence_encoding,
            selected_sent_embeddings=selected_sent_embeddings,
            global_attention_mask=global_attention_mask,
            encoder_outputs=encoder_outputs,
            encoder_section_outputs=encoder_section_outputs,
            topic_model_global_outputs=topic_model_global_outputs,
            topic_model_section_outputs=topic_model_section_outputs,
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
            step=step,
        )

        if self.task == 'ext':
            sect_scores, sent_scores = outputs
            loss_sent_fct = nn.BCELoss()
            loss_sect_fct = nn.MSELoss(reduction='sum')
            ext_labels = pad_sequence(ext_labels, batch_first=True, padding_value=-100).unsqueeze(0).cuda()
            section_scores = pad_sequence(section_scores, batch_first=True, padding_value=-100).unsqueeze(0).cuda()
            loss_sent = loss_sent_fct(sent_scores.expand_as(ext_labels.squeeze(0)), ext_labels.squeeze(0))
            loss_sect = loss_sect_fct(sect_scores.expand_as(section_scores.squeeze(0)), section_scores.squeeze(0))

            return Seq2SeqLMOutput(
                sect_loss=loss_sect,
                sent_loss=loss_sent,
            )

        elif self.task == 'all':
            # import pdb;pdb.set_trace()
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

            masked_lm_loss = None
            topic_loss = None
            loss_sent = None
            loss_sect = None
            if labels is not None:

                loss_fct = CrossEntropyLoss()
                if outputs.sent_scores is not None:
                    loss_sent_fct = nn.BCELoss()
                    loss_sect_fct = nn.MSELoss(reduction='sum')

                labels = pad_sequence(labels[0], batch_first=True, padding_value=-100).unsqueeze(0).cuda()
                if outputs.sent_scores is not None:
                    ext_labels = pad_sequence(ext_labels, batch_first=True, padding_value=-100).unsqueeze(0).cuda()
                    section_scores = pad_sequence(section_scores, batch_first=True, padding_value=-100).unsqueeze(0).cuda()


                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                if outputs.sent_scores is not None:
                    loss_sent = loss_sent_fct(outputs.sent_scores.expand_as(ext_labels.squeeze(0)), ext_labels.squeeze(0))
                    loss_sect = loss_sect_fct(outputs.sect_scores.expand_as(section_scores.squeeze(0)), section_scores.squeeze(0))
                    # masked_lm_loss += (loss_sent + loss_sect)

                if self.use_topic:
                    source_bow, word_dists, prior_mean, prior_variance, posterior_mean, \
                    posterior_variance, posterior_log_variance = outputs.topic_info

                    # backward pass
                    self.n_components = self.led.num_topics
                    kl_loss, rl_loss = self._topic_loss(
                        source_bow, word_dists, prior_mean, prior_variance, posterior_mean, posterior_variance,
                        posterior_log_variance
                    )

                    # this is gonna be added to the whole loss...
                    topic_loss = ((kl_loss + rl_loss)).sum()


            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            # if not self.training:
            #     import pdb;pdb.set_trace()
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                loss_topic=topic_loss if self.use_topic else None,
                sect_loss=loss_sect,
                sent_loss=loss_sent,
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
            src_bow_global=None,
            encoder_section_outputs=None,
            topic_model_global_outputs=None,
            topic_model_section_outputs=None,
            section_scores=None,
            ext_labels=None,
            input_ids=None,
            src_ids=None,
            sections_sentence_encoding=None,
            selected_sent_embeddings=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": src_ids,
            "sections_sentence_encoding": sections_sentence_encoding,
            "selected_sent_embeddings": selected_sent_embeddings,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "ext_labels": ext_labels,
            "section_scores": section_scores,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "src_bow_global": src_bow_global,
            "encoder_section_outputs": encoder_section_outputs,
            "topic_model_global_outputs": topic_model_global_outputs,
            "topic_model_section_outputs": topic_model_section_outputs,
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
        is_test = kwargs.pop("is_test", False)
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
        task = kwargs.pop("task", 'all')

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
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path, filename=filename, revision=revision, mirror=mirror
                )

                # archive_file_roberta = hf_bucket_url("roberta-large", filename=filename, revision=revision, mirror=mirror )
                # archive_file_ext = hf_bucket_url("/disk0/sajad/.cache/sci-trained-models/mup-led-arxiv-6144-ExtFinetuned/checkpoint-18000/",
                #                                  filename=filename, revision=revision, mirror=mirror )


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

                # resolved_archive_file_roberta = cached_path(
                #     archive_file_roberta,
                #     cache_dir=cache_dir,
                #     force_download=force_download,
                #     proxies=proxies,
                #     resume_download=resume_download,
                #     local_files_only=local_files_only,
                #     use_auth_token=use_auth_token,
                #     user_agent=user_agent,
                # )
                # resolved_archive_file_ext = cached_path(
                #     '/disk0/sajad/.cache/sci-trained-models/mup-led-arxiv-6144-ExtFinetuned/checkpoint-18000/pytorch_model.bin',
                #     cache_dir=cache_dir,
                #     force_download=force_download,
                #     proxies=proxies,
                #     resume_download=resume_download,
                #     local_files_only=local_files_only,
                #     use_auth_token=use_auth_token,
                #     user_agent=user_agent,
                # )


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
                # state_dict_ext = load_state_dict(resolved_archive_file_ext)

                # state_dict_roberta = load_state_dict(resolved_archive_file_roberta)

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
            model = cls(config, use_topic=use_topic, task=task, *model_args, **model_kwargs)
            # model.resize_token_embeddings(50267) # be causious
            # tokenizer = TGSumTokenizer.from_pretrained(
            #     '/disk0/sajad/.cache/sci-trained-models/mup-led-arxiv-6144-ExtFinetuned/checkpoint-18000/',
            #     cache_dir=model_args.cache_dir,
            #     use_fast=model_args.use_fast_tokenizer,
            #     revision=model_args.model_revision,
            #     use_auth_token=True if model_args.use_auth_token else None,
            # )
            # model.resize_token_embeddings(len(tokenizer))


        # import pdb;pdb.set_trace()
        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)


        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()

        if not is_test:
            added_params = {}
            for n, p in state_dict.items():
                if 'encoder_attn.' in n:
                    added_params[n.replace('encoder_attn', 'encoder_attn_section')] = p
                if 'encoder_attn_layer_norm.' in n:
                    # import pdb;pdb.set_trace()
                    added_params[n.replace('encoder_attn_layer_norm', 'encoder_attn_layer_norm_section')] = p
            state_dict.update(added_params)
        else:
            miskey = state_dict['led.topic_model.beta']


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

        if is_test:
            model.state_dict()['led.topic_model.beta'] = miskey

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
            # import pdb;pdb.set_trace()
            return model, loading_info

        return model