
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# from pytorch_transformers import BertModel
from models.neural import PositionwiseFeedForward, rnn_factory

#
# class Bert(nn.Module):
#     def __init__(self, temp_dir, finetune=False):
#         super(Bert, self).__init__()
#         self.model = BertModel.from_pretrained(temp_dir)
#
#         self.finetune = finetune
#
#     def forward(self, x, segs, mask):
#         if(self.finetune):
#             top_vec, _ = self.model(x, segs, attention_mask=mask)
#         else:
#             self.eval()
#             with torch.no_grad():
#                 top_vec, _ = self.model(x, segs, attention_mask=mask)
#         return top_vec


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):

        pe = torch.zeros(max_len, dim)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        position = torch.arange(0, max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, add_emb=None):
        emb = emb * math.sqrt(self.dim)
        if add_emb is not None:
            emb = emb + add_emb
        if (step):
            pos = self.pe[:, step][:, None, :]
            emb = emb + pos
        else:
            pos = self.pe[:, :emb.size(1)]
            emb = emb + pos
        emb = self.dropout(emb)
        return emb


class DistancePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # absolute position embedding
        ape = torch.zeros(max_len, dim // 2)
        # distance position embedding
        dpe = torch.zeros(max_len, dim // 2)

        ap = torch.arange(0, max_len).unsqueeze(1)
        dp = torch.abs(torch.arange(0, max_len).unsqueeze(1) - mid_pos)

        div_term = torch.exp((torch.arange(0, dim//2, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim * 2)))
        ape[:, 0::2] = torch.sin(ap.float() * div_term)
        ape[:, 1::2] = torch.cos(ap.float() * div_term)
        dpe[:, 0::2] = torch.sin(dp.float() * div_term)
        dpe[:, 1::2] = torch.cos(dp.float() * div_term)

        ape = ape.unsqueeze(0)
        super(DistancePositionalEncoding, self).__init__()
        self.register_buffer('ape', ape)
        self.register_buffer('dpe', dpe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        _, length, _ = emb.size()
        pe_seg = [len(ex) for ex in shift]
        medium_pos = [torch.cat([torch.tensor([0], device=device),
                                 (ex[1:] + ex[:-1]) // 2 + 1,
                                 torch.tensor([length], device=device)], 0)
                      for ex in shift]
        shift = torch.cat(shift, 0)
        index = torch.arange(self.mid_pos, self.mid_pos + length, device=device).\
            unsqueeze(0).expand(len(shift), length) - shift.unsqueeze(1)
        index = torch.split(index, pe_seg)
        dp_index = []
        for i in range(len(index)):
            dpi = torch.zeros([length], device=device)
            for j in range(len(index[i])):
                dpi[medium_pos[i][j]:medium_pos[i][j+1]] = index[i][j][medium_pos[i][j]:medium_pos[i][j+1]]
            dp_index.append(dpi.unsqueeze(0))
        dp_index = torch.cat(dp_index, 0).long()

        dpe = self.dpe[dp_index]
        ape = self.ape[:, :emb.size(1)].expand(emb.size(0), emb.size(1), -1)
        pe = torch.cat([dpe, ape], -1)
        emb = emb + pe
        return emb


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # relative position embedding
        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1) - mid_pos

        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        super(RelativePositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        bsz, length, _ = emb.size()
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(bsz, length) - shift.unsqueeze(1)
        pe = self.pe[index]
        emb = emb + pe
        return emb

    def get_emb(self, emb, shift):
        device = emb.device
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(emb.size(0), emb.size(1)) - shift.unsqueeze(1)
        return self.pe[index]


class MultiHeadedAttentionTopicAware(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True,
                 topic=False, topic_dim=100, split_noise=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear

        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

        self.use_topic = True

        self.split_noise = split_noise
        self.linear_topic_keys = nn.Linear(model_dim,
                                           head_count * self.dim_per_head)
        self.linear_topic_vecs = nn.Linear(topic_dim,
                                           head_count * self.dim_per_head)
        self.linear_topic_w = nn.Linear(head_count * self.dim_per_head * 3, head_count)

    def forward(self, key, value, query, mask=None, layer_cache=None,
                type=None, topic_vec=None, requires_att=False):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                topic_key = self.linear_topic_keys(key)
                topic_key = shape(topic_key)
                topic_vec = self.linear_topic_vecs(topic_vec)
                topic_vec = shape(topic_vec)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    if layer_cache["self_topic_vec"] is not None:
                        topic_vec = torch.cat(
                            (layer_cache["self_topic_vec"].to(device), topic_vec),
                            dim=2)
                    if layer_cache["self_topic_keys"] is not None:
                        topic_key = torch.cat(
                            (layer_cache["self_topic_keys"].to(device), topic_key),
                            dim=2)

                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
                    layer_cache["self_topic_vec"] = topic_vec
                    layer_cache["self_topic_keys"] = topic_key

        else:
            topic_key = self.linear_topic_keys(key)
            topic_key = shape(topic_key)
            topic_vec = self.linear_topic_vecs(topic_vec)
            topic_vec = shape(topic_vec)
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        try:
            scores = torch.matmul(query, key.transpose(2, 3))
        except:
            import pdb;pdb.set_trace()

        topic_vec = topic_vec / math.sqrt(dim_per_head)
        topic_scores = torch.matmul(topic_vec.unsqueeze(3), topic_key.unsqueeze(4)).squeeze_(-1)
        topic_scores = topic_scores.transpose(2, 3).expand_as(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)
            if self.use_topic:
                topic_scores = topic_scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)
        if self.use_topic:
            topic_attn = self.softmax(topic_scores)
            context_raw = torch.matmul(attn, value)
            context_topic = torch.matmul(topic_attn, value)

            p_vec = torch.cat([context_raw, context_topic, query], -1).transpose(1, 2)\
                .contiguous().view(batch_size, -1, head_count * dim_per_head * 3)
            topic_p = torch.sigmoid(self.linear_topic_w(p_vec).transpose(1, 2)).unsqueeze(-1)
            attn = topic_p * attn + (1-topic_p) * topic_attn
            """
            mean_key = torch.sum(topic_key.unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                torch.sum((1-mask).float(), dim=-1, keepdim=True)
            if self.split_noise:
                mean_topic = torch.sum(torch.cat(topic_vec, -1).unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                    torch.sum((1-mask).float(), dim=-1, keepdim=True)
                sigma_vec = torch.cat([query, mean_key, mean_topic], -1).transpose(1, 2)\
                    .contiguous().view(batch_size, -1, head_count * dim_per_head * 4)
            else:
                mean_topic = torch.sum(topic_vec.unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                    torch.sum((1-mask).float(), dim=-1, keepdim=True)
                sigma_vec = torch.cat([query, mean_key, mean_topic], -1).transpose(1, 2)\
                    .contiguous().view(batch_size, -1, head_count * dim_per_head * 3)
            sigma = torch.sigmoid(self.linear_topic_u(torch.tanh(self.linear_topic_w(sigma_vec)))).transpose(1, 2)
            topic_scores = -0.5 * ((1 - torch.sigmoid(topic_scores)) / sigma.unsqueeze(-1)).pow(2)
            scores = scores + topic_scores
            """

        if requires_att:
            required_att = attn.mean(1)
        else:
            required_att = None

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, required_att
        else:
            context = torch.matmul(drop_attn, value)
            return context, required_att

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        mask = mask.bool()

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.layer_norm_topic = nn.LayerNorm(topic_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
            # topic_norm = self.layer_norm_topic(topic_input)
        else:
            input_norm = inputs
            # topic_norm = topic_input

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                    # topic_vec=topic_norm,
                                    mask=mask, type='self')
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoder, self).__init__()
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, hidden_states, attention_mask=None):
        """ See :obj:`EncoderBase.forward()`"""
        # topic_emb = None
        x = self.pos_emb(hidden_states)

        for i in range(self.num_inter_layers):
            x = self.transformer[i](i, x, attention_mask)  # all_sents * max_tokens * dim
        output = self.layer_norm(x)

        return output
