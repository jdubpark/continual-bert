# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """


import logging
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.init import xavier_uniform_

from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import prune_linear_layer

from .bert import BertEmbeddings, BertPooler

logger = logging.getLogger(__name__)


def mish(x):
    return x * torch.tanh(F.softplus(x))


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
BertLayerNorm = nn.LayerNorm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # linear projections before scale dot-product attention
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # Scale to prevent softmax from reaching extremely small gradients
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, layerwise_hidden_att=None):
        hidden_states = self.dense(hidden_states)

        if layerwise_hidden_att is not None:
            if hidden_states.size(1) != layerwise_hidden_att.size(1):
                # print('unmatch att', hidden_states.size(), layerwise_hidden_att.size())
                hidden_states = hidden_states + layerwise_hidden_att[:, :hidden_states.size(1)]
            else:
                hidden_states = hidden_states + layerwise_hidden_att

        hidden_states = self.dropout(hidden_states)
        # residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT Attention using Self-Attention"""
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layerwise_hidden_att=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states, layerwise_hidden_att)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, layerwise_hidden_ffn=None):
        hidden_states = self.dense(hidden_states)

        if layerwise_hidden_ffn is not None:
            if hidden_states.size(1) != layerwise_hidden_ffn.size(1):
                # print('unmatch ffn', hidden_states.size(), layerwise_hidden_ffn.size())
                hidden_states = hidden_states + layerwise_hidden_ffn[:, :hidden_states.size(1)]
            else:
                hidden_states = hidden_states + layerwise_hidden_ffn

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    e.g. BERT_BASE layer structure:
    (0): BertLayer(
        ## Multi-Head Attention
        (attention): BertAttention(
          ### Scale Dot-Product Attention
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        ## Positon-wise FFN, Linear 1 & activation
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        ## Position-wise FFN, Linear 2 & norm and dropout
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    """
    def __init__(self, idx, device, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.device = device
        self.max_pos_embs = config.max_position_embeddings
        self.hidden_size = config.hidden_size

        """
        In Bert's normal dense and nonlinearity, the adaptor fosters connection from layerwise
        hidden states
        :math: `layerwise connection adaptor`
        \alpha_{i} \odot U_{i}\sigma (V_{i}h_{i-1}^{KB} + c_{i})

        h_{i}=\sigma(W_{i}h_{i-1} + \alpha_{i}\odot{U_{i}\sigma(V_{i}h_{i-1}^{KB} + c_{i})} + b_{i})

        where:
        \alpha_{i} is trainable vector of size equal to # of units in layer i
        U_{i}, V_{i} are weight matrices
        h_{i-1}^{KB} is activation from KB - here, we use hidden states
        """
        # All layers except first layer receives layerwise connection
        if idx > 0:
            # adaptor in: hidden_size from KB self.output()
            # adaptor out: hidden_size into AC self.output()
            self.adaptor_dense = {
                'att': nn.Linear(config.hidden_size, config.hidden_size).to(device),
                'ffn': nn.Linear(config.hidden_size, config.hidden_size).to(device),
            }
            self.adaptor_nonlinearity = ACT2FN['gelu']
            self.adaptor_weight2 = {
                'att': torch.rand((config.max_position_embeddings, config.hidden_size), requires_grad=True, device=device),
                'ffn': torch.rand((config.max_position_embeddings, config.hidden_size), requires_grad=True, device=device),
            }
            self.adaptor_alpha = {
                'att': torch.rand((config.max_position_embeddings,), requires_grad=True, device=device),
                'ffn': torch.rand((config.max_position_embeddings,), requires_grad=True, device=device),
            }

            xavier_uniform_(self.adaptor_dense['att'].weight)
            xavier_uniform_(self.adaptor_dense['ffn'].weight)

    def evm(self, v, M):
        """ Element-wise multiplication of vector v and matrix M
        """
        N = M.shape[0]
        shp = M.shape[1:]
        return torch.mul(v, M.view(N, -1).transpose(0, 1)).transpose(0, 1).view(N, *shp)

    def layerwise_adaptor(self, type, hidden_states):
        output = torch.squeeze(hidden_states, 0)
        # print('... lw adaptor {})'.format(type), output.size())
        # print('({} 1)'.format(type), hidden_states.size(), output.size())
        should_pad = False
        # print('orig', output.size())
        if output.size(0) != self.max_pos_embs:
            # print('need padding for this output', output.size())
            # should_pad = True
            orig_size = output.size(0)
            pad = torch.zeros((self.max_pos_embs, self.hidden_size), device=self.device)
            pad[:orig_size, :] = output
            output = pad
            # pad_top = math.floor((self.max_pos_embs-orig_size)/2)
            # pad_bot = self.max_pos_embs-pad_top
            # F.pad(input=output, pad=(0, 0, pad_top, pad_bot), mode='constant', value=0)
            # print('pad', output.size())

        output = self.adaptor_dense[type](output)
        output = self.adaptor_nonlinearity(output)
        # print('({} 2)'.format(type), output.size(), self.adaptor_weight2[type].size())
        output = torch.mul(self.adaptor_weight2[type], output)
        # vector-matrix elementwise multiplication
        # print('({} 3)'.format(type), output.size(), self.adaptor_alpha[type].view(-1, 1, 1).size())
        output = self.evm(self.adaptor_alpha[type], output) # ~ [:, None, None] * outputs
        # print('({} 3)'.format(type), output.size())
        # if should_pad:
        #     output = output[:orig_size]
        return torch.unsqueeze(output, 0)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layerwise_outputs_layer=None,
    ):
        layerwise_hidden_att = None
        layerwise_hidden_ffn = None
        if layerwise_outputs_layer is not None:
            layerwise_hidden_att = self.layerwise_adaptor('att', layerwise_outputs_layer['att'])
            layerwise_hidden_ffn = self.layerwise_adaptor('ffn', layerwise_outputs_layer['ffn'])

        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                                layerwise_hidden_att=layerwise_hidden_att)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, layerwise_hidden_ffn=layerwise_hidden_ffn)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(idx, device, config) for idx in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layerwise_outputs=None,
    ):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                # head_mask is [] so get head_mask[i]
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
                layerwise_outputs_layer=(layerwise_outputs[i-1] if i > 0 else None)
            )
            # print(layer_outputs)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModel(BertPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.knowledge_base = kwargs.pop('knowledge_base')

        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(kwargs.pop('device'), self.config)
        self.pooler = BertPooler(self.config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        """ Run knowledge base in eval mode to prepare hidden_states for layerwise connection
        """
        with torch.no_grad():
            layerwise_outputs = self.knowledge_base.forward_layerwise(
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )

        # Active Column forwarding from here
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            layerwise_outputs=layerwise_outputs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
