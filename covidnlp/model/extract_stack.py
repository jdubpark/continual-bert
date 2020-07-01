import copy
import math

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_bert import BertLayer


class Classifier(nn.Module):
    """ Simple classifier """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        # scores for each sentence
        scores = self.sigmoid(h) * mask_cls.float()
        return scores


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob, max_len=5000):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout_prob)

        # sinusoid positional encoding (from Transformer)
        pos_enc = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        frequency = position.float() * torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / dim))
        pos_enc[:, 0::2] = torch.sin(frequency)
        pos_enc[:, 1::2] = torch.cos(frequency)
        pos_enc = pos_enc.unsqueeze(0)

        # register
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, embeddings, step=None):
        embeddings = embeddings * math.sqrt(self.dim)
        if step:
            embeddings = embeddings + self.pos_enc[:, step][:, None, :]
        else:
            embeddings = embeddings + self.pos_enc[:, :embeddings.size(1)]

        embeddings = self.dropout(embeddings)
        return embeddings

    def get_embeddings(self, embeddings):
        return self.pos_enc[:, :embeddings.size(1)]


class TransformerStack(nn.Module):
    """ Transformer Encoder for extraction
        TODO: layerwise connections for this as well
    """
    def __init__(self, hidden_size, args):
        super().__init__()
        self.extract_num_layers = args.extract_num_layers
        self.pos_embeddings = PositionalEncoding(hidden_size, args.extract_dropout_prob)

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, layer_norm_eps=args.extract_layer_norm_eps,
                            hidden_dropout_prob=args.extract_dropout_prob, attention_probs_dropout_prob=args.extract_dropout_prob)
        self.encoder_stack = nn.ModuleList([BertLayer(config) for _ in range(args.extract_num_layers)])

        self.dropout = nn.Dropout(args.extract_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, args.extract_layer_norm_eps)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vectors, mask):
        batch_size, n_sentences = top_vectors.size(0), top_vectors.size(1)
        pos_embeddings = self.pos_embeddings.pos_enc[:, :n_sentences]
        x = top_vectors * mask[:, :, None].float()
        x = x + pos_embeddings

        for i, layer_module in enumerate(self.encoder_stack):
            # all_sentences * max_tokens * dim, ~ inverse mask
            # x = self.encoder_stack[i](i, x, x, (~mask).unsqueeze(1))
            inv_mask = ~mask
            # returns tuple (x), so [0]
            x = layer_module(x, attention_mask=inv_mask)[0]

        # last layer is sigmoid classifier
        x = self.layer_norm(x)
        x = self.linear(x)
        scores = self.sigmoid(x) # sentence scores
        scores = scores.squeeze(-1) * mask.float()
        return scores
