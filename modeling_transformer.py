#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
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
import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, fcn_hidden, dropout=0.0, causal=False):
        super().__init__()
        # Task 2.1 (1 point)
        # Create layers needed for Transformer Encoder Block
        # (5 layers in total, including dropout)
        # We recommend to use nn.Sequential for FFN instead of creating is layer-by-layer,
        # it will make your code more readable
        # YOUR CODE STARTS HERE  (our implementation is about 5-8 lines)
        #model = MultiHeadSelfAttention(input_size=7, hidden=9, num_heads=3, causal=True)
        #class MultiHeadSelfAttention(nn.Module):
        #def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0)
        self.self_attention = MultiHeadSelfAttention(input_size= hidden, hidden= hidden, num_heads= num_heads)
        self.linear_1 = nn.Linear(hidden, fcn_hidden)
        self.linear_2 = nn.Linear(fcn_hidden, hidden)
        self.norm_1 = nn.LayerNorm(hidden)
        self.norm_2 = nn.LayerNorm(hidden)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()    

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """Self-Attention -> residual -> LayerNorm -> FCN -> residual -> LayerNorm
        
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """

        # Task 2.2 (2 points)
        # Implement Transformer encoder block forward pass
        # You can implement residual connection this way:
        # residual = x
        # x = some_stuff_that_changes_x(x)
        # x = x + residual
        # YOUR CODE STARTS HERE (our implementation is about 6 lines)
         # multihead self attention
        attention_output = self.self_attention(x)
        attention_output = self.dropout_1(attention_output)
        x = self.norm_1(x + attention_output)

        # positional feedforward
        feed_forward_output = self.linear_1(x)
        feed_forward_output = self.relu(feed_forward_output)
        feed_forward_output = self.linear_2(feed_forward_output)
        feed_forward_output = self.dropout_2(feed_forward_output)
        x = self.norm_2(x + feed_forward_output)
        # YOUR CODE ENDS HERE
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1, causal=False):
        """A minimal implementation of Transformer Encoder
        
        Args:
            num_layer: number of encoder layer
            hidden: embedding size and hidden size of attentions
            fcn_hidden: hidden size of fully-connected networks inside transformer layers
            vocab_size: size of vocabulary
            max_seq_len: maximum length of input sequence
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.fcn_hidden = fcn_hidden
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        # Task 2.3 (2 points)
        # 1. Create embedding layer and positional embedding layer
        # Use nn.Embedding for that
        # 2. Create a linear layer logit_proj that will project contextualized representations
        # of size hidden to your vocabulary size.
        # 3. Create a dropout layer
        # 4. Create a list of encoder Layers
        # Note that you need to wrap it with nn.ModuleList,
        # so that the parameters of the layers would be counted as the paramertes of the model
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # Read more about ModuleList here:
        # https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
        # You can use for-loop of python list comprehension to create the list of layers
        # YOUR CODE STARTS HERE (our implementation is about 6 lines)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden)

        self.trans_layers = nn.ModuleList([TransformerEncoderLayer(hidden, num_heads, fcn_hidden, dropout) for _ in range(num_layers)])
        # YOUR CODE ENDS HERE

    def _add_positions(self, sequence_tensor):
        """Adds positional embeddings to the input tensor.

        Args:
            sequence_tensor: FloatTensor[batch_size, seq_len, hidden]
        
        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        #seq_len = sequence_tensor.shape[1]
        seq_len = sequence_tensor.size(1)

        # Task 2.5 (1 point)
        # Implement positional embedding which is a sum of:
        # 1. Embedding of the position (self.pos_emb)
        # 2. Embedding of the token (sequence_tensor)
        # Remember that is you create any tensors here,
        # you need to move them to the same device as sequence_tensor
        # You can get device of sequence_tensor with sequence_tensor.device
        # YOUR CODE STARTS HERE (our implementation is about 3 lines)
        #
        positions = torch.arange(0, seq_len, dtype=torch.long, device= sequence_tensor.device)
        positions = positions.unsqueeze(0).expand(1, -1)  # (1, seq_len)
        sequence_tensor = sequence_tensor + self.positional_embedding(positions)
        # YOUR CODE ENDS HERE

    def forward(self, input_ids=None):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len]
        
        Returns:
            FloatTensor[batch_size, src_seq_len, hidden]
        """
        # Task 2.6 (2 points)
        # Implement Transformer Encoder
        # Remember that Transformer Encoder is composed of:
        # 1. Embedding
        # 2. Positional Embedding (use self._add_positions)
        # 3. Transformer Encoder Layers
        # NOTE: Please write shape of the tensor for each line of code
        # YOUR CODE STARTS HERE (our implementation is about 4 lines)
        x = self.embedding(input_ids)
        self._add_positions(x)
        for layer in self.trans_layers:
            x = layer(x)
        return x
        # YOUR CODE ENDS HERE


class TransformerLM(nn.Module):
    def __init__(self, num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout=0.1):
        """Transformer Language Model"""
        super().__init__()
        self.dropout_rate = dropout

        # Task 2.7 (1 point)
        # Create a Transformer Encoder, output layer for language modeling, and a dropout layer
        # Remember that when we use Transformer for language modeling, it should be **causal** or it will cheat.
        # Output layer should predict the logits for all words in the vocabulary (size of logits = vocab_size)
        # YOUR CODE STARTS HERE (our implementation is about 2 lines)
        self.encoder = TransformerEncoder(num_layers, hidden, num_heads, fcn_hidden, vocab_size, max_seq_len, dropout)
        self.trans_dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(hidden, vocab_size)
        # YOUR CODE ENDS HERE
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: LongTensor[batch_size, src_seq_len], optional, encoder_embeds could be used instead
        Returns:
            FloatTensor[batch_size, src_seq_len, vocab_size] — logits over the vocabulary
        """
        assert input_ids.dim() == 2, "Input should be of size [batch_size, seq_len]"
        # Task 2.8 (1 point)
        # Implement Transformer Language Model
        # Remember that Transformer Language Model is composed of:
        # 1. Transformer Encoder
        # 2. Dropout
        # 3. Output Layer to produce logits over the classes (our vocabulary in case of language modeling)
        # YOUR CODE STARTS HERE (our implementation is 2 lines)
        output = self.encoder(input_ids)
        x = self.trans_dropout(output)
        logits = self.logits(x)
        return logits
        # YOUR CODE ENDS HERE

    def save_pretrained(self, save_path):
        """Save the model weights to a directory

        Args:
            save_path: directory to save the model
        """
        config = {
            "num_layers": self.encoder.num_layers,
            "hidden": self.encoder.hidden,
            "num_heads": self.encoder.num_heads,
            "fcn_hidden": self.encoder.fcn_hidden,
            "vocab_size": self.encoder.vocab_size,
            "max_seq_len": self.encoder.max_seq_len,
            "dropout": self.encoder.dropout_rate,
        }

        with open(os.path.join(save_path, "model_config.json"), "w") as f:
           json.dump(config, f)

        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, save_path):
        """Load the model weights from a directory

        Args:
            save_path: directory to load the model
        """
        with open(os.path.join(save_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_path, "model.pt"), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model
