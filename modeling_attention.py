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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden):
        """Self-attention module which computes softmax(xQ @ xK^T) @ xV

        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
        """
        # Task 1.1: Create layers requires for self-attention (1 point)
        # YOUR CODE STARTS HERE (~4 lines)
        super(SelfAttention, self).__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.dk = math.sqrt(hidden)

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """Softmax(xQ @ xK^T) @ xV

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        # Task 1.2: Compute Self Attention (3 points)
        # 1. Compute key, query and value matrices from your input x
        # 2. Compute the scores using query and key matrices
        # 3. Compute probabilities using softmax and scale the scores using
        # 4. Compute the output using probabilities and value matrices
        #
        # Write shape of each tensor for each line of code
        # for example:
        #       Suppose batch_size = 3 and seq_len = 5
        #       x = torch.zeros(3, 5) # shape [batch_size, seq_len] 
        #       x = x.unqueeze(1)     # shape [batch_size, 1, seq_len]
        # 
        # NOTE: Remmenber that we work with batches of data [batch_size, seq_len, hidden],
        # not just single examples [seq_len, hidden] as we did in the lecture. This changes your shapes a bit.
        #
        # YOUR CODE STARTS HERE (~ can be implemented in 4 lines or 3 if you combine steps 2 and 3 into one operation)
        key = self.k(x)  # [batch_size, seq_len, hidden]
        query = self.q(x)  # [batch_size, seq_len, hidden]
        value = self.v(x)  # [batch_size, seq_len, hidden]
        scores = torch.matmul(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        probabilities = F.softmax(scores / self.dk, dim=-1)  # [batch_size, seq_len, seq_len]
        output = torch.matmul(probabilities, value)  # [batch_size, seq_len, hidden]
        return output

        # YOUR CODE ENDS HERE

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0):
        """
        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
            num_heads: int, number of attention heads, should be a divisor of hidden
            causal: use causal masking (do not allow queires to look to the keys that correspond to the future tokens)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)# not to forget at scores calculation

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, x, return_attention=False):
        """Computes [Softmax(x Q_1 @ x K_1^T) @ x V_1 : ... : Softmax(x Q_heads @ x K_heads^T) @ x V_heads] @ U
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
            if return_attention is True, returns also FloatTensor[batch_size * num_heads, seq_len, seq_len]
        """
        bs, seq, _ = x.shape

        # Task 2.1 (3 points)
        # YOUR CODE STARTS HERE (Our implementation is in 3 lines, one for each for k, q and v)
        key = self.k(x)  # [batch_size, seq_len, hidden]
        query = self.q(x)  # [batch_size, seq_len, hidden]
        value = self.v(x)  # [batch_size, seq_len, hidden]

        #now I need to slice them. Torch.Tensor.view ~~ numpy.reshape
        #https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        k = key.view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, hidden, head_size)
        q = query.view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, hidden, head_size)
        v = value.view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, hidden, head_size)

        #calculate scores
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_size) # (batch_size, num_heads, hidden, hidden)


        # YOUR CODE ENDS HERE

        if self.causal:
            # Task 2.2 (1 point)
            # Apply casual mask to the scores
            # YOUR CODE STARTS HERE (Our implementation is in 2 lines)

            #mask also should be shaped as scores
            #(batch_size, num_heads, hidden, hidden) * torch.triu by default diagonal
            upper_triangular = torch.triu(torch.ones(seq, seq), diagonal=1)
            mask = upper_triangular.bool()
            mask = mask.expand_as(scores)
            #update scores
            scores = scores.masked_fill_(mask == 0, -float("inf"))


            # YOUR CODE ENDS HERE

        # Task 2.3 (2 points)
        # Compute probability (probs) and attention (att), remember to apply mixing matrix
        # YOUR CODE STARTS HERE (can be implemented in 4 lines)
        probs = nn.Softmax(dim=-1)(scores)
        #if self.dropout ==1:
        probs = self.dropout(probs)
        att = torch.matmul(probs,v)# that is wrong shape
        #we need to convert (batch_size,num_heads,..) -> (batch_size * num_heads,..)
        att = att.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.head_size)
        # YOUR CODE ENDS HERE

        if return_attention:
            return att, probs

        return att