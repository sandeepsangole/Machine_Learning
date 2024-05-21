# transformer.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import math

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = int(self.d_model / self.n_heads)
        self.query_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_matrix = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.out = nn.Linear(self.n_heads * self.head_dim, self.d_model)

    def forward(self, key, query, value, mask=None):
       
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)
        key = key.view(batch_size, seq_length, self.n_heads, self.head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads,
                           self.head_dim)
        value = value.view(batch_size, seq_length, self.n_heads,self.head_dim)
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))
        product = product / math.sqrt(self.head_dim)
        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(batch_size,
                                                          seq_length_query,
                                                          self.head_dim * self.n_heads)
        output = self.out(concat)
        return output



# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super(Transformer, self).__init__()
        self.embedding_layer = Embedder(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, num_positions,
                                                     True)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for i
                                     in range(num_layers)])
        self.Dropout = nn.Dropout(0.2)
        self.g = nn.ReLU()
        self.W1 = nn.Linear(d_model, num_positions)
        self.W2 = nn.Linear(num_positions, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x1 = torch.unsqueeze(indices, 0)  # add batch
        embed_out = self.embedding_layer(x1)
        out = self.positional_encoder(embed_out)
        out1 = torch.squeeze(out).transpose(0, 1)
        for layer in self.layers:
            out, atten_out = layer(out, out, out)
        out = self.W2(self.Dropout(self.g(self.W1(self.Dropout(self.g(out))))))
        res = self.log_softmax(out)
        # return out #32x10x512
        res = torch.squeeze(res)
        # atten_out1 = torch.squeeze(atten_out)
        res2 = atten_out.matmul(out1)
        res3 = F.softmax(res2, dim=-1)
        return res, res3


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)  # 32x10x512
        attention_residual_out = attention_out + value  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512
        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512
        return norm2_out, attention_out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):

    model = Transformer(27, 20, 256, 512, 3, 2)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    for t in range(0, num_epochs):
        i = 0
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            i += 1
            input = train[ex_idx].input_tensor
            label = train[ex_idx].output_tensor
            model.zero_grad()
            res, _ = model.forward(input)
            res = torch.squeeze(res)
            loss = loss_fcn(res, label)  # TODO: Run forward and compute loss
            loss_this_epoch += loss.item()
            if i % 1000 == 0:
                print("Epoch %i, index %i, total loss %f, loss %f" % (t, i, loss_this_epoch, loss))
            loss.backward()
            optimizer.step()

        model.eval()
        total_correct = 0
        total = 20 * len(dev)
        for i in range(0, len(dev)):
            ex = dev[i]
        log_probs, _ = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        gold = ex.output.astype(dtype=int)
        total_correct += sum(gold == predictions)
        print(f"Total accuracy {total_correct / total}")
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
