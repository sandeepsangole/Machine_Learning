# models.py

import numpy as np
import math

import torch
from torch import nn, Tensor
from scipy.special import softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset, Dataset, DataLoader
from torch import nn, optim


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_index, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, vocab_size):
        super().__init__()
        self.vocab_index = vocab_index
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.init_weights()
        self.vocab_size = vocab_size


    def get_next_char_log_probs(self, context):
        self.eval()
        # print("Context: " + str(context))
        context_tensor = []
        for c in context:
            context_tensor.append(self.vocab_index.index_of(c))
        context_tensor = torch.LongTensor(context_tensor).unsqueeze(0)

        if context_tensor.shape[1] > 0:
            # print("Context: " + str(context_tensor))
            with torch.no_grad():
                output = self(context_tensor)
                log_probs = softmax(output[:, -1, :])
                log_probs = np.log(log_probs)
                retlog = log_probs[0]
                # print("Retlog: " + str(retlog))
            return retlog
        else:
            return np.zeros(self.vocab_size)

    def get_log_prob_sequence(self, next_chars, context):
        # raise Exception("Implement me")
        self.eval()
        log_probs = 0.0
        for i in range(len(next_chars)):
            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            log_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
        return log_probs

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = torch.triu(torch.ones(src.size(0), src.size(0)) * float('-inf'), diagonal=1)
        output = self.transformer_encoder(src, src_mask, is_causal=True)
        output = self.linear(output)
        return output


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function
    that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.
    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(np.asarray(x)).int()


class CharacterDataset(Dataset):
    def __init__(self, text, vocab_index, chunk_length):
        self.text = text
        self.vocab_index = vocab_index
        self.chunk_length = chunk_length

    def __len__(self):
        return len(self.text) - self.chunk_length

    def __getitem__(self, idx):
        input_data = []
        target_data = []
        chunk = self.text[idx:idx + self.chunk_length + 1]

        input_seq = chunk[:-1]
        target_seq = chunk[1:]

        for c in input_seq:
            input_data.append(self.vocab_index.index_of(c))
        input_data = torch.LongTensor(input_data)

        for c in target_seq:
            target_data.append(self.vocab_index.index_of(c))
        target_data = torch.LongTensor(target_data)

        return input_data, target_data


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    vocab_size = 27
    d_model = 128
    num_head = 4
    num_encoder_layers = 3
    batch_size = 32
    num_epochs = 3
    chunk_length = 50

    train_dataset = CharacterDataset(train_text, vocab_index, chunk_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralLanguageModel(vocab_index, vocab_size, d_model, num_head, 2, num_encoder_layers, 0.1, vocab_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch

            optimizer.zero_grad()
            logits = model(inputs)

            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')

    model.eval()
    dev_dataset = CharacterDataset(dev_text, vocab_index, chunk_length)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    with torch.no_grad():
        for batch in dev_loader:
            inputs, targets = batch
            logits = model(inputs)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(dev_loader)
    print(f'Dev Loss: {average_loss}')
    return model
