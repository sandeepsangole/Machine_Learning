# models.py
import random

import numpy as np
import torch
from torch import nn
from scipy.special import softmax


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


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function
    that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.
    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(np.asarray(x)).int()


# class NeuralLanguageModel(LanguageModel):
class RNNLanguageModel(LanguageModel, nn.Module):
    # def __init__(self):
    # raise Exception("Implement me")
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.g = nn.ReLU()
        self.V = nn.Linear(hidden_size, 512)
        self.W = nn.Linear(512, 27)
        self.softmax = nn.Softmax(dim=0)
        self.init_weight()

    def init_weight(self):
        # This is a randomly initialized RNN.
        # Bounds from https://openreview.net/pdf?id=BkgPajAcY7
        # Note: this is to make a random LSTM; these are *NOT* necessarily good weights for initializing learning!
        nn.init.uniform_(self.lstm.weight_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.weight_ih_l0, a=-1.0 /
                                                   np.sqrt(self.hidden_size), b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, context):
        embedded_input = self.word_embedding(context)
        # RNN expects a batch
        embedded_input = embedded_input.unsqueeze(1)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers *
        # num
        # directions
        # x
        # batch_size
        # x
        # dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.lstm(embedded_input, init_state)
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x
        # batch_size
        # x
        # dimensionality
        output = output.squeeze()
        hidden_state = hidden_state.squeeze()
        out = self.W(self.Dropout(self.g(self.V(self.Dropout(self.g(output))))))
        hidden = self.W(self.Dropout(self.g(self.V(self.Dropout(self.g(hidden_state))))))
        return out, hidden

    def get_next_char_log_probs(self, context):
        # raise Exception("Implement me")
        x1 = []
        for ex in context:
            x1.append(self.vocab_index.index_of(ex))
        x = form_input(x1)
        out, hidden = self.forward(x)
        hidden1 = hidden.detach().numpy()
        hidden2 = softmax(hidden1)
        hidden3 = np.log(hidden2)
        # hidden1 = np.log(hidden.detach().numpy())
        return hidden3

    def get_log_prob_sequence(self, next_chars, context):
        # raise Exception("Implement me")
        log_probs = 0.0
        for i in range(len(next_chars)):
            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            log_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
        return log_probs


def read_chunk_data(train_text, dev_text, input_size):
    train_exs = []
    dev_exs = []
    num_train = len(train_text) // input_size
    num_dev = len(dev_text) // input_size
    for i in range(num_train):
        start = i * input_size
        end = (i + 1) * input_size
        train_exs.append(train_text[start: end])
    if num_train * input_size < len(train_text):
        temp = train_text[num_train * input_size: ]
        while len(temp) < input_size:
            temp.append(' ')
        train_exs.append(temp)

    for i in range(num_dev):
        start = i * input_size
        end = (i + 1) * input_size
        dev_exs.append(dev_text[start: end])
    if num_train * input_size < len(dev_text):
        temp = dev_text[num_train * input_size: ]
        while len(temp) < input_size:
            temp.append(' ')
        dev_exs.append(temp)
    return train_exs, dev_exs


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    # raise Exception("Implement me")
    # define hyperparameters
    batch_size = 1
    dict_size = 27
    input_size =20
    hidden_size = 50
    dropout = 0.5

    num_epochs = 9
    num_classes = 27
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    # process training data and label:
    train_exs, dev_exs = read_chunk_data(train_text, dev_text, input_size)
    rnnlm = RNNLanguageModel(dict_size, input_size, hidden_size, dropout,
                             vocab_index)
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=learning_rate)
    ex_idx = [i for i in range(len(train_exs))]

    for epoch in range(0, num_epochs):
        rnnlm.train()
        ex_indices = ex_idx.copy()
        random.shuffle(ex_indices)
        total_loss = 0.0
        i = 0
        for idx in ex_indices:
            i += 1
            out_seq = [vocab_index.index_of(ex) for ex in train_exs[idx]]
            input_seq = [vocab_index.index_of(' ')] + out_seq[:-1]
            x = form_input(input_seq)
            y = torch.tensor(np.asanyarray(out_seq), dtype=torch.long)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT
            # TO
            # DO
            # BEFORE
            # CALLING
            # BACKWARD() *
            rnnlm.zero_grad()
            out, _ = rnnlm.forward(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being
            # explicit
            # here
            loss = criterion(out, y)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            if i % 1000 == 0:
                print("Epoch loss on epoch %i, %i: %f" % (epoch + 1, i, loss))
            loss.backward()
            optimizer.step()
        print("Total training loss on epoch %i th: %f" % (epoch + 1, total_loss))
        # Evaluate on the train set
        rnnlm.eval()
        i = 0
        total_loss = 0.0
        for idx in range(len(dev_exs)):
            i += 1
            out_seq = [vocab_index.index_of(ex) for ex in dev_exs[idx]]
            input_seq = [vocab_index.index_of(' ')] + out_seq[:-1]
            x = form_input(input_seq)
            y = torch.tensor(np.asanyarray(out_seq), dtype=torch.long)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT
            # TO
            # DO
            # BEFORE
            # CALLING
            # BACKWARD() *
            out, _ = rnnlm.forward(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being
            # explicit
            # here
            loss = criterion(out, y)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            if i % 1000 == 0:
                print("Epoch loss on epoch %i, %i: %f" % (epoch + 1, i, loss))
        print("Total dev loss on epoch %i th: %f" % (epoch + 1, total_loss))
    return rnnlm

