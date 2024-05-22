from typing import Iterable
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PiApproximationNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(PiApproximationNN, self).__init__()
        self.hidden_layer_1 = nn.Linear(n_input, n_hidden)
        self.hidden_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        hidden_output_1 = F.relu(self.hidden_layer_1(input))
        hidden_output_2 = F.relu(self.hidden_layer_2(hidden_output_1))
        actions = self.output_layer(hidden_output_2)
        return F.softmax(actions, dim=-1)

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.num_actions = num_actions
        self.state_dims = state_dims
        self.alpha = alpha
        self.network = PiApproximationNN(state_dims, 32, num_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.probabilities = dict()

    def __call__(self, s) -> int:
        self.network.eval()
        action_probs = self.network(torch.from_numpy(s).float())
        m = Categorical(action_probs)
        action = m.sample().item()
        self.probabilities[(tuple(s), action)] = torch.log(action_probs.squeeze(0)[action])
        return action

    def update(self, s, a, gamma_t, delta):
        action_prob = self.network(torch.from_numpy(s).float())
        selected_action_log_prob = torch.log(action_prob.squeeze(0)[a])
        loss = -selected_action_log_prob * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(VApproximationNN, self).__init__()
        self.hidden_layer_1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(n_hidden, 32)
        self.output_layer = nn.Linear(32, n_output)

    def forward(self, input):
        output = self.hidden_layer_1(input)
        output = self.relu(output)
        output = self.hidden_layer_2(output)
        output = self.output_layer(output)
        return output

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__(state_dims)
        self.alpha = alpha
        self.network = VApproximationNN(state_dims, 64, 1)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.mse = nn.MSELoss()

    def __call__(self, s) -> float:
        self.network.eval()
        return self.network(torch.from_numpy(s).float()).item()

    def update(self, s, G):
        self.network.train()
        self.optimizer.zero_grad()
        pred = self.network(torch.from_numpy(s).float())
        loss = self.mse(pred, torch.tensor([G], dtype=torch.float32))
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    result = []

    for episode in range(num_episodes):
        episode_data = []
        state = env.reset()
        action = pi(state)
        done = False

        while not done:
            new_state, reward, done, info = env.step(action)
            episode_data.append((state, action, reward))
            state = new_state
            action = pi(state)

        total_steps = len(episode_data)

        for t, step in enumerate(episode_data):
            s, a, r = step
            total_return = sum([pow(gamma, idx) * episode_data[idx][2] for idx in range(t, total_steps)])

            if t == 0:
                result.append(total_return)

            delta = total_return - V(s)
            V.update(s, total_return)
            pi.update(s, a, pow(gamma, t), delta)

    return result

