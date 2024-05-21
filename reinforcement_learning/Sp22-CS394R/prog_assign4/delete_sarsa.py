import numpy as np
import math


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        self.feature_len = 1
        self.dimension_num = []
        for i in range(len(state_low)):
            self.dimension_num.append((math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1))
            self.feature_len *= (math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1)
        # Dimension of each tiling's feature vector
        self.single_feature_len = self.feature_len
        # Dimension of total tilings' feature vector
        self.feature_len *= num_actions
        self.single_feature_len_w_action = self.feature_len
        self.feature_len *= num_tilings

        # Number of tilings
        self.num_tilings = num_tilings
        # Other configs
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.feature_len

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        s_vec = np.zeros(self.feature_vector_len())
        if done:
            return s_vec
        for k in range(self.num_tilings):
            # k-th tilings
            # (low - tiling_index / # tilings * tile width)
            pos = 0
            total_dimension = self.single_feature_len
            for i in range(s.shape[0]):
                # i-th dimension
                idx = math.floor(
                    (s[i] - (self.state_low[i] - k / self.num_tilings * self.tile_width[i])) / self.tile_width[i])
                total_dimension = int(total_dimension / self.dimension_num[i])
                pos += total_dimension * idx
            # Flatten the axis coordinates into position in 1-dimension vector
            s_vec[pos + self.single_feature_len * a + self.single_feature_len_w_action * k] = 1.0
        return s_vec


def SarsaLambda(
        env,  # openai gym environment
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size
        X: StateActionFeatureVectorWithTile,
        num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    # TODO: implement this function
    # Loop for each episode
    for episode in range(num_episode):
        # Reset the environment
        state = env.reset()
        done = False
        next_a = epsilon_greedy_policy(state, done, w)
        x = X(state, done, next_a)
        z = np.zeros(X.feature_vector_len())
        q_old = 0.0
        while not done:
            state, r, done, info = env.step(next_a)
            next_a = epsilon_greedy_policy(state, done, w)
            x_prime = X(state, done, next_a)
            q = np.dot(w, x)
            q_prime = np.dot(w, x_prime)
            delta = r + gamma * q_prime - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
            q_old = q_prime
            x = x_prime

    return w