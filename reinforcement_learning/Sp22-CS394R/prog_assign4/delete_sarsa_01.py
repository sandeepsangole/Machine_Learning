import numpy as np
import math

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low.flatten()
        self.state_high = state_high.flatten()
        self.num_tilings = num_tilings
        self.tile_width = tile_width.flatten()
        self.offsets = []
        num_dim = len(state_high)
        accum = 1
        for i in range(num_dim-1,-1,-1):
            self.offsets.append(accum)
            accum *= math.ceil((state_high[i]-state_low[i])/(tile_width[i])) + 1
        
        self.num_tiles = accum
        self.num_actions = num_actions

    def __call__(self,s,done,a):
        state = np.zeros(self.feature_vector_len())
        if done:
            return state
        else:
            s = s.flatten()
            num_dims = len(s)
            offset = 0
            action_offset = self.num_tilings*self.num_tiles
            for i in range(self.num_tilings):
                offset = i*self.num_tiles + a*action_offset
                for j in range(num_dims):
                    low = (self.state_low[j]-(i/self.num_tilings)*self.tile_width[j])
                    val = s[j]
                    tile_id = math.floor((val-low)/self.tile_width[j])

                    offset += tile_id*self.offsets[j]
                state[offset] = 1
            return state

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions*self.num_tiles*self.num_tilings


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for _ in range(num_episode):
        s = env.reset()
        action = epsilon_greedy_policy(s,False,w)
        x = X(s,False,action)
        z = np.zeros_like(w)
        Qold = 0
        done = False
        while not done:
            s,r,done ,_= env.step(action)
            action = epsilon_greedy_policy(s,done,w)
            x_t = X(s,done,action)
            Q = w.dot(x)
            Q_t = w.dot(x_t)
            delta = r + gamma*Q_t - Q
            z = gamma*lam*z + (1-alpha*gamma*lam*(z.dot(x)))*x
            w = w+ alpha*(delta+Q-Qold)*z - alpha*(Q-Qold)*x
            Qold = Q_t
            x = x_t

    return w        