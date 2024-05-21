from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy




def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # Implement Value Prediction Algorithm (Hint: Sutton Book p.75)


    #####################

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    V = {}
    for state in env.spec.nS:
        V[state] = 0.
    while True:
        newV = {}
    # compute the new values (newV) given the olf values (V)
        for state in env.spec.nS:
            if


    #####################

    return V, pi
