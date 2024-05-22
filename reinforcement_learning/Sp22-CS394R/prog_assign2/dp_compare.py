from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class PiStar(Policy):

    def __init__(self, nS, nA):
        super(PiStar, self).__init__()
        self.p = np.zeros((nS, nA))

    def action_prob(self, state:int, action:int) -> float:
        return self.p[state][action]

    def action(self, state:int)-> int:
        return np.argmax(self.p[state])

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

    Q = np.zeros((env.spec.nS, env.spec.nA))
    delta2 = float('inf')

    while delta2 >= theta:
        delta = 0.0

        for state in range(env.spec.nS):
            v = initV[state]
            piActionState = 0

            for action in range(env.spec.nA):
                probActionState = pi.action_prob(state, action)

                for sPrime in range(env.spec.nS):
                    # current reward for state
                    r = env.R[state, action, sPrime]
                    # bellman equation
                    piActionState += probActionState * env.TD[state, action, sPrime] * (r + env.spec.gamma * initV[sPrime])

            # update value prediction
            initV[state] = piActionState

            # update delta
            delta = max(delta, abs(v - initV[state]))
        delta2 = delta

    # update Q values
    delta2 = float('inf')
    while delta2 >= theta:
        delta = 0.0

        for state in range(env.spec.nS):
            for action in range(env.spec.nA):

                # old q value
                q = Q[state][action]
                stateActionValue = 0

                for sPrime in range(env.spec.nS):
                    r = env.R[state, action, sPrime]
                    stateActionValue += env.TD[state, action, sPrime] * (r + env.spec.gamma * initV[sPrime])

                Q[state][action] = stateActionValue

                delta = max(delta, abs(q - Q[state][action]))
        delta2 = delta

    V = initV

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

    delta2 = float('inf')
    pi = PiStar(env.spec.nS, env.spec.nA)

    while delta2 >= theta:
        delta = 0.0

        for state in range(env.spec.nS):
            maxActionValue = - float('inf')

            for action in range(env.spec.nA):
                v = initV[state]
                actionValue = 0

                for sPrime in range(env.spec.nS):
                    r = env.R[state, action, sPrime]
                    actionValue += env.TD[state, action, sPrime] * (r + env.spec.gamma * initV[sPrime])
                    maxActionValue = max(maxActionValue, actionValue)

            initV[state] = maxActionValue

            # compute delta
            delta = max(delta, abs(v - initV[state]))

            # chose optimal action in given state
        delta2 = delta

    # create deterministic policy
    values = []
    for state in range(env.spec.nS):
        for action in range(env.spec.nA):
            actionValue = 0

            for sPrime in range(env.spec.nS):
                r = env.R[state, action, sPrime]
                actionValue += env.TD[state, action, sPrime] * (r + env.spec.gamma * initV[sPrime])
            values.append(actionValue)

        pi.p[state][np.argmax(values)] = 1.0
        values.clear()

    V = initV

    return V, pi