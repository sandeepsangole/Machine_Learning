import math

import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function

    # 1. Looping for each episode

    for episode in range(num_episode):
        #2. Initialize and store S0 != terminal
        t = 0
        T = math.inf
        s = env.reset()
        rewards = []
        states = [s]
        isTerminated = False

        # state = env.reset()
        while True:

            if t < T:
                #3. Take action according to policy given state
                action = pi.action(states[-1])
                #4. Observer and Store the next reward as Rt+1 and next State as St+1
                state, reward, isTerminated, info = env.step(action)
                rewards.append(reward)
                states.append(state)
                if isTerminated:
                    T = t + 1
            tau = t - n + 1

            if tau >=0:
                G = 0
                # G = sum t+ 1- min(r+n,T) - gamma^i-tau rewards(i)
                end_t = min(tau + n, T)
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                # if tau +n < T = G = G + gamma^n v(State(tau+n, v)
                    if tau + n < T:
                        G += (gamma ** n) * V(states[tau + n])

                # Updates weights/values
                    V.update(alpha, G, states[tau])

                if(tau + 1) == T:
                    break

                t += 1





