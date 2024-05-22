import sys
from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class GP(Policy):

    def __init__(self, opt_action_prob, opt_policy):
        # numpy array of shape [nS, nA]
        self.opt_action_prob = opt_action_prob
        # numpy array of shape [nS]
        self.opt_policy = opt_policy

    def action_prob(self, state, action):
        return self.opt_action_prob[state][action]

    def action(self, state):
        return self.opt_policy[state]

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################

    V = initV
    gamma = env_spec.gamma
    for traj in trajs:
        T = len(traj)
        t = 0
        while True:
            tau = t - n + 1
            if tau >= 0:
                end_t = min(tau + n, T)
                G = 0
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * traj[i - 1][2]
                if tau + n < T:
                    G += (gamma ** n) * V[traj[tau + n][0]]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])
            if tau + 1 == T:
                break
            t += 1

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)

    Q = initQ
    gamma = env_spec.gamma

    opt_action_prob = np.zeros((env_spec.nS, env_spec.nA))
    opt_policy = np.zeros(env_spec.nS)
    for s_i in range(env_spec.nS):
        q_val = []
        for a_i in range(env_spec.nA):
            q_val.append(Q[s_i][a_i])
        optimal_action = q_val.index(max(q_val))
        opt_action_prob[s_i][optimal_action] = 1.0
        opt_policy[s_i] = optimal_action

    pi = GP(opt_action_prob, opt_policy)

    for traj in trajs:
        T = len(traj)
        t = 0
        while True:
            tau = t - n + 1
            if tau >= 0:
                rho = 1
                end_t = min(tau + n, T - 1)
                for i in range(tau + 1, end_t + 1):
                    s_i = traj[i][0]
                    a_i = traj[i][1]
                    rho *= float(pi.action_prob(s_i, a_i) / bpi.action_prob(s_i, a_i))
                G = 0
                end_t = min(tau + n, T)
                for i in range(tau + 1, end_t + 1):
                    G += (gamma ** (i - tau - 1)) * traj[i - 1][2]
                if tau + n < T:
                    G += (gamma ** n) * Q[traj[tau + n][0]][traj[tau + n][1]]
                Q[traj[tau][0]][traj[tau][1]] += alpha * rho * (G - Q[traj[tau][0]][traj[tau][1]])
                q_val = []
                s_i = traj[tau][0]
                for a_i in range(env_spec.nA):
                    q_val.append(Q[s_i][a_i])
                optimal_action = q_val.index(max(q_val))
                for temp_a in range(env_spec.nA):
                    pi.opt_action_prob[s_i][temp_a] = 0.0
                pi.opt_action_prob[s_i][optimal_action] = 1.0
                pi.opt_policy[s_i] = optimal_action
            if tau + 1 == T:
                break
            t += 1

    return Q, pi

