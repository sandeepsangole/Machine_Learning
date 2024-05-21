import numpy as np
import sys

def argmax(q_values):
    top_value = float("-inf")
    ties = []

    top_value = max(q_values)
    for i in range(len(q_values)):

        if q_values[i] == top_value:
            ties.append(i)

    return np.random.choice(ties)

def sample_average(q_values, arm_count, last_action, reward):
    arm_count[last_action] += 1
    n = arm_count[last_action]
    q_values[last_action] +=  (reward - q_values[last_action])/n

def constant_step_size(q_values, arm_count, last_action, reward, alpha = 0.1):
    arm_count[last_action] += 1
    q_values[last_action] +=  alpha * (reward - q_values[last_action])

def testbed(strategy):
    num_iterations = 10000
    num_runs = 300

    optimal_choices = np.array([0 for i in range(num_iterations)])
    average_rewards = np.array([0 for i in range(num_iterations)])

    for j in range(num_runs):
        bandits = [0 for x in range(10)]
        # per step
        q_values = [0 for x in range(10)]
        arm_count = [0 for x in range(10)]
        for i in range(num_iterations):

            optimal_action = bandits.index(max(bandits))
            rand = np.random.random()
            # print(rand)
            if rand <= 0.1:  # explore
                action = np.random.choice(len(arm_count))
            else:
                action = argmax(q_values)

            reward = np.random.normal(bandits[action], 1)
            average_rewards[i] += reward
            if optimal_action == action:
                optimal_choices[i] += 1

            # update rewards
            for idx in range(len(bandits)):
                noise = np.random.normal(0.0, 0.01)
                bandits[idx] = bandits[idx] + noise

            strategy(q_values, arm_count, action, reward)

    optimal_choices = [i / num_runs for i in optimal_choices]
    average_rewards = [i / num_runs for i in average_rewards]

    return optimal_choices, average_rewards


if __name__ == "__main__":
    cs_optimal_choices, cs_average_rewards = testbed(strategy=constant_step_size)
    sa_optimal_choices, sa_average_rewards = testbed(strategy=sample_average)

    arr = np.stack([  sa_average_rewards,sa_optimal_choices, cs_average_rewards, cs_optimal_choices], axis=0)
    np.savetxt(sys.argv[1], arr, fmt='%s')
