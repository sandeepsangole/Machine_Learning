import numpy as np

# Define the MDP structure
states = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'D1', 'D2']
actions = ['Stand', 'Clap', 'Wave']

# Initialize Q-values to 0 for each state-action pair
Q_values = {(state, action): 0 for state in states for action in actions}

# Trajectories
trajectory_1 = [('A2', 'Clap', 'B2'), ('B2', 'Stand', 'B1'), ('B1', 'Clap', 'C1'), ('C1', 'Clap', 'D1'), ('D1', 'Clap', 'D1'), ('D1', 'Wave','D2')]
trajectory_2 = [('A1', 'Clap', 'B1'), ('B1', 'Clap', 'C1'), ('C1', 'Clap', 'D1'), ('D1', 'Wave','D2')]
trajectory_3 = [('A2', 'Clap', 'B2'), ('B2', 'Clap', 'C2'), ('C2', 'Stand', 'C1'), ('C1', 'Clap', 'D1'), ('D1', 'Wave','D2')]

# Q-learning parameters
discount_factor = 1
learning_rate = 0.1

# Q-learning updates
for trajectory in [trajectory_1, trajectory_2, trajectory_3]:
    for i in range(len(trajectory)):
        state, action, next_state = trajectory[i]
        reward = 0  # Assuming no immediate rewards for most transitions
        if state == 'D1' and action =='Wave':
            reward = 100  # Reward for the transition to D2
        max_next_Q = max(Q_values.get((next_state, a), 0) for a in actions)
        Q_values[(state, action)] += learning_rate * (reward + discount_factor * max_next_Q - Q_values[(state, action)])

# Estimated value of the greedy action in state C2
state_C2_values = [Q_values.get(('C2', a), 0) for a in actions]
estimated_value = max(state_C2_values)

# Print Q-values and estimated value
print("Q-values:")
for (state, action), value in Q_values.items():
    print(f"Q({state}, {action}) = {value:.2f}")

print("\nEstimated value of the greedy action in state C2:")
print(f"{estimated_value:.2f}")
