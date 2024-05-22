# Define the MDP
states = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
actions = ['Stand', 'Clap', 'Wave']
rewards = {
    'A1': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'A2': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'A3': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'B1': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'B2': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'B3': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'C1': {'Stand': 1, 'Clap': 1, 'Wave': 1},
    'C2': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'C3': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'D1': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'D2': {'Stand': 0, 'Clap': 0, 'Wave': 0},
    'D3': {'Stand': 0, 'Clap': 0, 'Wave': 0}
}
transitions = {
    'A1': {'Stand': 'A1', 'Clap': 'B1', 'Wave': 'A2'},
    'A2': {'Stand': 'Aa', 'Clap': 'B2', 'Wave': 'A3'},
    'A3': {'Stand': 'A2', 'Clap': 'B2', 'Wave': 'A3'},

    'B1': {'Stand': 'B1', 'Clap': 'C1', 'Wave': 'B2'},
    'B2': {'Stand': 'B1', 'Clap': 'C2', 'Wave': 'B1'},
    'B3': {'Stand': 'B2', 'Clap': 'C3', 'Wave': 'B2'},

    'C1': {'Stand': 'C1', 'Clap': 'D1', 'Wave': 'C2'},
    'C2': {'Stand': 'C1', 'Clap': 'D2', 'Wave': 'C3'},
    'C3': {'Stand': 'C2', 'Clap': 'D3', 'Wave': 'C3'},

    'D1': {'Stand': 'D1', 'Clap': 'D1', 'Wave': 'D2'},
    'D2': {'Stand': 'D1', 'Clap': 'A2', 'Wave': 'D3'},
    'D3': {'Stand': 'D2', 'Clap': 'D3', 'Wave': 'D3'}
}

# Define the starting state
start_state = 'D1'

# Calculate the maximum undiscounted return
max_return = 0
for action in actions:
    # Simulate the episode
    state = start_state
    total_reward = 0
    for _ in range(10):
        next_state = transitions[state][action]
        reward = rewards[state][action]
        total_reward += reward
        state = next_state

    if total_reward > max_return:
        max_return = total_reward

print("Maximum undiscounted return:", max_return)