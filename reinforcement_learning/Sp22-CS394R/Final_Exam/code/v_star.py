# Define the MDP structure
states = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'D1', 'D2']
actions = ['Stand', 'Clap', 'Wave']

# Q-values obtained from Q-learning updates
Q_values = {
    ('A2', 'Stand'): -1.0,
    ('A2', 'Clap'): 0.0,
    ('A2', 'Wave'): -1.0,
    # ... (similar entries for other state-action pairs)
}

# Discount factor
gamma = 1.0

# Calculate optimal values using the Bellman optimality equation
optimal_values = {state: max(Q_values.get((state, action), 0) for action in actions) for state in states}

# Print the optimal value of A2
optimal_value_A2 = optimal_values['A2']
print(f"Optimal value of A2 under the discount factor {gamma}: {optimal_value_A2:.2f}")
