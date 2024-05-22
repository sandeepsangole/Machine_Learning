# Given initial weights
w = [1, 1, 1, 1, 1, 1, 5]

# Learning rate α and discount factor γ
α = 0.1
γ = 0.95

# Updates for each weight
updates = [0] * 7  # Initialize an array to store updates

# List of state transitions
transitions = [(0, 6), (1, 6), (2, 6), (3, 6), (5, 6), (6, 6)]

for S, S_prime in transitions:
    # Calculate ∇v(S)
    gradient_S = [0 if i != S else 1 for i in range(7)]

    # Calculate ∇v(S')
    gradient_S_prime = [0 if i != S_prime else 1 for i in range(7)]

    # Calculate the update for each weight
    for i in range(7):
        updates[i] += α * (0 + γ * w[S_prime] - w[S]) * gradient_S[i]

# Update the weights
for i in range(7):
    w[i] += updates[i]

# Calculate the new values of specific states
new_value_state1 = 2 * w[0] + w[6]
new_value_state2 = w[0] + 2 * w[1]

# Round to 3 decimal places
w = [round(weight, 3) for weight in w]
new_value_state1 = round(new_value_state1, 3)
new_value_state2 = round(new_value_state2, 3)

# Print the new weights and state values
print("New weights after the update:")
print("w0:", w[0])
print("w1:", w[1])
print("w2:", w[2])
print("w3:", w[3])
print("w4:", w[4])
print("w5:", w[5])
print("w6:", w[6])

print("New value of state (2w0 + w6):", new_value_state1)
print("New value of state (w0 + 2w1):", new_value_state2)