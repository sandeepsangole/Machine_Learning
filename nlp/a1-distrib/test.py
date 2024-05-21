import torch

# Define the initial weight vectors
w_health = torch.tensor([0, 0, 0], dtype=torch.float)
w_sports = torch.tensor([0, 0, 0], dtype=torch.float)
w_science = torch.tensor([0, 0, 0], dtype=torch.float)

# Define the training examples
example1_features = torch.tensor([1, 1, 0], dtype=torch.float)
example2_features = torch.tensor([1, 0, 1], dtype=torch.float)

# True labels
true_label1 = 1
true_label2 = 2

# Predict the labels for the examples
def predict_label(features, w_health, w_sports, w_science):
    scores = [torch.dot(w_health, features), torch.dot(w_sports, features), torch.dot(w_science, features)]
    predicted_label = torch.argmax(torch.tensor(scores))
    return predicted_label

# Check if the prediction is correct and update weights if needed
def update_weights(features, true_label, w_health, w_sports, w_science):
    predicted_label = predict_label(features, w_health, w_sports, w_science)
    if predicted_label != true_label:
        if true_label == 1:
            w_health += features
        elif true_label == 2:
            w_sports += features
        else:
            w_science += features
    return w_health, w_sports, w_science

# Update weights for example 1
w_health, w_sports, w_science = update_weights(example1_features, true_label1, w_health, w_sports, w_science)

# Update weights for example 2
w_health, w_sports, w_science = update_weights(example2_features, true_label2, w_health, w_sports, w_science)

# Print the final weight vectors
print("w_health:", w_health)
print("w_sports:", w_sports)
print("w_science:", w_science)
