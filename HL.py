# Step 1: Import the necessary libraries
import numpy as np

# Step 2: Define the Hebbian learning function
def hebbian_learning(inputs, learning_rate=0.1, epochs=100):
    # Initialize weights randomly
    num_inputs = inputs.shape[1]
    num_outputs = inputs.shape[1]  # Adjusted to match the shape of the weights
    weights = np.random.randn(num_inputs, num_outputs)

    # Iterate through epochs
    for _ in range(epochs):
        # Iterate through each input pattern
        for input_pattern in inputs:
            # Compute the outer product of input pattern with its transpose
            outer_product = np.outer(input_pattern, input_pattern)
            # Update weights using Hebb's rule
            weights += learning_rate * outer_product

    return weights

# Step 3: Define the input patterns
# Each row represents an input pattern
inputs = np.array([
    [1, -1, 1],
    [-1, 1, 1],
    [1, 1, -1],
    [-1, -1, -1]
])

# Step 4: Perform Hebbian learning
learned_weights = hebbian_learning(inputs)

# Step 5: Print the learned weights
print("Learned Weights:")
print(learned_weights)