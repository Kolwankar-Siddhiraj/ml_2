import numpy as np

# Step 1: Define the McCulloch-Pitts neuron function
def mcculloch_pitts(inputs, weights, threshold):
    # Compute the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights)
    
    # Apply threshold function
    output = 1 if weighted_sum >= threshold else 0
    
    return output

# Step 2: Define the inputs, weights, and threshold
inputs = np.array([1, 0, 1])  # Example input pattern
weights = np.array([0.5, -0.5, 0.5])  # Example weights
threshold = 0.5  # Example threshold

# Step 3: Compute the output of the McCulloch-Pitts neuron
output = mcculloch_pitts(inputs, weights, threshold)

# Step 4: Print the output
print("Output:", output)