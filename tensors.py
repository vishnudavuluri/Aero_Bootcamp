import torch

# 1. Input data
inputs = torch.tensor([2.0, 3.0])
print(f"Input size: {inputs}")

# 2. Creating 3 neurons and giving them weights and biases
weights = torch.tensor([
    [0.8, -0.3],
    [0.5, 0.7],
    [-0.6, 0.1]
])

biases = torch.tensor([-0.1, 0.4, -0.3])

# 3. The nueral network calculation, y = Wx + b using matmul
outputs = torch.matmul(weights, inputs) + biases

# 4. The output
print(f"\nWeights matrix size: {weights.shape}")
print(f"Output size: {outputs.shape}")
print(f"Final nuerons output: {outputs}")