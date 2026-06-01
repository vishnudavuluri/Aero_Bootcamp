import torch

# 1. Input tensor and putting requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# 2. Forward pass, the equation
y = 3 * x**2 + 2 * x

print(f"Calculated value of y: {y.item()}")

# 3. Backward pass, the derivative - calculates all derivatives
y.backward()

# 4. Result
print(f"The exact derivative (dy/dx) at x = 2 is: {x.grad.item()}")