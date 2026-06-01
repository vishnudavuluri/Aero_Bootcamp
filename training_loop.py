import torch
import torch.nn as nn

# 1. Ground Data: mass (m) = 2kg, Newtons 2nd law of motion: F = ma
# Accelaration (a) and training data - actual forces
accelarations = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
actual_forces = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. Nueral layer architecture
model = nn.Linear(1, 1)

# 3. Loss function: Mean squared error
criterion = nn.MSELoss()

# 4. Optimization: Stochastic Gradient Decent and with learning rate of 9.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. Training loop
epochs = 100

print("Starting the training........")
for epoch in range(epochs):

    # Step 1: Forward pass (Let the AI guess the value)
    predicted_forces = model(accelarations)

    # Step: 2: Calculate the loss
    loss = criterion(predicted_forces, actual_forces)

    # Step 3: Zero the gradient (clear the pytorch memory from last epoch)
    optimizer.zero_grad()

    # Step 4: Backward pass (Make pytorch calculate the derivative of the loss function)
    loss.backward()

    # Step 5: Optimize (Update the weight and biases of the network)
    optimizer.step()

    # Progress for every 29 loops
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1} | Loss (error): {loss.item():.4f}")

# 5. Testing the AI on the data it has never see before
print("\n-----Trainig Complete------")
test_accelaration = torch.tensor([5.0])
predicted_force = model(test_accelaration)

print(f"If the a = 5.0, the AI predicts {predicted_force.item():.4f} N")
print("The actual force is 10.0 N")