import torch
import torch.nn as nn

# 1. Physics data: Aerodynamic drag = 0.5 * v ^2
# torch's linespace creates evenly spaced values with bound included
#.view() reshapes it into a tall coloumn vector so that torch can process it
velocities = torch.linspace(0, 10, 50).view(-1, 1) # -1 is a placeholder for dimension
actual_drag = 0.5 * velocities**2

# 2. Deep neural network architecture, .sequential connects layers
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 3. The loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
epochs = 500 # more epochs because the curve is complex

print("Traning the Deep nueral network on Drag Physics")
for epoch in range(epochs):
    predictions = model(velocities)
    loss = criterion(predictions, actual_drag)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

# 5. The test
print("\n------Training completed------")
test_v = torch.tensor([[5.0]])
predicted_drag = model(test_v)

# If v = 5, drag = 0.5 * 5^2 = 12.5
print(f"If velocity is 5 m/s \nPredicted drag {predicted_drag.item():.4f}N")
print("Actual drag is 12.5N")