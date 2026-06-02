import torch
import torch.nn as nn
import math

# The Neural Network Architecture
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 2. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Physical world setup - The boundary condition
x_bc = torch.tensor([[0.0]], requires_grad=True)
u_bc = torch.tensor([[1.0]])

# Collection of points
x_physics = torch.linspace(0, 5, 30).view(-1, 1).requires_grad_(True)

epochs = 500
print("Training has begun.......")
for epoch in range(epochs):
    optimizer.zero_grad()

    # 1. Data Loss - The boundary data point
    u_pred_bc = model(x_bc)
    loss_data = torch.mean((u_pred_bc - u_bc)**2)

    # 2. Physics Loss - PDE: du/dx + u = 0
    u_pred_physics = model(x_physics)

    # Using autograd to calculate exact continous derivatives
    dudx = torch.autograd.grad(u_pred_physics, x_physics,
                               grad_outputs=torch.ones_like(u_pred_physics),
                               create_graph=True)[0]
    
    # PDE rule
    pde_residual = u_pred_physics + dudx
    loss_physics = torch.mean((pde_residual) ** 2)

    # 3. Tug of war
    loss_total = loss_data + loss_physics
    
    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1} | Total loss: {loss_total.item()} | Data loss: {loss_data.item()} | Physics Loss: {loss_physics}")

# 4. Final test
print("------Training complete------")
x_test = torch.tensor([[2.0]])
u_test = model(x_test)

actual_ans = math.exp(-2.0)

print(f"If x = 2.0:")
print(f"PINN Predicted Output: {u_test.item():.4f}")
print(f"Exact Analytical Math: {actual_ans:.4f}")
