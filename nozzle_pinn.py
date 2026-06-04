import torch
import torch.nn as nn

# Properties of the fluid
R = 1.0
gamma = 1.4
Cp = 3.5

# Geometry of the cross-section
def get_area(x):
    return 1 + (x - 1.5)**2

# Derivative of the area dA/dx
def get_area_grad(x):
    return 2 * (x - 1.5)

# The Architecture
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 3)
)

# Opitimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Boundary conditions
x_bc = torch.tensor([[0.0]])
rho_in = torch.tensor([[1.0]])
u_in = torch.tensor([[0.5]])
T_in = torch.tensor([[1.0]])

# The Physics: 100 points across the nozzle
x_physics = torch.linspace(0, 3, 100).view(-1, 1).requires_grad_(True)

# Training Loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward Pass & data loss
    bc_pred = model(x_bc)
    rho_bc_pred = bc_pred[:, 0:1]
    u_bc_pred = bc_pred[:, 1:2]
    T_bc_pred = bc_pred[:, 2:3]

    loss_data = torch.mean((rho_in - rho_bc_pred)**2) + \
                torch.mean((u_in - u_bc_pred)**2) + \
                torch.mean((T_in - T_bc_pred)**2)

    # Forward Pass and Physics Loss
    phys_pred = model(x_physics)
    rho = phys_pred[:, 0:1]
    u = phys_pred[:, 1:2]
    T = phys_pred[:, 2:3]

    # Calculate pressure P
    p = rho * R * T

    # Area and its gradient
    A = get_area(x_physics)
    dAdx = get_area_grad(x_physics)

    dudx = torch.autograd.grad(u, x_physics, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    drhodx = torch.autograd.grad(rho, x_physics, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    dTdx = torch.autograd.grad(T, x_physics, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dpdx = torch.autograd.grad(p, x_physics, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    residual_mass = rho * A * dudx + A * u * drhodx + rho * u * dAdx
    residual_momentum = dpdx + rho * u * dudx
    residual_energy = Cp * dTdx + u * dudx

    loss_physics = torch.mean(residual_mass**2) + torch.mean(residual_momentum**2) + torch.mean(residual_energy**2)

    lambda_data = 100.0
    lambda_physics = 1.0

    loss_total = loss_data + loss_physics
    loss_total.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1} | Loss Total: {loss_total.item():.5f}")

print("\nTraining Complete.")

print(f"Actual value at throat section (x = 1.5), u = 1.1832 (speed of sound) , model predicts {model(torch.tensor([[1.5]]))[:, 1:2].item():.4f}")
