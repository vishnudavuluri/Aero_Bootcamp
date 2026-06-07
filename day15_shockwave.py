import torch
import torch.nn as nn

# Artificial viscosity
epsilon = 0.05

# Architecture
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 4)
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Domain: 1x1 box
N_phys = 1000
x_phys = (torch.rand(N_phys, 1) * 1.0).requires_grad_(True)
y_phys = (torch.rand(N_phys, 1) * 1.0).requires_grad_(True)

# Training Loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    inputs = torch.cat([x_phys, y_phys], dim=1)
    preds = model(inputs)

    rho = preds[:, 0:1]
    u = preds[:, 1:2]
    v = preds[:, 2:3]
    p = preds[:, 3:4]

    # mass fluxes
    mass_x = rho * u
    mass_y = rho * v

    # Momentum fluxes
    mom_xx = rho * u**2 + p
    mom_xy = rho * u * v

    # First derivatives
    dmass_x_dx = torch.autograd.grad(mass_x, x_phys, grad_outputs=torch.ones_like(mass_x), create_graph=True)[0]
    dmass_y_dy = torch.autograd.grad(mass_y, y_phys, grad_outputs=torch.ones_like(mass_y), create_graph=True)[0]

    # 1st Derivatives (Momentum)
    dmom_xx_dx = torch.autograd.grad(mom_xx, x_phys, grad_outputs=torch.ones_like(mom_xx), create_graph=True)[0]
    dmom_xy_dy = torch.autograd.grad(mom_xy, y_phys, grad_outputs=torch.ones_like(mom_xy), create_graph=True)[0]
    
    # We still need 1st & 2nd derivatives of velocity for the artificial viscosity
    dudx = torch.autograd.grad(u, x_phys, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dudy = torch.autograd.grad(u, y_phys, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    d2udx2 = torch.autograd.grad(dudx, x_phys, grad_outputs=torch.ones_like(dudx), create_graph=True)[0]
    d2udy2 = torch.autograd.grad(dudy, y_phys, grad_outputs=torch.ones_like(dudy), create_graph=True)[0]
    
    # PHYSICS LOSSES
    # 1. Continuity
    residual_continuity = dmass_x_dx + dmass_y_dy
    
    # 2. X-Momentum 
    residual_x_momentum = dmom_xx_dx + dmom_xy_dy - epsilon * (d2udx2 + d2udy2)
    
    loss_physics = torch.mean(residual_continuity**2) + torch.mean(residual_x_momentum**2)
    
    loss_physics.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1} | Phys Loss: {loss_physics.item():.5f}")

print("\nArchitecture Test Complete.")