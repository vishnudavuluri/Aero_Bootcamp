import torch
import torch.nn as nn

# Fluid properties
rho = 1.0
nu = 0.01

# Architecture
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 3)
)

# Optimizer & Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# Domain setup
N_bc = 100
N_physics = 500

# 2D collacation of 100 points independently
x_physics = (torch.rand((N_physics, 1)) * 2.0).requires_grad_(True)
y_physics = (torch.rand((N_physics, 1)) * 1.0).requires_grad_(True)

# Boundary points
# The wall
x_wall = torch.rand((N_bc, 1)) * 2.0
y_wall = torch.zeros((N_bc, 1))

# The inlet
x_inlet = torch.zeros((N_bc, 1))
y_inlet = torch.rand((N_bc, 1))

# Free stream
x_free = torch.rand((N_bc, 1)) * 2*0
y_free = torch.ones((N_bc, 1))

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()

    # Boundary losses
    # 1. Wall loss
    preds_wall = model(torch.cat([x_wall, y_wall], dim=1))
    loss_wall = torch.mean((preds_wall[:, 0:1] - 0.0)**2) + torch.mean((preds_wall[:, 1:2] - 0.0)**2)

    # 2. Inlet loss
    preds_inlet= model(torch.cat([x_inlet, y_inlet], dim=1))
    loss_inlet = torch.mean((preds_inlet[:, 0:1] - 1.0)**2) + torch.mean((preds_inlet[:, 1:2] - 0.0)**2)

    # 3. Free Loss
    preds_free = model(torch.cat([x_free, y_free], dim=1))
    loss_free = torch.mean((preds_free[:, 0:1] - 1.0)**2)

    loss_data = loss_free + loss_inlet + loss_wall

    # Stitching x, y coordinates to feed into the network
    inputs = torch.cat([x_physics, y_physics], dim=1)

    preds = model(inputs)
    u = preds[:, 0:1]
    v = preds[:, 1:2]
    p = preds[:, 2:3]

    # First derivatives
    dudx = torch.autograd.grad(u, x_physics, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dudy = torch.autograd.grad(u, y_physics, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dvdx = torch.autograd.grad(v, x_physics, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dvdy = torch.autograd.grad(v, y_physics, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dpdx = torch.autograd.grad(p, x_physics, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dpdy = torch.autograd.grad(p, y_physics, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Second derivatives
    d2udx2 = torch.autograd.grad(dudx, x_physics, grad_outputs=torch.ones_like(dudx), create_graph=True)[0]
    d2udy2 = torch.autograd.grad(dudy, y_physics, grad_outputs=torch.ones_like(dudy), create_graph=True)[0]
    d2vdy2 = torch.autograd.grad(dvdy, y_physics, grad_outputs=torch.ones_like(dvdy), create_graph=True)[0]
    d2vdx2 = torch.autograd.grad(dvdx, x_physics, grad_outputs=torch.ones_like(dvdx), create_graph=True)[0]

    # Physics loss
    residual_continuty = dudx + dvdy
    residual_x_momentum = u * dudx + v * dudy + 1 / rho * (dpdx) - nu * (d2udx2 + d2udy2)
    residual_y_momentum = u * dvdx + v * dvdy + 1 / rho * (dpdy) - nu * (d2vdx2 + d2vdy2)

    loss_physics = torch.mean(residual_continuty**2) + torch.mean(residual_x_momentum**2) + torch.mean(residual_y_momentum**2)

    # Total Loss
    loss_total = 20 * loss_data + loss_physics

    loss_total.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1} | Total Loss: {loss_total.item():.5f} | Data Loss: {loss_data.item():.5f} | Phys Loss: {loss_physics.item():.5f}")

print("\n--- Simulation Complete ---")

# Let's probe the boundary layer halfway down the plate (x = 1.0)
print("\nVelocity Profile at x = 1.0:")
test_y_coords = [0.0, 0.1, 0.3, 0.6, 1.0]

for y_val in test_y_coords:
    test_pt = torch.tensor([[1.0, y_val]])
    pred = model(test_pt)
    u_vel = pred[0, 0].item()
    print(f"Height y = {y_val:.1f} | u = {u_vel:.4f}")
