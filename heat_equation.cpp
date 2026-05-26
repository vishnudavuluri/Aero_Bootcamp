#include <iostream>
#include <vector>

int main() {
    // 1. Physics & Grid setup
    int numNodes = 20;              // Chop the rod into 20 discrete points
    double alpha = 0.01;            // Thermal diffusivity of the material
    double dt = 0.01;               // Time step duration
    double dx = 0.1;                // Distance between nodes

    // Mathmatical coffecient of our FTCS equation
    double r = (alpha * dt) / (dx * dx);

    // Intializing the vectors with zeros
    std::vector<double> u(numNodes, 0.0);
    std::vector<double> uNext(numNodes, 0.0);

    // 2. Boundary Conditions
    u[0] = 100;                     // Left end is hot
    u[numNodes - 1] = 0;            // Right end is cool

    // 3. Simulating 100 frames into the future
    int timeSteps = 100;

    for (int t = 0; t < timeSteps; t++) {

        // Spatial Loop: Calculating new temparature at every node
        // Avoiding the boundary temperatures
        for (int x = 1; x < numNodes - 1; x++) {
            uNext[x] = u[x] + r * (u[x + 1] - 2 * u[x] + u[x - 1]);
        }

        // Re-applying the boundary conditions to new time step
        uNext[0] = 100.0;
        uNext[numNodes - 1] = 0.0;

        // Switching the present to past
        u = uNext;
    }

    // 4. Outputting the final state
    for (int i = 0; i < numNodes; i++) {
        std::cout << "Node " << i << ": " << u[i] << " C\n";
    }

    return 0;
}
