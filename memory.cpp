#include <iostream>
#include <vector>

void accelarteFlow(std::vector<double>& velocities) {
    for (double& v : velocities) {
        v *= 2;
    }
}

int main() {
    std::vector<double> velocities = {15.5, 22.0, 45.3, 10.1};

    std::cout << "Original velocities\n";
    for (double v : velocities) {
        std::cout << v << " ";
    }
    std::cout << '\n';

    accelarteFlow(velocities);

    std:: cout << "Changed velocities\n";
    for (double v : velocities) {
        std::cout << v << " ";
    }

    return 0;
}