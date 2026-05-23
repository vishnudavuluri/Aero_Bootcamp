#include <iostream>
#include <cmath>

int main() {
    double density;
    double velocity;
    double dynamic_pressure;

    std::cout << "-----Dynamic Pressure Calculator-----\n";
    std::cout << "Enter density(kg/m^3): \n";
    std::cin >> density;
    std::cout << "Ënter velocity (m/s): \n";
    std::cin >> velocity;

    dynamic_pressure = 0.5 * density * std::pow(velocity, 2);

    std::cout <<"The dynamic pressure is "<< dynamic_pressure << " Pascals";

    return 0;
}
