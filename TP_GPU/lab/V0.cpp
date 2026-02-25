#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>

void step(float* r, const float* d, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = std::numeric_limits<float>::infinity();
            for (int k = 0; k < n; ++k) {
                float x = d[n*i + k];
                float y = d[n*k + j];
                float z = x + y;
                v = std::min(v, z);
            }
            r[n*i + j] = v;
        }
    }
}

int main() {
    int n = 3;
    int seed = 0;

    std::vector<float> d(n*n);
    d = {
        0, 8, 2,
        1, 0, 9,
        4, 5, 0,
    };

    std::vector<float> r(n*n);

    step(r.data(), d.data(), n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << d[i*n + j] << " ";
        }
        std::cout << "\n";
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << r[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}