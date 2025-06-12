#include <cmath>
#include <stdexcept>
#include "math.hpp"

double lambert_wm1(double x, double tol, int max_iter) {
    if (x >= 0 || x < -1.0 / std::exp(1.0)) {
        throw std::domain_error("W₋₁(x) is only defined for -1/e < x < 0");
    }

    // Initial guess for W₋₁, from Corless et al. (1996) // need to check if this is
    // correct
    double L1 = std::log(-x);
    double L2 = std::log(-L1);
    double w = L1 - L2 + (L2 / L1);

    for (int i = 0; i < max_iter; ++i) {
        double ew = std::exp(w);
        double wew = w * ew;
        double diff = (wew - x) / (ew * (w + 1) - ((w + 2) * (wew - x)) / (2 * w + 2));
        w -= diff;
        if (std::abs(diff) < tol) {
            return w;
        }
    }

    throw std::runtime_error("lambert_wm1 did not converge");
}
