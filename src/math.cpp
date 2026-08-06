#include <cmath>
#include <iostream>   // for std::cerr
#include "math.hpp"

double lambert_wm1(double x, double tol, int max_iter) {
    // Corless, R.M., Gonnet, G.H., Hare, D.E.G. et al. On the LambertW function.
    // Adv Comput Math 5, 329–359 (1996). https://doi.org/10.1007/BF02124750 

    if (x >= 0 || x < -1.0 / std::exp(1.0)) {
        std::cerr << "Warning in function '" << __func__
            << "': lambert_wm1 is only defined for -1/e < x < 0\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    double L1 = std::log(-x);
    double L2 = std::log(-L1);
    double w = L1 - L2 + (L2 / L1);

    for (int i = 0; i < max_iter; ++i) {
        double ew = std::exp(w);
        double w_ew = w * ew;
        double diff = (w_ew - x) / (ew * (w + 1) - ((w + 2) * (w_ew - x)) / (2 * w + 2));
        w -= diff;
        if (std::abs(diff) < tol) {
            return w;
        }
    }

    std::cerr << "Warning in function '" << __func__
        << "': lambert_wm1 did not converge.\n";
    return std::numeric_limits<double>::quiet_NaN();
}
