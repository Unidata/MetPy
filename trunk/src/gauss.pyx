import numpy as np
cimport numpy as np
cimport cython

cdef extern from 'math.h':
    float expf(float x)
    double exp(double x)

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def gauss_filter(np.ndarray[DTYPE_t, ndim=1] x,
        np.ndarray[DTYPE_t, ndim=1] y,
        np.ndarray[DTYPE_t, ndim=2] var,
        DTYPE_t sigmax, DTYPE_t sigmay,
        DTYPE_t masked_value=-9999., DTYPE_t min_weight=0.0001):

    cdef unsigned int xloc, yloc, xi, yi
    cdef DTYPE_t xdenom, ydenom
    cdef DTYPE_t weighted_sum, totalw, weights

    cdef unsigned int numx = var.shape[0]
    cdef unsigned int numy = var.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] var_fil = np.zeros([numx, numy], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] xweight = np.empty([numx, numx], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] yweight = np.empty([numy, numy], dtype=DTYPE)

    xdenom = -1. / (2 * sigmax**2)
    ydenom = -1. / (2 * sigmay**2)

    for xloc in range(numx):
        for xi in range(numx):
            xweight[xloc, xi] = exp(xdenom * (x[xloc] - x[xi])**2)

    for yloc in range(numy):
        for yi in range(numy):
            yweight[yloc, yi] = exp(ydenom * (y[yloc] - y[yi])**2)

    for xloc in range(numx):
        for yloc in range(numy):
            weighted_sum = 0.
            totalw = 0.
            if var[xloc, yloc] != masked_value:
                for xi in range(numx):
                    for yi in range(numy):
                        if var[xi, yi] != masked_value:
                            weights = xweight[xloc, xi] * yweight[yloc, yi]
                            if weights >= min_weight:
                                weighted_sum += var[xi, yi] * weights
                                totalw += weights

            if totalw > 0.:
                var_fil[xloc, yloc] = weighted_sum / totalw
            else:
                var_fil[xloc, yloc] = masked_value

    return var_fil
