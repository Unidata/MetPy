import numpy as np

__all__ = ['gaussian_filter']

try:
    from _gauss_filt import gauss_filter as _gauss
    def gaussian_filter(x_grid, y_grid, var, sigmax, sigmay, min_weight=0.0001):
        # Reduce dimensional grids to 1D
        if x_grid.ndim > 1:
            x_grid = x_grid[:, 0]
        if y_grid.ndim > 1:
            y_grid = y_grid[0, :]

        #Fill masked arrays:
        try:
            masked_value = var.fill_value
            var = var.filled()
            masked = True
        except AttributeError:
            masked = False
            masked_value = -9999

        filt_var = _gauss(x_grid.astype(np.float), y_grid.astype(np.float),
            var.astype(np.float), sigmax, sigmay, masked_value, min_weight)

        if masked:
            filt_var = np.ma.array(filt_var, mask=(filt_var == masked_value))
            filt_var.fill_value = masked_value

        return filt_var

except ImportError:
    def gaussian_filter(x_grid, y_grid, var, sigmax, sigmay, min_weight=0.0001):
        var_fil = np.empty_like(var)
        # Reduce dimensional grids to 1D
        if x_grid.ndim > 1:
            x_grid = x_grid[:, 0]
        if y_grid.ndim > 1:
            y_grid = y_grid[0, :]

        xw = np.exp(-((x_grid[:, np.newaxis] - x_grid)**2 / (2 * sigmax**2)))
        yw = np.exp(-((y_grid[:, np.newaxis] - y_grid)**2 / (2 * sigmay**2)))

        for ind in np.ndindex(var.shape):
            totalw = np.outer(yw[ind[0]], xw[ind[1]])
            totalw = np.ma.array(totalw, mask=var.mask|(totalw < min_weight))
            var_fil[ind] = (var * totalw).sum() / totalw.sum()

        # Optionally create a masked array
        try:
            var_fil[var.mask] = np.ma.masked
        except AttributeError:
            pass

        return var_fil

gaussian_filter_doc="""
    Smooth a 2D array of data using a 2D Gaussian kernel function.  This will
    ignore missing values.

    x_grid : array
        Locations of grid points along the x axis

    y_grid : array
        Locations of grid points along the y axis

    var : array
        2D array of data to be smoothed.  Should be arranged in x by y order.

    sigmax : scalar
        Width of kernel in x dimension.  At x = sigmax, the kernel will have
        a value of e^-1.

    sigmay : scalar
        Width of kernel in y dimension.  At y = sigmay, the kernel will have
        a value of e^-1.

    min_weight : scalar
        Minimum weighting for points to be included in the smoothing.  If a
        point ends up with a weight less than this value, it will not be
        included in the final weighted sum.

    Returns : array
        2D (optionally masked) array of smoothed values.
"""
gaussian_filter.__doc__ = gaussian_filter_doc

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 40)
    y = np.linspace(-15, 15, 60)
    Y,X = np.meshgrid(y,x)
    noise = np.random.randn(*X.shape) * 10
    data = X**2 + Y**2 + noise
    data = np.ma.array(data, mask=((X**2 + Y**2) < 0.4))

    data_filt = gaussian_filter(x, y, data, 4, 4)

    plt.subplot(1, 2, 1)
    plt.imshow(data.T, interpolation='nearest',
        extent=(x.min(), x.max(), y.min(), y.max()))

    plt.subplot(1, 2, 2)
    plt.imshow(data_filt.T, interpolation='nearest',
        extent=(x.min(), x.max(), y.min(), y.max()))

    plt.show()
