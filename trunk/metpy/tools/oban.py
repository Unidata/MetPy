import numpy as np
from metpy.cbook import iterable

__all__ = ['gaussian_filter', 'grid_data', 'barnes_weights', 'cressman_weights']

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

def grid_point_dists(grid_x, grid_y, ob_x, ob_y):
    "Calculates distances for each grid point to every ob point."
    return np.hypot(grid_x[..., np.newaxis] - ob_x[np.newaxis, np.newaxis],
        grid_y[..., np.newaxis] - ob_y[np.newaxis, np.newaxis])

def cressman_weights(dists, radius):
    "Calculates weights for Cressman interpolation."
    dist_sq = dists**2
    rad_sq = radius**2
    weights = (rad_sq - dist_sq) / (rad_sq + dist_sq)
    weights[dists > radius] = 0.0
    return weights

def barnes_weights(dists, data_spacing, kappa_star=0.5, gamma=1.0, min_weight=0.01):
    """Calculates weights for Barnes interpolation.

    The Barnes weighting function is a simple inverse exponential.

    Trapp and Doswell (2000, J. Atmos. Ocean. Tech., hereafter TD2000)
    and Koch et al. (1983, J. Clim. and Appl. Met.), hereafter K83 describe
    the theory behind the following prescription for choosing weights.

    Adopting the nomenclature of TD2000,
    w=exp(- r^2 / k)
    k= (k*) (2*DLT)^2

    DLT is the maximum data spacing, and k* is the nondimensional response
    parameter, specified nondimensionally in terms of the Nyquist.

    For subsequent passes, given initial k_0 as calculated above,
    k_1 = gamma * k_0
    (k_0 and k_1 can be either the dimensional or nondimensional forms,
    since k and k* are related by simple proportionality)

    TD2000 recommend k*=0.5, which attenuates about 30 pct at twice the
    spatial nyquist, though it almost comlpetely removes all variability
    below the nyquist. k*=0.1 is another intersting choice, as it achives
    a similar filter response in one pass as does a two-pass with k*=0.5
    and gamma=0.2

    # below is a demonstration of the above discussion of nondimensional
    # filter parameter tuning.

    import numpy as N

    def response_firstpass(lstar, kstar):
        # return first pass response given nondimensional wavelengths lstar
        # and nondimensional tuning parameter kstar

        R = N.exp(-kstar*((N.pi / lstar)**2.0))
        return R

    def response_secondpass(lstar, gamma, kstar):
        # calculate the second pass response given nondimensional
        # wavelengths lstar, nondimensional tuning parameter kstar
        # and smoothing adjustment parameter gamma
        R0 = response_firstpass(lstar, kstar)
        R =  R0 * (1 + R0**(gamma-1) - R0**gamma)
        return R

    #non-dim wavelengths
    l = N.arange(0, 10, 0.1)

    # Test single pass
    k = 0.1
    R_test1 = response_firstpass(l, k)

    # Test two pass
    k = 0.5
    g = 0.3
    R_test2 = response_secondpass(l, g, k)

    # Test two pass
    k = 0.5
    g = 0.2
    R_test3 = response_secondpass(l, g, k)

    from pylab import *
    figure()
    plot(l, R_test1)
    plot(l, R_test2)
    plot(l, R_test3)
    plot((1,1), (0,1), 'y')
    plot((0,10), (0.9,0.9), 'y')
    title('Nondimensinoal filter response for Barnes')
    legend(('one pass, k=0.1',
            'two pass, g=0.3, k=0.5',
            'two pass, g=0.2, k=0.5'), loc='lower right')
    show()

    """

    kappa = kappa_star * (2.0*data_spacing)**2.0
    weights = np.exp(-dists**2.0 / (kappa*gamma))
    # build in a hard cutoff on the weights
    weights[weights < min_weight] = 0.0

    # OLD
    # weights = np.exp(-dists**2 / (kappa0 * gamma))
    # critical_radius = np.sqrt(15 * kappa0)
    # weights[dists > critical_radius] = 0.0

    return weights


#def bilinear(x, y, data, xloc, yloc):
#    xind = find_axis_index(x, xloc)
#    yind = find_axis_index(y, yloc)
#    xw = (xloc - x[xind]) / (x[xind + 1] - x[xind])
#    x_weights = np.array([1 - xw, xw])
#    yw = (yloc - y[yind]) / (y[yind + 1] - y[yind])
#    y_weights = np.array([1 - yw, yw])
#    return np.dot(y_weights, np.dot(data[yind:yind + 2, xind:xind + 2],
#        x_weights))

#def find_axis_index(axis_vals, location):
#    if location > axis_vals[-1] or location < axis_vals[0]:
#        raise ValueError, "Location out of bounds"
#    return axis_vals.searchsorted(location)

def grid_data(ob_data, grid_x, grid_y, ob_x, ob_y, weight_func, params):
    '''
    Calculates a value at each grid point based on the observed data in
    ob_data.

    ob_data : 1D array
        The observation data

    grid_x : 2D array
        The x locations of the grid points

    grid_y : 2D array
        The y locations of the grid points

    ob_x : 1D array
        The x locations of the observation points

    ob_y : 1D array
        The y locations of the observation points

    weight_func : callable
        Any function that returns weights for the observations given their
        distance from a grid point.

    params : any object or tuple of objects
        Appropriate parameters to pass to *weight_func* after the distances.

    Returns  : 2D array
        The values for the grid points
    '''
    if not iterable(params):
        params = (params,)
    # grid_point_dists calculates a 3D array containing the distance for each
    # grid point to every observation.
    weights = weight_func(grid_point_dists(grid_x, grid_y, ob_x, ob_y), *params)
    total_weights = weights.sum(axis=2)
    final = (weights * ob_data).sum(axis=2) / total_weights
    final = np.ma.masked_array(final, mask=(total_weights==0.))
    return final

def analyze_grid_multipass(ob_data, grid_x, grid_y, ob_x, ob_y, num_passes,
    weight_func, params, background=None):
    '''
    Calculate a value at each grid point using multiple passes of an objective
    analysis technique
    '''
    if background is None:
        mod_param = (params[0], 1.0)
        background = analyze_grid(ob_data, grid_x, grid_y, ob_x, ob_y,
            weight_func, mod_param)
        num_passes -= 1
    for i in range(num_passes):
        ob_incs = get_ob_incs(ob_x, ob_y, ob_data, grid_x[0], grid_y[:,0],
            background)
        background += analyze_grid(ob_incs, grid_x, grid_y, ob_x, ob_y,
            weight_func, params)
    return background

from itertools import izip
def get_ob_incs(obx, oby, ob, grid_x, grid_y, field, cressman_radius = None):
  ob_inc = list()
  mask = np.zeros(ob.size)
  for x,y,value in izip(obx,oby,ob):
    try:
      interp_val = bilinear(grid_x, grid_y, field, x, y)
      ob_inc.append(value - interp_val)
    except ValueError:
      if cressman_radius is None:
        mask[len(ob_inc)] = 1
        ob_inc.append(0.0)
      else:
      #Ugly hack here to allow the one station off the grid to be interpolated
        xg,yg = np.meshgrid(grid_x, grid_y)
        interp_val = analyze_grid(field.flatten(), np.array(x, ndmin=2),
          np.array(y, ndmin=2), xg.flatten(), yg.flatten(),
          cressman_weights, cressman_radius)
        if np.isnan(interp_val):
          interp_val = value
#        mask[len(ob_inc) - 1] = 1
        ob_inc.append(value - interp_val.flatten()[0])
  return np.ma.array(ob_inc, mask=mask)

if __name__ == '__main__':
    from StringIO import StringIO
    from mpl_toolkits.basemap import Basemap, maskoceans
    import matplotlib.pyplot as plt
    from scipy.constants import kilo
    import metpy

    # name, weight function, and parameters for each type of objective analysis
    obans = {'barnes':(barnes_weights, (500.*kilo, 0.5)),
             'cressman':(cressman_weights, (600.*kilo))
             }

    # name of the oban type to test (key from obans dict)
    which_oban = 'cressman'

    # station, lat, lon, height, wdir, wspd
    data = '''72214,30.4,-84.3,5740.0,250.0,34.9
72274,32.2,-111.0,5730.0,45.0,13.3
72293,32.9,-117.1,5750.0,95.0,19.0
72649,44.8,-93.6,5210.0,315.0,25.7
78526,18.4,-66.0,5830.0,295.0,5.1
72201,24.6,-81.8,5830.0,275.0,7.7
74004,32.5,-114.0,5770.0,65.0,18.0
70219,60.8,-161.8,5100.0,240.0,11.3
70026,71.3,-156.8,5090.0,30.0,7.2
70316,55.2,-162.7,5140.0,270.0,3.6
70350,57.8,-152.5,5100.0,190.0,5.1
70261,64.8,-147.9,5130.0,230.0,17.4
70326,58.7,-156.6,5110.0,20.0,1.5
70231,63.0,-155.6,5090.0,215.0,14.4
70273,61.2,-150.0,5130.0,200.0,13.3
70398,55.0,-131.6,5550.0,250.0,34.4
70200,64.5,-165.4,5040.0,25.0,10.8
70133,66.9,-162.6,5050.0,35.0,17.4
70308,57.2,-170.2,5140.0,340.0,10.2
70361,59.5,-139.7,5300.0,230.0,38.5
72230,33.2,-86.8,5630.0,250.0,40.1
72340,34.8,-92.2,5530.0,260.0,40.1
72376,35.2,-111.8,5750.0,30.0,14.9
72493,37.7,-122.2,5810.0,355.0,9.7
72469,39.8,-104.9,5580.0,335.0,32.4
72476,39.1,-108.5,5680.0,360.0,25.7
72206,30.5,-81.7,5760.0,245.0,34.9
72202,25.8,-80.3,5840.0,260.0,9.2
72210,27.7,-82.4,5820.0,240.0,14.4
72215,33.4,-84.6,5640.0,250.0,39.0
74455,41.6,-90.6,5280.0,275.0,18.5
72681,43.6,-116.2,5750.0,345.0,20.0
74560,40.2,-89.3,5330.0,250.0,24.6
72451,37.8,-100.0,5500.0,330.0,25.7
72456,39.1,-95.6,5400.0,310.0,25.7
72240,30.1,-93.2,5700.0,255.0,38.0
72233,30.3,-89.8,5710.0,245.0,39.6
72248,32.4,-93.8,5620.0,255.0,37.5
74494,41.7,-70.0,5460.0,250.0,47.3
74389,43.9,-70.2,5340.0,245.0,34.9
72712,46.9,-68.0,5280.0,255.0,18.5
72634,44.9,-84.7,5230.0,250.0,18.5
72632,42.7,-83.5,5280.0,275.0,19.5
72747,48.6,-93.4,5170.0,320.0,12.8
72440,37.2,-93.4,5440.0,275.0,26.2
72235,32.3,-90.1,5640.0,255.0,38.5
72776,47.5,-111.4,5610.0,325.0,32.4
72768,48.2,-106.6,5460.0,335.0,46.2
72317,36.1,-79.9,5580.0,245.0,41.6
72764,46.8,-100.8,5350.0,330.0,30.3
72562,41.1,-100.7,5460.0,340.0,33.9
72558,41.3,-96.4,5370.0,320.0,26.2
72365,35.0,-106.6,5690.0,5.0,26.7
72582,40.9,-115.7,5770.0,345.0,13.3
72387,36.6,-116.0,5780.0,25.0,12.3
72501,40.9,-72.9,5440.0,250.0,52.4
72518,42.7,-73.8,5340.0,245.0,33.4
72528,42.9,-78.7,5280.0,265.0,25.7
72426,39.4,-83.8,5370.0,260.0,31.3
72357,35.2,-97.4,5520.0,280.0,27.7
72597,42.4,-122.9,5810.0,350.0,8.2
72694,44.9,-123.0,5810.0,320.0,12.8
72520,40.5,-80.3,5340.0,255.0,37.0
72208,32.9,-80.0,5700.0,255.0,40.6
72659,45.4,-98.4,5310.0,330.0,34.4
72662,44.0,-103.1,5460.0,330.0,39.0
72327,36.1,-86.7,5500.0,260.0,44.7
72363,35.2,-101.7,5580.0,320.0,21.0
72250,25.9,-97.4,5800.0,275.0,23.1
72251,27.8,-97.5,5770.0,265.0,27.7
72261,29.4,-100.9,5730.0,265.0,29.8
72249,32.8,-97.3,5610.0,270.0,39.6
72265,31.9,-102.2,5670.0,275.0,32.4
72572,40.8,-112.0,5720.0,355.0,21.6
72318,37.2,-80.4,5520.0,250.0,44.7
72403,38.9,-77.4,5470.0,250.0,48.8
72402,37.9,-75.5,5560.0,245.0,44.2
72786,47.7,-117.6,5710.0,325.0,27.2
72797,47.9,-124.6,5760.0,295.0,15.9
72645,44.5,-88.1,5220.0,255.0,16.9
72672,43.1,-108.4,5630.0,345.0,33.4
71119,53.5,-114.1,5480.0,315.0,47.8
71945,58.9,-125.8,5370.0,285.0,46.8
71203,50.0,-119.4,5690.0,315.0,28.8
71109,50.7,-127.4,5720.0,280.0,26.7
71867,54.0,-101.1,5220.0,325.0,28.2
71913,58.7,-94.1,5070.0,265.0,6.1
71815,48.5,-58.5,5210.0,290.0,24.6
71816,53.3,-60.4,5200.0,75.0,3.6
71600,43.9,-60.0,5450.0,265.0,49.8
71603,43.8,-66.1,5400.0,250.0,45.2
71917,80.0,-85.9,5110.0,195.0,14.4
71082,82.5,-62.3,5130.0,75.0,2.0
71926,64.3,-96.1,4990.0,240.0,5.6
71925,69.1,-105.1,4960.0,335.0,4.1
71957,68.3,-133.5,5030.0,285.0,13.3
71909,63.8,-68.6,5070.0,305.0,1.0
71934,60.0,-111.9,5190.0,305.0,34.4
71043,65.3,-126.8,5110.0,290.0,39.6
71915,64.2,-83.4,5050.0,180.0,4.6
71845,51.4,-90.2,5100.0,290.0,13.3
71823,53.8,-73.7,5150.0,210.0,12.3
71836,51.3,-80.6,5120.0,240.0,12.3
71906,58.1,-68.4,5140.0,255.0,17.4
71964,60.7,-135.1,5310.0,250.0,36.5
'''
    # Extracts data from IDV output and masks out stations outside of North America
    lat,lon,height,wdir,speed = np.loadtxt(StringIO(data), delimiter=',',
        unpack=True, usecols=(1,2,3,4,5))

    # Create a map for plotting
    bm = Basemap(projection='tmerc', lat_0=90.0, lon_0=-100.0, lat_ts=40.0,
        llcrnrlon=-121, llcrnrlat=24, urcrnrlon=-65, urcrnrlat=46,
        resolution='l')

    # Get U,V components from wind speed and direction
    u,v = metpy.get_wind_components(speed, wdir)

    # Rotate the vectors to be properly aligned in the map projection
    u,v = bm.rotate_vector(u, v, lon, lat)

    # Generate grid of x,y positions
    lon_grid, lat_grid, x_grid, y_grid = bm.makegrid(130, 60, returnxy=True)

    # Transform the obs to basemap space for gridding
    obx,oby = bm(lon, lat)

    # Perform analysis of height obs using Cressman weights
    heights_oban = grid_data(height, x_grid, y_grid, obx, oby,
        obans[which_oban][0], obans[which_oban][1])

    heights_oban = maskoceans(lon_grid, lat_grid, heights_oban)

    # Map plotting
    contours = np.arange(5000., 5800., 60.0)
    bm.drawstates()
    bm.drawcountries()
    bm.drawcoastlines()
    cp = bm.contour(x_grid, y_grid, heights_oban, contours)
    bm.barbs(obx, oby, u, v)
    plt.clabel(cp, fmt='%.0f', inline_spacing=0)
    plt.title('500mb Height map')
    plt.show()
