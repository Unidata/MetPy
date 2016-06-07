from metpy.mapping.points import interp_points
from metpy.cbook import get_test_data
#from mpl_toolkits.basemap import pyproj

import cartopy.crs as ccrs

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

from mpl_toolkits import basemap

def make_string_list(arr):
    return [s.decode('ascii') for s in arr]

def station_test_data(variable_name, proj_from=None, proj_to=None):

    f = get_test_data('station_data.txt')

    all_data = np.loadtxt(f, skiprows=1, delimiter=',',
                          usecols=(1, 2, 3, 4, 5, 6, 7, 17, 18, 19),
                          dtype=np.dtype([('stid', '3S'), ('lat', 'f'), ('lon', 'f'),
                                          ('slp', 'f'), ('air_temperature', 'f'),
                                          ('cloud_fraction', 'f'), ('dewpoint', 'f'),
                                          ('weather', '16S'),
                                          ('wind_dir', 'f'), ('wind_speed', 'f')]))

    all_stids = make_string_list(all_data['stid'])

    data = np.concatenate([all_data[all_stids.index(site)].reshape(1, ) for site in all_stids])

    value = data[variable_name]
    lon = data['lon']
    lat = data['lat']

    lon = lon[~np.isnan(value)]
    lat = lat[~np.isnan(value)]
    value = value[~np.isnan(value)]

    if proj_from != None and proj_to != None:

        try:

            proj_points = proj_to.transform_points(proj_from, lon, lat)
            return proj_points[:,0], proj_points[:,1], value

        except Exception as e:

            print(e)
            return None

    return lon, lat, value

def test_scipy_nearest():

    view = basemap.Basemap(width=4800000, height=3100000, projection='aea', resolution='l',
                           lat_1=28.5, lat_2=44.5, lat_0=38.5, lon_0=-97.,area_thresh=10000)

    lons, lats, z = create_test_data("air_temperature")

    xp, yp = view(lons, lats)

    xg, yg, img = interp_points(xp, yp, z, interp_type='nearest')

    img = np.ma.masked_where(np.isnan(img), img)

    view.pcolormesh(xg, yg, img)
    view.drawstates()
    view.drawcoastlines()

    return img

def test_scipy_linear():

    view = basemap.Basemap(width=4800000, height=3100000, projection='aea', resolution='l',
                           lat_1=28.5, lat_2=44.5, lat_0=38.5, lon_0=-97.,area_thresh=10000)

    lons, lats, z = create_test_data("air_temperature")

    xp, yp = view(lons, lats)

    xg, yg, img = interp_points(xp, yp, z, interp_type='linear')

    img = np.ma.masked_where(np.isnan(img), img)

    view.pcolormesh(xg, yg, img)
    view.drawstates()
    view.drawcoastlines()

    return img

def test_scipy_cubic():

    view = basemap.Basemap(width=4800000, height=3100000, projection='aea', resolution='l',
                           lat_1=28.5, lat_2=44.5, lat_0=38.5, lon_0=-97.,area_thresh=10000)

    lons, lats, z = create_test_data("air_temperature")

    xp, yp = view(lons, lats)

    xg, yg, img = interp_points(xp, yp, z, interp_type='cubic')

    img = np.ma.masked_where(np.isnan(img), img)

    view.pcolormesh(xg, yg, img)
    view.drawstates()
    view.drawcoastlines()

    return img

def test_natural_neighbor():

    view = basemap.Basemap(width=4800000, height=3100000, projection='aea', resolution='l',
                           lat_1=28.5, lat_2=44.5, lat_0=38.5, lon_0=-97.,area_thresh=10000)

    lons, lats, z = create_test_data("air_temperature")

    xp, yp = view(lons, lats)

    points = np.array(list(zip(xp, yp)))

    vor = Voronoi(points)

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

    center = points.mean(axis=0)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
            plt.plot([vor.vertices[i, 0], far_point[0]], [vor.vertices[i, 1], far_point[1]], 'k--')

    #xg, yg, img = interp_points(xp, yp, z, interp_type='cubic')

    #img = np.ma.masked_where(np.isnan(img), img)

    #view.pcolormesh(xg, yg, img)
    view.drawstates()
    view.drawcoastlines()

    #return img