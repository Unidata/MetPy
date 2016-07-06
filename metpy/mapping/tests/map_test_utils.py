from metpy.cbook import get_test_data
import cartopy.crs as ccrs
from metpy.mapping import points
from metpy.mapping import interpolation
import numpy as np


def make_string_list(arr):
    return [s.decode('ascii') for s in arr]


def station_test_data(variable_names, proj_from=None, proj_to=None):

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

    value = data[variable_names]
    lon = data['lon']
    lat = data['lat']

    # lon = lon[~np.isnan(value)]
    # lat = lat[~np.isnan(value)]
    # value = value[~np.isnan(value)]

    if proj_from is not None and proj_to is not None:

        try:

            proj_points = proj_to.transform_points(proj_from, lon, lat)
            return proj_points[:, 0], proj_points[:, 1], value

        except Exception as e:

            print(e)
            return None

    return lon, lat, value


def state_capitol_wx_stations():

    return {'Washington': 'KOLM', 'Oregon': 'KSLE', 'California': 'KSAC',
            'Nevada': 'KCXP', 'Idaho': 'KBOI', 'Montana': 'KHLN',
            'Utah': 'KSLC', 'Arizona': 'KDVT', 'New Mexico': 'KSAF',
            'Colorado': 'KBKF', 'Wyoming': 'KCYS', 'North Dakota': 'KBIS',
            'South Dakota': 'KPIR', 'Nebraska': 'KLNK', 'Kansas': 'KTOP',
            'Oklahoma': 'KPWA', 'Texas': 'KATT', 'Louisiana': 'KBTR',
            'Arkansas': 'KLIT', 'Missouri': 'KJEF', 'Iowa': 'KDSM',
            'Minnesota': 'KSTP', 'Wisconsin': 'KMSN', 'Illinois': 'KSPI',
            'Mississippi': 'KHKS', 'Alabama': 'KMGM', 'Nashville': 'KBNA',
            'Kentucky': 'KFFT', 'Indiana': 'KIND', 'Michigan': 'KLAN',
            'Ohio': 'KCMH', 'Georgia': 'KFTY', 'Florida': 'KTLH',
            'South Carolina': 'KCUB', 'North Carolina': 'KRDU',
            'Virginia': 'KRIC', 'West Virginia': 'KCRW',
            'Pennsylvania': 'KCXY', 'New York': 'KALB', 'Vermont': 'KMPV',
            'New Hampshire': 'KCON', 'Maine': 'KAUG', 'Massachusetts': 'KBOS',
            'Rhode Island': 'KPVD', 'Connecticut': 'KHFD', 'New Jersey': 'KTTN',
            'Delaware': 'KDOV'}


def run_test():

    from_proj = ccrs.Geodetic()
    to_proj = ccrs.AlbersEqualArea(central_longitude=-97.0000, central_latitude=38.0000)

    x, y, temp = station_test_data("air_temperature", from_proj, to_proj)

    x = x[~np.isnan(temp)]
    y = y[~np.isnan(temp)]
    temp = temp[~np.isnan(temp)]

    xres = 100000
    yres = 100000

    x += xres
    y -= yres

    grid_x, grid_y = points.generate_grid(xres, yres, points.get_boundary_coords(x, y))

    grids = points.generate_grid_coords(grid_x, grid_y)
    img = interpolation.natural_neighbor(x, y, temp, grids)

    img = img.reshape(grid_x.shape)

    return img
