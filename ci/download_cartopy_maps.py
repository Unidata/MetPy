#!/usr/bin/env python
# Copyright (c) 2021 MetPy Developers.
"""Explicitly download needed Cartopy maps."""
from cartopy.io import config, Downloader

AWS_TEMPLATE = ('https://naturalearth.s3.amazonaws.com/{resolution}_'
                '{category}/ne_{resolution}_{name}.zip')


def grab_ne(category, feature, res):
    """Download the correct Natural Earth feature using Cartopy."""
    download = Downloader.from_config(('shapefiles', 'natural_earth'))
    download.path({'category': category, 'name': feature, 'resolution': res, 'config': config})


if __name__ == '__main__':
    # Need to override the pre-Cartopy 0.20 URL to use S3
    config['downloaders'][('shapefiles', 'natural_earth')].url_template = AWS_TEMPLATE

    for feat in ['admin_0_boundary_lines_land', 'admin_1_states_provinces_lakes']:
        for r in ['110m', '50m', '10m']:
            grab_ne('cultural', feat, r)

    for feat, r in [('coastline', '10m'), ('coastline', '50m'), ('coastline', '110m'),
                    ('lakes', '10m'), ('lakes', '50m'),
                    ('land', '10m'), ('land', '50m'), ('land', '110m'),
                    ('ocean', '110m'), ('ocean', '50m'),
                    ('rivers_lake_centerlines', '10m'), ('rivers_lake_centerlines', '110m')]:
        grab_ne('physical', feat, r)
