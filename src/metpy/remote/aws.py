# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading known collections of data that are hosted on Amazon Web Services (AWS).

"""
import bisect
from datetime import datetime, timedelta
from functools import cached_property
import itertools
from pathlib import Path
import shutil

import boto3
import botocore
from botocore.client import Config
import xarray as xr

from ..io import Level2File, Level3File
from ..package_tools import Exporter

exporter = Exporter(globals())


class Product:
    def __init__(self, obj, reader):
        self.path = obj.key
        self._obj = obj
        self._reader = reader

    @property
    def url(self):
        return f'https://{self._obj.Bucket().name}.s3.amazonaws.com/{self.path}'

    @property
    def name(self):
        return Path(self.path).name

    @cached_property
    def file(self):
        return self._obj.get()['Body']

    def download(self, path=None):
        if path is None:
            path = Path() / self.name
        elif (path := Path(path)).is_dir():
            path = path / self.name
        else:
            path = Path(path)

        with open(path, 'wb') as outfile:
            shutil.copyfileobj(self.file, outfile)

    def parse(self):
        return self._reader(self)


def date_iterator(start, end, **step_kw):
    while start < end:
        yield start
        start = start + timedelta(**step_kw)


class S3DataStore:
    s3 = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED,
                                            user_agent_extra='Resource'))

    def __init__(self, bucket_name, delimiter):
        self.bucket_name = bucket_name
        self.delimiter = delimiter
        self._bucket = self.s3.Bucket(bucket_name)

    def common_prefixes(self, prefix, delim=None):
        delim = delim or self.delimiter
        try:
            return (p['Prefix'] for p in
                    self._bucket.meta.client.list_objects_v2(
                        Bucket=self.bucket_name, Prefix=prefix,
                        Delimiter=delim)['CommonPrefixes'])
        except KeyError:
            return []

    def objects(self, prefix):
        return self._bucket.objects.filter(Prefix=prefix)

    def _build_result(self, obj):
        return Product(obj, lambda s: None)


@exporter.export
class NEXRADLevel3Archive(S3DataStore):
    def __init__(self):
        super().__init__('unidata-nexrad-level3', '_')

    def sites(self):
        """Return sites available."""
        return (item.rstrip(self.delimiter) for item in self.common_prefixes(''))

    def product_ids(self, site='TLX'):
        """Return product_ids available.

        Takes a site, defaults to TLX.
        """
        return (item.split(self.delimiter)[-2] for item in
                self.common_prefixes(f'{site}{self.delimiter}'))

    def build_key(self, site, prod_id, dt, depth=None):
        parts = [site, prod_id, f'{dt:%Y}', f'{dt:%m}', f'{dt:%d}', f'{dt:%H}', f'{dt:%M}',
                 f'{dt:%S}']
        return self.delimiter.join(parts[slice(0, depth)])

    def dt_from_key(self, key):
        return datetime.strptime(key.split(self.delimiter, maxsplit=2)[-1],
                                 '%Y_%m_%d_%H_%M_%S')

    def get_range(self, site, prod_id, start, end):
        for dt in date_iterator(start, end, days=1):
            for obj in self.objects(self.build_key(site, prod_id, dt, depth=5)):
                if start <= self.dt_from_key(obj.key) < end:
                    yield self._build_result(obj)

    def get_product(self, site, prod_id, dt):
        search_key = self.build_key(site, prod_id, dt)
        bounding_keys = [self.build_key(site, prod_id, dt, 2) + self.delimiter]
        for depth in range(3, 8):
            prefixes = list(itertools.chain(*(self.common_prefixes(b) for b in bounding_keys)))
            loc = bisect.bisect_left(prefixes, search_key)
            rng = slice(loc - 1, loc + 1) if loc else slice(0, 1)
            bounding_keys = prefixes[rng]

        min_obj = min(itertools.chain(*(self.objects(p) for p in bounding_keys)),
                      key=lambda o: abs((self.dt_from_key(o.key) - dt).total_seconds()))

        return self._build_result(min_obj)

    def _build_result(self, obj):
        return Product(obj, lambda s: Level3File(s.file))


@exporter.export
class NEXRADLevel2Archive(S3DataStore):
    def __init__(self):
        super().__init__('noaa-nexrad-level2', '/')

    def sites(self, dt=None):
        """Return sites available for a date."""
        if dt is None:
            dt = datetime.utcnow()
        prefix = self.build_key('', dt, depth=3) + self.delimiter
        return (item.split('/')[-2] for item in self.common_prefixes(prefix))

    def build_key(self, site, dt, depth=None):
        parts = [f'{dt:%Y}', f'{dt:%m}', f'{dt:%d}', site, f'{site}{dt:%Y%m%d_%H%M%S}']
        return self.delimiter.join(parts[slice(0, depth)])

    def dt_from_key(self, key):
        return datetime.strptime(key.rsplit(self.delimiter, maxsplit=1)[-1][4:19],
                                 '%Y%m%d_%H%M%S')

    def get_range(self, site, start, end):
        for dt in date_iterator(start, end, days=1):
            for obj in self.objects(self.build_key(site, dt, depth=4)):
                try:
                    if start <= self.dt_from_key(obj.key) < end:
                        yield self._build_result(obj)
                except ValueError:
                    continue

    def get_product(self, site, dt):
        search_key = self.build_key(site, dt)
        prefix = search_key.split('_')[0]
        min_obj = min(self.objects(prefix),
                      key=lambda o: abs((self.dt_from_key(o.key) - dt).total_seconds()))

        return self._build_result(min_obj)

    def _build_result(self, obj):
        return Product(obj, lambda s: Level2File(s.file))


@exporter.export
class GOES16Archive(S3DataStore):
    def __init__(self, bucket_name='noaa-goes16'):
        super().__init__(bucket_name, delimiter='/')

    def product_ids(self):
        """Return product_ids available."""
        return (item.rstrip(self.delimiter) for item in self.common_prefixes(''))

    def build_key(self, product, dt, depth=None):
        parts = [product, f'{dt:%Y}', f'{dt:%j}', f'{dt:%H}', f'OR_{product}']
        return self.delimiter.join(parts[slice(0, depth)])

    def _subprod_prefix(self, prefix, mode, channel):
        subprods = set(item.rstrip('_').rsplit('-', maxsplit=1)[-1] for item in
                       self.common_prefixes(prefix + '-', '_'))
        if len(subprods) > 1:
            if modes := set(item[1] for item in subprods):
                if len(modes) == 1:
                    mode = next(iter(modes))
                if str(mode) in modes:
                    prefix += f'-M{mode}'
                else:
                    raise ValueError(
                        f'Need to specify a valid operating mode. Available options are '
                        f'{", ".join(sorted(modes))}')
            if channels := set(item[-2:] for item in subprods):
                if len(channels) == 1:
                    channel = next(iter(channels))
                if str(channel) in channels:
                    prefix += f'C{channel}'
                elif isinstance(channel, int) and f'{channel:02d}' in channels:
                    prefix += f'C{channel:02d}'
                else:
                    raise ValueError(
                        f'Need to specify a valid channel. Available options are '
                        f'{", ".join(sorted(channels))}')
        return prefix

    def dt_from_key(self, key):
        start_time = key.split('_')[-3]
        return datetime.strptime(start_time[:-1], 's%Y%j%H%M%S')

    def get_product(self, product, dt, mode=None, channel=None):
        prefix = self.build_key(product, dt)
        prefix = self._subprod_prefix(prefix, mode, channel)
        min_obj = min(self.objects(prefix),
                      key=lambda o: abs((self.dt_from_key(o.key) - dt).total_seconds()))

        return self._build_result(min_obj)

    def _build_result(self, obj):
        return Product(obj, lambda s: xr.open_dataset(s.url + '#mode=bytes', engine='netcdf4'))


@exporter.export
class GOES17Archive(GOES16Archive):
    def __init__(self):
        super().__init__('noaa-goes17')


@exporter.export
class GOES18Archive(GOES16Archive):
    def __init__(self):
        super().__init__('noaa-goes18')
