# Copyright (c) 2023 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for reading known collections of data that are hosted on Amazon Web Services (AWS)."""
import bisect
from datetime import datetime, timedelta, timezone
import itertools
from pathlib import Path
import shutil

import xarray as xr

from ..io import Level2File, Level3File
from ..package_tools import Exporter

exporter = Exporter(globals())


def ensure_timezone(dt):
    """Add UTC timezone if no timezone present."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


class AWSProduct:
    """Represent a product stored in an AWS S3 bucket."""

    def __init__(self, obj, reader):
        self.path = obj.key
        self._obj = obj
        self._reader = reader

    @property
    def url(self):
        """Provide the URL for directly accessing the product."""
        return f'https://{self._obj.Bucket().name}.s3.amazonaws.com/{self.path}'

    @property
    def name(self):
        """Provide the name for the product."""
        return Path(self.path).name

    @property
    def file(self):
        """Provide a file-like object for reading data from the product."""
        return self._obj.get()['Body']

    def download(self, path=None):
        """Download the complete product to a local file.

        Parameters
        ----------
        path : str, optional
            Location to save the product. If a directory, the product will be saved in that
            directory, using the product name as the filename. Otherwise, this should be a
            full path. Defaults to saving in the current directory with the product name.

        """
        if path is None:
            path = Path() / self.name
        elif (path := Path(path)).is_dir():
            path = path / self.name
        else:
            path = Path(path)

        with open(path, 'wb') as outfile:
            shutil.copyfileobj(self.file, outfile)

    def access(self):
        """Access the product and return a usable Python object.

        This is configured using the ``reader`` parameter, which can be used to open the
        product using e.g. `xarray.open_dataset` or `Level2File`.

        Returns
        -------
        The object created by parsing the product.

        """
        return self._reader(self)


def date_iterator(start, end, **step_kw):
    """Yield dates from ``start`` to ``end``, stepping as specified by ``step_kw``.

    Parameters
    ----------
    start : `datetime.datetime`
        Start date/time for iteration.
    end : datetime.datetime
        End date/time for iteration.
    step_kw :
        Keyword arguments to pass to `datetime.timedelta` to control step size.

    """
    while start < end:
        yield start
        start = start + timedelta(**step_kw)


class S3DataStore:
    """Facilitate access to a data store on AWS S3."""

    def __init__(self, bucket_name, delimiter='/'):
        """Initialize the data store.

        Parameters
        ----------
        bucket_name : str
            The name of the bucket
        delimiter : str, optional
            The delimiter used to split the key into distinct portions. Defaults to '/'.

        """
        import boto3
        import botocore  # noqa: I900
        from botocore.client import Config  # noqa: I900

        self.s3 = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED,
                                                     user_agent_extra='Resource'))
        self.bucket_name = bucket_name
        self.delimiter = delimiter
        self._bucket = self.s3.Bucket(bucket_name)

    def dt_from_key(self, key):
        """Parse date from key.

        Parameters
        ----------
        key : str
            The key to parse

        Returns
        -------
        datetime
            The parsed date

        """
        raise NotImplementedError()

    def common_prefixes(self, prefix, delim=None):
        """Return the common prefixes under a given prefix.

        Parameters
        ----------
        prefix : str
            The starting prefix to look under for common prefixes.
        delim : str, optional
            The delimiter used to split the key into distinct portions. If not specified,
            defaults to the one initially set on the client.

        """
        delim = delim or self.delimiter
        try:
            return (p['Prefix'] for p in
                    self._bucket.meta.client.list_objects_v2(
                        Bucket=self.bucket_name, Prefix=prefix,
                        Delimiter=delim)['CommonPrefixes'])
        except KeyError:
            return []

    def objects(self, prefix):
        """Return objects matching the given prefix.

        Parameters
        ----------
        prefix : str
            The prefix to match against.

        Returns
        -------
        Iterator of `botocore.client.Object`
            Objects matching the given prefix.

        """
        return self._bucket.objects.filter(Prefix=prefix)

    def _closest_result(self, it, dt):
        """Iterate over a sequence and return a result built from the closest match."""
        try:
            min_obj = min(it,
                          key=lambda o: abs((self.dt_from_key(o.key) - dt).total_seconds()))
        except ValueError as e:
            raise ValueError(f'No result found for {dt}') from e
        return self._build_result(min_obj)

    def _build_result(self, obj):
        """Build a basic product with no reader."""
        return AWSProduct(obj, lambda s: None)


@exporter.export
class NEXRADLevel3Archive(S3DataStore):
    """Access data from the NEXRAD Level 3 archive in AWS.

    These data consist of processed data from NWS NEXRAD, including:
    * Single elevation moment data
    * Estimated precipitation data
    * Feature detection (e.g. tornadoes, mesocyclones, hail)

    """

    def __init__(self):
        super().__init__('unidata-nexrad-level3', '_')

    def sites(self):
        """Return sites available.

        Returns
        -------
        List[str]
            Sites

        """
        return [item.rstrip(self.delimiter) for item in self.common_prefixes('')]

    def product_ids(self, site='TLX'):
        """Return product ids available.

        Parameters
        ----------
        site : str, optional
            Site to examine for product ids. Defaults to 'TLX'.

        Returns
        -------
        List[str]
            Product ids

        """
        return [item.split(self.delimiter)[-2] for item in
                self.common_prefixes(f'{site}{self.delimiter}')]

    def _build_key(self, site, prod_id, dt, depth=None):
        """Build a key up to a particular depth (number of sub parts)."""
        parts = [site, prod_id, f'{dt:%Y}', f'{dt:%m}', f'{dt:%d}', f'{dt:%H}', f'{dt:%M}',
                 f'{dt:%S}']
        return self.delimiter.join(parts[slice(0, depth)])

    def dt_from_key(self, key):  # noqa: D102
        # Docstring inherited
        return datetime.strptime(key.split(self.delimiter, maxsplit=2)[-1],
                                 '%Y_%m_%d_%H_%M_%S').replace(tzinfo=timezone.utc)

    def get_range(self, site, prod_id, start, end):
        """Yield products within a particular date/time range.

        Parameters
        ----------
        site : str
            The site to search for data
        prod_id : str
            The product ID to search for data
        start : `datetime.datetime`
            The start of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        end : `datetime.datetime`
            The end of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.

        See Also
        --------
        product_ids, sites, get_product

        """
        start = ensure_timezone(start)
        end = ensure_timezone(end)

        # We work with a list of keys/prefixes that we iteratively find that bound our target
        # key. To start, this only contains the site and product.
        bounding_keys = [self._build_key(site, prod_id, start, 2) + self.delimiter,
                         self._build_key(site, prod_id, end, 2) + self.delimiter]

        # Iteratively search with more specific keys, finding where our key fits within the
        # list by using the common prefixes that exist for the current bounding keys
        for depth in range(3, 8):
            # Get a key for the site/prod/dt that we're looking for, constrained by how deep
            # we are in the search i.e. site->prod->year->month->day->hour->minute->second
            search_start = self._build_key(site, prod_id, start, depth)
            search_end = self._build_key(site, prod_id, end, depth)

            # Get the next collection of partial keys using the common prefixes for our
            # candidates
            prefixes = list(itertools.chain(*(self.common_prefixes(b) for b in bounding_keys)))

            # Find where our target would be in the list and grab the ones on either side
            # if possible. This also handles if we're off the end.
            loc_start = bisect.bisect_left(prefixes, search_start)
            loc_end = bisect.bisect_left(prefixes, search_end)

            # loc gives where our target *would* be in the list. Therefore slicing from loc - 1
            # to loc + 1 gives the items to the left and right of our target. If get get 0,
            # then there is nothing to the left and we only need the first item.
            left = loc_start - 1 if loc_start else 0
            rng = slice(left, loc_end + 1)
            bounding_keys = prefixes[rng]

        for obj in itertools.chain(*(self.objects(p) for p in bounding_keys)):
            if start <= self.dt_from_key(obj.key) < end:
                yield self._build_result(obj)

    def get_product(self, site, prod_id, dt=None):
        """Get a product from the archive.

        Parameters
        ----------
        site : str
            The site to search for data
        prod_id : str
            The product ID to search for data
        dt : `datetime.datetime`, optional
            The desired date/time for the model run; the one closest matching in time will
            be returned. This should have the proper timezone included; if not specified, UTC
            will be assumed. If ``None``, defaults to the current UTC date/time.

        See Also
        --------
        product_ids, sites, get_range

        """
        dt = datetime.now(timezone.utc) if dt is None else ensure_timezone(dt)

        # We work with a list of keys/prefixes that we iteratively find that bound our target
        # key. To start, this only contains the site and product.
        bounding_keys = [self._build_key(site, prod_id, dt, 2) + self.delimiter]

        # Iteratively search with more specific keys, finding where our key fits within the
        # list by using the common prefixes that exist for the current bounding keys
        for depth in range(3, 8):
            # Get a key for the site/prod/dt that we're looking for, constrained by how deep
            # we are in the search i.e. site->prod->year->month->day->hour->minute->second
            search_key = self._build_key(site, prod_id, dt, depth)

            # Get the next collection of partial keys using the common prefixes for our
            # candidates
            prefixes = list(itertools.chain(*(self.common_prefixes(b) for b in bounding_keys)))

            # Find where our target would be in the list and grab the ones on either side
            # if possible. This also handles if we're off the end.
            loc = bisect.bisect_left(prefixes, search_key)

            # loc gives where our target *would* be in the list. Therefore slicing from loc - 1
            # to loc + 1 gives the items to the left and right of our target. If get get 0,
            # then there is nothing to the left and we only need the first item.
            rng = slice(loc - 1, loc + 1) if loc else slice(0, 1)
            bounding_keys = prefixes[rng]

        # At this point we've gone through to the minute, now just find the nearest product
        # from everything under the remaining minute options
        return self._closest_result(itertools.chain(*(self.objects(p) for p in bounding_keys)),
                                    dt)

    def _build_result(self, obj):
        """Build a product that opens the data using `Level3File`."""
        return AWSProduct(obj, lambda s: Level3File(s.file))


@exporter.export
class NEXRADLevel2Archive(S3DataStore):
    """Access data from the NEXRAD Level 2 archive in AWS.

    These data consist of complete volumes (i.e. multiple elevation cuts) from NWS NEXRAD.

    """

    def __init__(self, include_mdm=False):
        """Initialize the archive client.

        Parameters
        ----------
        include_mdm : bool, optional
            Whether Model Data Messages (MDM) should be included in results. Defaults to False.

        """
        super().__init__('noaa-nexrad-level2')
        self.include_mdm = include_mdm

    def sites(self, dt=None):
        """Return sites available for a particular date.

        Parameters
        ----------
        dt : datetime.datetime, optional
            The date to use for listing available sites. Defaults to the current date.

        Returns
        -------
        List[str]
            Sites

        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        prefix = self._build_key('', dt, depth=3) + self.delimiter
        return [item.split('/')[-2] for item in self.common_prefixes(prefix)]

    def _build_key(self, site, dt, depth=None):
        """Build a key for the bucket up to the desired point."""
        parts = [f'{dt:%Y}', f'{dt:%m}', f'{dt:%d}', site, f'{site}{dt:%Y%m%d_%H%M%S}']
        return self.delimiter.join(parts[slice(0, depth)])

    def dt_from_key(self, key):  # noqa: D102
        # Docstring inherited
        return datetime.strptime(key.rsplit(self.delimiter, maxsplit=1)[-1][4:19],
                                 '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)

    def get_range(self, site, start, end):
        """Yield products within a particular date/time range.

        Parameters
        ----------
        site : str
            The site to search for data
        start : `datetime.datetime`
            The start of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        end : `datetime.datetime`
            The end of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.

        See Also
        --------
        sites, get_product

        """
        start = ensure_timezone(start)
        end = ensure_timezone(end)
        for dt in date_iterator(start, end, days=1):
            for obj in self.objects(self._build_key(site, dt, depth=4)):
                try:
                    if (start <= self.dt_from_key(obj.key) < end
                            and (self.include_mdm or not obj.key.endswith('MDM'))):
                        yield self._build_result(obj)
                except ValueError:
                    continue

    def get_product(self, site, dt=None):
        """Get a product from the archive.

        Parameters
        ----------
        site : str
            The site to search for data
        dt : `datetime.datetime`, optional
            The desired date/time for the model run; the one closest matching in time will
            be returned. This should have the proper timezone included; if not specified, UTC
            will be assumed. If ``None``, defaults to the current UTC date/time.

        See Also
        --------
        sites, get_range

        """
        dt = datetime.now(timezone.utc) if dt is None else ensure_timezone(dt)
        search_key = self._build_key(site, dt)
        prefix = search_key.split('_')[0]
        objs = (self.objects(prefix) if self.include_mdm else
                filter(lambda o: not o.key.endswith('MDM'), self.objects(prefix)))

        return self._closest_result(objs, dt)

    def _build_result(self, obj):
        """Build a product that opens the data using `Level2File`."""
        return AWSProduct(obj, lambda s: Level2File(s.file))


@exporter.export
class GOESArchive(S3DataStore):
    """Access data from the NOAA GOES archive in AWS.

    This consists of individual GOES image files stored in netCDF format, across a variety
    of sectors, bands, and modes.

    """

    def __init__(self, satellite):
        """Initialize the archive client.

        Parameters
        ----------
        satellite : str or int
            The specific GOES satellite to access (e.g. 16, 17, 18).
        """
        super().__init__(f'noaa-goes{satellite}')

    def product_ids(self):
        """Return product ids available.

        Returns
        -------
        List[str]
            Product ids

        """
        return [item.rstrip(self.delimiter) for item in self.common_prefixes('')]

    def _build_time_prefix(self, product, dt):
        """Build the initial prefix for time and product."""
        # Handle that the meso sector products are grouped in the same subdir
        reduced_product = product[:-1] if product.endswith(('M1', 'M2')) else product
        parts = [reduced_product, f'{dt:%Y}', f'{dt:%j}', f'{dt:%H}', f'OR_{product}']
        return self.delimiter.join(parts)

    def _subprod_prefix(self, prefix, mode, band):
        """Build the full prefix with mode/band, choosing if unambiguous."""
        subprods = {item.rstrip('_').rsplit('-', maxsplit=1)[-1] for item in
                    self.common_prefixes(prefix + '-', '_')}
        if len(subprods) > 1:
            if modes := {item[1] for item in subprods}:
                if len(modes) == 1:
                    mode = next(iter(modes))
                if str(mode) in modes:
                    prefix += f'-M{mode}'
                else:
                    raise ValueError(
                        f'Need to specify a valid operating mode. Available options are '
                        f'{", ".join(sorted(modes))}')
            if bands := {item[-2:] for item in subprods}:
                if len(bands) == 1:
                    band = next(iter(bands))
                if str(band) in bands:
                    prefix += f'C{band}'
                elif isinstance(band, int) and f'{band:02d}' in bands:
                    prefix += f'C{band:02d}'
                else:
                    raise ValueError(
                        f'Need to specify a valid band. Available options are '
                        f'{", ".join(sorted(bands))}')
        return prefix

    def dt_from_key(self, key):  # noqa: D102
        # Docstring inherited
        start_time = key.split('_')[-3]
        return datetime.strptime(start_time[:-1], 's%Y%j%H%M%S').replace(tzinfo=timezone.utc)

    def get_product(self, product, dt=None, mode=None, band=None):
        """Get a product from the archive.

        Parameters
        ----------
        product : str
            The site to search for data
        dt : `datetime.datetime`, optional
            The desired date/time for the model run; the one closest matching in time will
            be returned. This should have the proper timezone included; if not specified, UTC
            will be assumed. If ``None``, defaults to the current UTC date/time.
        mode : str or int, optional
            The particular mode to select. If not given, the query will try to select an
            appropriate mode based on data available.
        band : str or int, optional
            The particular band (or channel) to select. Not all products have multiple bands.
            If not given, the query will try to select an appropriate band, but may error
            giving the channels available if multiple bands are available.

        See Also
        --------
        product_ids, get_range

        """
        dt = datetime.now(timezone.utc) if dt is None else ensure_timezone(dt)
        time_prefix = self._build_time_prefix(product, dt)
        prod_prefix = self._subprod_prefix(time_prefix, mode, band)
        return self._closest_result(self.objects(prod_prefix), dt)

    def get_range(self, product, start, end, mode=None, band=None):
        """Yield products within a particular date/time range.

        Parameters
        ----------
        product : str
            The site to search for data
        start : `datetime.datetime`
            The start of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        end : `datetime.datetime`
            The end of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        mode : str or int, optional
            The particular mode to select. If not given, the query will try to select an
            appropriate mode based on data available.
        band : str or int, optional
            The particular band (or channel) to select. Not all products have multiple bands.
            If not given, the query will try to select an appropriate band, but may error
            giving the channels available if multiple bands are available.

        See Also
        --------
        product_ids, get_product

        """
        start = ensure_timezone(start)
        end = ensure_timezone(end)
        for dt in date_iterator(start, end, hours=1):
            time_prefix = self._build_time_prefix(product, dt)
            prod_prefix = self._subprod_prefix(time_prefix, mode, band)
            for obj in self.objects(prod_prefix):
                if start <= self.dt_from_key(obj.key) < end:
                    yield self._build_result(obj)

    def _build_result(self, obj):
        """Build a product that opens the data using `xarray.open_dataset`."""
        return AWSProduct(obj,
                          lambda s: xr.open_dataset(s.url + '#mode=bytes', engine='netcdf4'))


@exporter.export
class MLWPArchive(S3DataStore):
    """Access data from the NOAA/CIRA Machine-Learning Weather Prediction archive in AWS.

    This consists of individual model runs stored in netCDF format, across a variety
    a collection of models (Aurora, FourCastNet, GraphCast, Pangu) and initial conditions
    (GFS or IFS).

    """

    _model_map = {'aurora': 'AURO', 'fourcastnet': 'FOUR',
                  'graphcast': 'GRAP', 'pangu': 'PANG'}

    def __init__(self):
        super().__init__('noaa-oar-mlwp-data')

    def _model_id(self, model, version, init):
        """Build a model id from the model name, version, and initial conditions."""
        init = init or 'GFS'
        model = self._model_map.get(model.lower(), model)
        if version is None:
            model_id = sorted(self.common_prefixes(model + '_', '_'))[-1]
        else:
            version = str(version)
            if len(version) < 3:
                version = version + '00'
            model_id = f'{model}_v{version}_'
        return f'{model_id}{init}'

    def _build_key(self, model_id, dt, depth=None):
        """Build a key for the bucket up to the desired point."""
        first_hour = 0
        last_hour = 240
        step_hours = 6
        parts = [model_id, f'{dt:%Y}', f'{dt:%m%d}',
                 f'{model_id}_{dt:%Y%m%d%H}_'
                 f'f{first_hour:03d}_f{last_hour:03d}_{step_hours:02d}.nc']
        return self.delimiter.join(parts[slice(0, depth)])

    def dt_from_key(self, key):  # noqa: D102
        # Docstring inherited
        # GRAP_v100_GFS_2025021212_f000_f240_06.nc
        dt = key.split('/')[-1].split('_')[3]
        return datetime.strptime(dt, '%Y%m%d%H').replace(tzinfo=timezone.utc)

    def get_product(self, model, dt=None, version=None, init=None):
        """Get a product from the archive.

        Parameters
        ----------
        model : str
            The selected model to get data for. Can be any of the four-letter codes supported
            by the archive (currently FOUR, PANG, GRAP, AURO), or the known names (
            case-insensitive): ``'Aurora'``, ``'FourCastNet'``, ``'graphcast'``, or
            ``'pangu'``.
        dt : `datetime.datetime`, optional
            The desired date/time for the model run; the one closest matching in time will
            be returned. This should have the proper timezone included; if not specified, UTC
            will be assumed. If ``None``, defaults to the current UTC date/time.
        version : str or int, optional
            The particular version of the model to select. If not given, the query will try
            to select the most recent version of the model.
        init : str, optional
            Selects the model run initialized with a particular set of initial conditions.
            Should be one of ``'GFS'`` or ``'IFS'``, defaults to ``'GFS'``.

        See Also
        --------
        get_range

        """
        dt = datetime.now(timezone.utc) if dt is None else ensure_timezone(dt)
        model_id = self._model_id(model, version, init)
        search_key = self._build_key(model_id, dt)
        prefix = search_key.rsplit('_', maxsplit=4)[0]
        return self._closest_result(self.objects(prefix), dt)

    def get_range(self, model, start, end, version=None, init=None):
        """Yield products within a particular date/time range.

        Parameters
        ----------
        model : str
            The selected model to get data for. Can be any of the four-letter codes supported
            by the archive (currently FOUR, PANG, GRAP, AURO), or the known names (
            case-insensitive): ``'Aurora'``, ``'FourCastNet'``, ``'graphcast'``, or
            ``'pangu'``.
        start : `datetime.datetime`
            The start of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        end : `datetime.datetime`
            The end of the date/time range. This should have the proper timezone included;
            if not specified, UTC will be assumed.
        version : str or int, optional
            The particular version of the model to select. If not given, the query will try
            to select the most recent version of the model.
        init : str, optional
            Selects the model run initialized with a particular set of initial conditions.
            Should be one of ``'GFS'`` or ``'IFS'``, defaults to ``'GFS'``.

        See Also
        --------
        get_product

        """
        start = ensure_timezone(start)
        end = ensure_timezone(end)
        model_id = self._model_id(model, version, init)
        for dt in date_iterator(start, end, days=1):
            prefix = self._build_key(model_id, dt, depth=3)
            for obj in self.objects(prefix):
                if start <= self.dt_from_key(obj.key) < end:
                    yield self._build_result(obj)

    def _build_result(self, obj):
        """Build a product that opens the data using `xarray.open_dataset`."""
        return AWSProduct(obj,
                          lambda s: xr.open_dataset(s.url + '#mode=bytes', engine='netcdf4'))
