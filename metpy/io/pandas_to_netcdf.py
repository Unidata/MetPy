# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Support reading a pandas dataframe to a DSG netCDF."""

import logging
from os import path

from numpy import arange
import xarray as xr

from ..package_tools import Exporter

exporter = Exporter(globals())

log = logging.getLogger(__name__)


@exporter.export
def dataframe_to_netcdf(df, sampling_var, sampling_data_vars, path_to_save, netcdf_format=None,
                        column_units=None, standard_names=None, long_names=None,
                        dataset_type=None):
    r"""Take a Pandas DataFrame and convert it to a netCDF file.

    If given a Pandas DataFrame, this function will first convert
    it to a xarray Dataset, attach attributes and metadata to it as
    provided by the user, and then save it as a CF-compliant discrete
    sampling geometry (DGS) netCDF file. Assumes each row of the DataFrame
    is a unique observation

    This function is ideal for point data, such as station observations,
    or for trajectory or profile data, which is discretely sampled at
    individual points

    Parameters
    ----------
    df : `pandas.DataFrame`
        Point data in pandas dataframe.

    sampling_var : str
        Column name that is the sampling dimension: for surface observations,
        this is the column that contains the station identifier/name

    sampling_data_vars : list
        List of all variables associated with the sampling variable that do not
        vary with time, such as latitude, longitude, and elevation for
        surface observations

    path_to_save : str
        Path, including filename, for where to save netCDF file.

    netcdf_format : str, optional

    column_units : dict, optional
        Dictionary of units to attach to columns of the dataframe. Overrides
        the units attribute if it is attached to the dataframe.

    standard_names : dict, optional
        Dictionary of variable descriptions that are CF-compliant

    long_names : dict, optional
        Dictionary of longer variable descriptions that provide more detail
        than standard_names

    dataset_type: str, optional
        Type of dataset to be converted. Options are 'timeSeries', 'profile',
        or 'trajectory'. While optional, this variable should be declared to create
        a CF-compliant DSG netCDF file.

    Returns
    -------
    NetCDF file saved to `path_to_save`.

    """
    # Verify_integrity must be true in order for conversion to netCDF to work
    # Return a TypeError if not provided a Pandas DataFrame
    try:
        # Create the dimensions for use later in netCDF file
        samplingindex = df.groupby([sampling_var], sort=False).ngroup()
        obs = arange(0, len(df))
        df.insert(0, 'samplingIndex', samplingindex)
        df.insert(1, 'observations', obs)

        # Handle the sampling location specific data
        sampling_data = df[sampling_data_vars]
        samples = sampling_data.groupby([sampling_var], sort=False).ngroup()
        sampling_data.insert(0, 'samples', samples)
        sampling_data = sampling_data.groupby('samples').first()
        dataset_samples = xr.Dataset.from_dataframe(sampling_data)

        # Create the dataset for the variables of each observation
        df = df.drop(sampling_data_vars, axis=1)
        df = df.set_index(['observations'], verify_integrity=True)
        dataset_var = xr.Dataset.from_dataframe(df)

        # Merge the two datasets together
        dataset_final = xr.merge([dataset_samples, dataset_var], compat='no_conflicts')

    except (AttributeError, ValueError, TypeError):
        raise TypeError('A pandas dataframe was not provided')

    # Attach variable-specific metadata
    _assign_metadata(dataset_final, column_units, standard_names, long_names)

    # Attach dataset-specific metadata
    if dataset_type:
        dataset_final.attrs['featureType'] = dataset_type
    else:
        log.warning('No dataset type provided - netCDF will not have appropriate metadata'
                    'for a DSG dataset.')
    if dataset_type:
        dataset_final[sampling_var].attrs['cf_role'] = dataset_type.lower() + '_id'
    dataset_final['samplingIndex'].attrs['instance_dimension'] = 'samples'

    # Determine mode to write to netCDF
    write_mode = 'w'
    if path.exists(path_to_save):
        # Eventually switch to 'a' to allow appending and delete error
        raise ValueError('File already exists - please delete and run again')

    # Check if netCDF4 is installed to see how many unlimited dimensions we can use
    # Need conditional import for checking due to Python 2
    try:
        from importlib.util import find_spec
        check_netcdf4 = find_spec('netCDF4')
    except ImportError:
        from imp import find_module
        check_netcdf4 = find_module('netCDF4')

    # Make sure path is a string to allow netCDF4 to be used - needed for tests to pass
    path_to_save = str(path_to_save)

    if check_netcdf4 is not None:
        unlimited_dimensions = ['samples', 'observations']
    else:
        # Due to xarray's fallback to scipy if netCDF4-python is not installed
        # only one dimension can be unlimited. This may cause issues for users
        log.warning('NetCDF4 not installed - saving as a netCDF3 file with only the'
                    'observations dimension as unlimited. If netCDF4 or multiple'
                    'dimensions are desired, run `pip install netCDF4`')
        unlimited_dimensions = ['observations']

    # Convert to netCDF
    dataset_final.to_netcdf(path=path_to_save, mode=write_mode, format=netcdf_format,
                            unlimited_dims=unlimited_dimensions, compute=True)


def _assign_metadata(dataset, units_dict, standard_names_dict, long_names_dict):
    if units_dict is not None:
        final_column_units = {}
        final_column_units['samples'] = ''
        final_column_units['observations'] = ''
        final_column_units['samplingIndex'] = ''
        final_column_units.update(units_dict)
        for var in dataset.variables:
            dataset[var].attrs['units'] = final_column_units[var]
    if standard_names_dict is not None:
        final_std_names = {}
        final_std_names['samples'] = ''
        final_std_names['observations'] = ''
        final_std_names['samplingIndex'] = ''
        final_std_names.update(standard_names_dict)
        for var in dataset.variables:
            dataset[var].attrs['standard_name'] = final_std_names[var]
    if long_names_dict is not None:
        final_long_names = {}
        final_long_names['samples'] = 'Sampling dimension'
        final_long_names['observations'] = 'Observation dimension'
        final_long_names['samplingIndex'] = 'Index of station for this observation'
        final_long_names.update(long_names_dict)
        for var in dataset.variables:
            dataset[var].attrs['long_name'] = final_long_names[var]
