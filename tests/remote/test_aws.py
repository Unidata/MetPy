# Copyright (c) 2025 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `metpy.remote.aws` module."""
from datetime import datetime
from pathlib import Path
import tempfile

from metpy.remote import GOESArchive, MLWPArchive, NEXRADLevel2Archive, NEXRADLevel3Archive
from metpy.testing import needs_aws


@needs_aws
def test_nexrad3_single():
    """Test getting a single product from the NEXRAD level 3 archive."""
    l3 = NEXRADLevel3Archive().get_product('FTG', 'N0Q', datetime(2020, 4, 1, 12, 30))
    assert l3.path == 'FTG_N0Q_2020_04_01_12_30_12'
    assert l3.access()


@needs_aws
def test_nexrad3_range():
    """Test getting a range of products from the NEXRAD level 3 archive."""
    prods = list(NEXRADLevel3Archive().get_range('FTG', 'N0B', datetime(2024, 12, 31, 23, 45),
                                                 datetime(2025, 1, 1, 1, 15)))
    names = [p.name for p in prods]
    assert names == ['FTG_N0B_2024_12_31_23_46_08', 'FTG_N0B_2024_12_31_23_53_12',
                     'FTG_N0B_2025_01_01_00_00_17', 'FTG_N0B_2025_01_01_00_07_22',
                     'FTG_N0B_2025_01_01_00_14_26', 'FTG_N0B_2025_01_01_00_21_30',
                     'FTG_N0B_2025_01_01_00_28_34', 'FTG_N0B_2025_01_01_00_35_38',
                     'FTG_N0B_2025_01_01_00_42_41', 'FTG_N0B_2025_01_01_00_49_45',
                     'FTG_N0B_2025_01_01_00_56_49', 'FTG_N0B_2025_01_01_01_03_54',
                     'FTG_N0B_2025_01_01_01_10_58']
    with tempfile.TemporaryDirectory() as tmpdir:
        prod = prods[2]
        prod.download(tmpdir)
        assert (Path(tmpdir) / 'FTG_N0B_2025_01_01_00_00_17').exists()

        prods[4].download(Path(tmpdir) / 'tempprod')
        assert (Path(tmpdir) / 'tempprod').exists()


@needs_aws
def test_nexrad2_single():
    """Test getting a single volume from the NEXRAD level 2 archive."""
    l2 = NEXRADLevel2Archive().get_product('KTLX', datetime(2013, 5, 20, 20, 15))
    assert l2.name == 'KTLX20130520_201643_V06.gz'


@needs_aws
def test_nexrad2_range():
    """Test getting a range of products from the NEXRAD level 2 archive."""
    vols = list(NEXRADLevel2Archive().get_range('KFTG', datetime(2024, 12, 14, 15, 15),
                                                datetime(2024, 12, 14, 16, 25)))
    names = [v.name for v in vols]
    assert names == ['KFTG20241214_151956_V06', 'KFTG20241214_152855_V06',
                     'KFTG20241214_153754_V06', 'KFTG20241214_154653_V06',
                     'KFTG20241214_155552_V06', 'KFTG20241214_160451_V06',
                     'KFTG20241214_161349_V06', 'KFTG20241214_162248_V06']


@needs_aws
def test_goes_single():
    """Test getting a single product from the GOES archive."""
    prod = GOESArchive(18).get_product('ABI-L1b-RadM1', datetime(2025, 1, 9, 23, 56), band=2)
    assert prod.url == ('https://noaa-goes18.s3.amazonaws.com/ABI-L1b-RadM/2025/009/23/'
                        'OR_ABI-L1b-RadM1-M6C02_G18_s20250092356254_e20250092356311_'
                        'c20250092356338.nc')
    assert prod.access().attrs['dataset_name'] == ('OR_ABI-L1b-RadM1-M6C02_G18_s20250092356254'
                                                   '_e20250092356311_c20250092356338.nc')


@needs_aws
def test_goes_range():
    """Test getting a range of products from the GOES archive."""
    prods = list(GOESArchive(16).get_range('ABI-L1b-RadC', datetime(2024, 12, 10, 1, 0),
                                           datetime(2024, 12, 10, 2, 15), band=1))
    names = [p.name for p in prods]
    truth = ['OR_ABI-L1b-RadC-M6C01_G16_s20243450101170_e20243450103543_c20243450103590.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450106170_e20243450108543_c20243450108594.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450111170_e20243450113543_c20243450113586.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450116170_e20243450118543_c20243450118589.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450121170_e20243450123543_c20243450123594.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450126170_e20243450128543_c20243450129025.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450131170_e20243450133543_c20243450133590.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450136170_e20243450138543_c20243450138585.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450141170_e20243450143543_c20243450143595.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450146170_e20243450148543_c20243450148579.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450151170_e20243450153543_c20243450154026.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450156170_e20243450158543_c20243450158585.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450201170_e20243450203543_c20243450204022.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450206170_e20243450208543_c20243450208597.nc',
             'OR_ABI-L1b-RadC-M6C01_G16_s20243450211170_e20243450213543_c20243450214031.nc']
    assert names == truth


@needs_aws
def test_mlwp_single():
    """Test getting a single product from the MLWP archive."""
    prod = MLWPArchive().get_product('graphcast', datetime(2025, 1, 30, 10))
    assert prod.url == ('https://noaa-oar-mlwp-data.s3.amazonaws.com/GRAP_v100_GFS/'
                        '2025/0130/GRAP_v100_GFS_2025013012_f000_f240_06.nc')


@needs_aws
def test_mlwp_range():
    """Test getting a single product from the MLWP archive."""
    prods = MLWPArchive().get_range('fourcastnet', datetime(2025, 2, 3), datetime(2025, 2, 6))
    names = [p.name for p in prods]
    truth = ['FOUR_v200_GFS_2025020300_f000_f240_06.nc',
             'FOUR_v200_GFS_2025020312_f000_f240_06.nc',
             'FOUR_v200_GFS_2025020400_f000_f240_06.nc',
             'FOUR_v200_GFS_2025020412_f000_f240_06.nc',
             'FOUR_v200_GFS_2025020500_f000_f240_06.nc',
             'FOUR_v200_GFS_2025020512_f000_f240_06.nc']
    assert names == truth
