import tiledb
import xarray as xr
import shutil
import urllib.request
import os
import numpy as np


# test file that I found in researching cfgrib
test_file = 'http://download.ecmwf.int/test-data/cfgrib/era5-levels-members.grib'
test_file_local = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'era5-levels-members.grib')
if not os.path.exists(test_file_local):
    with urllib.request.urlopen(test_file) as response:
        with open(test_file_local, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)
if not os.path.exists(test_file_local):
    raise ValueError(f'Unable to download {test_file} to {test_file_local}')

# now test the xarray grib stuff
ds = xr.open_dataset(test_file_local, engine='cfgrib')

# going from xarray to tiledb seems to be something that needs to be developed.
# https://github.com/TileDB-Inc/TileDB-CF-Py/issues/112
# sounds like it might be complex requiring a refactor of the writes to support a generic entry point, similar to open_dataset

# can try this, to go from pandas to tiledb, but the timedelta dtype is not supported
# tiledb.from_pandas(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'era5tiledb'), ds.to_dataframe())

# lets try to make sense of the existing dimensions.  I believe we need to collapse the three time dimensions into a 1d array
# where: finalized_time = time + (step * number)

# first, step for this test dataset is zero, let's set a real value here for testing so we can see the change
ds['step'] = xr.DataArray(np.timedelta64(65, 's'))


def build_finalized_time_from_dataset(ds: xr.Dataset):
    """
    Collapse the step/time/number dimensions into a single time dimension

    Parameters
    ----------
    ds
        xarray Dataset read from GRIB

    Returns
    -------
    np.ndarray
        Base time for each time
    np.ndarray
        finalized_time = time + (step * number), cast to integer seconds to satisfy tiledb coordinate requirements
    """

    base_time = np.repeat(ds.time.values, ds.number.shape[0])
    finalized_time = np.repeat(ds.time.values, ds.number.shape[0])
    time_offset = ds.step * ds.number
    time_dim = ds.time.shape[0]
    number_dim = ds.number.shape[0]
    for i in range(time_dim):
        t_range = (i * number_dim, (i + 1) * number_dim)
        finalized_time[t_range[0]:t_range[1]] = finalized_time[t_range[0]:t_range[1]] + time_offset.values
    # convert to utc timestamp, to integer seconds
    base_time = (base_time - np.datetime64(0, 's')) / np.timedelta64(1, 's')
    finalized_time = (finalized_time - np.datetime64(0, 's')) / np.timedelta64(1, 's')
    return base_time, finalized_time


# get the new time dimension arrays
basetime, newtime = build_finalized_time_from_dataset(ds)
# build the new dataset with the single time dimension
# ** note that float dims are not allowed here
new_ds = xr.Dataset(data_vars={'z': (['time', 'isobaricInhPa', 'latitude', 'longitude'], np.concatenate([ds.z.values[:, i, :, :] for i in range(ds.time.shape[0])])),
                               't': (['time', 'isobaricInhPa', 'latitude', 'longitude'], np.concatenate([ds.t.values[:, i, :, :] for i in range(ds.time.shape[0])]))},
                    coords={'time': newtime,
                            'isobaricInhPa': ds.isobaricInhPa.drop('step'), 'latitude': ds.latitude.drop('step'),
                            'longitude': ds.longitude.drop('step')}
                    )

# >>> new_ds
# <xarray.Dataset>
# Dimensions:        (time: 40, isobaricInhPa: 2, latitude: 61, longitude: 120)
# Coordinates:
#   * time           (time) float64 1.483e+09 1.483e+09 ... 1.483e+09 1.483e+09
#   * isobaricInhPa  (isobaricInhPa) float64 850.0 500.0
#   * latitude       (latitude) float64 90.0 87.0 84.0 81.0 ... -84.0 -87.0 -90.0
#   * longitude      (longitude) float64 0.0 3.0 6.0 9.0 ... 351.0 354.0 357.0
# Data variables:
#     z              (time, isobaricInhPa, latitude, longitude) float32 1.42e+0...
#     t              (time, isobaricInhPa, latitude, longitude) float32 252.7 ....

# save to tiledb
output_tiledb = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'era5tiledb')
ds_dataframe = new_ds.to_dataframe()

# >>> ds_dataframe
#                                                           z           t
# time         isobaricInhPa latitude longitude
# 1.483229e+09 850.0          90.0    0.0        14201.753906  252.663147
#                                     3.0        14201.753906  252.663147
#                                     6.0        14201.753906  252.663147
#                                     9.0        14201.753906  252.663147
#                                     12.0       14201.753906  252.663147
#                                                      ...         ...
# 1.483359e+09 500.0         -90.0    345.0      51149.941406  240.378220
#                                     348.0      51149.941406  240.378220
#                                     351.0      51149.941406  240.378220
#                                     354.0      51149.941406  240.378220
#                                     357.0      51149.941406  240.378220
# [585600 rows x 2 columns]

tiledb.from_pandas(output_tiledb, ds_dataframe)
# tiledb.cc.TileDBError: [TileDB::ArraySchema] Error: Cannot set domain; Dense arrays do not support dimension datatype 'FLOAT64'
# in addition, found other errors on trying to even use integer coordinates like
# IndexError: index out of bounds <todo>

# reload to check out the data (Never gets this far)
reload_ds = xr.open_dataset(output_tiledb, engine='tiledb')

# I believe this should work, but I keep running into issues with expected sizes of arrays written to disk
# I need to experiment more with creating the tiledb schema from scratch to better understand the format.

import tiledb
import numpy as np

x_size = 1000
y_size = 1000
schema = tiledb.ArraySchema(
    domain=tiledb.Domain(
        tiledb.Dim("x", domain=(0, x_size - 1), dtype=np.uint64),
        tiledb.Dim("y", domain=(0, y_size - 1), dtype=np.uint64)
    ),
    attrs=(
        tiledb.Attr("survey1", np.float64),),
)

uri1 = r'C:\Users\eyou1\Downloads\tiledb_test3'
tiledb.Array.create(uri1, schema)

with tiledb.open(uri1, mode="w") as array:
    array[:, :] = {"survey1": np.random.rand(x_size, y_size)}

tiledb.array_fragments(uri1)
# tiledb.array_fragments(uri1).timestamp_range
# ((1666972369899, 1666972369899), (1666973013988, 1666973013988))

with tiledb.open(uri1, mode="w") as array:
    array[1, 1] = {"survey1": np.ones((1, 1))}

with tiledb.open(uri1, mode="w") as array:
    array[900, 900] = {"survey1": np.ones((1, 1))}

tiledb.array_fragments(uri1)
# tiledb.array_fragments(uri1).timestamp_range
# ((1666972369899, 1666972369899), (1666973013988, 1666973013988))

tiledb.consolidate(uri1)
tiledb.vacuum(uri1)