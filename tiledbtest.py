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
    return base_time.astype(int), finalized_time.astype(int)


# get the new time dimension arrays
basetime, newtime = build_finalized_time_from_dataset(ds)
# build the new dataset with the single time dimension
# ** note that float dims are not allowed here
new_ds = xr.Dataset(data_vars={'z': (['time', 'isobaricInhPa', 'latitude', 'longitude'], np.concatenate([ds.z.values[:, i, :, :] for i in range(ds.time.shape[0])])),
                               't': (['time', 'isobaricInhPa', 'latitude', 'longitude'], np.concatenate([ds.t.values[:, i, :, :] for i in range(ds.time.shape[0])]))},
                    coords={'time': newtime,
                            'isobaricInhPa': ds.isobaricInhPa.drop('step').astype(int), 'latitude': ds.latitude.drop('step').astype(int),
                            'longitude': ds.longitude.drop('step').astype(int)}
                    )

# save to tiledb
output_tiledb = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'era5tiledb')
ds_dataframe = new_ds.to_dataframe()

tiledb.from_pandas(output_tiledb, ds_dataframe)

# reload to check out the data
reload_ds = xr.open_dataset(output_tiledb, engine='tiledb')

# I believe this should work, but I keep running into issues with expected sizes of arrays written to disk
# I need to experiment more with creating the tiledb schema from scratch to better understand the format.