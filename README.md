# tiledb_test
Test the xarray / tiledb workflow with GRIB files

So far, I've found a number of potential issues using xarray + tiledb including:
 - no support for floating point numbers in coordinate arrays in Xarray/tiledb implementation (currently assuming that we can cast these to int)
 - no support for timedelta/datetime objects in coordinate arrays (I collapsed the dims to resolve this)

Setup requires:
- conda install -c conda-forge tiledb-py
- pip install tiledb-cf
