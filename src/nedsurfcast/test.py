import xarray as xr
ds = xr.open_zarr("data/processed/northsea_training_data.zarr", consolidated=False)
for var in list(ds.data_vars):
    n_nan = ds[var].isnull().sum().values
    print(var, "NaNs:", n_nan, "/", ds[var].size)