"""
Helpers for wave direction, which is circular (0 and 360 degrees are the same)
"""

import numpy as np


# 2-component continuous representation of VMDR, for averaging and as model input/output.
# The datasets on disk store VMDR in degrees, like the buoys.
def add_direction_sin_cos(ds_wave):
    theta_rad = np.deg2rad(ds_wave["VMDR"])
    ds_wave = ds_wave.copy()
    ds_wave["VMDR_sin"] = np.sin(theta_rad)
    ds_wave["VMDR_cos"] = np.cos(theta_rad)
    return ds_wave

# Angular difference in [-180, 180): 350 - 10 is -20, not 340
def wrap_angle(deg):
    return (deg + 180) % 360 - 180
