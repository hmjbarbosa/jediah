# This script goal is to produce some plots to show an issue we found with the data. 
# When computing 1h averages, we noticed that signal decreases with altitude and eventually becomes negative. 
# As if the automatic BG removal removed too much signal.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates

plt.ion()

file = './ceilometer/2016/12/20161202_EMBRAPA_CHM160147.nc'

#--------------------------------------------------
# read ceilometer data
ds_lidar = xr.open_dataset(file)

#--------------------------------------------------
# grab the CBH and convert -1 into NaN
cbh = ds_lidar.cbh.values.astype(float)
cbh[cbh < 0] = np.nan

# grab first cloud base height
cbh1 = cbh[:, 0]

# filter dataset for cloud-free conditions
# Note: clouds typically start small, before the ceilometer can detect them. 
# Hence, we also exclude the profiles next to the clouds
neighb = 2
cloud_mask = cbh1 > 0
# have to use a copy, because we are modifying the mask while iterating over it
tmp = cloud_mask.copy()
for i in np.arange(0, len(cbh1)):
    if tmp[i]:
        cloud_mask[max(0, i-neighb):min(len(cbh1)-1, i+neighb+1)] = True
ds_lidar_clear = ds_lidar.isel(time=~cloud_mask)

p_calc_max = np.max(ds_lidar.p_calc)
stddev_max = np.max(ds_lidar.stddev)
base_max = np.max(ds_lidar.base)

#--------------------------------------------------
# Big figure showing the whole day
plt.figure(num=100, figsize=(10, 6), clear=True)
plt.pcolormesh(ds_lidar.time, ds_lidar.range, ds_lidar.beta_raw.T, vmin=-8e5, vmax=8e5, cmap='jet')
# add some control flags/diagnostics
plt.plot(ds_lidar.time, ds_lidar.p_calc*2000/p_calc_max + 6000, '-k')
plt.plot(ds_lidar.time, ds_lidar.stddev*2000/stddev_max + 8000, '-k')
plt.plot(ds_lidar.time, ds_lidar.base*2000/base_max + 10000, '-k')
#
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.title(file)
plt.ylabel('Range [m]')
plt.xlabel('Time [UTC]')
plt.grid()
cb = plt.colorbar()
cb.set_label('beta_raw')
cb.formatter = ScalarFormatter(useMathText=True)
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()


# Now one figure per hour, with two panels 
for hour in np.arange(0, 24):
    # left is the beta_raw for the hour, 
    fig, (ax1, ax2) = plt.subplots(1, 2, num=hour, figsize=(10, 6), clear=True)
    mask = ds_lidar.time.dt.hour == hour
    pc = ax1.pcolormesh(ds_lidar.time[mask], ds_lidar.range, ds_lidar.beta_raw[mask].T, vmin=-8e5, vmax=8e5, cmap='jet')
    # add the cloud bases
    ax1.plot(ds_lidar.time[mask], cbh[mask], 'om')
    # add some control flags/diagnostics
    ax1.plot(ds_lidar.time[mask], ds_lidar.p_calc[mask]*2000/p_calc_max + 6000, '-k')
    ax1.plot(ds_lidar.time[mask], ds_lidar.stddev[mask]*2000/stddev_max + 8000, '-k')
    ax1.plot(ds_lidar.time[mask], ds_lidar.base[mask]*2000/base_max + 10000, '-k')
    #
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.suptitle(file)
    ax1.set_title(f'Hour {hour:02d}')
    ax1.set_ylabel('Range [m]')
    ax1.set_xlabel('Time [UTC]')
    ax1.grid()
    cb = fig.colorbar(pc, ax=ax1)
    #cb.set_label('beta_raw')
    cb.formatter = ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    # right panel is the average signal for the cloud-free profiles in the hour
    mask = ds_lidar_clear.time.dt.hour == hour
    beta_raw = ds_lidar_clear.beta_raw[mask].values
    if np.sum(mask) > 0:
        beta_raw_mean = np.nanmean(beta_raw, axis=0)
        ax2.plot(beta_raw_mean, ds_lidar_clear.range, '-k')
    ax2.set_title(f'Num. cloud-free profiles = {np.sum(mask):02d}')
    ax2.set_xlabel('beta_raw')
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.xaxis.get_major_formatter().set_powerlimits((0, 0))
    #ax2.set_ylabel('Range [m]')
    ax2.set_xlim(-1e5, 8e5)
    ax2.set_ylim(ax1.get_ylim())
    ax2.grid() 

