#import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import TextIOWrapper
import glob
import re 
import xarray as xr
import os

plt.ion()

#ZIP_PATH = "c:/Users/jedia/Downloads/radiosonde.zip"
PATH = "sonde/daily_files/"
PATH_OUT = "sonde/daily_files/nc_files_interpolated/"
#PATH = "./82332/dat/"

def read_radiosonde_fixed_width(lines):
    """Parse one radiosonde profile into a DataFrame."""
    dash_count = 0
    start_idx, end_idx = None, None
    observation_time = None

    for i, line in enumerate(lines):
        if "----" in line:
            dash_count += 1
            if dash_count == 2:
                start_idx = i + 1
        if "Station information" in line:
            end_idx = i
        if "Observation time" in line:
            # Extract last item after the colon
            tmp = line.split(":")[-1].strip()
            year = int(tmp[:2]) + 1900
            month = int(tmp[2:4])
            day = int(tmp[4:6])
            hour = int(tmp[7:9])
            minute = int(tmp[9:11])
            observation_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

    if start_idx is None or end_idx is None:
        return pd.DataFrame()  # maybe return None instead?

    # extract only data lines
    data_lines = lines[start_idx:end_idx]

    colnames = [
        "PRES_hPa", "HGHT_m", "TEMP_C", "DWPT_C", "RELH_pct",
        "MIXR_gkg", "DRCT_deg", "SKNT_knot",
        "THTA_K", "THTE_K", "THTV_K"
    ]

    # fixed-width parsing
    df = pd.read_fwf(pd.io.common.StringIO("".join(data_lines)), names=colnames)

    # exclude rows with all NaNs
    df.dropna(how="all", inplace=True)

    # why is this necessary? 
    df = df.apply(pd.to_numeric, errors="coerce")

    # remove lines with missing essential data in the important columns
    # There are many files with one extra line at the beginning with 1000 hPa and < 84m altitude
    # This was probably added later, since the sonde station is at 84m. 
    df.dropna(subset=["PRES_hPa", "HGHT_m", "TEMP_C"], inplace=True)

    # attach timestamp to whole profile
    ds = xr.Dataset(
        {
            "PRES_hPa": (("time", "height"), df["PRES_hPa"].values[np.newaxis, :]),
            "TEMP_C": (("time", "height"), df["TEMP_C"].values[np.newaxis, :]),
            "Density": (("time", "height"), compute_density(df["PRES_hPa"].values, df["TEMP_C"].values)[np.newaxis, :]),
        },
        coords={
            "height": df["HGHT_m"].values,
            "time": [observation_time],
        }
    )
    #df.attrs["xr_dataset"] = ds
    #if observation_time is not None:
    #    df["time"] = observation_time

    return ds

def compute_density(pressure_hPa, temp_C):
    """Compute air density using the ideal gas law."""
    R_d = 287.05
    p = pressure_hPa * 100.0
    T = temp_C + 273.15
    return p / (R_d * T)

def check_profile_quality(ds):

    # this function was first written for DataFrame, now adapted for xarray Dataset
    # it assumes we still only have 1 single time per Dataset
    # if the user passes a DS with multiple times, the function will likely fail
    # in the future, we should expand it to handle multiple times (e.g. loop over time dimension)
    problems = []
    if ds is None:
        problems.append("Dataset is None")
        return ds, problems

    if len(ds.height) < 5: 
        problems.append(f"Not enough data points: {len(ds.height)}")
        return None, problems
    
    # Surface values
    # Since we now use datasets, we need to index properly
    surf_h = ds["height"].values[0]
    # the DS has two coordinates, height and time, although time has only one value
    surf_p = ds["PRES_hPa"].values[0, 0]
    surf_t = ds["TEMP_C"].values[0, 0]

    # Quality Checks
    if not (990 <= surf_p <= 1020):
        problems.append(f"Unrealistic surface pressure: {surf_p:.1f} hPa")
        return None, problems

    if surf_h != 84:
        problems.append(f"Wrong surface altitude: {surf_h:.1f} m")
        return None, problems
    
    if not (10 <= surf_t <= 40):
        problems.append(f"Unrealistic surface temperature: {surf_t:.1f} °C")
        return None, problems

    # Height checks using np.diff
    hdiff = np.diff(ds["height"].values)
    bad_idx = np.where((hdiff <= 0))[0] 

    if len(bad_idx) > 1:
        problems.append(f"Too many problems with the altitude values: {len(bad_idx)} (non-monotonic)")
        return None, problems
    
    elif len(bad_idx) == 1:
        if hdiff[bad_idx[0]] < -10:
            problems.append(f"There is 1 non-monotonic altitude, but difference too large: {hdiff[bad_idx[0]]:.1f} m")
            return None, problems
        
        else:
            # in this case, we try to fix it
            problems.append(f"Removed {len(bad_idx)} suspect levels (non-monotonic)")
            # Drop the "next" row after each bad difference
            # Remove the height(s) at bad_idx + 1 from the dataset
            drop_heights = ds["height"].values[bad_idx + 1]
            ds = ds.sel(height=~np.isin(ds["height"], drop_heights))
            #ds = ds.sel(height =~ ds["height"].values[bad_idx + 1])-

    if sorted(ds["height"].values)[0] != surf_h:
        print(ds["height"].values)
        print('test=',bad_idx)
        problems.append(f"weird low altitude value!")
        return None, problems

    max_h = ds["height"].max()
    if max_h < 15000:
        problems.append(f"Profile too shallow: max height {max_h:.0f} m")

    return ds, problems

def process_filtered_from_path(path, filter="*.dat"):
    """Read only files whose names contain one of the filter strings."""
    profiles = []
    for fname in sorted(glob.glob(os.path.join(path, filter))):
    # extract year from filename
        match = re.search(r"_(\d{4})_", fname)
        if match:
            year = int(match.group(1))
            with open(fname, 'r', encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                ds = read_radiosonde_fixed_width(lines)
                # basic quality checks
                #print('fname=',fname)
                #print(ds)
                #print(ds is None)
                #print(len(ds))
                ds, problems = check_profile_quality(ds)
                if problems:
                    print(f"Issues in {fname}: {problems}")
                else:
                    profiles.append(ds)
                if len(profiles) % 100 == 0:
                    print(f"Processed {len(profiles)} profiles... Wall time is: {pd.Timestamp.now()}")

    print(f"Loaded {len(profiles)} profiles matching filter: {filter}")
    return profiles

def vertical_interpolate_profiles(profiles, height_step=100, max_height=20000):
    """Interpolate profiles onto a common grid and return the list of interpolated profiles."""
    # new vertical grid
    # assume all profiles came from the same site, i.e., same surface height
    sfc_altitude = profiles[0]["height"].values[0]
    height_grid = np.arange(sfc_altitude, max_height + 1, height_step)

    # list of DS with interpolated profiles
    interp_profiles = []

    for ds in profiles:
        interp_profiles.append(ds.interp(height=height_grid))

    return interp_profiles

def average_profiles(combined_ds):
    """Compute average, std and count from an xarray.Dataset of profiles."""

    # Compute aggregated datasets for seasons that exist in the data
    means = combined_ds.mean(dim="time", skipna=True)
    stds = combined_ds.std(dim="time", skipna=True)
    counts = combined_ds.count(dim="time")

    # now we return xarray datasets, each one with the 4 seasons inside
    # but one dataset for "averages", one for "std" and one for "counts"
    return means, stds, counts

def get_season(month):
    """Return season label from month."""
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"
    
def seasonal_profiles(combined_ds):
    """
    Compute seasonal mean, std and count for each season from an xarray.Dataset.

    Parameters
    ----------
    combined_ds : xarray.Dataset
        Dataset must contain a 'time' coordinate of datetime64-like values.

    Returns
    -------
    seasonal_avgs : dict
        Mapping season -> xarray.Dataset with the mean values (averaged over time for that season), or None if no data.
    seasonal_stds : dict
        Mapping season -> xarray.Dataset with the standard deviation (over time for that season), or None.
    seasonal_counts : dict
        Mapping season -> xarray.Dataset with counts (number of samples contributing per variable/dimension), or None.
    """

    if not isinstance(combined_ds, xr.Dataset):
        raise TypeError("seasonal_profiles expects an xarray.Dataset as input")

    if "time" not in combined_ds.coords and "time" not in combined_ds.dims:
        raise ValueError("Dataset must have a 'time' coordinate/dimension")

    # Group by season using xarray's groupby on time.season
    grouped = combined_ds.groupby("time.season")

    # Compute aggregated datasets for seasons that exist in the data
    means = grouped.mean(dim="time", skipna=True)
    stds = grouped.std(dim="time", skipna=True)
    counts = grouped.count(dim="time")

    # now we return xarray datasets, each one with the 4 seasons inside
    # but one dataset for "averages", one for "std" and one for "counts"
    return means, stds, counts

def collect_surface_values(profiles):
    """Grab the first pressure, temp, density from each sonde."""
    pres, temp, dens = [], [], []
    for df in profiles:
        if not df is None:
            pres.append(df["PRES_hPa"].iloc[0])
            temp.append(df["TEMP_C"].iloc[0])
            dens.append(df["Density"].iloc[0])
    return np.array(pres), np.array(temp), np.array(dens)

def seasonal_surface_histograms(profiles):
    # Make histograms of surface values grouped by season with aligned bins (subplots + overlays).
    seasons = ["DJF", "MAM", "JJA", "SON"]
    colors = {"DJF": "blue", "MAM": "green", "JJA": "orange", "SON": "purple"}
    pres_by_season = {s: [] for s in seasons}
    temp_by_season = {s: [] for s in seasons}
    dens_by_season = {s: [] for s in seasons}

    # Collect surface values grouped by season
    for ds in profiles:
        if (not ds is None) and ("time" in ds):
            season = get_season(ds["time.month"].values[0])
            pres_by_season[season].append(ds["PRES_hPa"].values[0][0])
            temp_by_season[season].append(ds["TEMP_C"].values[0][0])
            dens_by_season[season].append(ds["Density"].values[0][0])

    # --- Pressure bins (aligned) ---
    all_pres = np.concatenate([pres_by_season[s] for s in seasons if pres_by_season[s]])
    pres_min, pres_max = np.min(all_pres), np.max(all_pres)
    pres_bins = np.arange(np.floor(pres_min/1)*1, np.ceil(pres_max/1)*1 + 1, 1)

    # --- Temperature bins (aligned) ---
    all_temp = np.concatenate([temp_by_season[s] for s in seasons if temp_by_season[s]])
    temp_min, temp_max = np.min(all_temp), np.max(all_temp)
    temp_bins = np.arange(np.floor(temp_min/0.5)*0.5, np.ceil(temp_max/0.5)*0.5 + 0.5, 0.5)

    # --- Density bins (aligned) ---
    all_dens = np.concatenate([dens_by_season[s] for s in seasons if dens_by_season[s]])
    dens_min, dens_max = np.min(all_dens), np.max(all_dens)
    dens_bins = np.arange(np.floor(dens_min*100)/100, np.ceil(dens_max*100)/100 + 0.01, 0.01)

    # Seperate per season
    plt.figure(figsize=(10, 8))
    for i, s in enumerate(seasons, 1):
        plt.subplot(2, 2, i)
        plt.hist(pres_by_season[s], bins=pres_bins, color="skyblue", edgecolor="black")
        plt.title(s)
        plt.xlabel("Surface Pressure (hPa)")
        plt.ylabel("Count")
    plt.suptitle("Seasonal Surface Pressure Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i, s in enumerate(seasons, 1):
        plt.subplot(2, 2, i)
        plt.hist(temp_by_season[s], bins=temp_bins, color="salmon", edgecolor="black")
        plt.title(s)
        plt.xlabel("Surface Temp (°C)")
        plt.ylabel("Count")
    plt.suptitle("Seasonal Surface Temperature Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i, s in enumerate(seasons, 1):
        plt.subplot(2, 2, i)
        plt.hist(dens_by_season[s], bins=dens_bins, color="lightgreen", edgecolor="black")
        plt.title(s)
        plt.xlabel("Surface Density (kg/m³)")
        plt.ylabel("Count")
    plt.suptitle("Seasonal Surface Density Distribution")
    plt.tight_layout()
    plt.show()

    # All seasons on same plot
    # Pressure overlay
    plt.figure(figsize=(8, 6))
    for s in seasons:
        plt.hist(pres_by_season[s], bins=pres_bins, alpha=0.5, label=s, color=colors[s], edgecolor="black")
    plt.xlabel("Surface Pressure (hPa)")
    plt.ylabel("Count")
    plt.title("Overlay: Seasonal Surface Pressure Distribution")
    plt.legend()
    plt.grid()
    plt.show()

    # Temperature overlay
    plt.figure(figsize=(8, 6))
    for s in seasons:
        plt.hist(temp_by_season[s], bins=temp_bins, alpha=0.5, label=s, color=colors[s], edgecolor="black")
    plt.xlabel("Surface Temp (°C)")
    plt.ylabel("Count")
    plt.title("Overlay: Seasonal Surface Temperature Distribution")
    plt.legend()
    plt.grid()
    plt.show()

    # Density overlay
    plt.figure(figsize=(8, 6))
    for s in seasons:
        plt.hist(dens_by_season[s], bins=dens_bins, alpha=0.5, label=s, color=colors[s], edgecolor="black")
    plt.xlabel("Surface Density (kg/m³)")
    plt.ylabel("Count")
    plt.title("Overlay: Seasonal Surface Density Distribution")
    plt.legend()
    plt.grid()
    plt.show()

def monthly_surface_histograms(profiles):
    # Make histograms of surface values grouped by month with aligned bins.
    months = range(1, 13)
    pres_by_month = {m: [] for m in months}
    temp_by_month = {m: [] for m in months}
    dens_by_month = {m: [] for m in months}

    # Collect values by month
    for ds in profiles:
        if (not ds is None) and ("time" in ds):
            month = ds["time.month"].values[0]
            pres_by_month[month].append(ds["PRES_hPa"].values[0][0])
            temp_by_month[month].append(ds["TEMP_C"].values[0][0])
            dens_by_month[month].append(ds["Density"].values[0][0])

    # Pressure bins 
    all_pres = np.concatenate([pres_by_month[m] for m in months if pres_by_month[m]])
    pres_min, pres_max = np.min(all_pres), np.max(all_pres)
    pres_bins = np.arange(np.floor(pres_min/1)*1, np.ceil(pres_max/1)*1 + 1, 1)

    # Temperature bins 
    all_temp = np.concatenate([temp_by_month[m] for m in months if temp_by_month[m]])
    temp_min, temp_max = np.min(all_temp), np.max(all_temp)
    temp_bins = np.arange(np.floor(temp_min/0.5)*0.5, np.ceil(temp_max/0.5)*0.5 + 0.5, 0.5)

    # Density bins 
    all_dens = np.concatenate([dens_by_month[m] for m in months if dens_by_month[m]])
    dens_min, dens_max = np.min(all_dens), np.max(all_dens)
    dens_bins = np.arange(np.floor(dens_min*100)/100, np.ceil(dens_max*100)/100 + 0.01, 0.01)

    # --- Pressure plots ---
    plt.figure(figsize=(12, 8))
    for m in months:
        plt.subplot(3, 4, m)
        plt.hist(pres_by_month[m], bins=pres_bins, color="skyblue", edgecolor="black")
        plt.title(f"Month {m}")
        plt.xlabel("Surface Pressure (hPa)")
        plt.ylabel("Count")
    plt.suptitle("Monthly Surface Pressure Distribution")
    plt.tight_layout()
    plt.show()

    # --- Temperature plots ---
    plt.figure(figsize=(12, 8))
    for m in months:
        plt.subplot(3, 4, m)
        plt.hist(temp_by_month[m], bins=temp_bins, color="salmon", edgecolor="black")
        plt.title(f"Month {m}")
        plt.xlabel("Surface Temp (°C)")
        plt.ylabel("Count")
    plt.suptitle("Monthly Surface Temperature Distribution")
    plt.tight_layout()
    plt.show()

    # --- Density plots ---
    plt.figure(figsize=(12, 8))
    for m in months:
        plt.subplot(3, 4, m)
        plt.hist(dens_by_month[m], bins=dens_bins, color="lightgreen", edgecolor="black")
        plt.title(f"Month {m}")
        plt.xlabel("Surface Density (kg/m³)")
        plt.ylabel("Count")
    plt.suptitle("Monthly Surface Density Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load all profiles (keep original vertical resolution)
    # result is a list of xarray Datasets, with different heights
    profiles = process_filtered_from_path(PATH, filter="*.dat")  # empty filter grabs all .dat
    if not profiles:
        raise RuntimeError("No valid profiles loaded. Exiting.")

    # Interpolate onto common grid
    interp_profiles = vertical_interpolate_profiles(profiles)
    # For each profile, save a netCDF file with day and month in the filename
    for i, ds in enumerate(interp_profiles):
        if "time" in ds.coords:
            # day_str and month_str should be derived from the filename, which is the format '82332_YYYY_MM_DD_... .dat'
            day_str = str(ds.time.values[0])[5:10]
            filename = f"radiosonde_interp_2015-{day_str}.nc"
            print(f"Saving profile {i} to {filename}...")
            ds.to_netcdf(os.path.join(PATH_OUT, filename))
        else:
            print(f"Profile {i} has no time coordinate. Skipping save.")


    # Checking if interp_profiles is for 1 day or multiple days
    #print(f"Number of interpolated profiles: {len(interp_profiles)}")
    # Combine into one xarray Dataset
    #interp_profiles.to_netcdf(r"daily1_avg.nc")
    

    

    