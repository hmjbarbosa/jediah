#import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import TextIOWrapper
import glob
import re 
import xarray as xr
import os
import zipfile

plt.ion()

# Modified the file to open the IGRA radiosonde files, instead of the 
# files downloaded from the Wyoming site.

# there is 1 ZIP file per sonde side. Each ZIP contains one TXT file, 
# with all sounding one after the other

#PATH = "./82332/dat/"

PATH = '/storage/DATA/www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por'

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

def read_radiosonde_igra_format(lines):
    # Parse one radiosonde profile in IGRA format into an xarray.Dataset.
    # Header format is fixed width with the following columns: 
    # ---------------------------------
    # HEADREC       1-  1  Character = "#"
    # ID            2- 12  Character
    # YEAR         14- 17  Integer
    # MONTH        19- 20  Integer
    # DAY          22- 23  Integer
    # HOUR         25- 26  Integer
    # RELTIME      28- 31  Integer written as HHMM
    # NUMLEV       33- 36  Integer
    # P_SRC        38- 45  Character
    # NP_SRC       47- 54  Character
    # LAT          56- 62  Integer
    # LON          64- 71  Integer
    # ---------------------------------
    header_line = lines[0]
    station_id = header_line[1:12].strip()
    year = int(header_line[13:17])
    month = int(header_line[18:20]) 
    day = int(header_line[21:23])
    hour = int(header_line[24:26])
    relhour = int(header_line[27:29])  # RELTIME in HHMM format
    relminute = int(header_line[29:31])  # RELTIME in HHMM format
    numlev = int(header_line[32:36])
    p_src = header_line[37:45].strip()
    np_src = header_line[46:54].strip()
    lat = int(header_line[55:62]) / 1000.0  # convert to degrees
    lon = int(header_line[63:71]) / 1000.0  # convert to degrees
    rel_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0)

    
    # this is followed by NUMLEV data records, each fixed width with the following columns: 
    # -------------------------------
    # LVLTYP1         1-  1   Integer
    # LVLTYP2         2-  2   Integer
    # ETIME           4-  8   Integer
    # PRESS          10- 15   Integer
    # PFLAG          16- 16   Character
    # GPH            17- 21   Integer
    # ZFLAG          22- 22   Character
    # TEMP           23- 27   Integer
    # TFLAG          28- 28   Character
    # RH             29- 33   Integer
    # DPDP           35- 39   Integer
    # WDIR           41- 45   Integer
    # WSPD           47- 51   Integer
    # -------------------------------
    data_lines = lines[1:numlev + 1]
    colnames = [
        "LVLTYP1", "LVLTYP2", "ETIME", "PRES_hPa", "PFLAG",
        "HGHT_m", "ZFLAG", "TEMP_C", "TFLAG", "RELH_pct",
        "DWPT_C", "WDIR_deg", "WSPD_knot"]
    colwidths = [1, 1, 5, 6, 1, 5, 1, 5, 1, 5, 5, 5, 5]
    
    df = pd.read_fwf(pd.io.common.StringIO("".join(data_lines)), names=colnames, widths=colwidths)
    # convert -8888 and -9999 to NaN
    df.replace([-8888, -9999], np.nan, inplace=True)
    # replace A with 1, B with 2, and blank with -1
    df["PFLAG"].replace({"A": 1, "B": 2, " ": -1}, inplace=True)
    df["ZFLAG"].replace({"A": 1, "B": 2, " ": -1}, inplace=True)
    df["TFLAG"].replace({"A": 1, "B": 2, " ": -1}, inplace=True)

    # exclude rows with all NaNs
    df.dropna(how="all", inplace=True)

    # convert to numeric, coerce errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # remove lines with missing essential data in the important columns
    #df.dropna(subset=["PRES_hPa", "HGHT_m", "TEMP_C"], inplace=True)
    # attach timestamp to whole profile
    observation_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0)

    # convert pd.DataFrame to xarray.Dataset including all variables from the data records
    # and including all header info as attributes
    ds = xr.Dataset(
        { 
            "PRES_hPa": (("time", "height"), df["PRES_hPa"].values[np.newaxis, :]),
            "TEMP_C": (("time", "height"), df["TEMP_C"].values[np.newaxis, :]),
            "Density": (("time", "height"), compute_density(df["PRES_hPa"].values, df["TEMP_C"].values)[np.newaxis, :]),
            "DWPT_C": (("time", "height"), df["DWPT_C"].values[np.newaxis, :]),
            "RELH_pct": (("time", "height"), df["RELH_pct"].values[np.newaxis, :]),
            "HGHT_m": (("time", "height"), df["HGHT_m"].values[np.newaxis, :]),
            "WDIR_deg": (("time", "height"), df["WDIR_deg"].values[np.newaxis, :]),
            "WSPD_knot": (("time", "height"), df["WSPD_knot"].values[np.newaxis, :]),
            "REL_HOUR": (("time"), relhour),
            "REL_MINUTE": (("time"), relminute),
            "NUMLEV": (("time"), numlev),
        },
        coords={
            "height": df["HGHT_m"].values,
            "time": [observation_time],
        },
        attrs={
            "station_id": station_id,
            "latitude": lat,
            "longitude": lon,
        }

    

def process_zipped_from_path(path, station, filter="*.zip"):
    profiles = []

    filelist = sorted(glob.glob(os.path.join(path, station + filter)))
    print(f"Found {len(filelist)} files for station: {station}")

    for fname in filelist:
        with zipfile.ZipFile(fname, 'r') as zf:
            # for IGRA, there is only one file per zip
            # however, we still loop over all files in the zip to be more general
            for zippedfile in zf.namelist():
                with zf.open(zippedfile) as f:
                    # Need to wrap binary file in TextIOWrapper for text reading
                    lines = TextIOWrapper(f, encoding='utf-8', errors='ignore').readlines()

                    # the issue with IGRA files is that there are multiple sonde launches
                    # in the same text file, one after the other. 

                    # we need to split the lines into chunks, one per sonde launch
                    # each sonde launch starts with a line containing "#station"
                    launch_indices = [i for i, line in enumerate(lines) if line.startswith("#station")]
                    launch_indices.append(len(lines))  # add end of file index

                    for start, end in zip(launch_indices[:-1], launch_indices[1:]):

                        ds = read_radiosonde_igra_format(lines[start:end])
                        # basic quality checks
                        #ds, problems = check_profile_quality(ds)
                        #if problems:
                        #    print(f"Issues in {fname}: {problems}")
                        #else:
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

    STNID = 'ZZXUAICE031'  # example station ID

    # Load all profiles (keep original vertical resolution)
    # result is a list of xarray Datasets, with different heights
    profiles = process_zipped_from_path(PATH, STNID)
    if not profiles:
        raise RuntimeError("No valid profiles loaded. Exiting.")

    # Interpolate onto common grid
    interp_profiles = vertical_interpolate_profiles(profiles)

    # Merge interpolated datasets into one big dataset
    combined_ds = xr.concat(interp_profiles, dim="time")

def notnow():
    # Seasonal averages
    # This returns three xarray Datasets, each with the 4 seasons inside
    seas_avg, seas_std, seas_count = seasonal_profiles(combined_ds)
    # can we achieve the same if we calll 
    # .... = average_profiles(combined_ds.groupby("time.season")) ?
    seas_avg.to_netcdf(r"seasonal_avg.nc")
    seas_std.to_netcdf(r"seasonal_std.nc")
    seas_count.to_netcdf(r"seasonal_count.nc")

    # Compute annual mean
    # this function averages everything over time. Since we are 
    # passing the whole dataset, it is the "annual average". 
    # however, if we need, we could pass a subset of the dataset 
    # and average that instead. 
    avg, std, count = average_profiles(combined_ds)

    # Save to NetCDF
    #annual_ds.to_netcdf(r"c:\DATA\sonde\annual.nc")
    avg.to_netcdf(r"annual_avg.nc")
    std.to_netcdf(r"annual_std.nc")
    count.to_netcdf(r"annual_count.nc")

    print("Saved annual climatology.")

    # compute monthly means
    monthly_avg, monthly_std, monthly_count = average_profiles(combined_ds.groupby("time.month"))
    monthly_avg.to_netcdf(r"monthly_avg.nc")
    monthly_std.to_netcdf(r"monthly_std.nc")
    monthly_count.to_netcdf(r"monthly_count.nc")

    # =============================================================================

    # Plot Temperature vs Height (seasonal)
    plt.figure(num=1, clear='True', figsize=(6, 8))
    for ss in seas_avg.season.values:
        plt.plot(seas_avg.sel(season=ss)["TEMP_C"], seas_avg.height, label=ss)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title("Seasonal Avg Temperature vs Height")
    plt.grid()
    plt.show()

    # Same for Density
    plt.figure(num=2, clear='True', figsize=(6, 8))
    for ss in seas_avg.season.values:
        plt.plot(seas_avg.sel(season=ss)["Density"], seas_avg.height, label=ss)
    plt.xlabel("Air Density (kg/m³)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title("Seasonal Avg Density vs Height")
    plt.grid()
    plt.show()

    # Same for Pressure
    plt.figure(num=3, clear=True, figsize=(6, 8))
    for ss in seas_avg.season.values:
        plt.plot(seas_avg.sel(season=ss)["PRES_hPa"], seas_avg.height, label=ss)
    plt.xlabel("Pressure (hPa)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title("Seasonal Avg Pressure vs Height")
    plt.grid()
    plt.show()

    # Monthly and seasonal surface histograms
    monthly_surface_histograms(profiles)
    seasonal_surface_histograms(profiles)
