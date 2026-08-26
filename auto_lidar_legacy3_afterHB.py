 # FIX: Take one lidar file and run the inversion 3 times. Run once with the annual average, once with the seasonal average and once with the monthly average.
 # Compare te inversion for the aerosal properties depending on which average you use.

#Save as an xarray saying that datetime varies. Then save as a netcdf. Go to the lidar script and check through the data to see if the AOD is there
#Define a time window +-1 hour from the lidar measurement
#Right now save the data so I can read it in the other script. Modify the script that does Newtons method (Michaels code).

import os
import glob
import sys
#import math
#import calendar
#from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#from scipy.optimize import curve_fit

from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular

# def season_label_from_month(m):
#     #Return 'DJF','MAM','JJA','SON' from month 1-12
#     if m in [12, 1, 2]:
#         return "DJF"
#     if m in [3, 4, 5]:
#         return "MAM"
#     if m in [6, 7, 8]:
#         return "JJA"
#     return "SON"

# def load_climo_profile(climo_type, ref_time):

#     # Load a (pres_hPa, temp_C, height_m) 1D profile from the specified climatology.
#     # climo_type in {'annual','seasonal','monthly'}.

#     ts = pd.Timestamp(ref_time)

#     if climo_type == "annual":
#         ds = xr.load_dataset(ANNUAL_CLIMO)
#         return(ds) 

#     elif climo_type == "seasonal":
#         season = season_label_from_month(ts.month)
#         ds = xr.load_dataset(SEASONAL_CLIMO)
#         return(ds.sel(season=season))

#     elif climo_type == "monthly":
#         month = ts.month
#         ds = xr.load_dataset(MONTHLY_CLIMO)
#         return(ds.sel(month=month))

#     else:
#         raise ValueError("climo_type must be one of {'annual','seasonal','monthly'}")

#     return pres, temp, height

def build_molecular(*,sonde_pres_hPa, sonde_temp_C, sonde_height_m, 
                    lidar_ground_m, lidar_range_m, lidar_zenith_deg, wavelength_nm):

    # Convert the (pres_hPa, temp_C, height_m) profile to molecular α, β on the lidar range grid.
    # Accounts for slant path: r = (z - z0) / cos(theta).

    # data from the sonde
    p_Pa = sonde_pres_hPa * 100.0
    T_K = sonde_temp_C + 273.15
    # slant range for the sonde data
    #29-oct-2025 fix: we should subtract the lidar ground altitude, not the sonde ground altitude.
    #r = (height_m - height_m[0]) / np.cos(np.deg2rad(zenith_deg))
    r_m = (sonde_height_m - lidar_ground_m) / np.cos(np.deg2rad(lidar_zenith_deg))

    df_mol = AlphaBetaMolecular(p_air=p_Pa, t_air=T_K, rangebin=r_m, 
                                wavelength=wavelength_nm).get_params(new_rangebin=lidar_range_m)

    return df_mol

def run_klett_constrained_by_aod(signal_processed, rangebin, df_mol, myzref, tau_target):

    # Binary search the lidar ratio so that the integrated extinction matches tau_target.

    LR1, LR2 = 5.0, 150.0
    TAU1 = Klett(signal=signal_processed, rangebin=rangebin,
                 molecular_data=df_mol, molecular_reference_region=myzref,
                 lidar_ratio=LR1).fit()[-1]
    TAU2 = Klett(signal=signal_processed, rangebin=rangebin,
                 molecular_data=df_mol, molecular_reference_region=myzref,
                 lidar_ratio=LR2).fit()[-1]

    # stop if tau_target is insane (avoid endless while)
    if not np.isfinite(tau_target) or tau_target <= 0:
        raise ValueError("tau_target must be finite and > 0 for AOD constrained inversion.")

    # Binary search

    # need to check: tau_target is the AOD from aeronet which is vertical
    while abs((TAU2 - TAU1) / tau_target) > 0.001:
        LRm = 0.5 * (LR1 + LR2)
        # this is actually optical thickness (i.e. along the lidar slant path)
        TAUm = Klett(signal=signal_processed, rangebin=rangebin,
                     molecular_data=df_mol, molecular_reference_region=myzref,
                     lidar_ratio=LRm).fit()[-1]
        if TAUm > tau_target:
            LR2, TAU2 = LRm, TAUm
        else:
            LR1, TAU1 = LRm, TAUm

    LR_opt = 0.5 * (LR1 + LR2)
    alpha_aer, beta_aer, TAU_opt = Klett(
        signal=signal_processed, rangebin=rangebin,
        molecular_data=df_mol, molecular_reference_region=myzref,
        lidar_ratio=LR_opt
    ).fit()

    return alpha_aer, beta_aer, TAU_opt, LR_opt

def run_hourly_klett_inversion(*, lidar, df_mol, myzref, lr_target=-1, tau_target=-1):

    # Runs hourly Klett inversion (clear-sky windows) for a given day and a given df_mol.
    # Returns an xarray.Dataset with time x range variables and metadata.
    
    hourly = {
        "time": [], "n_good_profiles": [], "n_profiles": [],
        "tau": [], "lr": [], "fixed_lr": [],
        "signal_processed": [], "alpha_aer": [], "beta_aer": [],
        "alpha_mol": [], "beta_mol": []
    }

    rng = lidar.range.values
    ref_min, ref_max = myzref
    ref_mask_global = (rng >= ref_min) & (rng <= ref_max)
    if ref_mask_global.sum() < 10:
        print(f"[warn] Ref band [{ref_min},{ref_max}] has <10 bins on this grid. Will still try per-hour checks.")


    for hour in range(24):
        base_day = pd.to_datetime(str(lidar.time[0].values)).normalize()
        current_time = base_day + pd.to_timedelta(hour, "h")
        time_mask = (lidar.time.dt.hour == hour)

        # Clear-sky screening
        if "sci" not in lidar.variables:
            print("Missing 'sci' var; skipping hour.")
            continue
        sci_hour = lidar.sci.sel(time=time_mask)
        clear_fraction = (sci_hour == 0).sum().item() / sci_hour.size
        if clear_fraction < 0.9:
            print(f"[warn] Hour {hour:02d}: only {clear_fraction*100:.1f}% clear profiles; skipping hour.")
            continue

        # Additional cloud screening using cbh
        cbh_selected = lidar.cbh.sel(time=time_mask)
        no_clouds_mask = (cbh_selected == -1).all(dim="layer")
        tmp = no_clouds_mask.values.astype(int)
        for i in range(1, len(tmp) - 1):
            if not tmp[i-1] or not tmp[i+1]:
                no_clouds_mask[i] = False

        if no_clouds_mask.sum() < 10:
            print(f"[warn] Hour {hour:02d}: <10 clear profiles after CBH screening; skipping hour.")
            continue
        
        # Prepare inversion
        signal_selected = lidar.beta_raw.sel(time=time_mask)
        signal_no_clouds = signal_selected.where(no_clouds_mask, drop=True)
        signal_avg = signal_no_clouds.mean(dim="time")
        signal_processed = signal_avg / (lidar.range ** 2)
        sig_arr = signal_processed.values

        # Reference-region check
        ref_mask = (rng >= ref_min) & (rng <= ref_max)
        if ref_mask.sum() < 10:
            continue

        # Copy and lightly clean model backscatter for reference region
        model_beta = df_mol.beta.data.copy()
        mfin = np.isfinite(model_beta)
        if (~mfin).any():
            ii = np.where(mfin)[0]
            if ii.size >= 2:
                model_beta[~mfin] = np.interp(np.where(~mfin)[0], ii, model_beta[ii])

        # Require both signal and model to be finite in the reference band
        ok_ref = ref_mask & np.isfinite(sig_arr) & np.isfinite(model_beta)
        if ok_ref.sum() < 10:
            continue

        # Inversion
        if lr_target > 0:
            alpha_aer, beta_aer, TAU = Klett(
                signal=sig_arr,
                rangebin=rng,
                molecular_data=df_mol,
                molecular_reference_region=[ref_min, ref_max],
                lidar_ratio=lr_target
            ).fit()
            used_lr = lr_target
            fixed_lr = 1
            used_tau = TAU
        elif tau_target > 0:
            alpha_aer, beta_aer, used_tau, used_lr = run_klett_constrained_by_aod(
                signal_processed=sig_arr,
                rangebin=rng,
                df_mol=df_mol,
                myzref=[ref_min, ref_max],
                tau_target=tau_target
            )
            fixed_lr = 0
        else:
            print("Neither lr_target nor tau_target set; set one and re-run.")
            return None

        # Store
        hourly["time"].append(current_time)
        hourly["n_good_profiles"].append(no_clouds_mask.sum().item())
        hourly["n_profiles"].append(len(no_clouds_mask))
        hourly["tau"].append(used_tau)
        hourly["lr"].append(used_lr)
        hourly["fixed_lr"].append(fixed_lr)
        hourly["signal_processed"].append(sig_arr)
        hourly["alpha_aer"].append(alpha_aer)
        hourly["beta_aer"].append(beta_aer)
        hourly["alpha_mol"].append(df_mol.alpha.data)
        hourly["beta_mol"].append(df_mol.beta.data)

    if len(hourly["time"]) == 0:
        print("No valid hours for this scenario.")
        return None

    ds = xr.Dataset(
        data_vars={
            "signal_processed": (("time", "range"), np.array(hourly["signal_processed"])),
            "alpha_aer": (("time", "range"), np.array(hourly["alpha_aer"])),
            "beta_aer": (("time", "range"), np.array(hourly["beta_aer"])),
            "alpha_mol": (("time", "range"), np.array(hourly["alpha_mol"])),
            "beta_mol": (("time", "range"), np.array(hourly["beta_mol"])),
            "tau": ("time", np.array(hourly["tau"])),
            "lr": ("time", np.array(hourly["lr"])),
            "fixed_lr": ("time", np.array(hourly["fixed_lr"])),
            "n_good_profiles": ("time", np.array(hourly["n_good_profiles"])),
            "n_profiles": ("time", np.array(hourly["n_profiles"])),
        },
        coords={
            "time": np.array(hourly["time"]),
            "range": rng
        }
    )
    return ds


def lidar_analysis(ds_lidar, ds_sonde, params):
    '''For one lidar dataset and one sonde, perform Kleet'''

    df_mol = build_molecular(
        sonde_pres_hPa=ds_sonde.PRES_hPa, 
        sonde_temp_C=ds_sonde.TEMP_C, 
        sonde_height_m=ds_sonde.height, 
        lidar_ground_m=params["GROUND_M"],
        lidar_range_m=ds_lidar.range, 
        lidar_zenith_deg=params["ZENITH_DEG"], 
        wavelength_nm=params["WAVELENGTH_NM"]
        )

    #FIX: compare lidar ratios to true AOD using Newtons method like Michael did. Look at the file look at the data we need and read the data with a script. 
    # With any given time that we have lidar data go to the tale and say do we have the right values withing 2 hours. Make a time series of the AOD with that plot.
    # New goal make the code automaticall choose the right AOD.
    # Run inversions

    ds_out = run_hourly_klett_inversion(
        lidar=ds_lidar,
        df_mol=df_mol,
        myzref=[params["REF_MIN"], params["REF_MAX"]],
        lr_target=params["LR_TARGET"],
        tau_target=params["TAU_TARGET"]
    )
    if ds_out is None:
        print(f"WARN: No output produced!")

    return(ds_out)

# def compare_day_three_climos(out1, out2, out, params):
#     #else:
#     #    ds_out.to_netcdf(os.path.join(out_dir, f"{climo_type}_inversion.nc"))
#     #    print(f"Saved: {os.path.join(out_dir, f'{climo_type}_inversion.nc')}")

#       # Combine the three inversion results (annual, seasonal, monthly) into one file to easily compare them later
#     scen_names = list(results.keys())
#     aligned = xr.concat([results[s] for s in scen_names], dim="scenario")
#     aligned = aligned.assign_coords(scenario=("scenario", scen_names))
#     combined_path = os.path.join(out_dir, "combined_inversions.nc")
#     aligned.to_netcdf(combined_path)
#     print(f"Saved combined: {combined_path}")

#     # 1) Overlaid α and β at a chosen hour (nearest)
#     def _plot_profile_overlay(varname, title):
#         plt.figure(figsize=(6, 8))
#         for s in scen_names:
#             ds = results[s]
#             if "time" not in ds.coords or ds.sizes["time"] == 0:
#                 continue
#             # pick closest time to PLOT_HOUR
#             times = pd.to_datetime(ds.time.values)
#             idx = np.argmin(np.abs(times.hour - PLOT_HOUR))
#             y = ds["range"].values
#             x = ds[varname].isel(time=idx).values
#             m = y <= PLOT_MAX_RANGE
#             plt.plot(x[m], y[m], label=s)
#         plt.xlabel(f"{varname} (units vary)")
#         plt.ylabel("Range (m)")
#         plt.title(f"{title} (nearest {PLOT_HOUR:02d}:00)")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
    
#     _plot_profile_overlay("alpha_aer", "Aerosol Extinction")
#     _plot_profile_overlay("beta_aer", "Aerosol Backscatter")

#     # 2) Time series of LR and TAU (if any scenario has them)
#     def _plot_timeseries(varname, ylabel):
#         plt.figure(figsize=(8, 4))
#         for s in scen_names:
#             ds = results[s]
#             if ds is None or ds.sizes.get("time", 0) == 0:
#                 continue
#             t = pd.to_datetime(ds.time.values)
#             v = ds[varname].values
#             plt.plot(t, v, marker="o", label=s)
#         plt.xlabel("Time")
#         plt.ylabel(ylabel)
#         plt.title(f"{ylabel} time series")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     _plot_timeseries("lr", "Lidar Ratio (sr)")
#     _plot_timeseries("tau", "AOD from Klett")

def find_one_ceilometer_file(path, year, month, day):
    #Pick the first lidar NetCDF for the given date.
    mon_path = os.path.join(path, f"{year:04d}", f"{month:02d}")
    nc_files = sorted(glob.glob(os.path.join(mon_path, f"{year:04d}{month:02d}{day:02d}*.nc")))
    if not nc_files:
        return None
    else: 
        if len(nc_files) > 1:
            print(f"[WARN] Multiple ceilometer files found for {year}-{month:02d}-{day:02d}, using first: {nc_files[0]}")
    return nc_files[0]

if __name__ == "__main__":

    #BASE = r"c:\DATA"
    BASE = r"/storage/jediah/"
    BASE_CEILO = os.path.join(BASE, "ceilometer")
    OUT_BASE = os.path.join(BASE, "output")

    # Climatology files produced by climatology script
    #ANNUAL_CLIMO = r"c:\DATA\sonde\annual_avg.nc"     # dims: height
    #SEASONAL_CLIMO = r"c:\DATA\sonde\seasonal_avg.nc" # dims: season, height
    #MONTHLY_CLIMO = r"c:\DATA\sonde\monthly_avg.nc"   # dims: month, height
    ANNUAL_CLIMO = os.path.join(BASE, "sonde", "annual_avg.nc")     # dims: height
    SEASONAL_CLIMO = os.path.join(BASE, "sonde", "seasonal_avg.nc") # dims: season, height
    MONTHLY_CLIMO = os.path.join(BASE, "sonde", "monthly_avg.nc")   # dims: month, height

    # Reference region for calibration 
    PARAM = {"REF_MIN": 6000, "REF_MAX": 14000}

    # Fixed lidar ratio example (set >0 to use fixed LR path, or set TAU_TARGET>0 for AOD constrained)
    # for set-16 2017 try using TAU=0.261
    # https://aeronet.gsfc.nasa.gov/cgi-bin/data_display_aod_v3?site=Manaus_EMBRAPA&nachal=0&year=2017&month=9&day=16&aero_water=0&level=2&if_day=0&if_err=0&place_code=10&year_or_month=0
    PARAM.update({"LR_TARGET": 60.0, "TAU_TARGET": -1 })

    # Lidar geometry, At Embrapa site
    PARAM.update({"GROUND_M": 100.0, "ZENITH_DEG": 25.0, "WAVELENGTH_NM": 1064.})

    # For plotting
    PARAM["PLOT_MAX_RANGE"] = 8000  # m
    PARAM["PLOT_HOUR"] = 12         # local hour to overlay α/β

    # Pick a day we know has lidar data
    YEAR, MONTH, DAY, SEASON = 2017, 9, 16, "SON"
    PARAM["YEAR"] = YEAR
    PARAM["MONTH"] = MONTH
    PARAM["DAY"] = DAY
    PARAM["SEASON"] = SEASON

    ds_sonde_annual = xr.load_dataset(ANNUAL_CLIMO)
    ds_sonde_seasonal = xr.load_dataset(SEASONAL_CLIMO).sel(season=SEASON)
    ds_sonde_month = xr.load_dataset(MONTHLY_CLIMO).sel(month=MONTH)

    # Finds the name of the first lidar file for that day
    lidar_file = find_one_ceilometer_file(BASE_CEILO, YEAR, MONTH, DAY)

    if lidar_file is None:
        print(f"No lidar files found for {YEAR}-{MONTH:02d}-{DAY:02d}.")
        sys.exit(0)

    ds_lidar = xr.load_dataset(lidar_file)

    out_dir = os.path.join(OUT_BASE, f"{YEAR:04d}", f"{MONTH:02d}")
    os.makedirs(out_dir, exist_ok=True)

    print("Using lidar:", lidar_file)
    print("Output dir :", out_dir)

    # Safety: don’t allow both constraints at once
    if PARAM["LR_TARGET"] > 0 and PARAM["TAU_TARGET"] > 0:
        print("Choose either fixed LR_TARGET or TAU_TARGET (not both).")
        sys.exit(1)

    output_annual = lidar_analysis(ds_lidar, ds_sonde_annual, PARAM)
    output_seasonal = lidar_analysis(ds_lidar, ds_sonde_seasonal, PARAM)
    output_month = lidar_analysis(ds_lidar, ds_sonde_month, PARAM)

    #
    # tput_annual, output_seasonal, output_month)

# eof file