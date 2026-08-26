import os
import glob
import sys
import re
import zipfile
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular

# JM: I removed _find_aeronet_header_line and just wrote it in read_aeronet_directsun since it's only used within this function
def read_aeronet_directsun(filepath, *, aod_wavelength_nm=1020):
    # Find the header row
    with open(filepath, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if ("Date(" in line and "Time(" in line) or ("Date" in line and "Time" in line and "," in line):
                header = i
                break
        else:
            raise ValueError(f"Could not find AERONET header row in: {filepath}")

    df = pd.read_csv(filepath, skiprows=header, engine="python")
    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
    time_col = next((c for c in df.columns if c.lower().startswith("time")), None)
    if not date_col or not time_col:
        raise ValueError(f"Could not identify Date/Time columns. Got: {df.columns[:20].tolist()}")

    dt = pd.to_datetime(
        df[date_col].str.strip() + " " + df[time_col].str.strip(),
        format="%d:%m:%Y %H:%M:%S", errors="coerce"
    )
    bad = dt.isna()
    if bad.any():
        dt.loc[bad] = pd.to_datetime(
            df.loc[bad, date_col].str.strip() + " " + df.loc[bad, time_col].str.strip(),
            format="%d:%m:%Y %H:%M", errors="coerce"
        )

    df["time"] = dt
    df = df.dropna(subset=["time"])

    # Pick AOD column for desired wavelength.
    wl = int(aod_wavelength_nm)
    aod_col = next(
        (c for c in [f"AOD_{wl}nm", f"AOD_{wl}", f"AOD_{wl}nm-Total", f"AOD_{wl}-Total"] if c in df.columns),
        None
    )
    if aod_col is None:
        matches = [c for c in df.columns if re.match(rf"^AOD_{wl}(\D.*)?$", c, re.IGNORECASE)]
        aod_col = matches[0] if matches else None
    if aod_col is None:
        aod_cols = [c for c in df.columns if c.upper().startswith("AOD_")]
        raise ValueError(f"No AOD column for {wl} nm in {filepath}. Available: {aod_cols[:25]}")

    out = df[["time", aod_col]].rename(columns={aod_col: "aod"}).copy()
    out["aod"] = pd.to_numeric(out["aod"], errors="coerce")
    return out.dropna(subset=["aod"]).sort_values("time").reset_index(drop=True)

# JM: I removed load_aeronet_years_directsun and wrote it directly in main, since it's only used once there
def build_molecular(*, sonde_pres_hPa, sonde_temp_C, sonde_height_m,
                    lidar_ground_m, lidar_range_m, lidar_zenith_deg, wavelength_nm):
    # Convert the (pres_hPa, temp_C, height_m) profile to molecular α, β on the lidar range grid.

    #29-oct-2025 fix: we should subtract the lidar ground altitude, not the sonde ground altitude.
    r_m = (sonde_height_m - lidar_ground_m) / np.cos(np.deg2rad(lidar_zenith_deg))
    return AlphaBetaMolecular(
        p_air=sonde_pres_hPa * 100.0,
        t_air=sonde_temp_C + 273.15,
        rangebin=r_m,
        wavelength=wavelength_nm
    ).get_params(new_rangebin=lidar_range_m)


def run_klett_constrained_by_aod(signal_processed, rangebin, df_mol, myzref, tau_target, zenith_deg):
    # Binary search the lidar ratio so that the integrated extinction matches tau_target.
    # JM: Created a helper function to shorten the Klett calls.
    def klett_tau(lr):
        return Klett(signal=signal_processed, rangebin=rangebin,
                     molecular_data=df_mol, molecular_reference_region=myzref,
                     lidar_ratio=lr).fit()[-1]

    LR1, LR2 = 5.0, 150.0
    TAU1, TAU2 = klett_tau(LR1), klett_tau(LR2)

    if not np.isfinite(tau_target) or tau_target <= 0:
        raise ValueError("tau_target must be finite and > 0.")

    tau_slant = tau_target / np.cos(np.deg2rad(zenith_deg))

    while abs((TAU2 - TAU1) / tau_slant) > 0.001:
        LRm = 0.5 * (LR1 + LR2)
        # this is actually optical thickness (i.e. along the lidar slant path)
        TAUm = klett_tau(LRm)
        if TAUm > tau_slant:
            LR2, TAU2 = LRm, TAUm
        else:
            LR1, TAU1 = LRm, TAUm

    LR_opt = 0.5 * (LR1 + LR2)
    alpha_aer, beta_aer, TAU_opt, coef_slope = Klett(
        signal=signal_processed, rangebin=rangebin,
        molecular_data=df_mol, molecular_reference_region=myzref,
        lidar_ratio=LR_opt
    ).fit()
    return alpha_aer, beta_aer, TAU_opt, LR_opt, coef_slope # Klett should also returns coef_slope


def run_hourly_klett_inversion(*, lidar, df_mol, myzref, lr_target=-1, tau_target=-1, zenith_deg=25.0):
    # Runs hourly Klett inversion (clear-sky windows) for a given day and a given df_mol.
    # Returns an xarray.Dataset with time x range variables and metadata.
    print(" HMJB: run_hourly_klett_inversion ", lr_target, tau_target)
    hourly = {
        "time": [], "n_good_profiles": [], "n_profiles": [],
        "tau": [], "lr": [], "fixed_lr": [],
        "signal_processed": [], "alpha_aer": [], "beta_aer": [],
        "alpha_mol": [], "beta_mol": [], "coef_slope": [] 
    }

    rng = lidar.range.values
    ref_min, ref_max = myzref
    ref_mask = (rng >= ref_min) & (rng <= ref_max)

    if ref_mask.sum() < 10:
        print(f"[warn] Ref band [{ref_min},{ref_max}] has <10 bins on this grid.")

    if "sci" not in lidar.variables:
        print("Missing 'sci' var; cannot process.")
        return None

    base_day = pd.to_datetime(str(lidar.time[0].values)).normalize()

    for hour in range(24):
        current_time = base_day + pd.to_timedelta(hour, "h")
        print(f"\n=== Hour {hour:02d} | {base_day.strftime('%Y-%m-%d')} ===")

        time_mask = lidar.time.dt.hour == hour
        sci_hour = lidar.sci.sel(time=time_mask)
        cbh_selected = lidar.cbh.sel(time=time_mask)

        no_clouds_mask = (cbh_selected == -1).all(dim="layer") & (sci_hour == 0)
        tmp = no_clouds_mask.values.astype(int)
        for i in range(1, len(tmp) - 1):
            if not tmp[i-1] or not tmp[i+1]:
                no_clouds_mask[i] = False

        if no_clouds_mask.sum() < 10:
            print(f"[warn] Hour {hour:02d}: <10 clear profiles; skipping.")
            continue

        sig_arr = lidar.beta_raw.sel(time=time_mask).where(no_clouds_mask, drop=True).mean(dim="time").values

        # Signal sanity checks
        # JM: Removed unecessary checks
        fin = np.isfinite(sig_arr)

        if fin.sum() < 100:
            print(f"[warn] Hour {hour:02d}: too few finite signal bins; skipping.")
            continue

        if (ref_mask & fin).sum() < 10:
            print(f"[warn] Hour {hour:02d}: not enough finite bins in reference region; skipping.")
            continue


        ref_vals = sig_arr[ref_mask]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        # require the reference region to be mostly positive
        if ref_vals.size < 10 or np.mean(ref_vals > 0) < 0.7:
            print(f"[warn] Hour {hour:02d}: reference region quality check failed; skipping.")
            continue

        # Clean molecular backscatter for reference region
        model_beta = df_mol.beta.data.copy()
        fin = np.isfinite(model_beta)
        if (~fin).any(): # ~fin means if any values are NOT finite
            ii = np.where(fin)[0]
            if ii.size >= 2:
                model_beta[~fin] = np.interp(np.where(~fin)[0], ii, model_beta[ii])

        if (ref_mask & np.isfinite(sig_arr) & np.isfinite(model_beta)).sum() < 10:
            continue

        # Inversion (nneves: now it also stores the calibration coefficient coef_slope for each hour, we can check if it is stable across hours)
        if lr_target > 0:
            print(" HMJB: lr_target is positive")
            alpha_aer, beta_aer, used_tau, coef_slope = Klett(
                signal=sig_arr, rangebin=rng, molecular_data=df_mol,
                molecular_reference_region=[ref_min, ref_max], lidar_ratio=lr_target
            ).fit()
            used_lr, fixed_lr = lr_target, 1
        else:
            print(" HMJB: lr_target is not positive, doing AOD constrained inversion")
            # if lr<0, then we are doing the AOD constrained inversion
            if isinstance(tau_target, pd.DataFrame):
                # Select AERONET points that fall within this hour 
                aod_this_hour = tau_target.loc[
                    (tau_target["time"] >= current_time) &
                    (tau_target["time"] < current_time + pd.Timedelta(hours=1)), "aod"
                ]
                # If no AOD values for this hour, skip to next hour
                if aod_this_hour.empty:
                    print(f"[info] Hour {hour:02d}: no AERONET AOD values; skipping.")
                    continue
                tau_1h = aod_this_hour.mean()
            else:
                tau_1h = float(tau_target)

            # hmjb: need to double check that used_tau is vertical or slanted #nneves: included in next steps
            alpha_aer, beta_aer, used_tau, used_lr, coef_slope = run_klett_constrained_by_aod(
                signal_processed=sig_arr, rangebin=rng, df_mol=df_mol,
                myzref=[ref_min, ref_max], tau_target=tau_1h, zenith_deg=zenith_deg
            )
            fixed_lr = 0

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
        hourly["coef_slope"].append(coef_slope)

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
            "coef_slope": (("time", "parameter"), np.array(hourly["coef_slope"])), # Provide a tuple of two dimensions to match the 2D array
        },
        coords={
            "time": np.array(hourly["time"]),
            "range": rng,
            "parameter": ["a", "b"] 
        }
    )
    return ds


def lidar_analysis(ds_lidar, ds_sonde, params):
    df_mol = build_molecular(
        sonde_pres_hPa=ds_sonde.PRES_hPa,
        sonde_temp_C=ds_sonde.TEMP_C,
        sonde_height_m=ds_sonde.height,
        lidar_ground_m=params["GROUND_M"],
        lidar_range_m=ds_lidar.range,
        lidar_zenith_deg=params["ZENITH_DEG"],
        wavelength_nm=params["WAVELENGTH_NM"]
    )
    ds_out = run_hourly_klett_inversion(
        lidar=ds_lidar, df_mol=df_mol,
        myzref=[params["REF_MIN"], params["REF_MAX"]],
        lr_target=params["LR_TARGET"],
        tau_target=params["TAU_TARGET"]
    )
    if ds_out is None:
        print("WARN: No output produced!")
    return ds_out


def iter_netcdf_members_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_members = sorted(n for n in zf.namelist() if n.lower().endswith(".nc"))
        if not nc_members:
            raise FileNotFoundError(f"No .nc files found inside {zip_path}")
        for nc_name in nc_members:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
            tmp_path = tmp.name
            tmp.close()
            with zf.open(nc_name) as src, open(tmp_path, "wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
            yield nc_name, tmp_path


def plot_aeronet_overview(df_aod, start_date, end_date, title="AERONET AOD (local time)"):
    dfp = df_aod[(df_aod["time"] >= start_date) & (df_aod["time"] < end_date + pd.Timedelta(days=1))]
    if dfp.empty:
        print("[warn] No AERONET data in requested date range.")
        return
    plt.figure(figsize=(10, 4))
    plt.plot(dfp["time"], dfp["aod"], marker=".", linestyle="none")
    plt.xlabel("Time"); plt.ylabel("AOD"); plt.title(title)
    plt.grid(True); plt.tight_layout(); plt.show()


def plot_alpha_profiles(ds_out, max_range_m=8000, label="annual"):
    if ds_out is None or ds_out.sizes.get("time", 0) == 0:
        return
    rng = ds_out["range"].values.astype(float)
    m = rng <= max_range_m
    plt.figure(figsize=(6, 8))
    for i, ti in enumerate(pd.to_datetime(ds_out["time"].values)):
        a = ds_out["alpha_aer"].isel(time=i).values.astype(float)
        if np.isfinite(a[m]).sum() >= 10:
            plt.plot(a[m], rng[m], label=ti.strftime("%Y-%m-%d %H:%M"))
    plt.xlabel("Aerosol extinction α (1/m)"); plt.ylabel("Range (m)")
    plt.title(f"α profiles [{label}]"); plt.grid(True)
    plt.legend(title="Hour", fontsize=8); plt.tight_layout(); plt.show()

# JM: Removed list_monthly_zip_files function and wrote it directly in the main block
if __name__ == "__main__":
    BASE = r"/storage/jediah"
    BASE_CEILO = os.path.join(BASE, "ceilometer")
    OUT_BASE = os.path.join(BASE, "output")
    # FIX: remember that now this script is for processing the ceilometer data when we have aod values from aeronet
    # From the climatology script (analyze_sonde6) we noticed no major variations between the different climatologies, so we're using the annual average.
    ANNUAL_CLIMO = os.path.join(BASE, "sonde", "annual_avg.nc")
    
    AERONET_2016 = r"aeronet/20160101_20161231_Manaus_EMBRAPA_level15.directsun"
    AERONET_2017 = r"aeronet/20170101_20171231_Manaus_EMBRAPA_level15.directsun"
    # JM: Concat combines the two AERONET files into a single dataframe with all the rows from both
    # JM: fp is the file path of each AERONET file
    df_aod = pd.concat(
        [read_aeronet_directsun(fp, aod_wavelength_nm=1020) for fp in [AERONET_2016, AERONET_2017]],
        ignore_index=True
    )
    # df_aod["time"] -= pd.Timedelta(hours=4)  # UTC → Manaus local (UTC-4)
    print(f"[info] Loaded {len(df_aod)} AERONET AOD rows")

    PARAM = {
        # Fixed lidar ratio vs AOD-constrained:
        #  - set LR_TARGET > 0 and TAU_TARGET < 0 for fixed LR
        #  - set LR_TARGET < 0 and TAU_TARGET > 0 for AOD-constrained path 
        "REF_MIN": 6000, "REF_MAX": 14000,
        "LR_TARGET": 50, "TAU_TARGET": -1,
        "GROUND_M": 100.0, "ZENITH_DEG": 25.0, "WAVELENGTH_NM": 1064.0,
        "PLOT_MAX_RANGE": 8000,
    }

    # Safety: don’t allow both constraints at once
    if PARAM["LR_TARGET"] > 0 and PARAM["TAU_TARGET"] > 0:
        print("Choose either fixed LR_TARGET or TAU_TARGET (not both).")
        sys.exit(1)

    # load climotologies
    ds_sonde_annual = xr.load_dataset(ANNUAL_CLIMO)

    zip_files = sorted(glob.glob(os.path.join(BASE_CEILO, "*.nc.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No *.nc.zip files found in {BASE_CEILO}")

    print(f"[info] Found {len(zip_files)} monthly zip files")

    for ZIP_PATH in zip_files:
        yyyymm = os.path.basename(ZIP_PATH)[:6]
        out_dir = os.path.join(OUT_BASE, yyyymm)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Processing {yyyymm} ===")

        monthly_outputs = []
        # Loop over each daily netcdf inside the zip
        for member_name, tmp_path in iter_netcdf_members_in_zip(ZIP_PATH):
            ds_lidar = None
            try:
                print("Day file:", member_name)
                ds_lidar = xr.open_dataset(tmp_path, engine="netcdf4")
                output_annual = lidar_analysis(ds_lidar, ds_sonde_annual, PARAM)
                if output_annual is not None and output_annual.sizes.get("time", 0) > 0:
                    monthly_outputs.append(output_annual)
            except Exception as e:
                print(f"[ERROR] {member_name}: {repr(e)}")
            finally:
                if ds_lidar is not None:
                    ds_lidar.close()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if len(monthly_outputs) == 0:
            print("[WARN] No valid outputs for month", yyyymm)
            continue

        ds_month = xr.concat(monthly_outputs, dim="time").sortby("time")
        out_nc = os.path.join(out_dir, f"{yyyymm}_klett_annual.nc")
        ds_month.to_netcdf(out_nc)
        print(f"[info] Saved: {out_nc} ({ds_month.sizes['time']} time points)")

        print(f"\n=== Plotting {yyyymm} ===")
        start_date = pd.to_datetime(ds_month.time.min().values).normalize()
        end_date   = pd.to_datetime(ds_month.time.max().values).normalize()
        plot_aeronet_overview(df_aod, start_date, end_date, title=f"AERONET AOD {yyyymm} (local)")
        plot_alpha_profiles(ds_month, max_range_m=PARAM["PLOT_MAX_RANGE"], label=yyyymm)

    print(f"\nAll done. Processed {len(zip_files)} monthly zips.")