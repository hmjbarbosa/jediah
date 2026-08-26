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


# NEW: Individual daily radiosonde reading

def compute_density(pressure_hPa, temp_C):
    """Compute air density using the ideal gas law."""
    R_d = 287.05
    p = pressure_hPa * 100.0
    T = temp_C + 273.15
    return p / (R_d * T)


def read_radiosonde_fixed_width(lines):
    """Parse one radiosonde profile (list of lines from a .dat file) into an xr.Dataset."""
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
            tmp = line.split(":")[-1].strip()
            year = int(tmp[:2]) + 1900
            month = int(tmp[2:4])
            day = int(tmp[4:6])
            hour = int(tmp[7:9])
            minute = int(tmp[9:11])
            observation_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

    if start_idx is None or end_idx is None:
        return None

    data_lines = lines[start_idx:end_idx]

    colnames = [
        "PRES_hPa", "HGHT_m", "TEMP_C", "DWPT_C", "RELH_pct",
        "MIXR_gkg", "DRCT_deg", "SKNT_knot",
        "THTA_K", "THTE_K", "THTV_K"
    ]

    df = pd.read_fwf(pd.io.common.StringIO("".join(data_lines)), names=colnames)
    df.dropna(how="all", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["PRES_hPa", "HGHT_m", "TEMP_C"], inplace=True)

    if df.empty:
        return None

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
    return ds


def check_profile_quality(ds):
    """Same QC rules used to build the climatologies. Assumes a single-time Dataset."""
    problems = []
    if ds is None:
        problems.append("Dataset is None")
        return None, problems

    if len(ds.height) < 5:
        problems.append(f"Not enough data points: {len(ds.height)}")
        return None, problems

    surf_h = ds["height"].values[0]
    surf_p = ds["PRES_hPa"].values[0, 0]
    surf_t = ds["TEMP_C"].values[0, 0]

    if not (990 <= surf_p <= 1020):
        problems.append(f"Unrealistic surface pressure: {surf_p:.1f} hPa")
        return None, problems

    if surf_h != 84:
        problems.append(f"Wrong surface altitude: {surf_h:.1f} m")
        return None, problems

    if not (10 <= surf_t <= 40):
        problems.append(f"Unrealistic surface temperature: {surf_t:.1f} °C")
        return None, problems

    hdiff = np.diff(ds["height"].values)
    bad_idx = np.where((hdiff <= 0))[0]

    if len(bad_idx) > 1:
        problems.append(f"Too many problems with the altitude values: {len(bad_idx)} (non-monotonic)")
        return None, problems
    elif len(bad_idx) == 1:
        if hdiff[bad_idx[0]] < -10:
            problems.append(f"1 non-monotonic altitude, difference too large: {hdiff[bad_idx[0]]:.1f} m")
            return None, problems
        else:
            problems.append(f"Removed {len(bad_idx)} suspect level(s) (non-monotonic)")
            drop_heights = ds["height"].values[bad_idx + 1]
            ds = ds.sel(height=~np.isin(ds["height"], drop_heights))

    if sorted(ds["height"].values)[0] != surf_h:
        problems.append("weird low altitude value!")
        return None, problems

    max_h = ds["height"].max()
    if max_h < 15000:
        problems.append(f"Profile too shallow: max height {max_h:.0f} m")

    return ds, problems


def get_season(month):
    """Return season label from month (matches the climatology script's grouping)."""
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"


def load_sonde_file(fpath):
    """Parse and QC a single radiosonde .dat file, given its full path.

    Returns an xr.Dataset (time=1, height=N) or None if the file fails QC.
    Note: unlike the old per-date matching version, this doesn't care what
    year the file is from -- the lidar signal we pair it with is fixed and
    doesn't need to correspond to the same date.
    """
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    ds = read_radiosonde_fixed_width(lines)
    ds, problems = check_profile_quality(ds)
    if problems:
        print(f"[warn] quality check notes for {os.path.basename(fpath)}: {problems}")
    if ds is None:
        return None
    return ds


def build_molecular(*, sonde_pres_hPa, sonde_temp_C, sonde_height_m,
                    lidar_ground_m, lidar_range_m, lidar_zenith_deg, wavelength_nm):
    # Convert the (pres_hPa, temp_C, height_m) profile to molecular α, β on the lidar range grid.
    # .reshape(-1) makes this work whether the input came from a climatology file
    # (already 1-D over height) or from an individual daily sonde Dataset, which
    # still carries a length-1 "time" dimension: (1, height) -> (height,).
    sonde_pres_hPa = np.asarray(sonde_pres_hPa).reshape(-1)
    sonde_temp_C = np.asarray(sonde_temp_C).reshape(-1)
    sonde_height_m = np.asarray(sonde_height_m).reshape(-1)

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
    return alpha_aer, beta_aer, TAU_opt, LR_opt, coef_slope


def run_hourly_klett_inversion(*, lidar, df_mol, myzref, lr_target=-1, tau_target=-1, zenith_deg=25.0):
    # Runs hourly Klett inversion (clear-sky windows) for a given day and a given df_mol.
    # Returns an xarray.Dataset with time x range variables and metadata.
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

        time_mask = lidar.time.dt.hour == hour
        sci_hour = lidar.sci.sel(time=time_mask)
        cbh_selected = lidar.cbh.sel(time=time_mask)

        no_clouds_mask = (cbh_selected == -1).all(dim="layer") & (sci_hour == 0)
        tmp = no_clouds_mask.values.astype(int)
        for i in range(1, len(tmp) - 1):
            if not tmp[i-1] or not tmp[i+1]:
                no_clouds_mask[i] = False

        if no_clouds_mask.sum() < 10:
            continue

        sig_arr = lidar.beta_raw.sel(time=time_mask).where(no_clouds_mask, drop=True).mean(dim="time").values

        fin = np.isfinite(sig_arr)
        if fin.sum() < 100:
            continue
        if (ref_mask & fin).sum() < 10:
            continue

        ref_vals = sig_arr[ref_mask]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if ref_vals.size < 10 or np.mean(ref_vals > 0) < 0.7:
            continue

        model_beta = df_mol.beta.data.copy()
        fin = np.isfinite(model_beta)
        if (~fin).any():
            ii = np.where(fin)[0]
            if ii.size >= 2:
                model_beta[~fin] = np.interp(np.where(~fin)[0], ii, model_beta[ii])

        if (ref_mask & np.isfinite(sig_arr) & np.isfinite(model_beta)).sum() < 10:
            continue

        if lr_target > 0:
            alpha_aer, beta_aer, used_tau, coef_slope = Klett(
                signal=sig_arr, rangebin=rng, molecular_data=df_mol,
                molecular_reference_region=[ref_min, ref_max], lidar_ratio=lr_target
            ).fit()
            used_lr, fixed_lr = lr_target, 1
        else:
            if isinstance(tau_target, pd.DataFrame):
                aod_this_hour = tau_target.loc[
                    (tau_target["time"] >= current_time) &
                    (tau_target["time"] < current_time + pd.Timedelta(hours=1)), "aod"
                ]
                if aod_this_hour.empty:
                    continue
                tau_1h = aod_this_hour.mean()
            else:
                tau_1h = float(tau_target)

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
        return None

    return xr.Dataset(
        data_vars={
            "signal_processed": (("time", "range"), np.array(hourly["signal_processed"])),
            "alpha_aer":        (("time", "range"), np.array(hourly["alpha_aer"])),
            "beta_aer":         (("time", "range"), np.array(hourly["beta_aer"])),
            "alpha_mol":        (("time", "range"), np.array(hourly["alpha_mol"])),
            "beta_mol":         (("time", "range"), np.array(hourly["beta_mol"])),
            "tau":              ("time", np.array(hourly["tau"])),
            "lr":               ("time", np.array(hourly["lr"])),
            "fixed_lr":         ("time", np.array(hourly["fixed_lr"])),
            "n_good_profiles":  ("time", np.array(hourly["n_good_profiles"])),
            "n_profiles":       ("time", np.array(hourly["n_profiles"])),
            "coef_slope": (("time", "coef_idx"), np.array(hourly["coef_slope"])),
        },
        coords={"time": np.array(hourly["time"]), "range": rng}
    )


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



def get_clear_sky_hour_signal(ds_lidar, ref_min, ref_max, preferred_hour=None):
    """Scan a single-day lidar Dataset's 24 hours (or just `preferred_hour` if
    given) and return the first hour whose clear-sky averaged signal passes
    the same quality checks used in run_hourly_klett_inversion. Returns a
    dict with timestamp/hour/signal/range, or None if nothing qualifies.
    """
    rng = ds_lidar.range.values
    ref_mask = (rng >= ref_min) & (rng <= ref_max)

    if "sci" not in ds_lidar.variables:
        return None

    base_day = pd.to_datetime(str(ds_lidar.time[0].values)).normalize()
    hours_to_try = [preferred_hour] if preferred_hour is not None else range(24)

    for hour in hours_to_try:
        time_mask = ds_lidar.time.dt.hour == hour
        sci_hour = ds_lidar.sci.sel(time=time_mask)
        cbh_selected = ds_lidar.cbh.sel(time=time_mask)
        if sci_hour.sizes.get("time", 0) == 0:
            continue

        no_clouds_mask = (cbh_selected == -1).all(dim="layer") & (sci_hour == 0)
        tmp = no_clouds_mask.values.astype(int)
        for i in range(1, len(tmp) - 1):
            if not tmp[i-1] or not tmp[i+1]:
                no_clouds_mask[i] = False

        if no_clouds_mask.sum() < 10:
            continue

        sig_arr = ds_lidar.beta_raw.sel(time=time_mask).where(no_clouds_mask, drop=True).mean(dim="time").values
        fin = np.isfinite(sig_arr)
        if fin.sum() < 100:
            continue
        if (ref_mask & fin).sum() < 10:
            continue
        ref_vals = sig_arr[ref_mask]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if ref_vals.size < 10 or np.mean(ref_vals > 0) < 0.7:
            continue

        return {"timestamp": base_day + pd.to_timedelta(hour, "h"), "hour": hour,
                "signal": sig_arr, "range": rng}

    return None


def find_reference_lidar_hour(zip_path, ref_min, ref_max,
                              day_override=None, hour_override=None):
    """Open one exact ceilometer ZIP and return a valid fixed lidar hour.

    The clear-sky mask and signal-quality tests remain inside
    get_clear_sky_hour_signal() and are not changed here.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Ceilometer ZIP not found: {zip_path}")

    for member_name, tmp_path in iter_netcdf_members_in_zip(zip_path):
        ds_lidar = None
        try:
            ds_lidar = xr.open_dataset(tmp_path, engine="netcdf4")

            if ds_lidar.sizes.get("time", 0) == 0:
                print(f"[skip] {member_name}: empty time dimension.")
                continue

            member_day = pd.to_datetime(ds_lidar.time.values[0]).strftime("%Y%m%d")

            # A six-digit override such as "201611" allows every day in the
            # month. An eight-digit override such as "20161105" selects one day.
            if day_override and not member_day.startswith(day_override):
                continue

            found = get_clear_sky_hour_signal(
                ds_lidar,
                ref_min,
                ref_max,
                preferred_hour=hour_override,
            )

            if found is not None:
                found["zip_path"] = zip_path
                found["member_name"] = member_name
                print(
                    f"[info] Reference lidar hour selected: "
                    f"{found['timestamp']} "
                    f"(from {os.path.basename(zip_path)} / {member_name})"
                )
                return found

            print(
                f"[info] {member_name}: no qualifying clear-sky hour, "
                "trying next day."
            )

        finally:
            if ds_lidar is not None:
                ds_lidar.close()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return None


def run_single_profile_klett(*, signal, rangebin, df_mol, myzref,
                              lr_target=-1, tau_target=-1, zenith_deg=25.0, reference_timestamp=None):
    """One Klett fit for a single fixed signal + single molecular profile."""
    ref_min, ref_max = myzref
    if lr_target > 0:
        alpha_aer, beta_aer, used_tau, coef_slope = Klett(
            signal=signal, rangebin=rangebin, molecular_data=df_mol,
            molecular_reference_region=[ref_min, ref_max], lidar_ratio=lr_target
        ).fit()
        used_lr, fixed_lr = lr_target, 1
    else:
        if isinstance(tau_target, pd.DataFrame):
            if reference_timestamp is None:
                raise ValueError("reference_timestamp is required when tau_target is a DataFrame.")
            aod_this_hour = tau_target.loc[
                (tau_target["time"] >= reference_timestamp) &
                (tau_target["time"] < reference_timestamp + pd.Timedelta(hours=1)), "aod"
            ]
            if aod_this_hour.empty:
                return None
            tau_1h = aod_this_hour.mean()
        else:
            tau_1h = float(tau_target)

        alpha_aer, beta_aer, used_tau, used_lr, coef_slope = run_klett_constrained_by_aod(
            signal_processed=signal, rangebin=rangebin, df_mol=df_mol,
            myzref=[ref_min, ref_max], tau_target=tau_1h, zenith_deg=zenith_deg
        )
        fixed_lr = 0

    return {"alpha_aer": alpha_aer, "beta_aer": beta_aer, "tau": used_tau,
            "lr": used_lr, "fixed_lr": fixed_lr, "coef_slope": coef_slope}


def run_fixed_signal_sonde_sensitivity(*, reference_hour, ds_monthly_avg, ds_seasonal_avg, ds_annual_avg,
                                        params, station_id, sonde_dir, sonde_year, target_month):
    rng = reference_hour["range"]
    sig = reference_hour["signal"]
    ts = reference_hour["timestamp"]
    season = get_season(target_month)

    sonde_pattern = os.path.join(
        sonde_dir,
        f"{station_id}_{sonde_year:04d}_{target_month:02d}_*_00Z.dat"
    )
    sonde_files = sorted(glob.glob(sonde_pattern))
    print(
        f"[info] Found {len(sonde_files)} candidate radiosondes for "
        f"{sonde_year:04d}-{target_month:02d}"
    )
    if not sonde_files:
        raise FileNotFoundError(f"No radiosondes matched: {sonde_pattern}")

    sonde_labels, alpha_ens, beta_ens = [], [], []
    for fpath in sonde_files:
        ds_sonde = load_sonde_file(fpath)
        if ds_sonde is None:
            print(f"[skip] {os.path.basename(fpath)}: failed the quality check.")
            continue

        df_mol = build_molecular(
            sonde_pres_hPa=ds_sonde.PRES_hPa, sonde_temp_C=ds_sonde.TEMP_C, sonde_height_m=ds_sonde.height,
            lidar_ground_m=params["GROUND_M"], lidar_range_m=rng, lidar_zenith_deg=params["ZENITH_DEG"],
            wavelength_nm=params["WAVELENGTH_NM"]
        )
        fit = run_single_profile_klett(
            signal=sig, rangebin=rng, df_mol=df_mol, myzref=[params["REF_MIN"], params["REF_MAX"]],
            lr_target=params["LR_TARGET"], tau_target=params["TAU_TARGET"],
            zenith_deg=params["ZENITH_DEG"], reference_timestamp=ts
        )
        if fit is None:
            print(f"[skip] {os.path.basename(fpath)}: no AOD available for the reference hour.")
            continue

        sonde_labels.append(os.path.basename(fpath))
        alpha_ens.append(fit["alpha_aer"])
        beta_ens.append(fit["beta_aer"])

    print(f"[info] Sondes used in ensemble: {len(sonde_labels)} / {len(sonde_files)}")
    if not sonde_labels:
        print("[WARN] No usable individual sondes. Nothing to compare.")
        return None

    alpha_ens = np.array(alpha_ens)  # (n_sondes, n_range)
    beta_ens = np.array(beta_ens)

    # Same fixed signal, run once each for the monthly / seasonal / annual mean sonde
    climo_profiles = {}
    for label, ds_clim_full, sel_kwargs in [
        ("monthly", ds_monthly_avg, {"month": target_month}),
        ("seasonal", ds_seasonal_avg, {"season": season}),
        ("annual", ds_annual_avg, {}),
    ]:
        ds_clim = ds_clim_full.sel(**sel_kwargs) if sel_kwargs else ds_clim_full
        df_mol_clim = build_molecular(
            sonde_pres_hPa=ds_clim.PRES_hPa, sonde_temp_C=ds_clim.TEMP_C, sonde_height_m=ds_clim.height,
            lidar_ground_m=params["GROUND_M"], lidar_range_m=rng, lidar_zenith_deg=params["ZENITH_DEG"],
            wavelength_nm=params["WAVELENGTH_NM"]
        )
        fit = run_single_profile_klett(
            signal=sig, rangebin=rng, df_mol=df_mol_clim, myzref=[params["REF_MIN"], params["REF_MAX"]],
            lr_target=params["LR_TARGET"], tau_target=params["TAU_TARGET"],
            zenith_deg=params["ZENITH_DEG"], reference_timestamp=ts
        )
        if fit is None:
            print(f"[warn] {label} mean-sonde run produced no result for the reference hour.")
            fit = {"alpha_aer": np.full_like(rng, np.nan), "beta_aer": np.full_like(rng, np.nan)}
        climo_profiles[label] = fit

    result = xr.Dataset(
        data_vars={
            "alpha_altogether": (("sonde", "range"), alpha_ens),
            "beta_altogether": (("sonde", "range"), beta_ens),
            "alpha_individual_mean": ("range", np.nanmean(alpha_ens, axis=0)),
            "alpha_individual_std": ("range", np.nanstd(alpha_ens, axis=0, ddof=1)),
            "beta_individual_mean": ("range", np.nanmean(beta_ens, axis=0)),
            "beta_individual_std": ("range", np.nanstd(beta_ens, axis=0, ddof=1)),
            "alpha_monthly_mean_profile": ("range", climo_profiles["monthly"]["alpha_aer"]),
            "beta_monthly_mean_profile": ("range", climo_profiles["monthly"]["beta_aer"]),
            "alpha_seasonal_mean_profile": ("range", climo_profiles["seasonal"]["alpha_aer"]),
            "beta_seasonal_mean_profile": ("range", climo_profiles["seasonal"]["beta_aer"]),
            "alpha_annual_mean_profile": ("range", climo_profiles["annual"]["alpha_aer"]),
            "beta_annual_mean_profile": ("range", climo_profiles["annual"]["beta_aer"]),
        },
        coords={"sonde": np.array(sonde_labels), "range": rng},
        attrs={
            "description": "Sonde sensitivity study with a FIXED lidar signal (one clear-sky hour "
                            "of one day) and individual radiosonde molecular profiles swapped in one "
                            "at a time, compared against the same fixed signal inverted with the "
                            "monthly/seasonal/annual mean sonde.",
            "reference_timestamp": str(ts),
            "station_id": station_id,
            "sonde_year": sonde_year,
            "target_month": target_month,
            "season": season,
            "n_sondes_used": len(sonde_labels),
        }
    )

    for variable in ["alpha", "beta"]:
        individual_mean = result[f"{variable}_individual_mean"]
        individual_std = result[f"{variable}_individual_std"]
        for label in ["monthly", "seasonal", "annual"]:
            comparison = result[f"{variable}_{label}_mean_profile"]
            difference = comparison - individual_mean
            result[f"{variable}_{label}_minus_individual_mean"] = difference
            result[f"{variable}_{label}_absolute_difference"] = np.abs(difference)
            result[f"{variable}_{label}_difference_in_std"] = xr.where(
                individual_std > 0,
                difference / individual_std,
                np.nan,
            )

    return result


def plot_comparison(result, max_range_m=8000, out_png=None):
    if result is None:
        return
    rng = result["range"].values
    m = rng <= max_range_m

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    for ax, var, xlabel in [
        (axes[0], "alpha", "Aerosol extinction α (1/m)"),
        (axes[1], "beta", "Aerosol backscatter β (1/m/sr)"),
    ]:
        mean = result[f"{var}_individual_mean"].values
        std = result[f"{var}_individual_std"].values
        ax.plot(mean[m], rng[m], color="black", lw=1.8, label="Individual sonde mean")
        ax.plot(result[f"{var}_monthly_mean_profile"].values[m], rng[m], label="November monthly mean", lw=1.5)
        ax.plot(result[f"{var}_seasonal_mean_profile"].values[m], rng[m], label="SON seasonal mean", lw=1.5)
        ax.plot(result[f"{var}_annual_mean_profile"].values[m], rng[m], label="Annual mean", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.grid(True)

    axes[0].set_ylabel("Range (m)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"November 2016 lidar hour: {result.attrs.get('reference_timestamp')}\n"
        f"November {result.attrs.get('sonde_year')} radiosondes, "
        f"n={result.attrs.get('n_sondes_used')}"
    )
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"[info] Saved: {out_png}")
    plt.close()


# Main
if __name__ == "__main__":
    BASE = r"c:\DATA"
    BASE_CEILO = os.path.join(BASE, "ceilo_zips")
    LIDAR_ZIP = os.path.join(
        BASE_CEILO, "201611_EMBRAPA_CHM160147.nc.zip"
    )
    OUT_BASE = os.path.join(BASE, "output_SURF")

    STATION_ID = "82332"
    SONDE_DAT_DIR = r"c:\DATA\sonde\82332\dat"

    ANNUAL_CLIMO = os.path.join(BASE, "sonde", "annual_avg.nc")
    MONTHLY_CLIMO = os.path.join(BASE, "sonde", "monthly_avg.nc")
    SEASONAL_CLIMO = os.path.join(BASE, "sonde", "seasonal_avg.nc")

    # Use each daily November 2015 radiosonde, such as: C:\DATA\sonde\82332\dat\82332_2015_11_01_00Z.dat
    SONDE_YEAR = 2015
    TARGET_MONTH = 11

    # Choode which day to be fixed
    LIDAR_DAY_OVERRIDE = "201611"
    LIDAR_HOUR_OVERRIDE = None

    PARAM = {
        "REF_MIN": 6000, "REF_MAX": 14000,
        "LR_TARGET": 50, "TAU_TARGET": -1,
        "GROUND_M": 100.0, "ZENITH_DEG": 25.0, "WAVELENGTH_NM": 1064.0,
        "PLOT_MAX_RANGE": 8000,
    }

    if PARAM["LR_TARGET"] > 0 and PARAM["TAU_TARGET"] > 0:
        print("Choose either fixed LR_TARGET or TAU_TARGET (not both).")
        sys.exit(1)

    ds_annual_avg = xr.load_dataset(ANNUAL_CLIMO)
    ds_monthly_avg = xr.load_dataset(MONTHLY_CLIMO)
    ds_seasonal_avg = xr.load_dataset(SEASONAL_CLIMO)

    if not os.path.isfile(LIDAR_ZIP):
        raise FileNotFoundError(f"Ceilometer ZIP not found: {LIDAR_ZIP}")

    reference_hour = find_reference_lidar_hour(
        LIDAR_ZIP,
        PARAM["REF_MIN"],
        PARAM["REF_MAX"],
        day_override=LIDAR_DAY_OVERRIDE,
        hour_override=LIDAR_HOUR_OVERRIDE,
    )
    if reference_hour is None:
        raise RuntimeError(
            "The November ZIP was opened, but no hour passed the existing "
            "clear-sky and reference-region quality checks. Review the "
            "[info] messages printed for each daily NetCDF member."
        )

    out_dir = os.path.join(
        OUT_BASE,
        f"november_{SONDE_YEAR}_fixed_2016_lidar_hour"
    )
    os.makedirs(out_dir, exist_ok=True)

    result = run_fixed_signal_sonde_sensitivity(
        reference_hour=reference_hour,
        ds_monthly_avg=ds_monthly_avg,
        ds_seasonal_avg=ds_seasonal_avg,
        ds_annual_avg=ds_annual_avg,
        params=PARAM,
        station_id=STATION_ID,
        sonde_dir=SONDE_DAT_DIR,
        sonde_year=SONDE_YEAR,
        target_month=TARGET_MONTH,
    )

    if result is not None:
        out_nc = os.path.join(out_dir, "november_sonde_comparison.nc")
        result.to_netcdf(out_nc)
        print(f"[info] Saved: {out_nc}")

        rows = []
        range_mask = result["range"] <= PARAM["PLOT_MAX_RANGE"]
        for variable in ["alpha", "beta"]:
            individual_mean = result[f"{variable}_individual_mean"].where(range_mask, drop=True)
            individual_std = result[f"{variable}_individual_std"].where(range_mask, drop=True)
            for label in ["monthly", "seasonal", "annual"]:
                comparison = result[f"{variable}_{label}_mean_profile"].where(range_mask, drop=True)
                difference = comparison - individual_mean
                rows.append({
                    "variable": variable,
                    "comparison": label,
                    "n_individual_sondes": result.attrs["n_sondes_used"],
                    "mean_individual_std": float(individual_std.mean(skipna=True)),
                    "mean_difference": float(difference.mean(skipna=True)),
                    "mean_absolute_difference": float(np.abs(difference).mean(skipna=True)),
                    "rmse": float(np.sqrt((difference ** 2).mean(skipna=True))),
                })
        metrics = pd.DataFrame(rows)
        metrics_path = os.path.join(out_dir, "november_comparison_metrics.csv")
        metrics.to_csv(metrics_path, index=False)
        print(f"[info] Saved: {metrics_path}")
        print(metrics.to_string(index=False))

        plot_comparison(
            result, max_range_m=PARAM["PLOT_MAX_RANGE"],
            out_png=os.path.join(out_dir, "november_sonde_comparison.png")
        )

    print("\nDone.")