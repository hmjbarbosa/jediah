import os
import glob
import sys
import re
import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import tempfile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import importlib
from lidarpy.inversion import Klett
from lidarpy.molecular import AlphaBetaMolecular
import matplotlib.dates as mdates
showplots = True
if showplots: 
    plt.ion()
    

def _find_aeronet_header_line(filepath):

    with open(filepath, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            # common AERONET header has "Date(dd:mm:yyyy)" and "Time(hh:mm:ss)"
            if ("Date(" in line and "Time(" in line) or ("Date" in line and "Time" in line and "," in line):
                return i
    raise ValueError(f"Could not find AERONET header row in file: {filepath}")

def read_aeronet_directsun(filepath, *, aod_wavelength_nm=1020, tz=None):

    hdr = _find_aeronet_header_line(filepath)

    df = pd.read_csv(
        filepath,
        skiprows=hdr,
        engine="python"
    )

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Find date/time column names
    date_col = None
    time_col = None
    for c in df.columns:
        if c.startswith("Date(") or c.lower().startswith("date"):
            date_col = c
        if c.startswith("Time(") or c.lower().startswith("time"):
            time_col = c

    if date_col is None or time_col is None:
        raise ValueError(f"Could not identify Date/Time columns in {filepath}. Columns: {df.columns.tolist()[:20]} ...")

    # Parse datetime. AERONET Date is usually dd:mm:yyyy and Time hh:mm:ss
    # Build a combined string then parse.
    dt = pd.to_datetime(
        df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
        format="%d:%m:%Y %H:%M:%S",
        errors="coerce"
    )

    # If Time sometimes is hh:mm (no seconds), attempt fallback
    bad = dt.isna()
    if bad.any():
        dt2 = pd.to_datetime(
            df.loc[bad, date_col].astype(str).str.strip() + " " + df.loc[bad, time_col].astype(str).str.strip(),
            format="%d:%m:%Y %H:%M",
            errors="coerce"
        )
        dt.loc[bad] = dt2

    df["time"] = dt
    df = df.dropna(subset=["time"]).copy()

    # Pick AOD column for desired wavelength.
    # AERONET column naming varies a bit, so we try several patterns.
    wl = int(aod_wavelength_nm)
    candidates = [
        f"AOD_{wl}nm",
        f"AOD_{wl}",
        f"AOD_{wl}nm-Total",   # sometimes “-Total” exists
        f"AOD_{wl}-Total",
    ]

    aod_col = None
    for c in candidates:
        if c in df.columns:
            aod_col = c
            break

    # If still not found, try match like AOD_1020
    if aod_col is None:
        pattern = re.compile(rf"^AOD_{wl}(\D.*)?$", re.IGNORECASE)
        matches = [c for c in df.columns if pattern.match(c)]
        if matches:
            aod_col = matches[0]

    if aod_col is None:
        # Helpful debug: show all AOD columns present
        aod_cols = [c for c in df.columns if c.upper().startswith("AOD_")]
        raise ValueError(
            f"Could not find an AOD column for {wl} nm in {filepath}.\n"
            f"Available AOD-like columns include: {aod_cols[:25]}{' ...' if len(aod_cols) > 25 else ''}"
        )

    out = df[["time", aod_col]].rename(columns={aod_col: "aod"}).copy()
    out["aod"] = pd.to_numeric(out["aod"], errors="coerce")
    out = out.dropna(subset=["aod"]).sort_values("time").reset_index(drop=True)

    return out

def load_aeronet_years_directsun(filepaths, *, aod_wavelength_nm=1020):
    """
    Read multiple DIRECTSUN files and return a single tidy df_aod(time, aod).
    """
    frames = []
    for fp in filepaths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"AERONET file not found: {fp}")
        frames.append(read_aeronet_directsun(fp, aod_wavelength_nm=aod_wavelength_nm))

    df_aod = pd.concat(frames, ignore_index=True)
    df_aod = df_aod.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df_aod

def build_molecular(*,sonde_pres_hPa, sonde_temp_C, sonde_height_m, 
                    lidar_ground_m, lidar_range_m, lidar_zenith_deg, wavelength_nm):
    # Convert the (pres_hPa, temp_C, height_m) profile to molecular α, β on the lidar range grid.
    # Accounts for slant path: r = (z - z0) / cos(theta).
    # data from the sonde
    p_Pa = sonde_pres_hPa * 100.0
    T_K = sonde_temp_C + 273.15
    r_m = (sonde_height_m - lidar_ground_m) / np.cos(np.deg2rad(lidar_zenith_deg))

    df_mol = AlphaBetaMolecular(p_air=p_Pa, t_air=T_K, rangebin=r_m, 
                                wavelength=wavelength_nm).get_params(new_rangebin=lidar_range_m)

    return df_mol

def run_klett_constrained_by_aod(signal_processed, rangebin, df_mol, myzref, tau_target, zenith_deg):

    # Binary search the lidar ratio so that the integrated extinction matches tau_target.

    #HMJB. sep 4th 2026 ---  this function is incorretly calling the .fit() 
    # for the AOD comparison, we cannot integrate the extinction starting from the ground because of the overlap. 
    # NEED to check new parameter in .fit(), hmin_aod, to make extinction constant in the bottom 200m. 

    # be careful: Klett computes AOD along the slant path, but tau_target is column (aeronet)
    #Convert AOD to slant path
    tau_target_slant = tau_target / np.cos(np.deg2rad(zenith_deg)) # nneves: Yeah, we're using tau slanted

    LR1, LR2 = 5.0, 150.0
    _, _, TAU1, _ = Klett(signal=signal_processed, rangebin=rangebin,
                 molecular_data=df_mol, molecular_reference_region=myzref,
                 lidar_ratio=LR1, correct_noise=False).fit()
    _, _, TAU2, _ = Klett(signal=signal_processed, rangebin=rangebin,
                 molecular_data=df_mol, molecular_reference_region=myzref,
                 lidar_ratio=LR2, correct_noise=False).fit()
    print(f'TAU1({LR1}sr): {TAU1}, TAU2({LR2}sr): {TAU2}, tau_target_slant: {tau_target_slant}')

    # stop if tau_target is insane (avoid endless while)
    if not np.isfinite(tau_target) or tau_target <= 0:
        raise ValueError("tau_target must be finite and > 0 for AOD constrained inversion.")
    
    # stop if tau_target is outside the range of TAU1 and TAU2 
    # if tau_target_slant < TAU1 or tau_target_slant > TAU2:   #### NNO: Commented this out to get the code running
    #     raise ValueError("tau_target_slant is outside the range of TAU1 and TAU2.") 

    # need to check: tau_target is the AOD from aeronet which is vertical
    while abs((TAU2 - TAU1) / tau_target_slant) > 0.001:
        LRm = 0.5 * (LR1 + LR2)
        # this is actually optical thickness (i.e. along the lidar slant path)
        _, _, TAUm, _ = Klett(signal=signal_processed, rangebin=rangebin,
                    molecular_data=df_mol, molecular_reference_region=myzref,
                    lidar_ratio=LRm, correct_noise=True).fit() # NNO: now we are correcting for noise in the final fit
        
        # NNO: Stopping the loop if TAU1 or the slope from the fit is negative, which indicates a bad molecular fit
        if TAU1 or coef_slope[0] < 0:
            print(f"[warn] Negative TAU or molecular fit slope calculated during AOD-constrained inversion; stopping search.")
            break
        
        if TAUm > tau_target_slant:
            LR2, TAU2 = LRm, TAUm
        else:
            LR1, TAU1 = LRm, TAUm
        print(f'TAU1({LR1}sr): {TAU1}, TAU2({LR2}sr): {TAU2}')

    LR_opt = 0.5 * (LR1 + LR2)
    tmp = Klett(
        signal=signal_processed, rangebin=rangebin,
        molecular_data=df_mol, molecular_reference_region=myzref,
        lidar_ratio=LR_opt, correct_noise=True # NNO: now we are correcting for noise in the final fit
    )
    alpha_aer, beta_aer, TAU_opt, coef_slope = tmp.fit()
    p_mol = tmp.get_model_mol()  # nn: attempt to get the molecular signal for plotting
        
    return alpha_aer, beta_aer, TAU_opt, LR_opt, coef_slope, p_mol # nneves: Klett should also return coef_slope

def run_hourly_klett_inversion(*, lidar, df_mol, myzref, lr_target=None, tau_target=None, zenith_deg=None):

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
    ref_mask_global = (rng >= ref_min) & (rng <= ref_max)
    if ref_mask_global.sum() < 10:
        print(f"[warn] Ref band [{ref_min},{ref_max}] has <10 bins on this grid. Will still try per-hour checks.")

    # we should probably check if the file has just 24h, or more data...
    for hour in range(24):
    #for hour in range(0):
        base_day = pd.to_datetime(str(lidar.time[0].values)).normalize()
        print(f"\n=== Hour {hour:02d} Day {base_day.strftime('%Y-%m-%d')} ===")
        print(" len(hour[coef_slope])", len(hourly["coef_slope"]))
        current_time = base_day + pd.to_timedelta(hour, "h")
        time_mask = (lidar.time.dt.hour == hour)

        # Clear-sky screening
        if "sci" not in lidar.variables:
            print("Missing 'sci' var; skipping hour.")
            continue
        sci_hour = lidar.sci.sel(time=time_mask)
        sci_mask = (sci_hour == 0)

        # Additional cloud screening using cbh
        cbh_selected = lidar.cbh.sel(time=time_mask)
        no_clouds_mask = (cbh_selected == -1).all(dim="layer") & sci_mask
        tmp = no_clouds_mask.values.astype(int)
        for i in range(1, len(tmp) - 1):
            if not tmp[i-1] or not tmp[i+1]:
                no_clouds_mask[i] = False
    
        if no_clouds_mask.sum() < 10: #FIX: Make sure this is saved in the output
            print(f"[warn] Hour {hour:02d}: <10 clear profiles after CBH screening; skipping hour.")
            continue
        
        # Prepare inversion
        signal_selected = lidar.beta_raw.sel(time=time_mask)
        signal_no_clouds = signal_selected.where(no_clouds_mask, drop=True)
        signal_avg = signal_no_clouds.mean(dim="time")
        r2 = (lidar.range ** 2)
        r2 = xr.where(r2 == 0, np.nan, r2)  # avoid divide by zero at range=0
        sig_arr = signal_avg.values  / r2.values

        # Signal sanity checks to avoid divide-by-zero inside Klett
        if not np.isfinite(sig_arr).any():
            print(f"[warn] Hour {hour:02d}: sig_arr has no finite values; skipping.")
            continue

        # Require enough finite points overall
        if np.isfinite(sig_arr).sum() < 100:
            print(f"[warn] Hour {hour:02d}: too few finite bins in sig_arr; skipping.")
            continue

        # Require reference region bins to be finite 
        ref_mask = (rng >= ref_min) & (rng <= ref_max)
        if (ref_mask & np.isfinite(sig_arr)).sum() < 10:
            print(f"[warn] Hour {hour:02d}: not enough finite bins in reference region; skipping.")
            continue

        # Avoid hours where the averaged signal is basically zero everywhere
        #hmjb: 2026-sep-23 I don't think this can ever be triggered because it requires the
        # signal to be ==0 everywhere
        if np.nanmax(np.abs(sig_arr)) <= 0:
            print(f"[warn] Hour {hour:02d}: signal is all zeros; skipping.")
            continue

        # Reference-region check
        ref_mask = (rng >= ref_min) & (rng <= ref_max)
        #hmjb: commented out on 2026-sep-23
        #if ref_mask.sum() < 10:
        #    continue

        ref_vals = sig_arr[ref_mask]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        print(f"[info] Hour {hour:02d}: {len(ref_vals)} finite bins in ref region [{ref_min},{ref_max}] m.")

        if ref_vals.size < 10:
            print(f"[warn] Hour {hour:02d}: too few finite ref bins; skipping.")
            continue

        # require the reference region to be mostly positive
        #HMJB: 2026-sep-23 Jediah and I found cases (2016-12-01 from 10UTC to 12UTC)
        # where the ceilometer background correction seems to be wrong. The signal
        # decreases with altitude, and become totally negative. Like as if the removed BG
        # was too high. Need to investigate this further. 
        # Will we have to check the BG removal that is "automated" in this instrument?!?

        ## TO TRY: REMOVE THIS CHECK, SO THAT THE INVERSION PROCEEDS. 
        ## BUT MODIFY THE MOLECULAR FIT TO USE 2 PARAMETERS A*X + B
        ### YOU WILL NEED TO DOUBLE-CHECK THAT WE ARE USING THE "B" TO CORRECT THE SIGNAL BACKGROUND. 
        ### FOR THE CASES WHERE THE SIGNAL IS NEGATIVE, B WILL BE NEGATIVE... WHICH MEANS THAT
        #### CORRECTED SIGNAL = SIGNAL - B => THIS WILL INCREASE THE SIGNAL, HENCE, IT BECOMES POSITIVE. 
        # pos_frac = np.mean(ref_vals > 0) #Fraction of bins in the reference region where the signal is positive
        # if pos_frac < 0.7:
        #     print(f"[warn] Hour {hour:02d}: ref region mostly non-positive (pos_frac={pos_frac:.2f}); skipping.")
        #     continue 
        # NNO: removing this filter for now

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
            print(f"[warn] Hour {hour:02d}: not enough finite bins in ref region after model check; skipping.")
            continue

        # passed all the tests, now we can do the inversion
        print(f"[info] Hour {hour:02d}: {no_clouds_mask.sum().item()} clear profiles, {ok_ref.sum()} finite bins in ref region; proceeding with inversion.")

        # Inversion (nneves: now it also stores the calibration coefficient coef_slope for each hour, we can check if it is stable across hours)
        if lr_target > 0:
            print(" HMJB: lr_target is positive")
            tmp = Klett(
                signal=sig_arr,
                rangebin=rng,
                molecular_data=df_mol,
                molecular_reference_region=[ref_min, ref_max],
                lidar_ratio=lr_target,
                correct_noise=True
            )

            alpha_aer, beta_aer, TAU, coef_slope = tmp.fit()
            p_mol = tmp.get_model_mol()  # nn: attempt to get the molecular signal for plotting

            #HMJB: if you return None for alpha and beta when the mol-Fit failed, you should test it here: 
            #if alpha_aer is None or Beta_aer is None:
            #   continue

            # NNO: Stopping the inversion if the coefficient slope is negative, which indicates a bad molecular fit
            if coef_slope[0] < 0:
                print(f"[warn] Hour {hour:02d}: negative coefficient slope; skipping hour!.")
                continue

            used_lr = lr_target
            fixed_lr = 1
            used_tau = TAU
        else:
            print(" HMJB: lr_target is not positive, doing AOD constrained inversion")
            # if lr<0, then we are doing the AOD constrained inversion
            # for the new version, tau_target is a dataframe with the times and aod measured by aeronet
            # we have to check here, if there are AOD values for this particular hour/day
            # if there is not, we "continue" to the loop (go to next hour)
            # if there is, then we average all the values within that hour and use that as the tau_target_1h for this hour

            # FIX - check tau_target (two columns: times, aod)
            # you need to average all aod values in the current hour
            # if there are no values, continue
            # if you have values, average them and use that as tau_target_1h

            if isinstance(tau_target, pd.DataFrame):
                print(' HMJB: tau_target is a dataframe, selecting AERONET points for this hour')
                # Select AERONET points that fall within this hour 
                aod_this_hour = tau_target.loc[
                    (tau_target["time"] >= current_time)
                    & (tau_target["time"] < current_time + pd.Timedelta(hours=1)),
                    "aod",
                ]

                # If no AOD values for this hour, skip to next hour
                if aod_this_hour.empty:
                    print(f"[info] Hour {hour:02d}: no AERONET AOD values, skipping hour.")
                    continue

                # Use the mean AOD as tau_target_1h (vertical AOD)
                tau_target_1h = aod_this_hour.mean()
                # sanity check
                if tau_target_1h < 0: 
                    print(f"[warn] Hour {hour:02d}: mean AERONET AOD is negative ({tau_target_1h}).")
                    print(' AOD values in hour: ', aod_this_hour.values)
                    raise ValueError("Mean AERONET AOD is negative, cannot proceed with inversion.")

            else:
                print(' HMJB: tau_target is not a dataframe, using it directly as tau_target_1h')
                # Just in case: treat tau_target as a simple array AOD
                tau_target_1h = float(tau_target)

            # hmjb: need to double check that used_tau is vertical or slanted #nneves: included in next steps
            alpha_aer, beta_aer, used_tau, used_lr, coef_slope, p_mol = run_klett_constrained_by_aod(
                signal_processed=sig_arr,
                rangebin=rng,
                df_mol=df_mol,
                myzref=[ref_min, ref_max],
                tau_target=tau_target_1h,
                zenith_deg=zenith_deg
            )
            fixed_lr = 0

            if coef_slope[0] < 0: # NNO: Stopping the inversion if the coefficient slope is negative (had to add it here again to work)
                print(f"[warn] Hour {hour:02d}: negative coefficient slope; skipping hour!.")
                continue

        #else:
        #    print("Neither lr_target nor tau_target set; set one and re-run.")
        #    return None
        # Write a function to read that AOD from the website 1020nm and make a plot to make sure everything is right
        #Save number of averaged profiles

        # Store
        hourly["time"].append(current_time) # hour start time
        hourly["n_good_profiles"].append(no_clouds_mask.sum().item()) # number of profiles averaged into sig_arr
        hourly["n_profiles"].append(len(no_clouds_mask)) # total number of profiles in this hour (before cloud mask)
        hourly["tau"].append(used_tau)
        hourly["lr"].append(used_lr)
        hourly["fixed_lr"].append(fixed_lr)
        hourly["signal_processed"].append(sig_arr)
        hourly["alpha_aer"].append(alpha_aer)
        hourly["beta_aer"].append(beta_aer)
        hourly["alpha_mol"].append(df_mol.alpha.data)
        hourly["beta_mol"].append(df_mol.beta.data)
        print(" before updating coef_slope")
        print(" len(hour[coef_slope])", len(hourly["coef_slope"]), hourly["coef_slope"])
        hourly["coef_slope"].append(coef_slope) # Store the calibration coefficient
        print(" AFTER updating coef_slope")
        print(" len(hour[coef_slope])", len(hourly["coef_slope"]), hourly["coef_slope"])
        
        print('SHAPE OF coef_slope', np.shape(coef_slope))

        # For this hour: Plot P (real signal) x P_mol (calculated molecular signal) for each hour, with the coefficients plotted on the graphs. Add fitting line.
        # Plot only on molecular region, and add the coefficient to the title of the plot.
        # only plot if there's a coeficient (i.e. if the hour was processed)
        if coef_slope is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(p_mol[ref_mask], sig_arr[ref_mask], marker=" ", linestyle="-", color="blue", label="Data points", markersize=5)
            # Add fitting line
            fit_line = coef_slope[0] * p_mol[ref_mask] + coef_slope[1]  # y = ax + b
            # plot the fitting line on the same plot, with a dashed line 
            # nno: now I used the molecular SIGNAL P_mol and got something more reasonable for the fit, I was using beta_mol before, which is not the same as the molecular signal P_mol
            plt.plot(p_mol[ref_mask], fit_line, label="Fitting line", color="green", linestyle="--")
            plt.xlabel("Molecular Signal Backscatter (P_mol)")
            plt.ylabel("Real Signal Backscatter (P)")
            plt.title(f"Hour {hour:02d} - Coef Slope: a = {coef_slope[0]:.2e}, b = {coef_slope[1]:.2e}, day: {base_day.strftime('%Y-%m-%d')} with fitting line y = ax + b")
            plt.grid(True)
            plt.legend()
            # save fig with hour and day in the name in a folder
            if not showplots:
                plt.savefig(f"P_P_mol_plots/PxP_mol_fit_hour_{hour:02d}_day_{base_day.strftime('%Y%m%d')}.png")


    print(" AFTER all 24hours")
    # Save curtain plot for this day

    # curtain plot
    plt.figure(1, clear='True', figsize=(12, 6))
    plt.pcolormesh(ds_lidar.time, ds_lidar.range, ds_lidar.beta_raw.T, norm='log',vmin=1e4,vmax=4e7, cmap='jet') 
    # plot cloud base height (cbh) as a line on top of the curtain plot
    # if "cbh" in ds_lidar.variables:
    #     cbh = ds_lidar.cbh.mean(dim="layer").values
    #     plt.plot(ds_lidar.time, cbh, color='black', linewidth=1.5, label='CBH')
    #     plt.plot(ds_lidar.time[::20], cbh[::20], 'o', color='black', markersize=3, label='CBH')
    #     plt.xticks
    #     plt.legend()
    plt.title(f"Curtain Plot - Day {base_day.strftime('%Y%m%d')}.png")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('Time (UTC)')
    plt.ylabel('Height (m)')
    plt.grid()
    cb = plt.colorbar()
    cb.set_label('beta_raw')
    if not showplots:
        plt.savefig(f"P_P_mol_plots/curtain_plot_day_{base_day.strftime('%Y%m%d')}.png")
    print(" len(hour[coef_slope])", len(hourly["coef_slope"]), hourly["coef_slope"])

    if len(hourly["time"]) == 0:
        print("No valid hours for this scenario.")
        return None

    print("coef_slope=",hourly["coef_slope"], " len(coef_slope)=", len(hourly["coef_slope"]))
    #print(hourly)
    print(type(hourly["coef_slope"]))
    print(type(hourly["n_profiles"]))
    
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
            "coef_slope": (("time", "parameter"), np.array(hourly["coef_slope"])), 
        },
        coords={
            "time": np.array(hourly["time"]),
            "range": rng,
            "parameter": ["a", "b"] # For the calibration coefficient, we now have two parameters: a and b
        }
    )
    #"coef_slope": (("time", "parameter"), np.array(hourly["coef_slope"])), # nneves: Use this if storing 'a' and 'b'
    # "parameter": ["a", "b"] 
    return ds


def lidar_analysis(ds_lidar, ds_sonde, params):
    '''For one lidar dataset and one sonde, perform Klett'''

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
        lr_target=params["LR_TARGET"],  # if < 0, do AOD constrained... we are passing a dataframe with many values
        tau_target=params["TAU_TARGET"], # might be one AOD value, or a dataframe with many values
        zenith_deg=params["ZENITH_DEG"]
    )
    if ds_out is None:
        print(f"WARN: No output produced!")

    return(ds_out)

def iter_netcdf_members_in_zip(zip_path):

    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_members = sorted([n for n in zf.namelist() if n.lower().endswith(".nc")])
        if not nc_members:
            raise FileNotFoundError(f"No .nc files found inside {zip_path}")

        for nc_name in nc_members:
            tmp = tempfile.NamedTemporaryFile(dir = "/tmp", delete=False, suffix=".nc")
            tmp_path = tmp.name
            tmp.close()

            # stream copy
            with zf.open(nc_name) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

            yield nc_name, tmp_path

def plot_aeronet_overview(df_aod, start_date, end_date, title="AERONET AOD (local time)"):
    dfp = df_aod[(df_aod["time"] >= start_date) & (df_aod["time"] < end_date + pd.Timedelta(days=1))].copy()
    if dfp.empty:
        print("[warn] No AERONET data in requested date range for overview plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(dfp["time"], dfp["aod"], marker=".", linestyle="none")
    plt.xlabel("Time")
    plt.ylabel("AOD")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if showplots:
        plt.show()
    else:
        plt.savefig("AERONET_AOD_overview.png")

def plot_alpha_profiles(ds_out, max_range_m=8000, label="annual"):

    if ds_out is None or ds_out.sizes.get("time", 0) == 0:
        return

    rng = ds_out["range"].values.astype(float)
    m = rng <= max_range_m
    t = pd.to_datetime(ds_out["time"].values)

    plt.figure(figsize=(6, 8))
    for i, ti in enumerate(t):
        a = ds_out["alpha_aer"].isel(time=i).values.astype(float)
        if np.isfinite(a[m]).sum() < 10:
            continue
        plt.plot(a[m], rng[m], label=ti.strftime("%Y-%m-%d %H:%M"))
    plt.xlabel("Aerosol extinction α (1/m)")
    plt.ylabel("Range (m)")
    plt.title(f"α profiles [{label}]")
    plt.grid(True)
    plt.legend(title="Hour", fontsize=8)
    plt.tight_layout()
    if showplots:
        plt.show()
    else:
        plt.savefig("alpha_profiles.png")

def list_monthly_zip_files(zip_dir):

    #Return sorted list of monthly ceilometer zip files
    zips = sorted(glob.glob(os.path.join(zip_dir, "*.nc.zip")))
    if not zips:
        raise FileNotFoundError(f"No *.nc.zip files found in {zip_dir}")
    return zips

if __name__ == "__main__":
    BASE = r"/storage/jediah"
    BASE_CEILO = os.path.join(BASE, "ceilometer")
    OUT_BASE = os.path.join(BASE, "output")

    ANNUAL_CLIMO   = os.path.join(BASE, "sonde", "annual_avg.nc")     # dims: height
    # SEASONAL_CLIMO = os.path.join(BASE, "sonde", "seasonal_avg.nc")   # dims: season, height
    # MONTHLY_CLIMO  = os.path.join(BASE, "sonde", "monthly_avg.nc")    # dims: month, height

    # Read AERONET DIRECTSUN AOD for Manaus_EMBRAPA
    AERONET_2016 = r"aeronet/20160101_20161231_Manaus_EMBRAPA_level15.directsun"
    AERONET_2017 = r"aeronet/20170101_20171231_Manaus_EMBRAPA_level15.directsun"

    df_aod = load_aeronet_years_directsun(
        [AERONET_2016, AERONET_2017],
        aod_wavelength_nm=1020
    )
    #raise ValueError("Stopping after loading AERONET AOD for testing.")

    print(f"[info] Loaded AERONET AOD rows: {len(df_aod)}")
    print(df_aod.head())

    # Reference region for calibration
    PARAM = {
        "REF_MIN": 4500, #nno lowered the altitudes for testing
        "REF_MAX": 6000,
        # Fixed lidar ratio vs AOD-constrained:
        #  - set LR_TARGET > 0 and TAU_TARGET < 0 for fixed LR
        #  - set LR_TARGET < 0 and TAU_TARGET > 0 for AOD-constrained path 
        "LR_TARGET": -77,  # keep <0 to use AOD-constrained path for now
        "TAU_TARGET": df_aod, #df_aod,   # either one AOD value, or a data frame with aod x time
        # Lidar geometry at EMBRAPA
        "GROUND_M": 100.0,
        "ZENITH_DEG": 25.0,
        "WAVELENGTH_NM": 1064.0,
        # Plot-related (still used inside run_hourly_klett_inversion)
        "PLOT_MAX_RANGE": 8000,
        "PLOT_HOUR": 12,
    }

    # Safety: don’t allow both constraints at once
    #if PARAM["LR_TARGET"] > 0 and PARAM["TAU_TARGET"] > 0:
    #    print("Choose either fixed LR_TARGET or TAU_TARGET (not both).")
    #    sys.exit(1)

    # load climotologies
    ds_sonde_annual   = xr.load_dataset(ANNUAL_CLIMO) # change to file of the specific day of the inversion
 
    #zip_files = list_monthly_zip_files(BASE_CEILO)
    zip_files = [BASE_CEILO + '/201612_EMBRAPA_CHM160147.nc.zip']

    print("\n==============")
    print("[info] Found monthly zip files:", len(zip_files))
    for z in zip_files[:5]:
        print("  ", os.path.basename(z))
    if len(zip_files) > 5:
        print("  ...")

    all_months = []   # list of (yyyymm, ds_month)

    # Process every month zip
   
    for ZIP_PATH in zip_files:
        print("\n===============================")
        print("Processing dataset zip:", os.path.basename(ZIP_PATH))

        yyyymm = os.path.basename(ZIP_PATH)[:6]   # '201709'
        out_dir = os.path.join(OUT_BASE, yyyymm)
        os.makedirs(out_dir, exist_ok=True)
        print("Output dir:", out_dir)

        monthly_outputs = []

        # nneves: Loop over each daily netcdf inside the zip (limit to first 5 files), only 3 first hours of that day will be processed
        
        idx = 0
        for member_name, tmp_path in iter_netcdf_members_in_zip(ZIP_PATH):
            print("=====================================================================================================")
            if idx >= 2:
                break
            else:
                idx += 1

            ds_lidar = None
            # 26-AUG-2026 - WE SHOULD NOT NEED A TRY: TO DO THE LIDAR ANALYSIS.
            # ALL THE POSSIBLE ERRORS SHOULD BE HANDLED INSIDE THE LIDAR ANALYSIS FUNCTION.
            #try:
            print("-------------------- TRY ")
            print("Day file:", member_name)
            ds_lidar = xr.open_dataset(tmp_path, engine="netcdf4")
            print("    Start time:", ds_lidar.time.values[0])
            print("    End time:", ds_lidar.time.values[-1])

            output_annual = lidar_analysis(ds_lidar, ds_sonde_annual, PARAM)

            if (output_annual is not None) and (output_annual.sizes.get("time", 0) > 0):
                monthly_outputs.append(output_annual)

            # except Exception as e:
            #     print("-------------------- EXCEPT ")
            #     print("[ERROR] failed on", member_name, "->", repr(e))
            #     continue
            # finally:
            #     print("-------------------- FINALLY ")
            #     if ds_lidar is not None:
            #         ds_lidar.close()
            #     if os.path.exists(tmp_path):
            #         os.remove(tmp_path)

            # DEBUG STOP AFTER DAY 2
            # print("-------------------- DEBUG STOP AFTER DAY 2 ")
            # if member_name =="20161203_EMBRAPA_CHM160147_000.nc":
            #     print("-------------------- EXIT AFTER DAY 2 ")
            #     break
            # else:
            #     print("--------------------     NOT MAR-2ND 2017    <<<<< =====================")

        # After this zip: concat + save +  plot
        if len(monthly_outputs) == 0:
            print("[WARN] No valid outputs for month", yyyymm)
            continue

        ds_month = xr.concat(monthly_outputs, dim="time").sortby("time")
        out_nc = os.path.join(out_dir, f"{yyyymm}_klett_annual.nc")
        ds_month.to_netcdf(out_nc)
        print("[info] Saved:", out_nc)

        print(f"\n=== Plotting {yyyymm} ===")
        start_date = pd.to_datetime(ds_month.time.min().values).normalize()
        end_date   = pd.to_datetime(ds_month.time.max().values).normalize()
        plot_aeronet_overview(df_aod, start_date, end_date, title=f"AERONET AOD {yyyymm} (local)")
        plot_alpha_profiles(ds_month, max_range_m=PARAM["PLOT_MAX_RANGE"], label=yyyymm)
    print("\nAll done for", len(zip_files), "monthly zips.")
