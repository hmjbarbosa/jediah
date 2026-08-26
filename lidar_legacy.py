# -*- coding: utf-8 -*-
"""
lidar_legacy.py
"""
#tau_target= how much light is expected to be absorbed or scattered by the atmosphere between the ground and a certain altitude


import glob
import math
import pickle
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapezoid as trapz
from scipy.optimize import curve_fit

from lidarpy.molecular import AlphaBetaMolecular
from lidarpy.utils.functions import z_finder

# --- Configuration ----------------------------------------------------------
# Replace absolute paths with relative ones; keep original code intact below
SONDE_FILE = r"C:\DATA\sonde\maosondewnpnM1.b1.20150326.053000.cdf"
LIDAR_FILE = r"C:\DATA\ceilo\2017\09\16\20170916_EMBRAPA_CHM160147_000.nc"

# --- Enable interactive plotting -------------------------------------------
plt.ion()
plt.interactive(True)

#====================================
# clean case - wet season Amazon March 2017rr
# Aeronet is good from 17 to 19 UTC (1 to 3 pm Local time)
if False:
    # Original absolute paths preserved here:
    sonde = xr.load_dataset(r"C:\DATA\sonde\maosondewnpnM1.b1.20150326.053000.cdf")
    lidar = xr.load_dataset(r"C:\DATA\ceilo\2017\09\16\20170916_EMBRAPA_CHM160147_000.nc")
    time_range = (lidar.time.dt.hour >= 17) & (lidar.time.dt.hour < 19)
    tau_target = 0.003 / np.cos(25 * np.pi/180) 
    myzref = [4000, 14000]

# Polluted case - dry season Amazon September 2017
# Aeronet is good from 11 to 12 UTC (7 to 8 am Local time)
if True:
    # Updated to use relative paths:
    sonde = xr.load_dataset(SONDE_FILE)
    lidar = xr.load_dataset(LIDAR_FILE)
    time_range = (lidar.time.dt.hour >= 11) & (lidar.time.dt.hour < 12)
    tau_target = 0.261 / np.cos(25 * np.pi/180)
    myzref = [6000, 14000]

#--- Molecular profile computation ------------------------------------------
l = 1064e-9  # laser wavelength
p = sonde.pres * 100        # hPa → Pa
t = sonde.tdry + 273.15     # °C → K
z = sonde.alt               # altitude (m)
# slant range along 25° beam
r = (z - z[0]) / np.cos(25. * np.pi / 180.)

# Compute molecular parameters
df_mol = AlphaBetaMolecular(
    p_air=p.values,
    t_air=t.values,
    rangebin=r.values,
    wavelength=1064
).get_params()

#--- Signal preprocessing ---------------------------------------------------
signal_raw = lidar.beta_raw
signal_avg = np.mean(signal_raw, axis=0)

# Quick plot of raw backscatter
plt.figure(figsize=(10, 6))
plt.pcolormesh(lidar.time, lidar.range, np.log10(lidar.beta_raw.T), shading='auto')
plt.colorbar(label='Log10(Backscatter)')
cbh_masked = lidar.cbh.where(lidar.cbh != -1)
for i in range(cbh_masked.shape[1]):
    plt.scatter(lidar.time, cbh_masked.isel(layer=i), color='red', s=10,
                label='Cloud Base' if i == 0 else "")
plt.xlabel("Time (UTC)")
plt.ylabel("Range (m)")
plt.title("Log10 of Backscatter with Cloud Base Heights")
plt.legend(loc='upper left')

# Select within the time range
signal_selected = lidar.beta_raw.sel(time=time_range)
cbh_selected = lidar.cbh.sel(time=time_range)

# Build a mask of clear-sky profiles
no_clouds_mask = (cbh_selected == -1).all(dim="layer")
tmp = no_clouds_mask.values * 1
print('good profs before additional mask =', np.sum(tmp))

# Exclude isolated clear profiles for continuity
for i in np.arange(1, len(tmp) - 1):
    if (tmp[i-1] == False) or (tmp[i+1] == False):
        no_clouds_mask[i] = False

goodprofs = np.sum(no_clouds_mask.values)
print('goodprofs =', goodprofs)
if goodprofs < 10:
    print('not enough profiles left!')
    raise KeyboardInterrupt

signal_no_clouds = signal_selected.where(no_clouds_mask, drop=True)
signal_avg_no_clouds = signal_no_clouds.mean(dim="time")
signal_selected_processed = signal_avg_no_clouds / (lidar.range ** 2)

plt.figure()
plt.plot(signal_avg_no_clouds.T, lidar.range)
plt.xlabel("Filtered Average Signal Intensity")
plt.ylabel("Range (m)")
plt.title("Lidar Signal Without Clouds - 11:00 - 12:00 UTC")
plt.grid()

#--- Calibration strategies -------------------------------------------------
def calib_strategy1(signal: np.array, model: np.array, reference: np.array):
    """
    Linear slope+offset calibration to align model → signal.
    """
    #HB: apr-22 2026 doesn't seem right to do this...
    #reference -= 1
    #if reference[-1] >= len(signal):
    #    reference = np.arange(reference[0], len(signal))

    reference -= 1
    if reference[-1] >= len(signal):
        reference = np.arange(reference[0], len(signal))
    mean_signal = np.mean(signal[reference])
    signal = signal / mean_signal
    mean_model = np.mean(model[reference])
    model = model / mean_model
    coef_slope_linear, *_ = curve_fit(
        f=lambda x, a, b: a * x + b, # This includes b (backscatter) which is why is gets noise_corrected later
        xdata=model[reference],
        ydata=signal[reference]
    )
    signal = (signal - coef_slope_linear[1]) / coef_slope_linear[0]
    coef_slope_linear = [
        coef_slope_linear[0] * mean_signal / mean_model,
        coef_slope_linear[1] * mean_signal
    ]
    return signal, model, coef_slope_linear

def calib_strategy2(signal: np.array, model: np.array, reference: np.array):
    """
    Simple scaling calibration: align mean(signal) to mean(model).
    """
    mean_signal = np.mean(signal[reference])
    signal = signal / mean_signal
    mean_model = np.mean(model[reference])
    model = model / mean_model
    coef_slope, *_ = curve_fit(
        f=lambda x, a: a * x, 
        xdata=model[reference],
        ydata=signal[reference]
    )
    signal = signal / coef_slope[0]
    coef_slope = [coef_slope[0] * mean_signal / mean_model]
    return signal, model, coef_slope

#--- Klett inversion --------------------------------------------------------
class Klett:
    """
    Implements the Klett inversion for combined molecular + aerosol backscatter.
    """
    def __init__(self, signal, rangebin, molecular_data, z_ref, lidar_ratio, correct_noise=True):
        self.signal = signal.copy()
        self.rangebin = rangebin.copy()
        self.ref = z_ref # why is z_ref defined multiple times? (z_ref vs myzref vs self.ref)
        self._calib_strategy = {True: calib_strategy1, False: calib_strategy2}[correct_noise]
        # store molecular profiles
        self._alpha = {'mol': molecular_data.alpha.data}
        self._beta = {'mol': molecular_data.beta.data}
        self._lr = {
            'mol': molecular_data.lidar_ratio.data,
            'aer': lidar_ratio
        }

        # Shorten to common array length
        # Finding the array with the lowest length and using that as the new length for all arrays (4/28/26)
        min_len = min(len(self._alpha['mol']), len(self.rangebin), len(self._beta['mol'])) 
        self._alpha['mol'] = self._alpha['mol'][:min_len]
        self._beta['mol'] = self._beta['mol'][:min_len]
        self.rangebin = self.rangebin[:min_len]
        if isinstance(self._lr['mol'], np.ndarray) and len(self._lr['mol']) > 1: #jm: If the molecular lidar ratio is an array instead of one number, trim it to match the other arrays.
            self._lr['mol'] = self._lr['mol'][:min_len]

    def get_model_mol(self):
        """Compute range-corrected molecular backscatter."""
        tau = -2 * cumtrapz(self._alpha['mol'], self.rangebin, initial=0)
        return self._beta['mol'] * np.exp(tau) / (self.rangebin ** 2) # if it's range corrected then why is it divided by rangebin^2? (4/28/26)

    def _calibration(self, signal):
        """Calibrate raw signal at reference altitude against molecular model."""
        ref = z_finder(self.rangebin, self.ref) # z finder gets the index of the molecular reference altitude (4/28/26)
        model = self.get_model_mol() 
        #jm: If the reference is a range of heights, use that full reference region for calibration.
        if len(ref) > 1: # the reference range needs to have a range (min, max) to perform the calibration (4/28/26)
            sig_cal, mod_norm, self.fit_parameters = self._calib_strategy(
                signal=signal.copy(),
                model=model,
                reference=np.arange(*ref)
            )
            #hb: apr-22 2026 - we need to understand why luan is doing this normalization here
            # The calibration strategy already normalizes the signal and model to their means in the reference region, 
            # so this additional normalization seems redundant.
            beta_ref = self._beta['mol'][ref[0]] * sig_cal[ref[0]] / mod_norm[ref[0]]
            self.norm_model = mod_norm.copy()
            self.norm_signal = sig_cal.copy()
        else:
            sig_cal = signal
            beta_ref = self._beta['mol'][ref[0]]
        return beta_ref, sig_cal, ref[0]

    def fit(self):
        """
        Perform Klett inversion.
        Returns (alpha_aer, beta_aer, lr_aer, beta_mol, alpha_mol, tau).
        """
        beta_ref, sig, ref0 = self._calibration(self.signal)
        #jm: Multiplies by range squared to undo geometric spreading in the lidar signal or something like that
        corrected = sig * (self.rangebin ** 2)
        #hb: spp seems to be the numerator of eq. 4.11 in weitkamp, Update: yes, it makes sense
        spp = corrected * np.exp(-2 * cumtrapz(
            x=self.rangebin,
            y=(self._lr['aer'] - self._lr['mol']) * self._beta['mol'],
            initial=0
        ))
        sppr = spp / spp[ref0] #In the book it means S(R)/S(R0), the exp terms cancel out (4/28/26)
        # It looks like Luan divided the whole denominator by S(R0) to avoid numerical issues (4/28/26)

        integral = cumtrapz(
            x=self.rangebin,
            y=2 * self._lr['aer'] * sppr,
            initial=0
        )
        denom = (1 / beta_ref) - (
            integral - trapz(
                x=self.rangebin[:ref0],
                y=2 * (self._lr['aer'] * np.ones_like(sppr)[:ref0]) * sppr[:ref0] # this is most likely the transmittivity (integral)
            )
        )
        self._beta['tot'] = sppr / denom
        self._beta['aer'] = self._beta['tot'] - self._beta['mol']
        self._alpha['aer'] = self._beta['aer'] * self._lr['aer']
        # Force near-ground extinction constant (<200 m)
        self._alpha['aer'][:12] = self._alpha['aer'][12]
        self._alpha['tot'] = self._alpha['mol'] + self._alpha['aer']

        maxlev = int(self.ref[0] / self.rangebin[0])
        print(' maxlev=', maxlev)
        self.tau = trapz(self._alpha['aer'][:maxlev], self.rangebin[:maxlev])
        return (
            self._alpha['aer'].copy(),
            self._beta['aer'].copy(),
            self._beta['mol'].copy(),
            self._alpha['mol'].copy(),
            self.tau.copy()
        )

#--- Binary search for optimal lidar ratio ----------------------------------
tau_rel_error = 0.001
maxlev = int(((myzref[0] + myzref[1]) / 2) / lidar.range.values[0])
print('maxlev=', maxlev)

LR1 = 5   # sr
#jm: Runs Klett once using the lower lidar-ratio bound.
results1 = Klett(
    signal=signal_selected_processed.values,
    rangebin=lidar.range.values,
    molecular_data=df_mol,
    z_ref=myzref,
    lidar_ratio=LR1
).fit()
alpha_aer1, beta_aer1, beta_mol1, alpha_mol1, TAU1 = results1

LR2 = 150  # sr
#jm: Runs Klett once using the upper lidar-ratio bound.
results2 = Klett(
    signal=signal_selected_processed.values,
    rangebin=lidar.range.values,
    molecular_data=df_mol,
    z_ref=myzref,
    lidar_ratio=LR2
).fit()
alpha_aer2, beta_aer2, beta_mol2, alpha_mol2, TAU2 = results2
# Where Newtons method (binary search) is applied
#jm: Repeatedly narrows the lidar-ratio range until the modeled AOD is close enough to the target AOD.
while abs((TAU2 - TAU1) / tau_target) > tau_rel_error:
    LRm = 0.5 * (LR1 + LR2)
    alpha_aerm, beta_aerm, beta_molm, alpha_molm, TAUm = Klett(
        signal=signal_selected_processed.values,
        rangebin=lidar.range.values,
        molecular_data=df_mol,
        z_ref=myzref,
        lidar_ratio=LRm
    ).fit()
    if TAUm > tau_target:
        LR2, TAU2 = LRm, TAUm
    else:
        LR1, TAU1 = LRm, TAUm
    print(LR1, LR2, TAU1, TAU2)

# Final optimal run
#jm: Runs Klett one last time using the final midpoint lidar ratio after the search converges on a solution
alpha_aer_selected, beta_aer_selected, beta_mol_selected, alpha_mol_selected, TAU_opt = Klett(
    signal=signal_selected_processed.values,
    rangebin=lidar.range.values,
    molecular_data=df_mol,
    z_ref=myzref,
    lidar_ratio=0.5 * (LR1 + LR2)
).fit()

#--- Plotting ---------------------------------------------------------------
def plot_profiles(beta_aer, beta_mol, alpha_aer, alpha_mol, rangebin, max_range=8000):
    """
    Plot aerosol & molecular backscatter/extinction profiles up to max_range.
    """
    max_ind = np.searchsorted(rangebin, max_range)
    plt.figure()
    plt.plot(beta_aer[:max_ind], rangebin[:max_ind], label='Aerosol Backscatter')
    plt.plot(beta_mol[:max_ind], rangebin[:max_ind], label='Molecular Backscatter')
    plt.ylim(0, max_range)
    plt.xlabel('Beta (m⁻¹ sr⁻¹)')
    plt.ylabel('Range (m)')
    plt.title('Klett Inversion Results - Backscatter')
    plt.legend()
    plt.grid()
    plt.show(block=True)

    plt.figure()
    plt.plot(alpha_aer[:max_ind], rangebin[:max_ind], label='Aerosol Extinction')
    plt.plot(alpha_mol[:max_ind], rangebin[:max_ind], label='Molecular Extinction')
    plt.ylim(0, max_range)
    plt.xlabel('Alpha (m⁻¹)')
    plt.ylabel('Range (m)')
    plt.title('Klett Inversion Results - Extinction')
    plt.legend()
    plt.grid()
    plt.show(block=True)

plot_profiles(
    beta_aer_selected,
    beta_mol_selected,
    alpha_aer_selected,
    alpha_mol_selected,
    lidar.range.values
)