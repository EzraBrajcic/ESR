#%%
import numpy as np
from wlinreg import *


# Calibration data (Voltage -> Magnetic Field)
measurements = 10

V_offset = np.arange(1.00, float(measurements - 3), 1)

T1 = np.array([10.9, 23.3, 35.4, 47.0, 58.3, 68.9]) * 1e-4  # Tesla
T2 = np.array([11.0, 22.7, 34.4, 46.0, 57.8, 69.0]) * 1e-4  # Tesla

T_avg = (T1 + T2) / 2
T_avg_error = np.sqrt((0.2e-4)**2 + (0.2e-4)**2) / 2  # Error propagation

# Perform calibration regression
s_cal, b_cal, s_cal_std, b_cal_std = plot_regression(
    V_offset, T_avg, T_avg_error,
    'Voltage Offset (V)', 'Magnetic Field (T)',
    'Calibration Plot 1.svg', 'Figure 1: Experiment 1 calibration measurements of voltage offset relationship with magnetic field strengths of helmholtz coils. Offset voltages going in increments of 1.00V from 1.00V to 6.00V.'
)

# Resonance data (Voltage -> Frequency)
V_C_offset = np.array([2.67, 2.70, 3.04, 3.36, 3.65, 3.95, 4.26, 4.55, 4.85, 5.13, 5.44])
V_C_offset_error = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.02])

f = np.array([90, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]) * 1e6  # Frequency in Hz
f_error = np.full(measurements + 1, 1000)  # Assume small frequency errors

# Perform resonance regression (Frequency vs Voltage)
s_frequency, s_frequency_intercept, s_frequency_std, _ = plot_regression(
    V_C_offset, f, f_error,
    'Voltage (V)', 'Frequency (Hz)',
    'Resonance Plot 1.svg', 'Figure 2: Experiment 1 capacitor resonance voltage offset relationship with generated radio frequency going from 90MHz to 180 MHz in 10MHz increments.'
)

# Calculate effective slope s_opt = s_cal / s (Tesla/Hz)
s_opt = s_cal / s_frequency
# Propagate errors in s_opt = s_cal / s_frequency
s_opt_std = s_opt * np.sqrt(
    (s_cal_std / s_cal)**2 + 
    (s_frequency_std / s_frequency)**2 +
    (np.mean(V_C_offset_error)**2 / np.mean(V_C_offset)**2)  # Voltage error term
)

# Calculate g-factor and uncertainty
h = 6.626e-34  # Planck's constant
mu_B = 9.274e-24  # Bohr magneton
g = (h / mu_B) / s_opt
g_std = g * (s_opt_std / s_opt)  # Relative error propagation

print(f"Slope (s_opt): {s_opt:.2e} ± {s_opt_std:.2e} T/Hz")
print(f"g-Factor: {g:.4f} ± {g_std:.4f}")
# %%