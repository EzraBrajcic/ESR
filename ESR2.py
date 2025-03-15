#%%
import numpy as np
from Regression import *

# Calibration data
measurements = 10

# Constants
h = 6.626e-34  # Planck's constant (J·s)
mu_B = 9.274e-24  # Bohr magneton (J/T)

V_offset = np.arange(1.00, float(measurements - 2), 1)

# Original magnetic field measurements (in Tesla)
T1 = np.array([10.9, 23.3, 35.4, 47.0, 58.3, 68.9, 82.9]) * 1e-4 
T2 = np.array([11.0, 22.7, 34.4, 46.0, 57.8, 69.0, 81.0]) * 1e-4

# Calculate average magnetic field (in Tesla)
T_avg = (T1 + T2) / 2
T_avg_error = np.sqrt((0.2e-4)**2 + (0.2e-4)**2) / 2  # Error propagation

# Perform calibration regression (using raw Tesla values, not multiplied by mu_B)
s_cal, b_cal, s_cal_std, b_cal_std = plot_regression(
    V_offset, T_avg, T_avg_error,
    'Voltage Offset (V)', 'Magnetic Field (T)',
    'Calibration Plot 1.png', 'Figure 1: Experiment 1 calibration measurements of voltage offset relationship with magnetic field strengths of helmholtz coils. Offset voltages going in increments of 1.00V from 1.00V to 7.00V.',
    x_err = None
)

print('Magnetic field and offset voltage slope data in form of: slope, intercept, slope error, intercept error.')
print(f'{s_cal:.3e}\t{b_cal:.3e}\t{s_cal_std:.3e}\t{b_cal_std:.3e}\n')



# Resonance data
V_C_offset = np.array([2.67, 2.70, 3.04, 3.36, 3.65, 3.95, 4.26, 4.55, 4.85, 5.13, 5.44])
V_C_offset_error = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.02])

# Frequencies in Hz
f = np.array([90, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]) * 1e6

# For frequency vs voltage, we consider voltage as the independent variable (x) with errors
# and frequency as the dependent variable (y) with no significant errors
# Using ODR to handle x errors properly
s_frequency, b_frequency, s_frequency_std, b_frequency_std = plot_regression(
    V_C_offset, f, np.full_like(f, 1),
    'Voltage (V)', 'Frequency (Hz)',
    'Resonance Plot 1.png', 'Figure 2: Experiment 1 capacitor resonance voltage offset relationship with generated radio frequency going from 90MHz to 180 MHz in 10MHz increments.',
    x_err = V_C_offset_error
)

print('Frequency and offset voltage slope data in form of: slope, intercept, slope error, intercept error.')
print(f'{s_frequency:.3e}\t{b_frequency:.3e}\t{s_frequency_std:.3e}\t{b_frequency_std:.3e}\n')



# Calculate effective slope s_opt = s_cal / s_frequency (Tesla/Hz)
s_opt = s_cal / s_frequency

# Propagate errors in s_opt = s_cal / s_frequency
s_opt_std = s_opt * np.sqrt(
    (s_cal_std / s_cal)**2 + 
    (s_frequency_std / s_frequency)**2
)

# Calculate g-factor and uncertainty using the traditional method
g_indirect = (h / mu_B) / s_opt
g_indirect_std = g_indirect * (s_opt_std / s_opt)  # Relative error propagation

print(f"Method 1 - Indirect calculation:")
print(f"Slope (s_opt): {s_opt:.3e} ± {s_opt_std:.3e} T/Hz")
print(f"g-Factor: {g_indirect:.4f} ± {g_indirect_std:.4f}\n")



# Need paired values of B and f for each measurement point
# First, calculate B values from V_C_offset calibration data
B_values = s_cal * V_C_offset + b_cal

# Calculate errors in B values
B_errors = np.sqrt(
    (s_cal * V_C_offset_error)**2 + 
    (V_C_offset * s_cal_std)**2 + 
    b_cal_std**2
)

# Preparing the data for the h*f vs mu_B*B plot
# Direct regression to calculate g-factor
# The g-factor is the slope when plotting h*f vs mu_B*B
g_direct, g_intercept, g_direct_std, g_intercept_std = plot_regression(
    mu_B * B_values, h * f, np.full_like(f, h*1e-6),
    r'$\mu_B B_0$ (J)', 
    r'$\hbar \omega$ (J)',
    'g-factor Plot 1.png', 'Figure 3: Direct measurement of g-factor as the slope of h·f vs μ_B·B plot for Experiment 1. The slope of this line gives the g-factor directly.',
    x_err = mu_B * B_errors
)

print(f"\nMethod 2 - Direct calculation:")
print(f"g-Factor (slope): {g_direct:.4f} ± {g_direct_std:.4f}")
print(f"Intercept: {g_intercept:.3e} ± {g_intercept_std:.3e} J")

# For comparison, calculate g-factor from the expected value (~2.0036)
g_expected = 2.0036
g_diff_percent_expected = abs(g_direct - g_expected) / g_expected * 100
g_diff_percent_methods = abs(g_direct - g_indirect) / ((g_direct + g_indirect) / 2) * 100

print(f"\nComparison with expected g-factor (2.0036):")
print(f"Difference from expected (Direct method): {g_diff_percent_expected:.2f}%")
print(f"Difference between methods: {g_diff_percent_methods:.2f}%")
# %%