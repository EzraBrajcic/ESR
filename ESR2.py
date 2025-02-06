#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define a weighted linear regression function
def weighted_linear_regression(x, y, y_err):

    # Define the linear model
    def linear_model(X, a, b):
        return a * X + b
    
    # Use the inverse of the squared errors as weights
    weights = 1 / y_err**2
    
    # Perform the weighted linear regression
    popt, pcov = curve_fit(linear_model, x, y, sigma=weights)
    a_opt, b_opt = popt
    a_std, b_std = np.sqrt(np.diag(pcov))
    
    return a_opt, b_opt, a_std, b_std

# Plotting function for weighted linear regression
def plot_weighted_linear_regression(x, y, y_err, xlabel, ylabel, title):

    a_opt, b_opt, a_std, b_std = weighted_linear_regression(x, y, y_err)
    
    # Generate points for the fitted line and uncertainty range
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = a_opt * x_fit + b_opt
    y_fit_upper = (a_opt + a_std) * x_fit + (b_opt + b_std)
    y_fit_lower = (a_opt - a_std) * x_fit + (b_opt - b_std)
    
    # Plot data with error bars
    plt.errorbar(x, y, yerr=y_err, fmt='o', label='Data with error bars')
    
    # Plot the fitted line
    plt.plot(x_fit, y_fit, label='Fitted line', color='c')
    
    # Plot the uncertainty range
    plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.2, label='Uncertainty range')
    
    # Labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    # Show plot
    plt.show()
    
    return a_opt, b_opt, a_std, b_std

'''Day 1 calibration run #1 measurements'''

measurements = 10

V_offset = np.arange(1.00, 9, 1)
T1 = np.array([10.9, 23.3, 35.4, 47.0, 58.3, 68.9, 82.9, 89.8])
T1 = T1 * 10**-4  # Convert to Tesla
T1_error = np.empty(measurements - 2)
T1_error.fill(0.2 * 10**-4)  # Convert to Tesla

'''Day 1 calibration run #2 measurements'''
T2 = np.array([11.0, 22.7, 34.4, 46.0, 57.8, 69.0, 81.0, 90.2])
T2 = T2 * 10**-4  # Convert to Tesla
T2_error = np.empty(measurements - 2)
T2_error.fill(0.2 * 10**-4)  # Convert to Tesla

# Calculate the average of T1 and T2
T_avg = (T1 + T2) / 2
T_avg_error = np.sqrt(T1_error**2 + T2_error**2) / 2  # Correct error propagation

# Perform weighted linear regression for the calibration data
a_cal, b_cal, a_cal_std, b_cal_std = plot_weighted_linear_regression(
    V_offset, T_avg, T_avg_error,
    xlabel='Voltage Offset (V)',
    ylabel='Magnetic Field Strength (T)',
    title='Voltage Offset VS Magnetic Field Strength'
)

'''Day 1 capacitor resonance voltage offset'''
V_C_offset = np.array([2.70, 3.04, 3.36, 3.65, 3.95, 4.26, 4.55, 4.85])
V_C_offset_error = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02])
f = np.arange(90000000, 10000000, 10000000)  # Frequency in Hz
f_error = np.empty(measurements)
f_error.fill(1000)

# Convert capacitor resonance voltages to magnetic field using the calibration slope
T_res = a_cal * V_C_offset + b_cal  # Magnetic field from resonance
T_res_error = a_cal * V_C_offset_error  # Propagate errors

# Perform weighted linear regression for the resonance data
a_opt, b_opt, a_std, b_std = plot_weighted_linear_regression(
    f, T_res, T_res_error,
    xlabel='Frequency (Hz)',
    ylabel='Resonance Magnetic Field Strength (T)',
    title='Frequency VS Resonance Magnetic Field Strength'
)

# Print the new slope value
print(f"New slope (a_opt) from the linear regression: {a_opt:.2e} ± {a_std:.2e} T/Hz")

# Calculate the g-factor
h = 6.626e-34  # Planck's constant in J·s
bohr = 9.274e-24  # Bohr magneton in J/T
g_factor = (h / bohr) * (1 / a_opt)  # g-factor formula
g_factor_uncertainty = (h / bohr) * (a_std / a_opt**2)  # Uncertainty in g-factor

print(f'G Factor for part 2 is {g_factor:.4f} ± {g_factor_uncertainty:.4f}')
# %%