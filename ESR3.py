#%%
import os
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

'''Day 2 calibration run #1 measurements'''

measurements = 10

V_offset = np.arange(1.00, measurements + 1, 1)
T1 = np.array([12.8, 29.8, 36.5, 48.0, 59.6, 70.7, 80.2, 90.6, 96.9, 97.2])
T1 = T1 * 10**-4  # Convert to Tesla
T1_error = np.empty(measurements)
T1_error.fill(0.2 * 10**-4)  # Convert to Tesla

'''Day 2 calibration run #2 measurements'''
T2 = np.array([13.0, 24.3, 36.0, 47.3, 58.9, 69.8, 82.0, 90.0, 95.2, 97.2])
T2 = T2 * 10**-4  # Convert to Tesla
T2_error = np.empty(measurements)
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

'''Day 2 capacitor resonance voltage offset'''
folder_name = './Data'

files = [f for f in os.listdir(folder_name) if f.endswith('.lvm')]

fig, ax = plt.subplots(1, 1, figsize=(8,6))

freq_data = []

for file in files:
    mhz = int(os.path.basename(file).lower().removesuffix('mhz.lvm'))
    
    n, v, current, idk = np.loadtxt(os.path.join(folder_name, file), skiprows=22, delimiter='\t', unpack=True)
    
    freq_data.append((mhz, n, v, current, idk))

freq_data.sort(key=lambda e: e[0])

mhz_data = []
v_data = []

for mhz, n, v, current, idk in freq_data:
    upper_percentile = np.percentile(current, 95)
    average = 0  # np.average(current[-100:])
    
    print(upper_percentile, average)
    
    found_upper = False
    zero_index = -1
    
    for n, c in enumerate(current):
        if not found_upper:
            if c >= upper_percentile:
                found_upper = True
        else:
            if c <= average:
                zero_index = n
                break
    
    if zero_index == -1:
        print(f'Error 1 with the algorithm for {mhz}MHz')
    
    x1, y1 = v[zero_index], current[zero_index]
    x2, y2 = v[zero_index - 1], current[zero_index - 1]
    m = (y2 - y1) / (x2 - x1)
    final_x = (average - y1 + m*x1) / m
    final_y = average
    
    plt.plot(v, current, label=f'{mhz}MHZ')
    plt.scatter([final_x], [final_y], zorder=100)
    print(final_x)
    
    mhz_data.append(mhz)
    v_data.append(final_x)

plt.title('Voltage offset sweep for each frequency')
plt.xlabel('Voltage (V)')
plt.ylabel('Amplitude')

plt.plot([0, 10], [0,0], linestyle='--')

plt.legend()

plt.show()

# Perform weighted linear regression on the extracted data
mhz_data = np.array(mhz_data)
v_data = np.array(v_data)
v_data_error = np.full_like(v_data, 0.01)  # Assuming a constant error for demonstration

a_cal, b_cal, a_std, b_std = weighted_linear_regression(mhz_data, v_data, v_data_error)

# Plot the data and the fitted line
plt.errorbar(mhz_data, v_data, yerr=v_data_error, fmt='o', label='Data with error bars')
x_fit = np.linspace(min(mhz_data), max(mhz_data), 1000)
plt.plot(x_fit, linear_model(x_fit, a_cal, b_cal), label='Fitted line', color='c')

# Labels and legend
plt.xlabel('Frequency (MHz)')
plt.ylabel('Voltage Offset (V)')
plt.title('Frequency VS Voltage Offset')
plt.legend()

# Show plot
plt.show()

# Print the slope value
print(f"Slope (a_cal) from the linear regression: {a_cal:.2e}")

# Calculate the g-factor
h = 6.626e-34  # Planck's constant in J·s
bohr = 9.274e-24  # Bohr magneton in J/T
g_factor = (h / bohr) * (1 / a_cal)  # g-factor formula
g_factor_uncertainty = (h / bohr) * (a_std / a_cal**2)  # Uncertainty in g-factor

print(f'G Factor for part 2 is {g_factor:.4f} ± {g_factor_uncertainty:.4f}')
# %%
