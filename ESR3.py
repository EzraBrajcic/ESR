#%%
import os
import numpy as np
from Regression import *

'''Day 2 calibration run #1 measurements'''
measurements = 10

# Constants
h = 6.626e-34  # Planck's constant (J·s)
mu_B = 9.274e-24  # Bohr magneton (J/T)

V_offset = np.arange(1.00, float(measurements - 1), 1)

# Magnetic field measurements in Tesla (not multiplied by mu_B)
T1 = np.array([12.8, 29.8, 36.5, 48.0, 59.6, 70.7, 80.2, 90.6]) * 1e-4  # Convert to Tesla
T1_error = np.empty(measurements - 2)
T1_error.fill(0.2 * 1e-4)  # Error in Tesla

'''Day 2 calibration run #2 measurements'''
T2 = np.array([13.0, 24.3, 36.0, 47.3, 58.9, 69.8, 82.0, 90.0]) * 1e-4  # Convert to Tesla
T2_error = np.empty(measurements - 2)
T2_error.fill(0.2 * 1e-4)  # Error in Tesla

# Calculate the average of T1 and T2
T_avg = (T1 + T2) / 2
T_avg_error = np.sqrt(T1_error**2 + T2_error**2) / 2  # Correct error propagation

# Perform weighted linear regression for the calibration data
s_cal, b_cal, s_cal_std, b_cal_std = plot_regression(
    V_offset, T_avg, T_avg_error,
    'Voltage Offset (V)', 'Magnetic Field (T)',
    'Calibration Plot 2.png', 'Figure 3: Experiment 2 calibration measurements of voltage offset relationship with magnetic field strengths of helmholtz coils. Offset voltages going in increments of 1.00V from 1.00V to 8.00V',
    x_err = None
)
print('Magnetic field and offset voltage slope data in form of: slope, intercept, slope error, intercept error.')
print(f'{s_cal:.3e}\t{b_cal:.3e}\t{s_cal_std:.3e}\t{b_cal_std:.3e}\n')



'''Day 2 capacitor resonance voltage offset'''
folder_name = './Data'

files = [f for f in os.listdir(folder_name) if f.endswith('.lvm')]

fig, ax = plt.subplots(1, 1, figsize=(8,6))

freq_data = []

for file in files:
    mhz = int(os.path.basename(file).lower().removesuffix('mhz.lvm'))
    
    n, v, amplitude, current = np.loadtxt(os.path.join(folder_name, file), skiprows=22, delimiter='\t', unpack=True)
    
    freq_data.append((mhz, n, v, amplitude, current))

freq_data.sort(key=lambda e: e[0])

mhz_data = []
v_data = []
v_data_errors = []  # Add error estimates for voltage measurements

for mhz, n, v, amplitude, current in freq_data:
    upper_percentile = np.percentile(amplitude, 95)
    average = 0
    
    found_upper = False
    zero_index = -1
    
    for n, c in enumerate(amplitude):
        if not found_upper:
            if c >= upper_percentile:
                found_upper = True
        else:
            if c <= average:
                zero_index = n
                break
    
    if zero_index == -1:
        print(f'Error 1 with the algorithm for {mhz}MHz')
    
    x1, y1 = v[zero_index], amplitude[zero_index]
    x2, y2 = v[zero_index - 1], amplitude[zero_index - 1]
    m = (y2 - y1) / (x2 - x1)
    final_x = (average - y1 + m*x1) / m
    final_y = average
    
    plt.plot(v, amplitude, label=f'{mhz}MHZ')
    plt.scatter([final_x], [final_y], zorder=100)
    
    mhz_data.append(mhz)
    v_data.append(final_x)
    v_data_errors.append(0.01)  # Assuming a standard uncertainty of 0.01V

plt.xlabel('Voltage (V)')
plt.ylabel('Amplitude')

plt.plot([0, 10], [0,0], linestyle='--')

plt.legend()

# Add caption below the figure
plt.figtext(0.5, -0.1, 'Figure 4: Experiment 2 voltage offset sweep for each frequency, with resonance points. Voltage offset ranges from 1.00V to 10.00V over 3 minutes and frequencies sweep from 90MHz to 180MHz in 10MHz increments', wrap=True, horizontalalignment='center', fontsize=8)
plt.savefig('Resonance Intercepts.png', bbox_inches='tight', dpi=600)
plt.show()
plt.close()

print('Offset voltage sweep voltage intercept values in V\n')
for i in range(len(v_data)):
    print(f'{v_data[i]:.3e}\n')

# Convert MHz to Hz for analysis
f_values = np.array(mhz_data) * 1e6  # Frequencies in Hz

v_data = np.array(v_data)
v_data_error = np.array(v_data_errors)

# Perform resonance regression (Frequency vs Voltage)
# Use small non-zero errors for numerical stability
s_frequency, b_frequency, s_frequency_std, b_frequency_std = plot_regression(
    v_data, f_values, np.full_like(f_values, 1),  # Small non-zero error for numerical stability
    'Voltage (V)', 'Frequency (Hz)',
    'Resonance Plot 2.png', 'Figure 5: Experiment 2 capacitor resonance voltage offset relationship with generated radio frequency going from 90MHz to 180 MHz in 10MHz increments.',
    x_err=v_data_error
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

# Calculate g-factor and uncertainty using traditional method
g_indirect = (h / mu_B) / s_opt
g_indirect_std = g_indirect * (s_opt_std / s_opt)  # Relative error propagation

print(f"Method 1 - Indirect calculation:")
print(f"Slope (s_opt): {s_opt:.3e} ± {s_opt_std:.3e} T/Hz")
print(f"g-Factor: {g_indirect:.4f} ± {g_indirect_std:.4f}")



# Calculate magnetic field values corresponding to each voltage
# Use the calibration relationship: B = s_cal * V + b_cal
B_values = s_cal * v_data + b_cal  # B in Tesla

# Calculate errors in B values
B_errors = np.sqrt(
    (s_cal * v_data_error)**2 + 
    (v_data * s_cal_std)**2 + 
    b_cal_std**2
)

# Prepare data for h*f vs mu_B*B plot
mu_B_times_B = mu_B * B_values
mu_B_times_B_errors = mu_B * B_errors

h_times_f = h * f_values 
# Use small non-zero errors for numerical stability
h_times_f_errors = np.full_like(h_times_f, h * 1)

# Direct regression to calculate g-factor
# g-factor is the slope of h*f vs mu_B*B
g_direct, g_intercept, g_direct_std, g_intercept_std = plot_regression(
    mu_B_times_B, h_times_f, h_times_f_errors,
    r'$\mu_B B_0$ (J)', r'$\hbar\omega$ (J)',
    'g-factor Plot 2.png', 'Figure 6: Direct measurement of g-factor as the slope of h·f vs μ_B·B plot for Experiment 2. The slope of this line gives the g-factor directly.',
    x_err=mu_B_times_B_errors
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
#%%