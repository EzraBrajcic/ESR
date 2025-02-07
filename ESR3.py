#%%
import os
import numpy as np
from wlinreg import *

'''Day 2 calibration run #1 measurements'''
measurements = 10

V_offset = np.arange([1.00, float(measurements - 1), 1])

T1 = np.array([12.8, 29.8, 36.5, 48.0, 59.6, 70.7, 80.2, 90.6]) * 1e-4 #Magnetic field in tesla

T1_error = np.empty(measurements - 2)
T1_error.fill(0.2 * 1e-4)  # Convert to Tesla



'''Day 2 calibration run #2 measurements'''
T2 = np.array([13.0, 24.3, 36.0, 47.3, 58.9, 69.8, 82.0, 90.0]) * 1e-4 #Magnetic field in tesla

T2_error = np.empty(measurements - 2)
T2_error.fill(0.2 * 1e-4)  # Convert to Tesla



# Calculate the average of T1 and T2
T_avg = (T1 + T2) / 2
T_avg_error = np.sqrt(T1_error**2 + T2_error**2) / 2  # Correct error propagation



# Perform weighted linear regression for the calibration data
s_cal, b_cal, s_cal_std, b_cal_std = plot_regression(
    V_offset, T_avg, T_avg_error,
    'Voltage Offset (V)', 'Magnetic Field (T)',
    'Calibration Plot 2.svg', 'Figure 3: Experiment 2 calibration measurements of voltage offset relationship with magnetic field strengths of helmholtz coils. Offset voltages going in increments of 1.00V from 1.00V to 8.00V'
)



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

for mhz, n, v, amplitude, current in freq_data:
    upper_percentile = np.percentile(amplitude, 95)
    average = 0
    
    #print(upper_percentile, average,'\n')
    
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
    #print(final_x)
    
    mhz_data.append(mhz)
    v_data.append(final_x)

plt.xlabel('Voltage (V)')
plt.ylabel('Amplitude')

plt.plot([0, 10], [0,0], linestyle='--')

plt.legend()

# Add caption below the figure
plt.figtext(0.5, -0.1, 'Figure 4: Experiment 2 voltage offset sweep for each frequency, with resonance points. Voltage offset ranges from 1.00V to 10.00V over 3 minutes and frequencies sweep from 90MHz to 180Hz in 10MHz increments', wrap = True, horizontalalignment = 'center', fontsize = 8)
plt.savefig('Resonance Intercepts.svg' , bbox_inches = 'tight', dpi=600)
plt.show()
plt.close()

# Perform weighted linear regression on the extracted data
mhz_data = np.array(mhz_data) * 1e6

v_data = np.array(v_data)
v_data_error = np.full_like(v_data, 0.01)  # Assuming a constant error of ±0.01V

# Perform resonance regression (Frequency vs Voltage)
s_frequency, s_frequency_intercept, s_frequency_std, _ = plot_regression(
    v_data, mhz_data, v_data_error,
    'Voltage (V)', 'Frequency (Hz)',
    'Resonance Plot 2.svg', 'Figure 5: Experiment 2 capacitor resonance voltage offset relationship with generated radio frequency going from 90MHz to 180 MHz in 10MHz increments.'
)

# Calculate effective slope s_opt = s_cal / s (Tesla/Hz)
s_opt = s_cal / s_frequency
# Propagate errors in s_opt = s_cal / s_frequency
s_opt_std = s_opt * np.sqrt(
    (s_cal_std / s_cal)**2 + 
    (s_frequency_std / s_frequency)**2 +
    (np.mean(v_data_error)**2 / np.mean(v_data)**2)  # Voltage error term
)

# Calculate g-factor and uncertainty
h = 6.626e-34  # Planck's constant
mu_B = 9.274e-24  # Bohr magneton
g = (h / mu_B) / s_opt
g_std = g * (s_opt_std / s_opt)  # Relative error propagation

print(f"Slope (s_opt): {s_opt:.2e} ± {s_opt_std:.2e} T/Hz")
print(f"g-Factor: {g:.4f} ± {g_std:.4f}")
#%%
