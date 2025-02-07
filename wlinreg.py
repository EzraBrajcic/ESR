import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the linear model
def linear_model(X, s, b):
    return s * X + b

def weighted_linear_regression(x, y, y_err):
    """Weighted linear regression with error propagation."""
    weights = 1 / y_err**2
    popt, pcov = curve_fit(linear_model, x, y, sigma=weights)
    s_opt, b_opt = popt
    s_std, b_std = np.sqrt(np.diag(pcov))
    return s_opt, b_opt, s_std, b_std

def plot_regression(x, y, y_err, xlabel, ylabel, filename, caption):
    """Plot data, fitted line, and uncertainty."""
    s_opt, b_opt, s_std, b_std = weighted_linear_regression(x, y, y_err)
    x_fit = np.linspace(min(x), max(x), 10000)
    y_fit = linear_model(x_fit, s_opt, b_opt)
    
    plt.errorbar(x, y, yerr = y_err, fmt = 'o', label = 'Data', color = 'purple')
    plt.plot(x_fit, y_fit, 'navy', label='Fit')
    plt.fill_between(x_fit, 
                     linear_model(x_fit, s_opt + s_std, b_opt + b_std),
                     linear_model(x_fit, s_opt - s_std, b_opt - b_std),
                     color = 'navy', alpha = 0.2, label = 'Uncertainty')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Add caption below the figure
    plt.figtext(0.5, -0.1, caption, wrap = True, horizontalalignment = 'center', fontsize = 8)
    plt.savefig(filename, bbox_inches = 'tight', dpi=600)
    plt.show()
    plt.close()
    return s_opt, b_opt, s_std, b_std