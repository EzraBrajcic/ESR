import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt

def linear_model(X, s, b):
    return s * X + b

def weighted_linear_regression(x, y, y_err):
    
    sigma = 1 / y_err
    
    popt, pcov = curve_fit(linear_model, x, y, sigma = sigma)
    s_opt, b_opt = popt
    s_std, b_std = np.sqrt(np.diag(pcov))
    return s_opt, b_opt, s_std, b_std

def orthogonal_distance_regression(x, y, x_err, y_err):
    """
    Perform orthogonal distance regression with improved error handling.
    """
    # Data normalization to prevent numerical instability
    x_scale = np.std(x)
    y_scale = np.std(y)
    x_norm = x / x_scale
    y_norm = y / y_scale
    
    # Error scaling
    x_err_norm = x_err / x_scale
    y_err_norm = y_err / y_scale
    
    # Get initial parameter estimate from weighted linear regression
    popt, pcov = curve_fit(linear_model, x_norm, y_norm)
    s_opt, b_opt = popt
    beta0 = s_opt, b_opt
    
    # Define linear model for ODR
    odr_model = Model(lambda beta, x: beta[0] * x + beta[1])
    
    # Create data object
    data = RealData(x_norm, y_norm, sx=x_err_norm, sy=y_err_norm)
    
    # Create ODR object with appropriate settings
    odr = ODR(data, odr_model, beta0=beta0, maxit=100)
    
    # Force more iterations by tightening convergence criteria
    odr.sstol = 1e-50  # Sum of squares tolerance (default 1e-8)
    odr.partol = 1e-50  # Parameter tolerance (default 1e-10)
    
    # Force the algorithm to take additional steps
    odr.stpb = (2, 2)  # Initial step bounds for parameters
    odr.sclb = [1.0, 1.0]  # Scale factors for parameters
    
    # Use a more rigorous ODR method
    odr.set_job(fit_type=2)  # Explicit ODR with weights
    
    # Perform the regression
    output = odr.run()
    
    # Check for successful convergence
    if output.info > 3:
        print(f"Warning: ODR did not converge properly. Info code: {output.info}")
    
    # Check if covariance matrix is reliable
    if output.cov_beta is None or np.linalg.cond(output.cov_beta) > 1e6:
        print("Warning: ODR covariance matrix may be unstable")
        
        # Use a more robust approach to estimate uncertainties
        residuals = y_norm - (output.beta[0] * x_norm + output.beta[1])
        dof = len(x) - 2
        variance = np.sum(residuals**2) / dof
        J = np.column_stack([x_norm, np.ones_like(x_norm)])
        cov = variance * np.linalg.inv(J.T @ J)
    else:
        cov = output.cov_beta
    
    # Denormalize parameters and errors
    s_opt = output.beta[0] * y_scale / x_scale
    b_opt = output.beta[1] * y_scale
    s_std = np.sqrt(cov[0,0]) * y_scale / x_scale
    b_std = np.sqrt(cov[1,1]) * y_scale
    
    # Print diagnostic information
    print(f'ODR summary:')
    output.pprint()
    print(f'Slope = {s_opt:.6e} ± {s_std:.6e}')
    print(f'Intercept = {b_opt:.6e} ± {b_std:.6e}')
    
    return s_opt, b_opt, s_std, b_std

def data_plot(x, y, x_err, y_err, s_opt, b_opt, s_std, b_std, filename, xlabel, ylabel, caption):
    
    x_fit = np.linspace(min(x), max(x), 10000)
    y_fit = linear_model(x_fit, s_opt, b_opt)
    
    plt.fill_between(x_fit, 
                     linear_model(x_fit, s_opt + s_std, b_opt + b_std),
                     linear_model(x_fit, s_opt - s_std, b_opt - b_std),
                     color='navy', alpha=0.2, label='Uncertainty')
    
    plt.plot(x_fit, y_fit, 'navy', label='Fit')
    
    plt.errorbar(x, y, xerr = x_err, yerr = y_err, fmt='o', label='Data', color='purple', ms = 3, capsize = 3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=8)
    plt.savefig(filename, dpi=600)
    plt.show()
    plt.close()
    
    return

def plot_regression(x, y, y_err, xlabel, ylabel, filename, caption, x_err=None):
    # Use ODR if x errors are provided
    if x_err is not None and np.any(x_err > 0):
        s_opt, b_opt, s_std, b_std = orthogonal_distance_regression(x, y, x_err, y_err)
    else:
        s_opt, b_opt, s_std, b_std = weighted_linear_regression(x, y, y_err)
    
    data_plot(x, y, x_err, y_err, s_opt, b_opt, s_std, b_std, filename, xlabel, ylabel, caption)

    return s_opt, b_opt, s_std, b_std