import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path


def read_clipboard():
    """
    :brief: Reads the user clipboard and extracts plot data

    :return: (tuple) Returns either
                     (x, xError, y, yError)
                     or (x, y)
                     or y
                     depending on the number of detected columns

    :author: Baran Duendar
    """

    try:
        df = pd.read_clipboard()
        if len(df.columns) == 4:
            x = np.array(df.iloc[:, 0])
            xError = np.array(df.iloc[:, 1])
            y = np.array(df.iloc[:, 2])
            yError = np.array(df.iloc[:, 3])
            return x, xError, y, yError
        elif len(df.columns) == 2:
            x = np.array(df.iloc[:, 0])
            y = np.array(df.iloc[:, 1])
            return x, y
        elif len(df.columns) == 1:
            y = np.array(df.iloc[:, 0])
            return y
    except Exception as e:
        print(f"Error reading Clipboard: {e}")


def read_csv_file(FilePath):
    """
    :brief: Reads a csv or txt file and extracts plot data

    :param FilePath: (string) Filepath to the data file. Raw strings are recommended

    :return: (tuple) Returns either
                     (x, xError, y, yError)
                     or (x, y)
                     or y
                     depending on the number of detected columns

    :author: Baran Duendar
    """

    if '\\' not in FilePath:
        FilePath = next((Path.home() / "Desktop").rglob(FilePath), None)

    try:
        df = pd.read_csv(FilePath, sep=r'\s+')
        if len(df.columns) == 4:
            x = np.array(df.iloc[:, 0])
            xError = np.array(df.iloc[:, 1])
            y = np.array(df.iloc[:, 2])
            yError = np.array(df.iloc[:, 3])
            return x, xError, y, yError
        elif len(df.columns) == 2:
            x = np.array(df.iloc[:, 0])
            y = np.array(df.iloc[:, 1])
            return x, y
        elif len(df.columns) == 1:
            y = np.array(df.iloc[:, 0])
            return y
    except Exception as e:
        print(f"Error reading CSV file: {e}")


def linear_regression(x, y):
    """
    :brief: Performs a linear regression on the given x and y data

    :param x: (np.array/list) X values
    :param y: (np.array/list) Y values

    :return: (tuple) (x_fit, y_fit, slope, delta_slope, intercept, delta_intercept)

    :author: Baran Duendar
    """
    res = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = res.slope * x_fit + res.intercept

    print('Slope:', res.slope)
    print('Delta Slope:', res.stderr)
    print('Intercept:', res.intercept)
    print('Delta Intercept:', res.intercept_stderr)

    return x_fit, y_fit, res.slope, res.stderr, res.intercept, res.intercept_stderr


def exponential_fit(model, x, y):
    """
    :brief: Fits an exponential function to the given data

    :param model: (int) Selects the functional form
                  model 1 -> a * exp(b * x)
                  model 2 -> a * exp(-x / b)
    :param x: (list/np.array) Values on the X axis
    :param y: (list/np.array) Values on the Y axis

    :return: (tuple) (x_fit, y_fit, a_fit, a_err, b_fit, b_err)

    :author: Baran Duendar
    """

    def exponential_model(x, a, b):
        if model == 1:
            return a * np.exp(x * b)
        elif model == 2:
            return a * np.exp(-x / b)

    initial_guess = [y[0], x[0]]
    popt, pcov = curve_fit(exponential_model, x, y, p0=initial_guess)

    a_fit, b_fit = popt

    perr = np.sqrt(np.diag(pcov))
    a_err, b_err = perr

    print(f"a = {a_fit} $\\pm$ {a_err}")
    print(f"b = {b_fit} $\\pm$ {b_err}")

    x_fit = np.linspace(min(x), max(x), 300)
    y_fit = exponential_model(x_fit, a_fit, b_fit)

    return x_fit, y_fit, a_fit, a_err, b_fit, b_err


def parabolic_fit(x, y):
    """
    :brief: Fits a parabolic function f(x) = a x^2 + b x + c to the given data

    :param x: (list/np.array) Values on the X axis
    :param y: (list/np.array) Values on the Y axis

    :return: (tuple) (x_fit, y_fit, a_fit, a_err, b_fit, b_err, c_fit, c_err)

    :author: Baran Duendar
    """

    def parabolic_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    initial_guess = [x[0], y[0], 0.0]
    popt, pcov = curve_fit(parabolic_model, x, y, p0=initial_guess)

    a_fit, b_fit, c_fit = popt

    perr = np.sqrt(np.diag(pcov))
    a_err, b_err, c_err = perr

    print(f"a = {a_fit} $\\pm$ {a_err}")
    print(f"b = {b_fit} $\\pm$ {b_err}")
    print(f"b = {c_fit} $\\pm$ {c_err}")

    x_fit = np.linspace(min(x), max(x), 300)
    y_fit = parabolic_model(x_fit, a_fit, b_fit, c_fit)

    return x_fit, y_fit, a_fit, a_err, b_fit, b_err, c_fit, c_err


def strip_leading_zeros(num):
    import re
    """
    :brief: Removes leading zeros of a given number

    :param num: (float/str) Number with leading zeros

    :return: (int) Number without leading zeros

    :author: Baran Duendar 
    """
    if type(num) == int or type(num) == float:
        num = str(num)

    return int(re.sub(r'^0*\.0*', '', num))


def check_for_positive_intercept(intercept):
    """
    :brief: Checks the sign of the intercept and returns the correct symbol

    :param intercept: (float or int) Intercept value

    :return: (string) '+' if the intercept is positive otherwise '-'

    :author: Baran Duendar
    """

    if float(intercept)>=0:
        return '+'
    else:
        return '-'


def smooth_connect(x, y, kind):
    """
    :brief: Creates a smooth connection between data points using interpolation

    :param x: (np.array or list) X values
    :param y: (np.array or list) Y values
    :param kind: (string) Interpolation type supported by scipy interp1d

    :return: (tuple) (x_smooth, y_smooth) Interpolated data for smooth plotting

    :author: Baran Duendar
    """

    x_new = np.linspace(x.min(), x.max(), 500)
    f = interp1d(x, y, kind=kind)
    y_smooth = f(x_new)
    return x_new, y_smooth


def round_uncertainty(uncertainty):
    """
    :brief: Takes an uncertainty and rounds it according to DIN 1319-3 regulations

    :param uncertainty: (float/int) uncertainty to be rounded

    :return: (float) rounded uncertainty

    :author: Baran Duendar
    """

    if uncertainty == 0:
        return 0.0

    uncertainty = np.abs(uncertainty)
    exp10 = np.floor(np.log10(uncertainty))
    lead = uncertainty / (10 ** (exp10 - 1))

    if lead < 30:
        sigs = 2
        rounded = np.ceil(lead) * (10 ** (exp10 - 1))
    else:
        sigs = 1
        rounded = np.ceil(lead / 10) * (10 ** exp10)

    decimals = int(max(0, -(exp10 - (sigs - 1))))
    rounded = float(f"{rounded:.{decimals}f}")

    return rounded


def round_to_digit(value, digit, output_str=False):
    """
    :brief: Normal arithmetic rounding (half away from zero) to a configurable digit.

    :param value: (float/int) value to be rounded
    :param digit: (int) digit to round to
                digit > 0  -> round to decimals
                digit = 0  -> round to ones
                digit < 0  -> round to tens, hundreds, etc.
    :param: output_str: (boolean) Switch between String or float output

    :return: (float/string) rounded value

    :author: Baran Duendar
    """

    factor = 10 ** digit
    y = value * factor

    # arithmetic rounding: half away from zero
    if y >= 0:
        y = np.floor(y + 0.5)
    else:
        y = np.ceil(y - 0.5)

    rounded = y / factor

    if output_str:
        if digit > 0:
            rounded_str = f"{rounded:.{digit}f}"
        else:
            rounded_str = f"{rounded:.0f}"

        return rounded_str
    else:
        return rounded


def round_din1333(value, uncertainty):
    """
    :brief: Takes a value and its uncertainty and rounds both according to DIN 1333 style rules.

    :param value: (float/int) value
    :param uncertainty: (float/int) uncertainty of value

    :return: (tuple) (rounded value, rounded uncertainty)

    :author: Baran Duendar
    """

    uncertainty = round_uncertainty(uncertainty) # round uncertainty
    if uncertainty == 0:
        return float(value), 0.0

    exp10 = int(np.floor(np.log10(uncertainty))) # calculate digit to round to

    uncertainty_digits = uncertainty / (10 ** exp10)
    tol = 1e-9
    has_one_sig_digit = abs(uncertainty_digits - round(uncertainty_digits)) < tol
    if uncertainty < 1 and not has_one_sig_digit:
        digit = -exp10 + 1
    else:
        digit = -exp10

    value = round_to_digit(value, digit, True) # round value

    return value, uncertainty