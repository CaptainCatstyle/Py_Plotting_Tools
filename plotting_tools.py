import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from pathlib import Path


def read_clipboard():
    """
    :brief: Reads users clipboard data to get needed plot data

    :return: (np.Array) x, Delta x, (np.Array) y, Delta y

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
    :brief: Reads csv or txt file to extract plot data

    :param FilePath: (String) Filepath to the Csv or txt file. Recommended is the Input as a raw String

    :return: (np.Array) x, Delta x, (np.Array) y, Delta y

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
    :brief: Linear regression of the given x and y values.

    :param x: (np.array/List) X-Values
    :param y: (np.array/List) Y-Values

    :return: (float) Slope, (float) Y-Intercept, (float) Delta Slope, (float) Delta Y-Intercept, (np.array) x-fit, (np.array) y-fit

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


def round_din1333(value: float, uncertainty: float):
    """
    :brief: Takes a value and the corresponding uncertainty and rounds it according to Din 1333 regulations
    :disclaimer: This function is error susceptible and the output should always be checked for correctness

    :param value: (float) nominal value
    :param uncertainty: (float) uncertainty of nominal value
    :return: (float) rounded nominal value, (float) rounded uncertainty

    :author: ChatGPT
    """
    import math  # local import keeps the function self-contained

    # ---------- local helpers ----------
    def _round_uncertainty(u: float) -> float:
        if u == 0:
            return 0.0
        sgn = 1 if u > 0 else -1
        u = abs(u)

        exp10 = math.floor(math.log10(u))
        lead = u / 10**exp10          # in [1, 10)
        first = int(lead)

        if first in (1, 2):           # keep two digits
            block = int(lead * 10 + 1)  # always upward
            if block >= 100:          # 19 → 20, etc.
                block, exp10 = 10, exp10 + 1
            rounded = block / 10 * 10**exp10
        else:                         # keep one digit
            block = first + 1
            if block == 10:
                block, exp10 = 1, exp10 + 1
            rounded = block * 10**exp10
        return sgn * rounded

    def _half_up(x: float, step: float) -> float:
        q = x / step
        return (math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)) * step

    def _step_from_u(u: float) -> float:
        u = abs(u)
        exp10 = math.floor(math.log10(u))
        first = int(u / 10**exp10)
        sigs = 2 if first in (1, 2) else 1
        return 10 ** (exp10 - (sigs - 1))

    def _fmt(x: float, step: float) -> str:
        dec = 0 if step >= 1 else int(round(-math.log10(step)))
        return f"{x:.{dec}f}"

    # ---------- main algorithm ----------
    u_r = _round_uncertainty(uncertainty)
    step = _step_from_u(u_r)
    v_r = _half_up(value, step)
    return _fmt(v_r, step), _fmt(u_r, step)



