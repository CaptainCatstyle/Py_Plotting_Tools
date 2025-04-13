"""
Created on Fri Apr  4 16:30:16 2025

@author: Baran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

"""
:Brief: Reads data from clipboard and saves it as a pandas dataframe

:return: Variables x, xError, y, yError
:rtype: Numpy array, numpy Array, numpy Array, numpy Array 
"""
def read_clipboard():
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

"""
:Brief: Reads specified CSV file and saves the data as a pandas dataframe
    
:param FilePath: CSV Filepath
:type FilePath: String
    
:return: Variables x, xError, y, yError
:rtype: Numpy array, numpy Array, numpy Array, numpy Array 
"""
def read_csv_file(FilePath):
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


def plot_errorbars(x, xError, y, yError, xlabel='X-Axis', ylabel='Y-Axis', title='Title', legend='', marker='o', linestyle='', color='black', capsize=5):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.errorbar(x, y, xerr=xError, yerr=yError, marker=marker, linestyle=linestyle, color=color, capsize=capsize,
                label=legend)

def plot_regression(x, y,):
    slope, intercept, r_value, p_value, std_err_slope = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept

    n = len(x)
    x_mean = np.mean(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    s_e = np.sqrt(np.sum(residuals ** 2) / (n - 2))  # std error of regression
    sum_x_deviation_squared = np.sum((x - x_mean) ** 2)
    std_err_intercept = s_e * np.sqrt(1 / n + (x_mean ** 2 / sum_x_deviation_squared))

    plt.plot(x_fit, y_fit)
    return slope, std_err_slope, intercept, std_err_intercept



