import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import scienceplots
import scatter as sc

plt.style.use(['science', 'notebook', 'grid'])

rows = 1
columns = 1
figWidth = 12
figHeight = 6

if rows == 1 and columns == 1:
    fig, ax = plt.subplots(rows, columns, figsize=(figWidth, figHeight))
elif rows == 1 and columns > 1:
    fig, axes = plt.subplots(rows, columns, figsize=(figWidth, figHeight))
    ax = axes[0]
    plt.sca(ax)
else:
    fig, axes = plt.subplots(rows, columns, figsize=(figWidth, figHeight))
    ax = axes[0][0]
    plt.sca(ax)

plt.sca(ax)
x ,xError, y, yError = sc.read_clipboard()
sc.plot_errorbars(x, xError, y, yError, 'X Values', 'Y Values', 'Test Plot')
slope, std_err_slope, intercept, std_err_intercept = sc.plot_regression(x, y)
plt.legend([f"Slope = ({slope:.3f} $\\pm$ {std_err_slope:.3f}) \n Y-Intercept = ({intercept:.3f} $\\pm$ {std_err_intercept:.3f})"],loc='upper left', fancybox=False, edgecolor='black')

plt.show()