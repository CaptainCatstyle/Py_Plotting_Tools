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
else:
    fig, axes = plt.subplots(rows, columns, figsize=(figWidth, figHeight))
    ax = axes[0][0]


x ,xError, y, yError = sc.read_clipboard()
sc.plot_errorbars(x, xError, y, yError, 'X Values', 'Y Values', 'Test Plot')


plt.legend(loc='upper left', fancybox=False, edgecolor='black')
plt.show()