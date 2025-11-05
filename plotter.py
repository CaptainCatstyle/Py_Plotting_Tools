import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import plotting_tools as pt


plt.style.use(['science', 'notebook', 'grid'])

rows = 1
columns = 1
figWidth = 12
figHeight = 6
sharey = False
sharex = False
figDpi = 150

if rows == 1 and columns == 1:
    fig, ax = plt.subplots(rows, columns, figsize=(figWidth, figHeight), dpi=figDpi)
elif rows == 1 and columns > 1:
    fig, axes = plt.subplots(rows, columns, figsize=(figWidth, figHeight), sharey=sharey, sharex=sharex, dpi=figDpi)
    ax = axes[0]
    plt.sca(ax)
else:
    fig, axes = plt.subplots(rows, columns, figsize=(figWidth, figHeight), sharey=sharey, sharex=sharex, dpi=figDpi)
    ax = axes[0][0]
    plt.sca(ax)

x ,xError, y, yError = pt.read_clipboard()
ax.errorbar(x, y, xerr=xError, yerr=yError, linestyle='', marker='o', capsize=4, color='black', markersize=3)
x_fit, y_fit, slope, std_err_slope, intercept, std_err_intercept = pt.linear_regression(x, y)
slope, std_err_slope = pt.round_din1333(slope, std_err_slope)
intercept, std_err_intercept = pt.round_din1333(intercept, std_err_intercept)
std_err_slope = pt.strip_leading_zeros(std_err_slope)
std_err_intercept = pt.strip_leading_zeros(std_err_intercept)
ax.plot(x_fit, y_fit, linestyle='-', label=f"fit: y = {slope}({std_err_slope})$\\,$(Einheit) {pt.check_for_positive_intercept(intercept)} {np.abs(float(intercept))}({std_err_intercept})$\\,$(Einheit)")

ax.legend(loc='upper left', fancybox=False, edgecolor='black')
ax.set_xlabel('X-Achse $Größe\\,$(Einheit)')
ax.set_ylabel('Y-Achse $Größe\\,$(Einheit)')
ax.set_title('Title')

plt.show()