# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:30:16 2025

@author: Baran
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import scienceplots

"""Variable initialization """
df = None 
x = None 
xError = None 
y = None
yError = None
FilePath = None


"""Basic plot parameter initialization"""
plt.style.use(['science', 'notebook', 'grid', 'no-latex'])

rows = 1
columns = 1
figWidth = 12
figHeight = 6


if rows == 1 and columns == 1:
    fig, ax = plt.subplots(rows,columns, figsize=(figWidth,figHeight))
else:
    fig, axes = plt.subplots(rows,columns, figsize=(figWidth,figHeight))
    ax = axes[0][0]


ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_title('Title')
marker = 'o'
#linestyle = '-'
color = 'black'
capsize = 5 #Errorbarsize
legendLabel = 'Dataset'
legendLocation='upper left'



def ReadClipboard():
    try:
        global df, x, xError, y, yError
        df = pd.read_clipboard()
        x = np.array(df.iloc[:,0])
        xError = np.array(df.iloc[:,1])
        y = np.array(df.iloc[:,2])
        yError = np.array(df.iloc[:,3])
        print(df)
    except Exception as e:
        print(f"Error reading Clipboard: {e}")
    
def ReadCsvFile():
    global df, x, xError, y, yError, FilePath
    FilePath = input('Please Input File Path: ')
    try:
        df = pd.read_csv(FilePath, sep=r'\s+')
        x = np.array(df.iloc[:,0])
        xError = np.array(df.iloc[:,1])
        y = np.array(df.iloc[:,2])
        yError = np.array(df.iloc[:,3])
        print(df)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        

if FilePath is not None:
    ReadCsvFile()
elif x is None and y is None:
    ReadClipboard()

#m, b = np.polyfit(x,y, deg=1)
#print(m,b)
#ax.plot(x_fit, y_fit, 'r-', label=f"Fit: y = {slope:.4f}x + {intercept:.4f}")



slope, intercept, r_value, p_value, std_err_slope = linregress(x, y)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = slope * x_fit + intercept

x_mean = np.mean(x)
sum_x_deviation_squared = np.sum((x - x_mean) ** 2)
std_err_intercept = std_err_slope * np.sqrt(1 / len(x) + (x_mean ** 2 / sum_x_deviation_squared))


confidence_interval = 1.96 * std_err_slope * x_fit
ax.plot(x_fit, y_fit, 'r-', label=f"Fit: y = ({slope:.2f} ± {std_err_slope:.2f})x + ({intercept:.2f} ± {std_err_intercept:.2f})")
ax.fill_between(x_fit, y_fit - confidence_interval, y_fit + confidence_interval, 
                color='red', alpha=0.2, label="Regression 95% CI")






ax.errorbar(x, y, xerr=xError, yerr=yError, marker=marker, color=color, capsize=capsize, label=legendLabel)
ax.legend(loc=legendLocation, fancybox=False, edgecolor='black')

