#!/usr/bin/env python
"""
(1) Random intercepts bare-distance model

build_model()

This plots the seasonal bare fractional cover values against distance from
artifical water points. It uses a linear mixed model, with paddock as a random
effect, allowing different intercept values. I followed this guide:
https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

It outputs:
- the results of all regression analyses in a CSV file
- density plots of bare versus distance for each season, showing regression lines
- scatter plots and histograms of model residuals

plot_results()

This plots the time series of slope and intercept values (where the p-value for
slope is significant, < 0.05), as well as seasonal rainfall. It also uses
Anderson-Darling tests to see if the distribution of values is statistically
different before and after conservation. The results were:
- n = 143 (pastoral = 88, conservation = 55)
- There is a significant difference for slope (p = 0.01) and intercept
  (p = 0.01), but not for rainfall (p = 0.17).

2. Random slopes and intercepts model

build_single_paddock_model()

Does the same as for (1) but also allows the slope for each paddock to vary. The
density plots have subplots for each paddock.

4. Slope-rainfall models


"""


import os
import sys
import datetime
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from scipy import stats
from scipy import signal
import statsmodels.api as sm
import statsmodels.formula.api as smf


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def build_model():
    
    # Get seasonal dates
    start = 198712198802
    end = 202409202411
    dateList = []
    for y1 in range(1987, 2025):
        for m1 in range(3, 13, 3):
            if m1 < 12:
                y2 = y1
                m2 = m1 + 2
            else:
                y2 = y1 + 1
                m2 = 2
            date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
            if date >= start and date <= end:
                dateList.append(date)
    
    # Read in data
    csvfile = r'D:\witchelina\3_seasonal_analyses\awp_seasonal_analysis_epsg3577_1987_2024.csv'
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    data['Distance'] = data['Distance'] / 1000.0

    # Model relationships between bare and distance to water
    outbase = r'D:\witchelina\3_seasonal_analyses\plots'
    cover = 'Bare'
    
    outCsv = os.path.join(outbase, r'seasonal_bare_distance_model.csv')
    with open(outCsv, 'w') as f:
        f.write('season,n,'+
                'slope,slope_stder,slope_95ci_lower,slope_95ci_upper,slope_pvalue,'+
                'intercept,intercept_stderr,intercept_95ci_lower,intercept_95ci_upper,intercept_pvalue\n')
    
    for date in dateList:
        colname = '%s_%s'%(cover, date)
        outplot = os.path.join(outbase, r'%s.png'%colname)
        fig = plt.figure()
        fig.set_size_inches((3, 3))
        ax = plt.axes([0.2, 0.2, 0.7, 0.7])
        ax.set_facecolor('k')
        ax.set_title(date, fontsize=10)
        fc = data[colname]
        d = data['Distance']
        h = ax.hist2d(d, fc, bins=[50, 50], range=[[0, 8], [0, 100]], cmap='Greys')
        ax.set_xlabel('Distance to water point (km)')            
        ax.set_ylabel('%s (%%)'%cover)
        ax.set_xlim([0, 8])
        ax.set_ylim([0, 100])
        
        # Remove 255 from data
        if np.max(data[colname]) == 255:
            nodata_rows = (data[colname] == 255)
            subset = data[:][~nodata_rows]
        else:
            subset = data
        
        # Fit the linear mixed effects model
        n = subset[colname].size
        md = smf.mixedlm(formula=f"{colname} ~ Distance", data=subset, groups=subset["Paddock"])
        mdf = md.fit(method=["powell", "lbfgs"])
        
        intercept = mdf.fe_params['Intercept']
        intercept_stderr = mdf.bse_fe['Intercept']
        intercept_95ci = (mdf.conf_int()[0]['Intercept'], mdf.conf_int()[1]['Intercept'])
        intercept_pvalue = mdf.pvalues['Intercept']
        
        slope = mdf.fe_params['Distance']
        slope_stder = mdf.bse_fe['Distance']
        slope_95ci = (mdf.conf_int()[0]['Distance'], mdf.conf_int()[1]['Distance'])
        slope_pvalue = mdf.pvalues['Distance']
        
        for paddock in np.unique(subset["Paddock"]):
            p_intercept_effect = mdf.random_effects[paddock]["Group"]
            maxD = np.max(subset['Distance'][subset["Paddock"] == paddock])
            ax.plot([0, maxD], [intercept+p_intercept_effect, maxD*slope+intercept+p_intercept_effect], ls='-', c='r', lw=0.5)
        
        plt.savefig(outplot, dpi=300)
        plt.close()
        
        with open(outCsv, 'a') as f:
            line = '%i,%i'%(date, n)
            line += ',%.4f,%.4f,%.4f,%.4f,%.12f'%(slope, slope_stder, slope_95ci[0], slope_95ci[1], slope_pvalue)
            line += ',%.4f,%.4f,%.4f,%.4f,%.12f\n'%(intercept, intercept_stderr, intercept_95ci[0], intercept_95ci[1], intercept_pvalue)
            f.write(line)
        
        # Make residual plots (residuals vs predicted values and histogram of residuals)
        res_folder = os.path.join(outbase, 'residual_plots')
        if os.path.exists(res_folder) is False:
            os.mkdir(res_folder)
        predictions = mdf.predict()
        residuals = mdf.resid
        resplot = os.path.join(res_folder, r'%s_residuals.png'%colname)
        fig = plt.figure()
        fig.set_size_inches((6, 3))
        fig.text(0.5, 0.95, '%s %i'%(cover, date), fontsize=10, horizontalalignment='center')
        ax1 = plt.axes([0.1, 0.2, 0.35, 0.7])
        ax1.plot(predictions, residuals, 'o', c='0.7', markersize=2)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        (slope, intercept, r, p, se) = stats.linregress(predictions, residuals)
        x = [np.min(predictions), np.max(predictions)]
        y = [intercept + slope * x[0], intercept + slope * x[1]]
        ax1.plot(x, y, 'r--', lw=1)
        ax1.set_xlabel('Predicted values')            
        ax1.set_ylabel('Model residuals')
        ax2 = plt.axes([0.6, 0.2, 0.35, 0.7])
        ax2.hist(residuals, bins=100)
        ax2.set_xlabel('Residuals')
        plt.savefig(resplot, dpi=300)
        plt.close()


def build_paddock_models():
    
    # Get seasonal dates
    start = 198712198802
    end = 202409202411
    dateList = []
    for y1 in range(1987, 2025):
        for m1 in range(3, 13, 3):
            if m1 < 12:
                y2 = y1
                m2 = m1 + 2
            else:
                y2 = y1 + 1
                m2 = 2
            date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
            if date >= start and date <= end:
                dateList.append(date)
    
    # Read in data
    #csvfile = r'D:\witchelina\3_seasonal_analyses\awp_seasonal_analysis_epsg3577_1987_2024.csv'
    csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\awp_seasonal_analysis_epsg3577_1987_2024.csv'
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    data['Distance'] = data['Distance'] / 1000.0

    # Model relationships between bare and distance to water separately for each paddock
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\paddocks'
    cover = 'Bare'
    
    outCsv = os.path.join(outbase, r'paddock_models.csv')
    with open(outCsv, 'w') as f:
        f.write('paddock,season,n,'+
                'slope,slope_stder,slope_pvalue,'+
                'intercept,intercept_stderr,intercept_pvalue\n')
    
    for paddock in np.unique(data["Paddock"]):
        p_dir = os.path.join(outbase, paddock.replace(' ', '_'))
        if os.path.exists(p_dir) is False:
            os.mkdir(p_dir)
        
        for date in dateList:
            colname = '%s_%s'%(cover, date)
            outplot = os.path.join(p_dir, r'%s_%s.png'%(paddock, colname))
            fig = plt.figure()
            fig.set_size_inches((3, 3))
            ax = plt.axes([0.2, 0.2, 0.7, 0.7])
            ax.set_facecolor('k')
            ax.set_title('%s %i'%(paddock, date), fontsize=10)
            ax.set_xlabel('Distance to water point (km)')            
            ax.set_ylabel('%s (%%)'%cover)
            ax.set_xlim([0, 8])
            ax.set_ylim([0, 100])
            
            # Remove 255 from data
            subset = data[:][data['Paddock'] == paddock]
            if np.max(subset[colname]) == 255:
                nodata_rows = (subset[colname] == 255)
                subset = subset[:][~nodata_rows]
            
            n = subset[colname].size
            if n > 100:
                
                print(paddock, date, n)
                
                # Add data to plot
                h = ax.hist2d(subset['Distance'], subset[colname], bins=[50, 50], range=[[0, 8], [0, 100]], cmap='Greys')
                
                # Fit the linear mixed effects model
                md = sm.OLS(subset[colname], sm.add_constant(subset['Distance']))
                mdf = md.fit()
                
                intercept = mdf.params[0]
                intercept_stderr = mdf.bse[0]
                intercept_pvalue = mdf.pvalues[0]
                
                slope = mdf.params[1]
                slope_stder = mdf.bse[1]
                slope_pvalue = mdf.pvalues[1]
                
                maxD = np.max(subset['Distance'])
                ax.plot([0, maxD], [intercept, maxD*slope+intercept], ls='-', c='r', lw=0.5)
                
            else:
                slope = 255
                slope_stder = 255
                slope_pvalue = 255
                intercept = 255
                intercept_stderr = 255
                intercept_pvalue = 255
                
            with open(outCsv, 'a') as f:
                line = '%s,%i,%i'%(paddock, date, n)
                line += ',%.4f,%.4f,%.12f'%(slope, slope_stder, slope_pvalue)
                line += ',%.4f,%.4f,%.12f\n'%(intercept, intercept_stderr, intercept_pvalue)
                f.write(line)
            
            plt.savefig(outplot, dpi=300)
            plt.close()
            

            
def plot_results():
    
    # Read in regression data and create central date for seasons
    #outbase = r'D:\witchelina\3_seasonal_analyses\plots'
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\plots'
    cover = 'Bare'
    csvfile = os.path.join(outbase, r'seasonal_bare_distance_model.csv')
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    
    seasonalDates = []
    for x in data['season']:
       year = int(str(x)[0:4])
       month = int(str(x)[4:6]) + 1
       if month == 13:
           year += 1
           month = 1
       seasonalDates.append(datetime.date(year=year, month=month, day=15))
    seasonalDates = np.array(seasonalDates, dtype=np.datetime64)
    data = rfn.append_fields(data, 'Date', seasonalDates)
    
    # Read in rainfall and calculate seasonal totals
    #csvFile = r'D:\witchelina\3_seasonal_analyses\witchelina_rainfall.csv'
    csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\witchelina_rainfall.csv'
    rainData = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=None, encoding=None)
    seasonalRain = []
    for x in data['season']:
        year1 = int(str(x)[0:4])
        month1 = int(str(x)[4:6])
        if month1 < 12:
            year2 = year1
            month2 = month1 + 1
            year3 = year1
            month3 = month1 + 2
        else:
            year2 = year1 + 1
            month2 = 1
            year3 = year1 + 1
            month3 = 2
        seasonalRain.append(rainData['Precipitation'][(rainData['Year'] == year1) & (rainData['Month'] == month1)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year2) & (rainData['Month'] == month2)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year3) & (rainData['Month'] == month3)][0])
    seasonalRain = np.array(seasonalRain, dtype=np.float32)
    data = rfn.append_fields(data, 'rainfall', seasonalRain)
    
    # Remove rows where slope is not significant
    nonsig_rows = (data['slope_pvalue'] > 0.05)
    data = data[:][~nonsig_rows]
    
    # Make time series plot
    outplot = csvfile.replace('.csv', '.png')     
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 5),
                            gridspec_kw={'width_ratios':  [4.5, 1.5],
                                         'height_ratios': [1.5, 1.5, 1.5],
                                         'wspace': 0.4,
                                         'hspace': 0.4})
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.05, hspace=0.01)
    
    axs[0, 0].errorbar(data['Date'], data['slope'], yerr=data['slope_stder'], fmt='ko', markersize=2)
    axs[0, 0].axvline(x=datetime.date(2010, 1, 1), color='k', linestyle='--', linewidth=1)
    axs[0, 0].set_ylabel('Slope')
    axs[0, 0].set_xlim([datetime.date(1987, 6, 1), datetime.date(2024, 12, 1)])
    axs[0, 0].set_xticklabels([])

    axs[1, 0].errorbar(data['Date'], data['intercept'], yerr=data['intercept_stderr'], fmt='ko', markersize=2)
    axs[1, 0].axvline(x=datetime.date(2010, 1, 1), color='k', linestyle='--', linewidth=1)
    axs[1, 0].set_ylabel('Intercept')
    axs[1, 0].set_xlim([datetime.date(1987, 6, 1), datetime.date(2024, 12, 1)])
    axs[1, 0].set_xticklabels([])
    
    axs[2, 0].bar(data['Date'], data['rainfall'], color='lightblue', width=93, align='edge')
    axs[2, 0].axvline(x=datetime.date(2010, 1, 1), color='k', linestyle='--', linewidth=1)
    axs[2, 0].set_ylabel('Seasonal\nrainfall (mm)')
    axs[2, 0].set_xlim([datetime.date(1987, 6, 1), datetime.date(2024, 12, 1)])
    axs[2, 0].set_ylim([0, 330])

    # Add boxplots for before/after
    # Use Anderson-Darling tests to determine if samples have the same distribution
    pastoral = data['slope'][data['Date'] < datetime.date(2010, 1, 1)]
    conservation = data['slope'][data['Date'] >= datetime.date(2010, 1, 1)]
    
    # Use violin plots instead?
    #axs[0, 1].violinplot([pastoral, conservation], showmeans=False, showmedians=True)
    
    bp = axs[0, 1].boxplot([pastoral, conservation], patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color='black')
    colors = ['lightgrey', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axs[0, 1].set_xticklabels([])
    print(stats.anderson_ksamp([pastoral, conservation], method=stats.PermutationMethod(n_resamples=100, random_state=np.random.default_rng())))
    
    pastoral = data['intercept'][data['Date'] < datetime.date(2010, 1, 1)]
    conservation = data['intercept'][data['Date'] >= datetime.date(2010, 1, 1)]
    bp = axs[1, 1].boxplot([pastoral, conservation], patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color='black')
    colors = ['lightgrey', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axs[1, 1].set_xticklabels([])
    print(stats.anderson_ksamp([pastoral, conservation], method=stats.PermutationMethod(n_resamples=100, random_state=np.random.default_rng())))
    
    pastoral = data['rainfall'][data['Date'] < datetime.date(2010, 1, 1)]
    conservation = data['rainfall'][data['Date'] >= datetime.date(2010, 1, 1)]
    bp = axs[2, 1].boxplot([pastoral, conservation], patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color='black')
    colors = ['lightgrey', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axs[2, 1].set_xticklabels(['1987-\n2009', '2010-\n2024'])
    axs[2, 1].set_ylim([0, 330])
    print(stats.anderson_ksamp([pastoral, conservation], method=stats.PermutationMethod(n_resamples=100, random_state=np.random.default_rng())))
    
    plt.savefig(outplot, dpi=300)
    plt.close()
    
    print(pastoral.size, conservation.size)


def plot_paddock_results():
    
    # Read in regression data and create central date for seasons
    #outbase = r'D:\witchelina\3_seasonal_analyses\plots'
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\paddocks'
    cover = 'Bare'
    csvfile = os.path.join(outbase, r'paddock_models.csv')
    outplot = csvfile.replace('.csv', '.png')   
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    
    seasonalDates = []
    for x in data['season']:
       year = int(str(x)[0:4])
       month = int(str(x)[4:6]) + 1
       if month == 13:
           year += 1
           month = 1
       seasonalDates.append(datetime.date(year=year, month=month, day=15))
    seasonalDates = np.array(seasonalDates, dtype=np.datetime64)
    data = rfn.append_fields(data, 'Date', seasonalDates)
    
    # Read in rainfall and calculate seasonal totals
    #csvFile = r'D:\witchelina\3_seasonal_analyses\witchelina_rainfall.csv'
    csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\witchelina_rainfall.csv'
    rainData = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=None, encoding=None)
    seasonalRain = []
    for x in data['season']:
        year1 = int(str(x)[0:4])
        month1 = int(str(x)[4:6])
        if month1 < 12:
            year2 = year1
            month2 = month1 + 1
            year3 = year1
            month3 = month1 + 2
        else:
            year2 = year1 + 1
            month2 = 1
            year3 = year1 + 1
            month3 = 2
        seasonalRain.append(rainData['Precipitation'][(rainData['Year'] == year1) & (rainData['Month'] == month1)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year2) & (rainData['Month'] == month2)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year3) & (rainData['Month'] == month3)][0])
    seasonalRain = np.array(seasonalRain, dtype=np.float32)
    data = rfn.append_fields(data, 'rainfall', seasonalRain)
    
    # Make time series plot
    outplot = csvfile.replace('.csv', '.png')     
    fig, axs = plt.subplots(nrows=18, ncols=2, figsize=(8, 16),
                            gridspec_kw={'width_ratios':  [6.8, 1],
                            'wspace': 0.2, 'hspace': 0.1})
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95)
    
    for p, paddock in enumerate(np.unique(data['paddock'])):
        
        p_rows = (data['paddock'] != paddock)
        subset = data[:][~p_rows]
        nonsig_rows = (subset['slope_pvalue'] > 0.05)
        subset = subset[:][~nonsig_rows]
        
        axs[p, 0].set_title(paddock, x=0.01, y=0.6, horizontalalignment='left', c='r')
        axs[p, 0].errorbar(subset['Date'], subset['slope'], yerr=subset['slope_stder'], fmt='ko', markersize=2)
        axs[p, 0].axvline(x=datetime.date(2010, 1, 1), color='k', linestyle='--', linewidth=1)
        axs[p, 0].set_ylabel('Slope')
        axs[p, 0].set_xlim([datetime.date(1987, 6, 1), datetime.date(2024, 12, 1)])
        axs[p, 0].set_xticklabels([])
    
        pastoral = subset['slope'][subset['Date'] < datetime.date(2010, 1, 1)]
        conservation = subset['slope'][subset['Date'] >= datetime.date(2010, 1, 1)]
        
        bp = axs[p, 1].boxplot([pastoral, conservation], patch_artist=True)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='black', marker='+')
        plt.setp(bp['medians'], color='black')
        colors = ['lightgrey', 'lightgrey']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axs[p, 1].set_xticklabels([])
    
    axs[17, 0].bar(data['Date'], data['rainfall'], color='lightblue', width=93, align='edge')
    axs[17, 0].axvline(x=datetime.date(2010, 1, 1), color='k', linestyle='--', linewidth=1)
    axs[17, 0].set_ylabel('Seasonal\nrainfall (mm)')
    axs[17, 0].set_xlim([datetime.date(1987, 6, 1), datetime.date(2024, 12, 1)])
    axs[17, 0].set_ylim([0, 330])

    pastoral = data['rainfall'][data['Date'] < datetime.date(2010, 1, 1)]
    conservation = data['rainfall'][data['Date'] >= datetime.date(2010, 1, 1)]
    bp = axs[17, 1].boxplot([pastoral, conservation], patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.setp(bp['medians'], color='black')
    colors = ['lightgrey', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axs[17, 1].set_xticklabels(['1987-\n2009', '2010-\n2024'])
    axs[17, 1].set_ylim([0, 330])

    plt.savefig(outplot, dpi=300)
    plt.close()

def analyse_with_rain():
    
    # Read in regression data and create central date for seasons
    #outbase = r'D:\witchelina\3_seasonal_analyses\plots'
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\plots'
    cover = 'Bare'
    csvfile = os.path.join(outbase, r'seasonal_bare_distance_model.csv')
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    
    seasonalDates = []
    for x in data['season']:
       year = int(str(x)[0:4])
       month = int(str(x)[4:6]) + 1
       if month == 13:
           year += 1
           month = 1
       seasonalDates.append(datetime.date(year=year, month=month, day=15))
    seasonalDates = np.array(seasonalDates, dtype=np.datetime64)
    data = rfn.append_fields(data, 'Date', seasonalDates)
    
    # Read in rainfall and calculate seasonal totals
    csvFile = r'D:\witchelina\3_seasonal_analyses\witchelina_rainfall.csv'
    rainData = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=None, encoding=None)
    seasonalRain = []
    for x in data['season']:
        year1 = int(str(x)[0:4])
        month1 = int(str(x)[4:6])
        if month1 < 12:
            year2 = year1
            month2 = month1 + 1
            year3 = year1
            month3 = month1 + 2
        else:
            year2 = year1 + 1
            month2 = 1
            year3 = year1 + 1
            month3 = 2
        seasonalRain.append(rainData['Precipitation'][(rainData['Year'] == year1) & (rainData['Month'] == month1)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year2) & (rainData['Month'] == month2)][0] +
                            rainData['Precipitation'][(rainData['Year'] == year3) & (rainData['Month'] == month3)][0])
    seasonalRain = np.array(seasonalRain, dtype=np.float32)
    data = rfn.append_fields(data, 'rainfall', seasonalRain)
    
    # Remove rows where slope is not significant
    nonsig_rows = (data['slope_pvalue'] > 0.05)
    data = data[:][~nonsig_rows]
    
    # Calculate correlation of slope and rainfall with different lags and plot
    p = data['rainfall']
    q = data['slope']
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    ccf = np.correlate(p, q, 'full')
    lags = signal.correlation_lags(len(p), len(q))
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.25, 0.25, 0.7, 0.7])
    ax.bar(lags, ccf, width=0.8, color='0.7')
    ax.set_xlim([-24.5, 0.5])
    ax.set_xlabel('Lag (seasons)')
    ax.set_ylabel('Cross correlation coefficient')
    plt.savefig(r'D:\witchelina\3_seasonal_analyses\plots\slope_rainfall_cross_correlation.png', dpi=300)
    plt.close()

    # Scatter plot of slope and rain coloured by pastoral and conservation
    pastoral = data[:][data['Date'] < datetime.date(2010, 1, 1)]
    conservation = data[:][data['Date'] >= datetime.date(2010, 1, 1)]
    
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.25, 0.25, 0.7, 0.7])
    ax.plot(pastoral['rainfall'], pastoral['slope'], 'o', c='0.7', markersize=2)
    ax.plot(conservation['rainfall'], conservation['slope'], 'o', c='r', markersize=2)
    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Slope')
    plt.savefig(r'D:\witchelina\3_seasonal_analyses\plots\slope_rainfall.png', dpi=300)
    plt.close()



def build_single_paddock_model():
    
    # Get seasonal dates
    start = 198712198802
    end = 202409202411
    dateList = []
    for y1 in range(1987, 2025):
        for m1 in range(3, 13, 3):
            if m1 < 12:
                y2 = y1
                m2 = m1 + 2
            else:
                y2 = y1 + 1
                m2 = 2
            date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
            if date >= start and date <= end:
                dateList.append(date)
    
    # Read in data
    csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\awp_seasonal_analysis_epsg3577_1987_2024.csv'
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    data['Distance'] = data['Distance'] / 1000.0

    # Model relationships between bare and distance to water
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\3_seasonal_analyses\single_paddock_model'
    cover = 'Bare'
    
    outCsv = os.path.join(outbase, r'seasonal_bare_distance_model.csv')
    with open(outCsv, 'w') as f:
        f.write('paddock,season,n,'+
                'slope,slope_stder,slope_pvalue,'+
                'intercept,intercept_stderr,intercept_pvalue\n')
    
    for date in dateList:
        colname = '%s_%s'%(cover, date)
        # outplot = os.path.join(outbase, r'%s.png'%colname)
        # fig = plt.figure()
        # fig.set_size_inches((3, 3))
        # ax = plt.axes([0.2, 0.2, 0.7, 0.7])
        # ax.set_facecolor('k')
        # ax.set_title(date, fontsize=10)
        # fc = data[colname]
        # d = data['Distance']
        # h = ax.hist2d(d, fc, bins=[50, 50], range=[[0, 8], [0, 100]], cmap='Greys')
        # ax.set_xlabel('Distance to water point (km)')            
        # ax.set_ylabel('%s (%%)'%cover)
        # ax.set_xlim([0, 8])
        # ax.set_ylim([0, 100])
        
        # Remove 255 from data
        if np.max(data[colname]) == 255:
            nodata_rows = (data[colname] == 255)
            subset = data[:][~nodata_rows]
        else:
            subset = data
        
        # Fit the linear mixed effects model
        n = subset[colname].size
        md = smf.mixedlm(formula=f"{colname} ~ Distance", data=subset, groups=subset["Paddock"], re_formula="~Distance")
        mdf = md.fit(method=["powell", "lbfgs"])
        
        intercept = mdf.fe_params['Intercept']
        intercept_stderr = mdf.bse_fe['Intercept']
        intercept_pvalue = mdf.pvalues['Intercept']
        
        slope = mdf.fe_params['Distance']
        slope_stder = mdf.bse_fe['Distance']
        slope_pvalue = mdf.pvalues['Distance']
        
        for paddock in np.unique(subset["Paddock"]):
            p_intercept_effect = mdf.random_effects[paddock]["Group"]
            p_slope_effect = mdf.random_effects[paddock]["Distance"]
            
            
            
            print("%s %s %.2f %.2f"%(date, paddock, intercept+p_intercept_effect, slope+p_slope_effect))
            
            #maxD = np.max(subset['Distance'][subset["Paddock"] == paddock])
            #ax.plot([0, maxD], [intercept+p_intercept_effect, maxD*slope+intercept+p_intercept_effect], ls='-', c='r', lw=0.5)
        
        #plt.savefig(outplot, dpi=300)
        #plt.close()
        
        sys.exit()
        
        with open(outCsv, 'a') as f:
            line = '%i,%i'%(date, n)
            line += ',%.4f,%.4f,%.4f,%.4f,%.12f'%(slope, slope_stder, slope_95ci[0], slope_95ci[1], slope_pvalue)
            line += ',%.4f,%.4f,%.4f,%.4f,%.12f\n'%(intercept, intercept_stderr, intercept_95ci[0], intercept_95ci[1], intercept_pvalue)
            f.write(line)
        
        # Make residual plots (residuals vs predicted values and histogram of residuals)
        res_folder = os.path.join(outbase, 'residual_plots')
        if os.path.exists(res_folder) is False:
            os.mkdir(res_folder)
        predictions = mdf.predict()
        residuals = mdf.resid
        resplot = os.path.join(res_folder, r'%s_residuals.png'%colname)
        fig = plt.figure()
        fig.set_size_inches((6, 3))
        fig.text(0.5, 0.95, '%s %i'%(cover, date), fontsize=10, horizontalalignment='center')
        ax1 = plt.axes([0.1, 0.2, 0.35, 0.7])
        ax1.plot(predictions, residuals, 'o', c='0.7', markersize=2)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        (slope, intercept, r, p, se) = stats.linregress(predictions, residuals)
        x = [np.min(predictions), np.max(predictions)]
        y = [intercept + slope * x[0], intercept + slope * x[1]]
        ax1.plot(x, y, 'r--', lw=1)
        ax1.set_xlabel('Predicted values')            
        ax1.set_ylabel('Model residuals')
        ax2 = plt.axes([0.6, 0.2, 0.35, 0.7])
        ax2.hist(residuals, bins=100)
        ax2.set_xlabel('Residuals')
        plt.savefig(resplot, dpi=300)
        plt.close()


#build_model()
#plot_results()
#analyse_with_rain()

#build_paddock_models()
#plot_paddock_results()

build_single_paddock_model()