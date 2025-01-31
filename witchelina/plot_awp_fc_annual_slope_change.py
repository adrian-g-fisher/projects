#!/usr/bin/env python
"""
This plots the annual bare fractional cover time series statistics against
distance from artifical water points. It also outputs the results of the
regression analyses in a CSV file. To include paddock as a random effect,
allowing different intercept values for each paddock, I followed this guide:
https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

It then plots the annual intercepts and slopes over time with rainfall.

"""


import os
import sys
sys.path.remove('C:\\Users\\Adrian\\AppData\\Roaming\\Python\\Python312\\site-packages')

import datetime
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def build_model():
    # Read in data
    csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\awp_analysis_epsg3577_1988_2023.csv'
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    data['Distance'] = data['Distance'] / 1000.0

    # Model annual relationships between cover and distance to water
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\plots\annual'
    for cover in ['Bare']: #['NPV', 'PV', 'Bare']
        outdir_cover = os.path.join(outbase, cover)
        if os.path.exists(outdir_cover) is False:
            os.mkdir(outdir_cover)
        for stat in ['mean']: #['max', 'mean', 'min']
            outdir_stat = os.path.join(outdir_cover, stat)
            if os.path.exists(outdir_stat) is False:
                os.mkdir(outdir_stat)
                
            outCsv = os.path.join(outdir_stat, r'model_%s_%s.csv'%(cover, stat))
            with open(outCsv, 'w') as f:
                f.write('year,n,'+
                        'slope,slope_stder,slope_95ci_lower,slope_95ci_upper,slope_pvalue,'+
                        'intercept,intercept_stderr,intercept_95ci_lower,intercept_95ci_upper,intercept_pvalue\n')
            
            for date in range(1988, 2024):
                colname = '%s_%s_%s'%(cover, stat, date)
                outplot = os.path.join(outdir_stat, r'%s.png'%colname)
                fig = plt.figure()
                fig.set_size_inches((3, 3))
                ax = plt.axes([0.2, 0.2, 0.7, 0.7])
                ax.set_facecolor('k')
                ax.set_title(date, fontsize=10)
                fc = data[colname]
                d = data['Distance']
                h = ax.hist2d(d, fc, bins=[50, 50], range=[[0, 8], [50, 100]], cmap='Greys')
                ax.set_xlabel('Distance to water point (km)')            
                ax.set_ylabel('%s %s (%%)'%(stat.capitalize(), cover))
                ax.set_xlim([0, 8])
                ax.set_ylim([50, 100])
                
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
                res_folder = os.path.join(outdir_stat, 'residual_plots')
                if os.path.exists(res_folder) is False:
                    os.mkdir(res_folder)
                predictions = mdf.predict()
                residuals = mdf.resid
                resplot = os.path.join(res_folder, r'%s_residuals.png'%colname)
                fig = plt.figure()
                fig.set_size_inches((6, 3))
                fig.text(0.5, 0.95, '%s %s %i'%(cover, stat, date), fontsize=10, horizontalalignment='center')
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
                
            
def plot_results():
    
    # Read in rainfall and calculate 12 month total between March-Feb
    csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\rainfall\witchelina_rainfall.csv'
    rainData = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=None, encoding=None)
    adjYear = np.copy(rainData['Year'])
    adjYear[rainData['Month'] <= 2] += -1
    rainYears = np.array(np.unique(rainData['Year']))
    annualRain = np.zeros_like(rainYears).astype(np.float32)
    for i, y in enumerate(rainYears):
        annualRain[i] = np.sum(rainData['Precipitation'][adjYear == y])
    
    outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\plots\annual'
    for cover in ['Bare']: #['NPV', 'PV', 'Bare']
        outdir_cover = os.path.join(outbase, cover)
        if os.path.exists(outdir_cover) is False:
            os.mkdir(outdir_cover)
        for stat in ['mean']: #['max', 'mean', 'min']
            
            outdir_stat = os.path.join(outdir_cover, stat)
            csvfile = os.path.join(outdir_stat, r'model_%s_%s.csv'%(cover, stat))
            data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
                        
            # Make time series plot
            outplot = csvfile.replace('.csv', '.png')     
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 5),
                                    gridspec_kw={'width_ratios':  [4.5, 1.5],
                                                 'height_ratios': [1.5, 1.5, 1.5],
                                                 'wspace': 0.4,
                                                 'hspace': 0.4})
            plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.05, hspace=0.01)
            
            axs[0, 0].errorbar(data['year']+0.5, data['slope'], yerr=data['slope_stder'], fmt='ko', markersize=2)
            axs[0, 0].axvline(x=2010, color='k', linestyle='--', linewidth=1)
            axs[0, 0].set_ylabel('Slope')
            axs[0, 0].set_xlim([1988, 2024])
            axs[0, 0].set_xticklabels([])

            axs[1, 0].errorbar(data['year']+0.5, data['intercept'], yerr=data['intercept_stderr'], fmt='ko', markersize=2)
            axs[1, 0].axvline(x=2010, color='k', linestyle='--', linewidth=1)
            axs[1, 0].set_ylabel('Intercept')
            axs[1, 0].set_xlim([1988, 2024])
            axs[1, 0].set_xticklabels([])
            
            axs[2, 0].bar(rainYears, annualRain, color='lightblue', width=1, align='edge')
            axs[2, 0].axvline(x=2010, color='k', linestyle='--', linewidth=1)
            axs[2, 0].set_ylabel('Annual\nrainfall (mm)')
            axs[2, 0].set_xlim([1988, 2024])

            # Add boxplots for before/after
            # Use Anderson-Darling tests to determine if samples have the same distribution
            pastoral = data['slope'][data['year'] < 2010]
            conservation = data['slope'][data['year'] >= 2010]
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
            
            pastoral = data['intercept'][data['year'] < 2010]
            conservation = data['intercept'][data['year'] >= 2010]
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
            
            pastoral = annualRain[(rainYears < 2010) & (rainYears > 1987)]
            conservation = annualRain[(rainYears >= 2010) & (rainYears < 2024)]
            bp = axs[2, 1].boxplot([pastoral, conservation], patch_artist=True)
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], color='black', marker='+')
            plt.setp(bp['medians'], color='black')
            colors = ['lightgrey', 'lightgrey']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            axs[2, 1].set_xticklabels(['1987-\n2009', '2010-\n2023'])
            print(stats.anderson_ksamp([pastoral, conservation], method=stats.PermutationMethod(n_resamples=100, random_state=np.random.default_rng())))
            
            plt.savefig(outplot, dpi=300)
            plt.close()
            
            # # Run linear mixed models predicting slope from rainfall with
            # # grazing (before/after) as a random effect (intercept)
            # model_folder = os.path.join(outdir_stat, 'slope_vs_rainfall')
            # if os.path.exists(model_folder) is False:
                # os.mkdir(model_folder)

            # model_folder2 = os.path.join(outdir_stat, 'intercept_vs_rainfall')
            # if os.path.exists(model_folder2) is False:
                # os.mkdir(model_folder2)

            # data = rfn.append_fields(data, ['rain', 'grazing'],
                                     # [np.zeros_like(data['slope']),
                                      # np.where(data['year'] < 2010, 1, 0)])
            
            # for accumulation in range(1, 6):
                # for year in np.unique(data['year']):
                    # data['rain'][data['year'] == year] = np.sum(annualRain[(rainYears > year-accumulation) & (rainYears <= year)])
                
                # outplot = os.path.join(model_folder, r'%s_%s_%i.png'%(cover, stat, accumulation))
                # fig = plt.figure()
                # fig.set_size_inches((3, 3))
                # ax = plt.axes([0.25, 0.2, 0.7, 0.7])
                # ax.set_xlabel('Rainfall (mm)\n(Accumulation = %i)'%(accumulation))            
                # ax.set_ylabel('Slope')
                # ax.errorbar(data['rain'][data['year'] < 2010],
                            # data['slope'][data['year'] < 2010],
                            # yerr=data['slope_stder'][data['year'] < 2010], fmt='ko', markersize=2)
                            
                # ax.errorbar(data['rain'][data['year'] >= 2010],
                            # data['slope'][data['year'] >= 2010],
                            # yerr=data['slope_stder'][data['year'] >= 2010], fmt='o', markersize=2, ecolor='grey', color='grey')
                
                # md = smf.mixedlm(formula=f"slope ~ rain", data=data, groups=data["grazing"], re_formula="~rain")
                # mdf = md.fit(method=["powell", "lbfgs"])
                
                # intercept = mdf.fe_params['Intercept']
                # intercept_stderr = mdf.bse_fe['Intercept']
                # intercept_pvalue = mdf.pvalues['Intercept']
                
                # slope = mdf.fe_params['rain']
                # slope_stder = mdf.bse_fe['rain']
                # slope_pvalue = mdf.pvalues['rain']
                
                # for g in np.unique(data["grazing"]):
                    # intercept_effect = mdf.random_effects[g]["Group"]
                    # slope_effect = mdf.random_effects[g]["rain"]
                    # minR = np.min(data['rain'][data["grazing"] == g])
                    # maxR = np.max(data['rain'][data["grazing"] == g])
                    # g_slope = slope + slope_effect
                    # g_intercept = intercept + intercept_effect
                    # ax.plot([minR, maxR], [minR*g_slope+g_intercept, maxR*g_slope+g_intercept], ls='-', c='r', lw=0.5)
                
                # fig.text(0.25, 0.96, 'Intercept ' + r'$\it{p}$' + ' = %.3f'%intercept_pvalue, fontsize=8)
                # fig.text(0.25, 0.92, 'Slope ' + r'$\it{p}$' + ' = %.3f'%slope_pvalue, fontsize=8)
                
                # plt.savefig(outplot, dpi=300)
                # plt.close()

                # outplot = os.path.join(model_folder2, r'%s_%s_%i.png'%(cover, stat, accumulation))
                # fig = plt.figure()
                # fig.set_size_inches((3, 3))
                # ax = plt.axes([0.25, 0.2, 0.7, 0.7])
                # ax.set_xlabel('Rainfall (mm)\n(Accumulation = %i)'%(accumulation))            
                # ax.set_ylabel('Intercept')
                # ax.errorbar(data['rain'][data['year'] < 2010],
                            # data['intercept'][data['year'] < 2010],
                            # yerr=data['intercept_stderr'][data['year'] < 2010], fmt='ko', markersize=2)
                            
                # ax.errorbar(data['rain'][data['year'] >= 2010],
                            # data['intercept'][data['year'] >= 2010],
                            # yerr=data['intercept_stderr'][data['year'] >= 2010], fmt='o', markersize=2, ecolor='grey', color='grey')
                
                # # md = smf.mixedlm(formula=f"intercept ~ rain", data=data, groups=data["grazing"], re_formula="~rain")
                # # mdf = md.fit(method=["powell", "lbfgs"])
                
                # # intercept = mdf.fe_params['Intercept']
                # # intercept_stderr = mdf.bse_fe['Intercept']
                # # intercept_pvalue = mdf.pvalues['Intercept']
                
                # # slope = mdf.fe_params['rain']
                # # slope_stder = mdf.bse_fe['rain']
                # # slope_pvalue = mdf.pvalues['rain']
                
                # # for g in np.unique(data["grazing"]):
                    # # intercept_effect = mdf.random_effects[g]["Group"]
                    # # slope_effect = mdf.random_effects[g]["rain"]
                    # # minR = np.min(data['rain'][data["grazing"] == g])
                    # # maxR = np.max(data['rain'][data["grazing"] == g])
                    # # g_slope = slope + slope_effect
                    # # g_intercept = intercept + intercept_effect
                    # # ax.plot([minR, maxR], [minR*g_slope+g_intercept, maxR*g_slope+g_intercept], ls='-', c='r', lw=0.5)
                
                # # fig.text(0.25, 0.96, 'Intercept ' + r'$\it{p}$' + ' = %.3f'%intercept_pvalue, fontsize=8)
                # # fig.text(0.25, 0.92, 'Slope ' + r'$\it{p}$' + ' = %.3f'%slope_pvalue, fontsize=8)
                
                # plt.savefig(outplot, dpi=300)
                # plt.close()
                    
#build_model()
plot_results()