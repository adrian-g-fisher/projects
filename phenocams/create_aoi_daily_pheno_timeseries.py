#!/usr/bin/env python
"""
This needs the full image time series, and the aoi time series, so they can be
scaled the same when converting to 0-1 using min-max.
"""

import glob
import argparse
import os, sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}


def read_ts_csv(csvFile):
    ts = []
    with open(csvFile, 'r') as f:
        f.readline()
        for line in f:
            d = line.strip().split(',')[1]
            t = line.strip().split(',')[2]
            rcc = float(line.strip().split(',')[3])
            gcc = float(line.strip().split(',')[4])
            bcc = float(line.strip().split(',')[5])
            if int(t[:2]) >= 11 and int(t[:2]) <= 13:
                d = datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]))
                t = datetime.time(hour=int(t[:2]), minute=int(t[3:5]), second=int(t[6:]))
                ts.append([d, t, rcc, gcc, bcc])
    return(np.array(ts))


def calc_daily_ts(ts):
    """
    Calculate daily time series
     - 90th percentile of values for +/- 2 days (11:00-13:00)
     - masking out nodata days
    """
    startDay = np.min(ts[:, 0])
    endDay = np.max(ts[:, 0])
    numdays = (endDay - startDay).days
    days = [startDay + datetime.timedelta(days=x) for x in range(numdays)]
    tsDaily = []
    n_count = []
    for d in days:
        rccDay = ts[:,2][(ts[:,0] >= d - datetime.timedelta(days=2)) & (ts[:,0] <= d + datetime.timedelta(days=2))]
        gccDay = ts[:,3][(ts[:,0] >= d - datetime.timedelta(days=2)) & (ts[:,0] <= d + datetime.timedelta(days=2))]
        bccDay = ts[:,4][(ts[:,0] >= d - datetime.timedelta(days=2)) & (ts[:,0] <= d + datetime.timedelta(days=2))]
        if gccDay.size > 0:
            n_count.append(gccDay.size)
            rcc = np.mean(rccDay)
            gcc = np.mean(gccDay)
            bcc = np.mean(bccDay)
                
        else:
            rcc = 999
            gcc = 999
            bcc = 999
        tsDaily.append([d, rcc, gcc, bcc])
    tsDaily = np.ma.masked_equal(np.array(tsDaily), 999)
    return(tsDaily, days)


def main(mainCSV, aoiCSV):
    """
    """
    dailyCSV = aoiCSV.replace('.csv', '_daily.csv')
    
    # Read in the data 
    main = read_ts_csv(mainCSV)
    aoi = read_ts_csv(aoiCSV)
    
    # Calculate daily time series
    main_daly, main_days = calc_daily_ts(main)
    aoi_daily, aoi_days = calc_daily_ts(aoi)
    
    # Calculate normalised daily time series
    tsNorm = np.ma.copy(main_daly)
    aoiNorm = np.ma.copy(aoi_daily)
    for i in range(1, 4):
        minCC = np.min(tsNorm.data[:, i])
        maxCC = np.max(tsNorm.data[:, i][tsNorm.data[:, i] != 999])
        tsNorm.data[:, i] = (tsNorm.data[:, i] - minCC) / (maxCC - minCC)
        aoiNorm.data[:, i] = (aoiNorm.data[:, i] - minCC) / (maxCC - minCC)
    
    # Write normalised data to a new CSV file
    with open(dailyCSV, 'w') as f:
        f.write('date,rcc,gcc,bcc\n')
        for i in range(len(aoi_days)):
            d = aoiNorm.data[i, 0]
            rcc = aoiNorm.data[i, 1]
            gcc = aoiNorm.data[i, 2]
            bcc = aoiNorm.data[i, 3]
            f.write('%s,%.4f,%.4f,%.4f\n'%(d, rcc, gcc, bcc))
    
    # Make graph of daily mean GCC for aoi and main
    fig = plt.figure()
    fig.set_size_inches((10, 4))
    axPheno = plt.axes([0.15, 0.10, 0.80, 0.80])
    axPheno.grid(which='major', axis='x', c='0.9')
    axPheno.set_ylabel('Green chromatic coordinate')
    main = axPheno.plot(tsNorm[:, 0], tsNorm[:, 2], color='g', linewidth=1, label='Main plot')
    aoi = axPheno.plot(aoiNorm[:, 0], aoiNorm[:, 2], color='g', linestyle='--', linewidth=1, label='Bare area')
    axPheno.legend(loc='upper right', frameon=False)
    
    plt.savefig(dailyCSV.replace('.csv', '.png'), dpi=300)
    

def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates daily time series of chromatic coordinates "+
                         "for an area of interest (AOI)"))
    p.add_argument("-m", "--mainCSV", dest="mainCSV", default=None,
                   help=("Input CSV file for the main area"))
    p.add_argument("-a", "--aoiCSV", dest="aoiCSV", default=None,
                   help=("Input CSV file for the area of interest"))
    cmdargs = p.parse_args()
    if cmdargs.mainCSV is None or cmdargs.aoiCSV is None:
        p.print_help()
        print("Must name input files")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.mainCSV, cmdargs.aoiCSV)