#!/usr/bin/env python
"""
The input direcory should contain the CSV files produced by
extract_pheno_timeseries.py

Output PNG graphs are written to the input directory

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

def main(inDir):
    """
    """
    for csvFile in glob.glob(os.path.join(inDir, '*_timeseries.csv')):
        site = os.path.basename(csvFile).replace('_timeseries.csv', '')
        print(site)
        
        # Read in daily data if already present
        dailyCSV = csvFile.replace('.csv', '_daily.csv')
        if os.path.exists(dailyCSV) is True:
            tsDaily = []
            with open(dailyCSV, 'r') as f:
                f.readline()
                for line in f:
                    d = line.strip().split(',')[0]
                    d = datetime.date(year=int(d[:4]), month=int(d[5:7]), day=int(d[8:]))
                    rcc = float(line.strip().split(',')[1])
                    gcc = float(line.strip().split(',')[2])
                    bcc = float(line.strip().split(',')[3])
                    tsDaily.append([d, rcc, gcc, bcc])
            tsDaily = np.ma.masked_equal(np.array(tsDaily), 999)
        
        else:
            # Read in all extracted data 
            ts = []
            with open(csvFile, 'r') as f:
                f.readline()
                for line in f:
                    d = line.strip().split(',')[1]
                    t = line.strip().split(',')[2]
                    rcc = float(line.strip().split(',')[3])
                    gcc = float(line.strip().split(',')[4])
                    bcc = float(line.strip().split(',')[5])
                    if int(t[:2]) >= 11 and int(t[:2]) < 13:
                        d = datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]))
                        t = datetime.time(hour=int(t[:2]), minute=int(t[3:5]), second=int(t[6:]))
                        ts.append([d, t, rcc, gcc, bcc])
            ts = np.array(ts)
        
            # Calculate daily time series (11:00-13:00) masking out nodata days
            startDay = np.min(ts[:, 0])
            endDay = np.max(ts[:, 0])
            numdays = (endDay - startDay).days
            days = [startDay + datetime.timedelta(days=x) for x in range(numdays)]
            tsDaily = []
            for d in days:
                rccDay = ts[:,2][ts[:,0] == d]
                gccDay = ts[:,3][ts[:,0] == d]
                bccDay = ts[:,4][ts[:,0] == d]
                if gccDay.size > 0:
                    rcc = np.mean(rccDay)
                    gcc = np.mean(gccDay)
                    bcc = np.mean(bccDay)
                else:
                    rcc = 999
                    gcc = 999
                    bcc = 999
                tsDaily.append([d, rcc, gcc, bcc])
            tsDaily = np.ma.masked_equal(np.array(tsDaily), 999)
        
            # Write daily data to a new CSV file
            with open(dailyCSV, 'w') as f:
                f.write('date,rcc,gcc,bcc\n')
                for i in range(len(days)):
                    d = tsDaily.data[i, 0]
                    rcc = tsDaily.data[i, 1]
                    gcc = tsDaily.data[i, 2]
                    bcc = tsDaily.data[i, 3]
                    f.write('%s,%.4f,%.4f,%.4f\n'%(d, rcc, gcc, bcc))
        
        # Make graph of daily mean RCC, GCC, and BCC for each site
        fig = plt.figure()
        fig.set_size_inches((10, 4))
        fig.text(0.5, 0.97, site, horizontalalignment='center')
        axPheno = plt.axes([0.15, 0.10, 0.80, 0.80])
        
        axPheno.grid(which='major', axis='x', c='0.9')
        axPheno.set_ylabel('Phenocam chromatic coordinates')

        x = tsDaily[:, 0]
        rcc = tsDaily[:, 1].astype(np.float32)
        axPheno.plot(x, rcc, color='r', alpha=0.2, linewidth=1)
        rccfiltered = ndimage.median_filter(rcc, size=9)
        rccfiltered[rccfiltered > np.max(rcc)] = 999
        rccfiltered = np.ma.masked_equal(rccfiltered, 999)
        axPheno.plot(x, rccfiltered, color='r', linewidth=1)
            
        gcc = tsDaily[:, 2].astype(np.float32)
        axPheno.plot(x, gcc, color='g', alpha=0.2, linewidth=1)
        gccfiltered = ndimage.median_filter(gcc, size=9)
        gccfiltered[gccfiltered > np.max(gcc)] = 999
        gccfiltered = np.ma.masked_equal(gccfiltered, 999)
        axPheno.plot(x, gccfiltered, color='g', linewidth=1)
            
        bcc = tsDaily[:, 3].astype(np.float32)
        axPheno.plot(x, bcc, color='b', alpha=0.2, linewidth=1)
        bccfiltered = ndimage.median_filter(bcc, size=9)
        bccfiltered[bccfiltered > np.max(bcc)] = 999
        bccfiltered = np.ma.masked_equal(bccfiltered, 999)
        axPheno.plot(x, bccfiltered, color='b', linewidth=1)
        
        maxY = max([np.max(rcc), np.max(gcc), np.max(bcc)])
        maxY = (int(maxY / 0.05, ) * 0.05) + 0.05 # Round maxY to nearest 0.5
        axPheno.set_ylim([0, maxY])
        
        plt.savefig(dailyCSV.replace('.csv', '.png'), dpi=300)
    

def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Graphs chromatic coordinate time series"))
    p.add_argument("-i", "--inDir", dest="inDir", default=None,
                   help=("Input directory with CSV files"))
    cmdargs = p.parse_args()
    if cmdargs.inDir is None:
        p.print_help()
        print("Must name input directory")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.inDir)