#!/usr/bin/env python
"""
The input direcory should contain the CSV files produced by
extract_pheno_timeseries.py

Outputs:
 - daily time series CSV files are written to the input directory
 - PNG time series graphs are written to the input directory

Note that the calculation of the daily time series includes several steps
outlined in Browning et al (2017).
 - Create mask for vegetation of interest.
 - Calculate mean chomatic coordinates for each image.
 - The daily GCC value was calculated as the mean of a five day window (-2 days
   and +2 days), of all GCC values between 11:00-13:00.
 - Each daily GCC series was then normalised using the method of Keenan et al
   (2014) which involves subtracting the minimum value and dividing by the
   range.

Browning, DM, Karl, JW, Morin, D, Richardson, AD, Tweedie, CE, (2017) Phenocams
Bridge the Gap between Field and Satellite Observations in an Arid Grassland
Ecosystem. Remote Sensing, 9, 1071 https://doi.org/10.3390/rs9101071 

Keenan, TF et al. (2014) Net carbon uptake has increased through warming-induced
changes in temperate forest phenology. Nature Climate Change, 4, 598–604,
https://doi.org/10.1038/nclimate2253

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
    for csvFile in glob.glob(os.path.join(inDir, '*.csv')):
        dailyCSV = csvFile.replace('.csv', '_daily.csv')
        site = os.path.basename(csvFile).replace('.csv', '')
        print(site)
        
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
                if int(t[:2]) >= 11 and int(t[:2]) <= 13:
                    d = datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]))
                    t = datetime.time(hour=int(t[:2]), minute=int(t[3:5]), second=int(t[6:]))
                    ts.append([d, t, rcc, gcc, bcc])
        ts = np.array(ts)
        
        # Calculate daily time series
        # - 90th percentile of values for +/- 2 days (11:00-13:00)
        # - masking out nodata days
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
        
        #print(np.mean(n_count))
        
        # Calculate normalised daily time series
        tsNorm = np.ma.copy(tsDaily)
        for i in range(1, 4):
            minCC = np.min(tsNorm.data[:, i])
            maxCC = np.max(tsNorm.data[:, i][tsNorm.data[:, i] != 999])
            tsNorm.data[:, i] = (tsNorm.data[:, i] - minCC) / (maxCC - minCC)
            
        # Write normalised data to a new CSV file
        with open(dailyCSV, 'w') as f:
            f.write('date,rcc,gcc,bcc\n')
            for i in range(len(days)):
                d = tsNorm.data[i, 0]
                rcc = tsNorm.data[i, 1]
                gcc = tsNorm.data[i, 2]
                bcc = tsNorm.data[i, 3]
                f.write('%s,%.4f,%.4f,%.4f\n'%(d, rcc, gcc, bcc))
        
        # Make graph of daily mean RCC, GCC, and BCC for each site
        fig = plt.figure()
        fig.set_size_inches((10, 4))
        fig.text(0.5, 0.97, site, horizontalalignment='center')
        axPheno = plt.axes([0.15, 0.10, 0.80, 0.80])
        
        axPheno.grid(which='major', axis='x', c='0.9')
        axPheno.set_ylabel('Phenocam chromatic coordinates')

        x = tsNorm[:, 0]
        rcc = tsNorm[:, 1].astype(np.float32)
        axPheno.plot(x, rcc, color='r', linewidth=1)
            
        gcc = tsNorm[:, 2].astype(np.float32)
        axPheno.plot(x, gcc, color='g', linewidth=1)
            
        bcc = tsNorm[:, 3].astype(np.float32)
        axPheno.plot(x, bcc, color='b', linewidth=1)
        
        maxY = max([np.max(rcc), np.max(gcc), np.max(bcc)])
        maxY = (int(maxY / 0.05, ) * 0.05) + 0.05 # Round maxY to nearest 0.5
        axPheno.set_ylim([0, maxY])
        
        plt.savefig(dailyCSV.replace('.csv', '.png'), dpi=300)
    

def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates daily time series of chromatic coordinates"))
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