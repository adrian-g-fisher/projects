import glob
import os, sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}

# Read in pheno data, discarding times outside 11-1
sites = []
tsList = []
baseDir = r'C:\Users\Adrian\Documents\fowlers_game_cameras\kangaroo_shades'
for csvFile in glob.glob(os.path.join(baseDir, '*.csv')):
    site = os.path.basename(csvFile).replace('_timeseries.csv', '')
    sites.append(site)
    data = []
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
                data.append([d, t, rcc, gcc, bcc])
    tsList.append(np.array(data))

# Make mean daily RCC, GCC, and BCC time series, masking out nodata days
tsDaily = []
for ts in tsList:
    startDay = np.min(ts[:, 0])
    endDay = np.max(ts[:, 0])
    numdays = (endDay - startDay).days
    days = [startDay + datetime.timedelta(days=x) for x in range(numdays)]
    data = []
    for i, d in enumerate(days):
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
        data.append([d, rcc, gcc, bcc])
    data = np.ma.masked_equal(np.array(data), 999)
    tsDaily.append(data)

# Read in FC data and sort by date
fc_sites = []
fc_data = []
with open(r'pheno_s2_fc_extract.csv', 'r') as f:
    f.readline()
    for line in f:
        if line.strip().split(',')[2] != '999':
            fc_sites.append(line.strip().split(',')[0])
            fc_data.append([float(f) for f in line.strip().split(',')[1:]])
fc_sites = np.array(fc_sites)
fc_data = np.array(fc_data)
fc_sites = fc_sites[fc_data[:, 0].argsort()]
fc_data = fc_data[fc_data[:, 0].argsort()]

# Make graph of daily mean RCC, GCC, and BCC for each site, and add a panel with
# sentinel-2 fractional cover.

siteNames = ['1c', '1s', '1w',
             '2c', '2s', '2w',
             '3c', '3s', '3w',
             '4c', '4s', '4w',
             '5c', '5s', '5w']

minDate = datetime.date(year=2020, month=6, day=1)
maxDate = datetime.date(year=2021, month=3, day=1)
minRCC = min([min(x[:, 1]) for x in tsDaily])
maxRCC = max([max(x[:, 1]) for x in tsDaily])
minGCC = min([min(x[:, 2]) for x in tsDaily])
maxGCC = max([max(x[:, 2]) for x in tsDaily])
minBCC = min([min(x[:, 3]) for x in tsDaily])
maxBCC = max([max(x[:, 3]) for x in tsDaily])
minY = min([minRCC, minGCC, minBCC])
maxY = max([maxRCC, maxGCC, maxBCC])

for i, site in enumerate(siteNames):
    
    fcValues = fc_data[fc_sites == site, 1:]
    fcDates = fc_data[fc_sites == site, 0]
    fcDates = [datetime.date(year=int(str(d)[0:4]),
                             month=int(str(d)[4:6]),
                             day=int(str(d)[6:8])) for d in fcDates]
    
    fig = plt.figure(i)
    fig.set_size_inches((8, 6))
    fig.text(0.5, 0.97, 'Site %s'%site, horizontalalignment='center')
    axPheno = plt.axes([0.15, 0.10, 0.80, 0.40])
    axPheno.set_xlim([minDate, maxDate])
    axPheno.set_ylim([minY, maxY])
    axPheno.grid(which='major', axis='x', c='0.9')
    axPheno.set_ylabel('Phenocam chromatic coordinates')
    for j, ts in enumerate(tsDaily):
        tsName = sites[j]
        if tsName[:2] == site:
            x = ts[:, 0]
            rcc = ts[:, 1].astype(np.float32)
            axPheno.plot(x, rcc, color='r', alpha=0.2, linewidth=1)
            rccfiltered = ndimage.median_filter(rcc, size=3)
            rccfiltered[rccfiltered > maxY] = 999
            rccfiltered = np.ma.masked_equal(rccfiltered, 999)
            axPheno.plot(x, rccfiltered, color='r', linewidth=1)
            
            gcc = ts[:, 2].astype(np.float32)
            axPheno.plot(x, gcc, color='g', alpha=0.2, linewidth=1)
            gccfiltered = ndimage.median_filter(gcc, size=3)
            gccfiltered[gccfiltered > maxY] = 999
            gccfiltered = np.ma.masked_equal(gccfiltered, 999)
            axPheno.plot(x, gccfiltered, color='g', linewidth=1)
            
            bcc = ts[:, 3].astype(np.float32)
            axPheno.plot(x, bcc, color='b', alpha=0.2, linewidth=1)
            bccfiltered = ndimage.median_filter(bcc, size=3)
            bccfiltered[bccfiltered > maxY] = 999
            bccfiltered = np.ma.masked_equal(bccfiltered, 999)
            axPheno.plot(x, bccfiltered, color='b', linewidth=1)
    
    axFC = plt.axes([0.15, 0.55, 0.80, 0.40])
    axFC.set_xlim([minDate, maxDate])
    axFC.set_ylim([0, 80])
    axFC.set_ylabel('Sentinel-2 fractional cover (%)')
    axFC.set_xticklabels([])
    axFC.grid(which='major', axis='x', c='0.9')
    axFC.fill_between(fcDates, fcValues[:, 0]-fcValues[:, 1],
                               fcValues[:, 0]+fcValues[:, 1],
                      alpha=0.2, facecolor='r', linewidth=0.0, edgecolor='r')
    axFC.plot(fcDates, fcValues[:, 0], color='r', linewidth=1)                  
    axFC.fill_between(fcDates, fcValues[:, 2]-fcValues[:, 3],
                               fcValues[:, 2]+fcValues[:, 3],
                      alpha=0.2, facecolor='g', linewidth=0.0, edgecolor='g')
    axFC.plot(fcDates, fcValues[:, 2], color='g', linewidth=1)
    axFC.fill_between(fcDates, fcValues[:, 4]-fcValues[:, 5],
                               fcValues[:, 4]+fcValues[:, 5],
                      alpha=0.2, facecolor='b', linewidth=0.0, edgecolor='b')
    axFC.plot(fcDates, fcValues[:, 4], color='b', linewidth=1)
    
    plt.savefig(r'pheno_timeseries_%s.png'%site, dpi=300)