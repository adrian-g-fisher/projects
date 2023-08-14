#!/usr/bin/env python
"""

"""


import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

params = {'font.sans-serif': "Arial", 'font.family': "sans-serif",
          'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out'}
plt.rcParams.update(params)


code2landsystem = {'P' : 'Paradise (stony hills and flats)',
                   'M' : 'Myrtle (sand dunes and swales)',
                   'S' : 'Stuarts Creek (plains with widely spaced dunes)'}

# Read in data
indata = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series\witchelina_fc_rain.csv'
seasonDates = []
landsystems = []
position = []
polygons = []
fc = []
rain = []
with open(indata, 'r') as f:
    f.readline()
    for line in f:
        y = int(line.split(',')[2][0:4])
        m = int(line.split(',')[2][4:6])
        seasonDates.append(datetime.date(year=y, month=m, day=15) + datetime.timedelta(days=30))
        landsystems.append(line.split(',')[3])
        polygons.append(line.split(',')[4])
        position.append(line.split(',')[5])
        fc.append([float(x) for x in line.split(',')[7:13]])
        rain.append([float(x) for x in line.strip().split(',')[13:]])
seasonDates = np.array(seasonDates)
landsystems = np.array(landsystems)
polygons = np.array(polygons)
position = np.array(position)
fc = np.array(fc)
rain = np.array(rain)

# Get rain from MI7 (it has the full time series)
rainDates = seasonDates[polygons == 'MI7']
rain_season = rain[polygons == 'MI7', 0]
rain_p03m = rain[polygons == 'MI7', 1]
rain_p06m = rain[polygons == 'MI7', 2]
rain_p12m = rain[polygons == 'MI7', 3]
rain_p24m = rain[polygons == 'MI7', 4]
rain_p36m = rain[polygons == 'MI7', 5]
names = ['Season', 'Previous 3 months', 'Previous 6 months',
         'Previous 12 months', 'Previous 24 months', 'Previous 36 months']
         
# Make rain plot
fig = plt.figure()
fig.set_size_inches((8, 7))
fig.text(0.005, 0.5, 'Rainfall (mm)', verticalalignment='center', rotation=90)
gs = fig.add_gridspec(6, hspace=0.2)
ax = gs.subplots(sharex=True)
ax[0].set_title('Witchelina rainfall')
for i, a in enumerate(ax):
    x = seasonDates[polygons == 'MI7']
    y = rain[polygons == 'MI7', i]
    a.plot(x, y, color='skyblue', linewidth=2)
    a.grid()
    a.set_ylim([0, 1000])
    a.text(datetime.date(2017, 1, 1), 790, names[i],
           bbox=dict(facecolor='w', edgecolor='w', pad=0))
    a.vlines(datetime.date(2010, 1, 1), 0, 1000, colors='k', linestyles='dashed')
    a.label_outer()
gs.tight_layout(fig)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series\witchelina_rain.png', dpi=300)

# Make mean cover time series for each landsystem/position with rainfall
fig = plt.figure()
fig.set_size_inches((8, 10))
fig.text(0.02, 0.66, 'Fractional cover (%)', verticalalignment='center', rotation=90)
gs = fig.add_gridspec(4, hspace=0.2)
ax = gs.subplots(sharex=True)
for i, l in enumerate(np.unique(landsystems)):
    landsystem = code2landsystem[l]
    
    a = ax[i]
    
    iDates = np.unique(seasonDates[(landsystems == l) & (position == 'Inside')])
    iFC = np.zeros((iDates.size, 6), dtype=np.float32)
    for d in iDates:
        ifc = fc[(landsystems == l) & (position == 'Inside') & (seasonDates == d), :]
        iFC[iDates==d, 0:3] = np.mean(ifc[:, 0:3], axis=0)
        iFC[iDates==d, 3] = np.sqrt(np.sum(ifc[:, 3]**2)/ifc[:, 3].size)
        iFC[iDates==d, 4] = np.sqrt(np.sum(ifc[:, 4]**2)/ifc[:, 4].size)
        iFC[iDates==d, 5] = np.sqrt(np.sum(ifc[:, 5]**2)/ifc[:, 5].size)
    
    a.fill_between(iDates, iFC[:, 1] - iFC[:, 4], iFC[:, 1] + iFC[:, 4],
                   alpha=0.2, facecolor='darkgreen',
                   linewidth=0.0, edgecolor='darkgreen')
    a.plot(iDates, iFC[:, 1], color='darkgreen', linewidth=1, label='Inside PV')
    
    a.fill_between(iDates, iFC[:, 2] - iFC[:, 5], iFC[:, 2] + iFC[:, 5],
                   alpha=0.2, facecolor='saddlebrown',
                   linewidth=0.0, edgecolor='saddlebrown')
    a.plot(iDates, iFC[:, 2], color='saddlebrown', linewidth=1, label='Inside NPV')
    
    oDates = np.unique(seasonDates[(landsystems == l) & (position == 'Outside')])
    oFC = np.zeros((oDates.size, 6), dtype=np.float32)
    for d in oDates:
        ofc = fc[(landsystems == l) & (position == 'Outside') & (seasonDates == d), :]
        oFC[oDates==d, 0:3] = np.mean(ofc[:, 0:3], axis=0)
        oFC[oDates==d, 3] = np.sqrt(np.sum(ofc[:, 3]**2)/ofc[:, 3].size)
        oFC[oDates==d, 4] = np.sqrt(np.sum(ofc[:, 4]**2)/ofc[:, 4].size)
        oFC[oDates==d, 5] = np.sqrt(np.sum(ofc[:, 5]**2)/ofc[:, 5].size)
        
    a.fill_between(oDates, oFC[:, 1] - oFC[:, 4], oFC[:, 1] + oFC[:, 4],
                   alpha=0.2, facecolor='palegreen',
                   linewidth=0.0, edgecolor='palegreen')
    a.plot(oDates, oFC[:, 1], color='palegreen', linewidth=1, label='Outside PV')
    
    a.fill_between(oDates, oFC[:, 2] - oFC[:, 5], oFC[:, 2] + oFC[:, 5],
                   alpha=0.2, facecolor='sandybrown',
                   linewidth=0.0, edgecolor='sandybrown')
    a.plot(oDates, oFC[:, 2], color='sandybrown', linewidth=1, label='Outside NPV')
    
    a.grid()
    a.set_ylim([0, 60])
    a.legend(loc='upper right', fancybox=False, framealpha=1, facecolor='w', edgecolor='w')
    
    a.set_title(landsystem, y=0.85, x=0.01, horizontalalignment='left', backgroundcolor='w')
    
    a.axvline(datetime.date(2010, 1, 1), 0, 1, c='k', ls='dashed')
    a.label_outer()

# Add rain
a = ax[3]
a.plot(rainDates, rain_season, color='skyblue', linewidth=2, label='seasonal')
a.plot(rainDates, rain_p36m, color='skyblue', linewidth=2, ls='--', label='previous 3 years')
a.axvline(datetime.date(2010, 1, 1), 0, 1, c='k', ls='dashed')
a.label_outer()
a.set_ylabel('Rainfall (mm)')
a.grid()
a.legend(loc='upper right', fancybox=False, framealpha=1, facecolor='w', edgecolor='w')
gs.tight_layout(fig)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series' +
             '\mean_time_series.png', dpi=300)

# Create figures for bare, green, and dead for each landsystem
colours = [['darkred', 'tomato'],
           ['darkgreen', 'palegreen'],
           ['saddlebrown', 'sandybrown']]
ylimits = [[0, 100], [0, 60], [0, 80]]
for c, cover in enumerate(['bare', 'green', 'dead']):
    for l in np.unique(landsystems):
        cs = colours[c]
        landsystem = code2landsystem[l]
        insidePolys = np.unique(polygons[(landsystems == l) & (position == 'Inside')])
        outsidePolys = np.unique(polygons[(landsystems == l) & (position == 'Outside')])

        fig = plt.figure()
        fig.set_size_inches((8, 14))
        fig.text(0, 0.5, '%s cover (%%)'%cover.title(), verticalalignment='center', rotation=90)
        gs = fig.add_gridspec(insidePolys.size, hspace=0.2)
        ax = gs.subplots(sharex=True)
        ax[0].set_title(landsystem)
        for i, a in enumerate(ax):
            ip = insidePolys[i]
            ix = seasonDates[polygons == ip]
            iy = fc[:, c][polygons == ip]
            iy_upper = iy + fc[:, c+3][polygons == ip]
            iy_lower = iy - fc[:, c+3][polygons == ip]
            a.fill_between(ix, iy_lower, iy_upper, alpha=0.2, facecolor=cs[0],
                           linewidth=0.0, edgecolor=cs[0])
            a.plot(ix, iy, color=cs[0], linewidth=1, label='Inside (%s)'%ip)
            op = ip.replace('I', 'O')
            ox = seasonDates[polygons == op]
            oy = fc[:, c][polygons == op]
            oy_upper = oy + fc[:, c+3][polygons == op]
            oy_lower = oy - fc[:, c+3][polygons == op]
            a.fill_between(ox, oy_lower, oy_upper, alpha=0.2, facecolor=cs[1],
                           linewidth=0.0, edgecolor=cs[1])
            a.plot(ox, oy, color=cs[1], linewidth=1, label='Outside (%s)'%op)
            a.grid()
            a.set_ylim(ylimits[c])
            if cover == 'bare':
                pos = 'lower left'
            else:
                pos = 'upper left'
            a.legend(loc=pos, fancybox=False, framealpha=1, facecolor='w', edgecolor='w')
            a.axvline(datetime.date(2010, 1, 1), 0, 1, c='k', ls='dashed')
            a.label_outer()
        gs.tight_layout(fig)
        plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series' +
                    '\%s_%s.png'%(landsystem.split(' ')[0], cover), dpi=300)

# Create figures for the difference in bare, green, and dead for each landsystem
cs = ['darkred', 'darkgreen', 'saddlebrown']
for c, cover in enumerate(['bare', 'green', 'dead']):
    for l in np.unique(landsystems):
        landsystem = code2landsystem[l]
        insidePolys = np.unique(polygons[(landsystems == l) & (position == 'Inside')])
        outsidePolys = np.unique(polygons[(landsystems == l) & (position == 'Outside')])

        fig = plt.figure()
        fig.set_size_inches((8, 14))
        fig.text(0, 0.5, 'Difference in %s cover (%%)'%cover, verticalalignment='center', rotation=90)
        gs = fig.add_gridspec(insidePolys.size, hspace=0.2)
        ax = gs.subplots(sharex=True)
        ax[0].set_title(landsystem)
        for i, a in enumerate(ax):
            
            # Get inside
            ip = insidePolys[i]
            ix = seasonDates[polygons == ip]

            # Get outside
            op = ip.replace('I', 'O')
            ox = seasonDates[polygons == op]
            
            # Get common dates 
            iDates = np.zeros(ix.size)
            oDates = np.zeros(ox.size)
            for d in ix:
                if d in ox:
                    iDates[ix == d] = 1
                    oDates[ox == d] = 1
            
            # Now get common values
            iy = fc[:, c][polygons == ip][iDates == 1]
            iy_std = fc[:, c+3][polygons == ip][iDates == 1] 
            oy = fc[:, c][polygons == op][oDates == 1]
            oy_std = fc[:, c+3][polygons == op][oDates == 1]
            
            # Get difference time series
            diff = iy - oy
            diff_std = np.sqrt(iy_std**2 + oy_std**2)
            
            # Get x values
            x = seasonDates[polygons == ip][iDates == 1]
            
            a.fill_between(x, diff - diff_std, diff + diff_std, alpha=0.2, facecolor=cs[c],
                           linewidth=0.0, edgecolor=cs[c])
            a.plot(x, diff, color=cs[c], linewidth=1, label='Inside (%s) - Outside (%s)'%(ip, op))
            
            a.set_ylim([-40, 40])
            a.legend(loc='upper left', fancybox=False, framealpha=1, facecolor='w', edgecolor='w')
            a.grid()
            a.axvline(datetime.date(2010, 1, 1), 0, 1, c='k', ls='dashed')
            a.axhline(0, 0, 1, c='k')
            a.label_outer()
        gs.tight_layout(fig)
        plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series' +
                    '\Difference_%s_%s.png'%(landsystem.split(' ')[0], cover), dpi=300)
