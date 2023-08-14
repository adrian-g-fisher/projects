#!/usr/bin/env python
"""

"""


import os
import sys
import datetime
import numpy as np

# Create dictionary to convert date to sequential number
date2id = {}
startSeason = datetime.date(year=1987, month=12, day=15) + datetime.timedelta(days=30)
endSeason = datetime.date(year=2022, month=12, day=15) + datetime.timedelta(days=30)
i = 1
for y in range(1987, 2024):
    for m in [3, 6, 9, 12]:
        d = datetime.date(year=y, month=m, day=15) + datetime.timedelta(days=30)
        if d >= startSeason and d <= endSeason:
            date2id[d] = i
            i += 1

# Read in polygon data
seasonCentralDate = []
polygons = []
data = []
with open(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series\witchelina_poly_fc.csv', 'r') as f:
    header = f.readline().strip()
    for line in f:
        data.append(line.strip())
        y = int(line.split(',')[0][0:4])
        m = int(line.split(',')[0][4:6])
        seasonCentralDate.append(datetime.date(year=y, month=m, day=15) + datetime.timedelta(days=30))
        polygons.append(line.split(',')[2])
seasonCentralDate = np.array(seasonCentralDate)
polygons = np.array(polygons)
data = np.array(data)

# Read in rainfall data
rainDates = []
rain = []
with open(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series\witchelina_rain.csv', 'r') as f:
    f.readline()
    for line in f:
        y = int(line.strip().split(',')[0])
        m = int(line.strip().split(',')[1])
        rainDates.append(datetime.date(year=y, month=m, day=15))
        rain.append(float(line.strip().split(',')[2]))
rainDates = np.array(rainDates)
rain = np.array(rain)

# Convert rain from monthly to seasonal
seasonalDates = []
seasonalRain = []
startSeason = datetime.date(year=1987, month=12, day=15) + datetime.timedelta(days=30)
endSeason = datetime.date(year=2022, month=12, day=15) + datetime.timedelta(days=30)
for y in range(1980, 2024):
    for m in [3, 6, 9, 12]:
        d = datetime.date(year=y, month=m, day=15) + datetime.timedelta(days=30)
        if d >= startSeason and d <= endSeason:
            seasonalDates.append(d)
            if m == 12:
                d1 = datetime.date(year=y, month=m, day=15)
                d2 = datetime.date(year=y+1, month=1, day=15)
                d3 = datetime.date(year=y+1, month=2, day=15)
            else:
                d1 = datetime.date(year=y, month=m, day=15)
                d2 = datetime.date(year=y, month=m+1, day=15)
                d3 = datetime.date(year=y, month=m+2, day=15)
            r1 = rain[rainDates == d1][0]
            r2 = rain[rainDates == d2][0]
            r3 = rain[rainDates == d3][0]
            seasonalRain.append(r1 + r2 + r3)
seasonalDates = np.array(seasonalDates)
seasonalRain = np.array(seasonalRain)

# Get cumulative rain for different periods
cumRain= []
for i, d in enumerate(seasonalDates):
    x = []
    x.append(seasonalRain[i])                         # season
    x.append(seasonalRain[i-1])                       # previous 3 months
    x.append(seasonalRain[i-1] + seasonalRain[i-2])   # previous 6 months
    x.append(seasonalRain[i-1] + seasonalRain[i-2] +
             seasonalRain[i-3] + seasonalRain[i-4])   # previous 12 months
    x.append(seasonalRain[i-1] + seasonalRain[i-2] +
             seasonalRain[i-3] + seasonalRain[i-4] +
             seasonalRain[i-5] + seasonalRain[i-6] +
             seasonalRain[i-7] + seasonalRain[i-8])   # previous 24 months
    x.append(seasonalRain[i-1] + seasonalRain[i-2] +
             seasonalRain[i-3] + seasonalRain[i-4] +
             seasonalRain[i-5] + seasonalRain[i-6] +
             seasonalRain[i-7] + seasonalRain[i-8] +
             seasonalRain[i-9] + seasonalRain[i-10] +
             seasonalRain[i-11] + seasonalRain[i-12]) # previous 36 months
    cumRain.append(x)
cumRain = np.array(cumRain)

# Create unique number for each pair of polygons
polygons = np.array([p.replace('O', 'I') for p in polygons])
pairs = np.zeros(polygons.size, dtype=np.byte)
uniquePairs = np.unique(polygons)
for i, p in enumerate(uniquePairs):
    pairs[polygons == p] = i+1

# Match FC data to rain data and create CSV file
header = 'sequence,pair,%s,%s\n'%(header, 'rain_season,rain_prev3m,rain_prev6m,rain_prev12m,rain_prev24m,rain_prev36m')
with open(r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\fc_time_series\witchelina_fc_rain.csv', 'w') as f:
    f.write(header)
    for i, d in enumerate(data):
        s = seasonCentralDate[i]
        p = pairs[i]
        r = ','.join(['%.2f'%x for x in cumRain[seasonalDates == s][0]])
        Id = date2id[s]
        line = '%i,%i,%s,%s\n'%(Id, p, d, r)
        f.write(line)