#!/usr/bin/env python
"""

Combines data from field observations with the fractional cover time series from
each site. It essentially adds 27 new columns to each field observation,
which contain the mean bare, PV and NPV percentages for the corresponding season
and all seasons from the previous 2 years.
"""


import os
import sys
import datetime
import numpy as np


def date2season(d):
    """
    Converts a datetime date into the corresponding sesaon, returning a new
    datetime date.
    """
    y = d.year
    m = d.month
    if m in [12, 1, 2]:
        new_m = 12
    elif m in [3, 4, 5]:
        new_m = 3
    elif m in [6, 7, 8]:
        new_m = 6
    elif m in [9, 10, 11]:
        new_m = 9
    d = datetime.date(year=y, month=new_m, day=15) + datetime.timedelta(days=30)
    
    startDate = '%i%02d'%(y, new_m)
    if new_m == 12:
        endDate = '%i%02d'%(y+1, 2)
    else:
        endDate = '%i%02d'%(y, new_m+2)
    season_string = '%s%s'%(startDate, endDate)
    
    return(d, int(season_string))


# Read in FC data
dates = []
Id = []
bare = []
green = []
dead = []
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\seasonal_fc_extract.csv'
with open(csvfile, 'r') as f:
    #Id,date,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead
    f.readline()
    for line in f:
        l = line.strip().split(',')
        Id.append(int(l[0]))
        dates.append(int(l[1]))
        bare.append(float(l[3]))
        green.append(float(l[5]))
        dead.append(float(l[7]))
Id = np.array(Id)
bare = np.array(bare)
green = np.array(green)
dead = np.array(dead)
datetimes = np.array([datetime.date(year=int(str(d)[:4]),
                                    month=int(str(d)[4:6]), day=15) +
                                    datetime.timedelta(days=30) for d in dates])

# Read in site and Id values and make dictionary
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\pw_site_ids.csv'
site2id = {}
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        ident, site = line.strip().split(',')
        site2id[site] = int(ident)

# Read in field data
field_csv = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\PW_Project_BestData.csv'
field_data = []
field_sites = []
field_dates = []
with open(field_csv, 'r') as f:
    header = f.readline().strip()
    for line in f:
        field_data.append(line.strip())
        field_sites.append(line.strip().split(',')[0])
        l = line.strip().split(',')[2]
        y = 2000 + int(l[6:])
        m = int(l[3:5])
        d = int(l[0:2])
        field_dates.append(datetime.date(y, m, d))
field_data = np.array(field_data)
field_sites = np.array(field_sites)
field_dates = np.array(field_dates)

# Find FC values for each date
sesaons = np.zeros(field_dates.size, dtype=np.uint64)
fc_values = np.zeros((field_dates.size, 27), dtype=np.float32)
for i in range(field_dates.size):
    s = field_sites[i]
    bare_subset = bare[Id == site2id[s]]
    green_subset = green[Id == site2id[s]]
    dead_subset = dead[Id == site2id[s]]
    date_subset = datetimes[Id == site2id[s]]
    (d, season_string) = date2season(field_dates[i])
    sesaons[i] = season_string
    fc_list = np.zeros(27, dtype=np.float32)
    for lag in range(0, 9):
        d_lag, q = date2season(d - datetime.timedelta(days=30*lag))
        fc_list[lag*3]     = bare_subset[date_subset == d_lag][0]
        fc_list[lag*3 + 1] = green_subset[date_subset == d_lag][0]
        fc_list[lag*3 + 2] = dead_subset[date_subset == d_lag][0]
    fc_values[i, :] = fc_list

# Add new column descriptions to the header
seasons = ['field_season_date', 'field_season_bare', 'field_season_green',
           'field_season_dead']
for s in range(1, 9):
    seasons += ['previous_%i_season_bare'%s, 'previous_%i_season_green'%s,
                'previous_%i_season_dead'%s]
header = '%s,%s\n'%(header, ','.join(seasons))

# Create new CSV file
new_csv = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\PW_data_plus_FC.csv'
with open(new_csv, 'w') as f:
    f.write(header)
    for i in range(field_dates.size):
        fc_data = ','.join(['%.2f'%v for v in fc_values[i, :]])
        line = '%s,%s,%s\n'%(field_data[i], sesaons[i], fc_data)
        f.write(line)