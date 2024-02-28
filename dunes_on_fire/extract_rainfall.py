#!/usr/bin/env python
"""
This only works in the netcdf env

conda create -n netcdf -c conda-forge xarray dask netCDF4 bottleneck geopandas rioxarray gdal=3.1

"""
import os, sys
import rioxarray
import xarray as xr
import geopandas as gpd
import numpy as np


def extract_rainfall(shapefile):
    """
    Uses Andres's code to extracts BOM 1 second rainfall for dunefield polygons.
    """
    # Go through each ploygon from the shapefile
    shp_df = gpd.read_file(shapefile)
    for index, row in shp_df.iterrows():
    
        print(f'Feature: {index + 1}/{len(shp_df)}')  
        
        # Get site ID
        ID = str(row['Id'])
        
        # Get name
        Name = str(row['Code'])
        
        # Check if its already extracted
        outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\rainfall'
        outfile = os.path.join(outdir, 'rainfall_%s.csv'%(Name))
        if os.path.exists(outfile) is False:
        
            # Create a csv file to save results
            with open(outfile, 'w') as f:
                f.write('ID,Date,Mean_rain,Stdev_rain,Pixel_count\n')
        
            # Go through each year and open NetCFD
            for year in range(1995, 2023):
        
                # Load the monthly rainfall data for all months in the year
                nc_fo = r'S:\aust\bom_climate_grids\bom_monthly_rainfall\1second\NetCDF\agcd_v2_precip_total_r001_monthly_{:d}.nc'.format(year)
            
                # Read the NetCDF and select the datarray for the variable of intrest and subset for AOI
                nc_ds = xr.open_dataset(nc_fo)
                ds = nc_ds['precip']
            
                # Get all months/timestamps
                timestamps = nc_ds['time'].values
            
                # Go through each timestamp within the NetCDF
                for x in timestamps:
                
                    #get values of variable per month
                    nc_arr = ds.sel(time=x)
                
                    # Format time to just by YYYYMM
                    y = x.astype(str)[0:4]
                    m = x.astype(str)[5:7]
                    yyyymm = '%s%s'%(y, m)
                
                    # Add missing crs and write polygon geometry to a list for clip tool to read
                    nc_arr_geo = nc_arr.rio.write_crs(4326)
                    my_geom = [shp_df.iloc[index].geometry]
                
                    # Clip to polygon, compute statistics and write to the csv
                    clipped = nc_arr_geo.rio.clip(my_geom, shp_df.crs, all_touched=True)
                    stats = []
                    stats.append('%.2f'%np.nanmean(clipped.values))
                    stats.append('%.2f'%np.nanstd(clipped.values))
                    stats.append('%.i'%np.count_nonzero(~np.isnan(clipped)))
                    with open(outfile, 'a') as f:
                        f.write('%s,%s,%s\n'%(ID, yyyymm, ','.join(stats)))        


# Get rainfall data
shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Australian_Dunefields_202402\dunefields_sinusoidal_multipart.shp'
extract_rainfall(shapefile)