#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rios import applier
from scipy import stats


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def makeLandsatNDVI(info, inputs, outputs, otherargs):
    """
    """
    red = inputs.dbg[3]
    nir = inputs.dbg[4]
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi = (nir - red) / sumRedNir
    ndvi[red + nir == 0] = 2
    ndvi[red == 32767] = 2
    outputs.ndvi = np.array([ndvi]).astype(np.float32)


def make_landsat_ndvi_images():
    srcDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\dbg"
    dstDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\ndvi"
    for i, inimage in enumerate(glob.glob(os.path.join(srcDir, "*.img"))):
        outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".img", "_ndvi.img"))
        if os.path.exists(outimage) is False:
            infiles = applier.FilenameAssociations()
            infiles.dbg = inimage
            outfiles = applier.FilenameAssociations()
            outfiles.ndvi = outimage
            otherargs = applier.OtherInputs()
            controls = applier.ApplierControls()
            controls.setStatsIgnore(2)
            controls.setCalcStats(True)
            applier.apply(makeLandsatNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)


def NDVIvsPV(info, inputs, outputs, otherargs):
    """
    """
    ndvi = inputs.ndvi[0]
    pv = inputs.fc[1] - 100
    pv[pv < 0] = 0
    pv[pv > 100] = 100
    valid = (ndvi != 2) & (pv != 0)
    pv = pv[valid]
    ndvi = ndvi[valid]
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    ax.set_title(otherargs.date, fontsize=10)
    h = ax.hist2d(ndvi, pv, bins=[80, 80], range=[[-1, 1], [0, 100]],
                  norm=mpl.colors.LogNorm())
    ax.set_xlabel('NDVI')            
    ax.set_ylabel('PV (%)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 100])
    (slope, intercept, r, p, se) = stats.linregress(ndvi, pv)
    ax.plot([-1, 1], [-slope+intercept, slope+intercept], ls='-', c='r', lw=0.5)
    
    rsquared = r**2
    pred = intercept + slope * ndvi
    rmse = np.sqrt(np.mean((pv - pred)**2))
    ax.text(-0.95, 95, 'y = %.2f + %.2fx'%(intercept, slope), fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 88, '$r^2$ = %.2f'%rsquared, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 81, 'RMSE = %.2f'%rmse, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 74, 'n = %i'%pv.size, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    plt.savefig(otherargs.plot, dpi=300)
    plt.close()
    otherargs.ndvi.append(ndvi)
    otherargs.pv.append(pv)
    

def compare_landsat_ndvi_pv():
    plotDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\plots"
    subset = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\l8olre_p096r081_20160928_dbg_subset.img"
    ndviDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\ndvi"
    fcDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\dil"
    NDVI = []
    PV = []
    for i, ndvi_image in enumerate(glob.glob(os.path.join(ndviDir, "*.img"))):
        fc_image = os.path.join(fcDir, os.path.basename(ndvi_image).replace("_dbgm4_ndvi", "_dilm4"))
        infiles = applier.FilenameAssociations()
        infiles.ndvi = ndvi_image
        infiles.fc = fc_image
        infiles.aoi = subset
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        date = os.path.basename(ndvi_image).split('_')[2]
        otherargs.date = date
        otherargs.plot = os.path.join(plotDir, '%s_ndvi_pv.png'%date)
        controls = applier.ApplierControls()
        controls.setReferenceImage(infiles.aoi)
        controls.setWindowXsize(1002)
        controls.setWindowYsize(1002)
        otherargs.ndvi = NDVI
        otherargs.pv = PV
        applier.apply(NDVIvsPV, infiles, outfiles, otherArgs=otherargs, controls=controls)
        
    ndvi = np.concatenate(otherargs.ndvi)
    pv = np.concatenate(otherargs.pv)
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    h = ax.hist2d(ndvi, pv, bins=[80, 80], range=[[-1, 1], [0, 100]], norm=mpl.colors.LogNorm())
    ax.set_xlabel('NDVI')            
    ax.set_ylabel('PV (%)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 100])
    (slope, intercept, r, p, se) = stats.linregress(ndvi, pv)
    ax.plot([-1, 1], [-slope+intercept, slope+intercept], ls='-', c='r', lw=0.5)
    rsquared = r**2
    pred = intercept + slope * ndvi
    rmse = np.sqrt(np.mean((pv - pred)**2))
    ax.text(-0.95, 95, 'y = %.2f + %.2fx'%(intercept, slope), fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 88, '$r^2$ = %.2f'%rsquared, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 81, 'RMSE = %.2f'%rmse, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    ax.text(-0.95, 74, 'n = %i'%pv.size, fontsize=8, bbox={"ec":"white", "fc": "white", "pad": 0})
    plt.savefig(r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\plots\total_ndvi_pv.png", dpi=300)
    plt.close()


def makeDroneNDVI(info, inputs, outputs, otherargs):
    """
    """
    red = inputs.drone[2]
    nir = inputs.drone[4]
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi = (nir - red) / sumRedNir
    ndvi[red + nir == 0] = 2
    ndvi[(red == -10000) | (nir == -10000)] = 2
    outputs.ndvi = np.array([ndvi]).astype(np.float32)


def make_drone_ndvi_images():
    droneList = ['S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\conservation_control\\conservation_control_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\conservation_exclosure\\conservation_exclosure_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\warrens_control\\warrens_control_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\warrens_exclosure\\warrens_exclosure_20230318_multiband.tif']
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_ausplots\\conservation\\conausplot_20240321_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_ausplots\\emuexclosure\\emuausplot_20240323_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_ausplots\\southsandstone\\southsandausplot_20240322_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_ausplots\\emugrazed\\emugrazedausplot_20240322_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_exclosures\\mosaics\\concon_20240310_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_exclosures\\mosaics\\conex_20240310_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_exclosures\\mosaics\\warcon_20240310_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2024\\202403_exclosures\\mosaics\\warex_20240311_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2022\\p4m_conausplot_20220514\\p4m_conausplot_20220514_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2022\\p4m_emuausplot_20220512\\p4m_emuausplot_20220512_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2022\\p4m_fg03_20220513\\p4m_fg03_20220513_multiband.tif',
                 
    
    dstDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi"
    for inimage in droneList:
        outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".tif", "_ndvi.img"))
        if os.path.exists(outimage) is False:
            infiles = applier.FilenameAssociations()
            infiles.drone = inimage
            outfiles = applier.FilenameAssociations()
            outfiles.ndvi = outimage
            otherargs = applier.OtherInputs()
            controls = applier.ApplierControls()
            controls.setStatsIgnore(2)
            controls.setCalcStats(True)
            applier.apply(makeDroneNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)

def rescaleDrone(info, inputs, outputs, otherargs):
    """
    """
    rescaled = inputs.drone[0]
    polys = inputs.polys[0]
    rescaled[polys == 0] = 2
    outputs.rescaled = np.array([rescaled])
    
    landsat_ndvi = inputs.landsat[0][polys == 1]
    drone_ndvi = rescaled[polys == 1]
    for i in range(landsat_ndvi.size):
        with open(otherargs.csvfile, 'a') as f:
            f.write('%s,%.4f,%.4f\n'%(otherargs.site, drone_ndvi[i], landsat_ndvi[i]))


def scale_drone_to_landsat():
    droneDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi"
    landsatNDVI = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\ndvi\l9olre_p096r081_20230316_dbgm4_ndvi.img"
    subset = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\l8olre_p096r081_20160928_dbg_subset.img"
    polys = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\landsat_drone_pixel_sites.shp"
    csvfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison.csv"
    with open(csvfile, 'w') as f:
        f.write('site,drone_ndvi,landsat_ndvi\n')
    for droneImage in glob.glob(os.path.join(droneDir, "*.img")):
        rescaled = droneImage.replace(".img", "_30m.img")
        infiles = applier.FilenameAssociations()
        infiles.drone = droneImage
        infiles.landsat = landsatNDVI
        infiles.aoi = subset
        infiles.polys = polys
        outfiles = applier.FilenameAssociations()
        outfiles.rescaled = rescaled
        otherargs = applier.OtherInputs()
        otherargs.csvfile = csvfile
        otherargs.site = '%s %s'%(os.path.basename(droneImage).split('_')[0], os.path.basename(droneImage).split('_')[0])
        controls = applier.ApplierControls()
        controls.setReferenceImage(infiles.aoi)
        controls.setResampleMethod('average')
        controls.setStatsIgnore(2)
        controls.setCalcStats(True)
        applier.apply(rescaleDrone, infiles, outfiles, otherArgs=otherargs, controls=controls)

def plot_landsat_vs_drone():
    csvfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison.csv"
    d = np.genfromtxt(csvfile, names=True, delimiter=',')
    output = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison.png"
    
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    
    #h = ax.scatter(d['drone_ndvi'], d['landsat_ndvi'], marker='o', edgecolor='0.5', facecolor='0.5', s=2)
    
    h = ax.hist2d(d['drone_ndvi'], d['landsat_ndvi'], bins=[80, 80], range=[[0, 0.35], [0, 0.35]],
                  norm=mpl.colors.LogNorm())
    
    ax.set_xlabel('Drone NDVI')
    ax.set_ylabel('Landsat NDVI')
    ax.set_xlim([0, 0.35])
    ax.set_ylim([0, 0.35])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.plot([0, 0.35], [0, 0.35], ls='-', c='k', lw=0.5)
    
    (slope, intercept, r, p, se) = stats.linregress(d['drone_ndvi'], d['landsat_ndvi'])
    
    minX = np.min(d['drone_ndvi'])
    maxX = np.max(d['drone_ndvi'])
    ax.plot([minX, maxX], [(slope*minX)+intercept, (slope*maxX)+intercept], ls='-', c='r', lw=0.5)
    rsquared = r**2
    pred = intercept + slope * d['drone_ndvi']
    rmse = np.sqrt(np.mean((d['landsat_ndvi'] - pred)**2))
    ax.text(0.01, 0.33, 'y = %.2f + %.2fx'%(intercept, slope), fontsize=8)
    ax.text(0.01, 0.31, '$r^2$ = %.2f'%rsquared, fontsize=8)
    ax.text(0.01, 0.29, 'RMSE = %.2f'%rmse, fontsize=8)
    ax.text(0.01, 0.27, 'n = %i'%d['drone_ndvi'].size, fontsize=8)
    plt.savefig(output, dpi=300)
    plt.close()


def makeDronePV(info, inputs, outputs, otherargs):
    """
    """
    ndvi = inputs.ndvi[0]
    nodata = (ndvi == 2)
    pv = (90.71*ndvi) - 5.61
    pv[pv < 0] = 0
    pv[pv > 100] = 100
    pv[ndvi == 2] = 255
    outputs.pv = np.array([pv]).astype(np.uint8)


def make_drone_pv_images():
    droneList = ['S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\conservation_control\\conservation_control_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\conservation_exclosure\\conservation_exclosure_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\warrens_control\\warrens_control_20230318_multiband.tif',
                 'S:\\fowlers_gap\\imagery\\drone\\2023\\202303\\mosaics\\warrens_exclosure\\warrens_exclosure_20230318_multiband.tif']
    
    ndviDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi"
    pvDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_pv"
    for inimage in droneList:
        ndvi_image = os.path.join(ndviDir, os.path.basename(inimage).replace(".tif", "_ndvi.img"))
        pv_image = os.path.join(pvDir, os.path.basename(inimage).replace(".tif", "_pv.img"))
        if os.path.exists(pv_image) is False:
            infiles = applier.FilenameAssociations()
            infiles.ndvi = ndvi_image
            outfiles = applier.FilenameAssociations()
            outfiles.pv = pv_image
            otherargs = applier.OtherInputs()
            controls = applier.ApplierControls()
            controls.setStatsIgnore(255)
            controls.setCalcStats(True)
            applier.apply(makeDronePV, infiles, outfiles, otherArgs=otherargs, controls=controls)


#make_landsat_ndvi_images()
#compare_landsat_ndvi_pv()
#make_drone_ndvi_images()
#scale_drone_to_landsat()
#plot_landsat_vs_drone()
make_drone_pv_images()