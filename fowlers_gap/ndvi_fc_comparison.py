#!/usr/bin/env python

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rios import applier
from scipy import stats, ndimage


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
    
    nodata = info.getNoDataValueFor(inputs.drone)
    red = inputs.drone[otherargs.red]
    nir = inputs.drone[otherargs.nir]
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi = (nir - red) / sumRedNir
    ndvi[red + nir == 0] = 2
    ndvi[(red == nodata) | (nir == nodata)] = 2
    outputs.ndvi = np.array([ndvi]).astype(np.float32)


def make_drone_ndvi_images():
    droneList = [r'S:\fowlers_gap\imagery\drone\2024\202403_ausplots\conservation\conausplot_20240321_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_ausplots\emuexclosure\emuausplot_20240323_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_ausplots\southsandstone\southsandausplot_20240322_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_ausplots\emugrazed\emugrazedausplot_20240322_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_exclosures\mosaics\concon_20240310_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_exclosures\mosaics\conex_20240310_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_exclosures\mosaics\warcon_20240310_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2024\202403_exclosures\mosaics\warex_20240311_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2023\202303\mosaics\concon_20230318_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2023\202303\mosaics\conex_20230318_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2023\202303\mosaics\warcon_20230318_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2023\202303\mosaics\warex_20230318_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_conausplot_20220514\p4m_conausplot_20220514_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_emuausplot_20220512\p4m_emuausplot_20220512_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg03_20220513\p4m_fg03_20220513_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg04_20220513\p4m_fg04_20220513_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg05_20220513\p4m_fg05_20220513_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg08_20220512\p4m_fg08_20220512_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg11_20220514\p4m_fg11_20220514_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg13_20220515\p4m_fg13_20220515_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg14_20220512\p4m_fg14_20220512_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2022\p4m_fg15_20220515\p4m_fg15_20220515_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\con1_20210430_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\con2_20210430_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\emu1_20210502_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\emu2_20210502_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\man1_20210503_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\man2_20210503_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\war1_20210501_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\war2_20210501_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\war3_20210504_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2021\war4_20210504_multiband.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\COP_ST32ST4_3DRSO_MSPSEQ_20160913T0030_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\COV_ST51ST52ST54_3DRSO_MSPSEQ_20160914T0030_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\EMU_ST21_3DRSO_MSPSEQ_20160915T1215_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\EMU_ST22_3DRSO_MSPSEQ_20160914T0330_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\EMU_ST23_3DRSO_MSPSEQ_20160915T2230_L1_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\EMU_ST24_3DRSO_MSPSEQ_20160915T2300_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\LAK_ST41_3DRSO_MSPSEQ_20160915T0100_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\LAK_ST43_3DRSO_MSPSEQ_20160915T0040_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\LAK_ST44_3DRSO_MSPSEQ_201609150020_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\SSP_ST11_3DRSO_MSPSEQ_20160914T0815_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\SSP_ST12_3DRSO_MSPSEQ_20160914T0315_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\SSP_ST13_3DRSO_MSPSEQ_20160913T0500_L1S_OM.tif',
                 r'S:\fowlers_gap\imagery\drone\2016\SSP_ST14_3DRSO_MSPSEQ_20160914T0730_L1S_OM.tif']
    
    dstDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi"
    for inimage in droneList:
        outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".tif", "_ndvi.img"))
        print('Creating %s'%os.path.basename(outimage))
        infiles = applier.FilenameAssociations()
        infiles.drone = inimage
        outfiles = applier.FilenameAssociations()
        outfiles.ndvi = outimage
        otherargs = applier.OtherInputs()
        # If ends in _multiband.tif then P4 (5 band) otherwise Sequoia (4 band)
        if inimage.split('_')[-1] == 'multiband.tif':
            otherargs.red = 2
            otherargs.nir = 4
        else:
            otherargs.red = 1
            otherargs.nir = 3
        controls = applier.ApplierControls()
        controls.setStatsIgnore(2)
        controls.setCalcStats(True) 
        applier.apply(makeDroneNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)


def rescaleDrone(info, inputs, outputs, otherargs):
    """
    """
    rescaled = inputs.drone[0]
    binary = inputs.binary[0]
    rescaled[binary == 0] = 2
    outputs.rescaled = np.array([rescaled])
    
    landsat_ndvi = inputs.landsat[0][binary == 100]
    drone_ndvi = rescaled[binary == 100]
    for i in range(landsat_ndvi.size):
        with open(otherargs.csvfile, 'a') as f:
            f.write('%s,%s,%.4f,%.4f\n'%(otherargs.dates[0],
                                         otherargs.dates[1],
                                         drone_ndvi[i], landsat_ndvi[i]))


def scale_drone_to_landsat():
    
    # Get all drone images and dates
    droneDir = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi'
    droneImages = []
    droneDates = []
    for droneImage in glob.glob(os.path.join(droneDir, '*_ndvi.img')):
        droneImages.append(droneImage)
        if len(os.path.basename(droneImage).split('_')) == 4:
            droneDates.append(int(os.path.basename(droneImage).split('_')[1]))
        elif len(os.path.basename(droneImage).split('_')) == 5:
            droneDates.append(int(os.path.basename(droneImage).split('_')[2]))
        else:
            droneDates.append(int(os.path.basename(droneImage).split('_')[4][0:8]))
    droneImages = np.array(droneImages)
    droneDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                         month=int(str(x)[4:6]),
                                         day=int(str(x)[6:]))
                                         for x in droneDates], dtype=np.datetime64)
    
    # Get all Landsat images and dates
    landsatDir= r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\ndvi'
    landsatImages = []
    landsatDates = []
    for landsatImage in glob.glob(os.path.join(landsatDir, '*_ndvi.img')):
        landsatImages.append(landsatImage)
        landsatDates.append(int(os.path.basename(landsatImage).split('_')[2]))
    landsatImages = np.array(landsatImages)
    landsatDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                           month=int(str(x)[4:6]),
                                           day=int(str(x)[6:]))
                                           for x in landsatDates], dtype=np.datetime64)
    
    # Match drone and landsat using minimum date difference
    droneLandsat = []
    matchedDates = []
    for i in range(len(droneImages)):
        dateDiff = np.array([abs((droneDates[i] - d)) for d in landsatDates], dtype=np.datetime64)
        landsatImage = landsatImages[dateDiff == min(dateDiff)][0]
        landsatDate = landsatDates[dateDiff == min(dateDiff)][0]
        droneLandsat.append([droneImages[i], landsatImage])
        matchedDates.append([droneDates[i], landsatDate])
    droneLandsat = np.array(droneLandsat)
    matchedDates = np.array(matchedDates)

    # Create output file
    csvfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison.csv"
    with open(csvfile, 'w') as f:
        f.write('drone_date,landsat_date,drone_ndvi,landsat_ndvi\n')
    
    # Resample drone and extract values
    polyDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\polygons"
    for i in range(len(droneImages)):
        droneImage = droneLandsat[i, 0]
        landsatImage = droneLandsat[i, 1]
        droneDate = np.datetime_as_string(matchedDates[i, 0], unit='D').replace('-', '')
        landsatDate = np.datetime_as_string(matchedDates[i, 1], unit='D').replace('-', '')
        binaryFile = os.path.join(polyDir, os.path.basename(droneImage).replace('_ndvi.img', '_binary_30m.img'))
        rescaled = droneImage.replace(".img", "_30m.img")
        infiles = applier.FilenameAssociations()
        infiles.drone = droneImage
        infiles.landsat = landsatImage
        infiles.binary = binaryFile
        outfiles = applier.FilenameAssociations()
        outfiles.rescaled = rescaled
        otherargs = applier.OtherInputs()
        otherargs.csvfile = csvfile
        otherargs.dates = [droneDate, landsatDate]
        controls = applier.ApplierControls()
        controls.setReferenceImage(infiles.landsat)
        controls.setResampleMethod('average')
        controls.setStatsIgnore(2)
        controls.setCalcStats(True)
        applier.apply(rescaleDrone, infiles, outfiles, otherArgs=otherargs, controls=controls)


def format_date(d_bytes):
    s = d_bytes.decode('utf-8')
    return datetime.datetime.strptime(s, '%Y%m%d').date()


def make_comparison_plot(x, y, plotfile):
    fig = plt.figure()
    fig.set_size_inches((3, 3))
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    h = ax.hist2d(x, y, bins=[100, 100], range=[[0, 1], [0, 1]], norm=mpl.colors.LogNorm())
    ax.set_xlabel('Drone NDVI')
    ax.set_ylabel('Landsat NDVI')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], ls='-', c='k', lw=0.5)
    (slope, intercept, r, p, se) = stats.linregress(x, y)
    minX = np.min(x)
    maxX = np.max(x)
    ax.plot([minX, maxX], [(slope*minX)+intercept, (slope*maxX)+intercept], ls='-', c='r', lw=0.5)
    rsquared = r**2
    pred = intercept + slope * x
    rmse = np.sqrt(np.mean((y - pred)**2))
    ax.text(0.05, 0.9, 'y = %.2f + %.2fx'%(intercept, slope), fontsize=8)
    ax.text(0.05, 0.85, '$r^2$ = %.2f'%rsquared, fontsize=8)
    ax.text(0.05, 0.8, 'RMSE = %.2f'%rmse, fontsize=8)
    ax.text(0.05, 0.75, 'n = %i'%x.size, fontsize=8)
    plt.savefig(plotfile, dpi=300)
    plt.close()


def plot_landsat_vs_drone():
    csvfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison.csv"
    d = np.genfromtxt(csvfile, names=True, delimiter=',', dtype=None,
                      converters={0: format_date, 1: format_date})

    x = d['drone_ndvi']
    y = d['landsat_ndvi']
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\all_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)
    
    x = d['drone_ndvi'][d['drone_date'] < datetime.date(2020, 1, 1)]
    y = d['landsat_ndvi'][d['drone_date'] < datetime.date(2020, 1, 1)]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\seq_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)
    
    x = d['drone_ndvi'][d['drone_date'] > datetime.date(2020, 1, 1)]
    y = d['landsat_ndvi'][d['drone_date'] > datetime.date(2020, 1, 1)]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\p4m_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)

    x = d['drone_ndvi'][d['drone_date'] > datetime.date(2024, 1, 1)]
    y = d['landsat_ndvi'][d['drone_date'] > datetime.date(2024, 1, 1)]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\2024_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)
    
    x = d['drone_ndvi'][(d['drone_date'] > datetime.date(2023, 1, 1)) & (d['drone_date'] < datetime.date(2024, 1, 1))]
    y = d['landsat_ndvi'][(d['drone_date'] > datetime.date(2023, 1, 1)) & (d['drone_date'] < datetime.date(2024, 1, 1))]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\2023_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)

    x = d['drone_ndvi'][(d['drone_date'] > datetime.date(2022, 1, 1)) & (d['drone_date'] < datetime.date(2023, 1, 1))]
    y = d['landsat_ndvi'][(d['drone_date'] > datetime.date(2022, 1, 1)) & (d['drone_date'] < datetime.date(2023, 1, 1))]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\2022_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)

    x = d['drone_ndvi'][(d['drone_date'] > datetime.date(2021, 1, 1)) & (d['drone_date'] < datetime.date(2022, 1, 1))]
    y = d['landsat_ndvi'][(d['drone_date'] > datetime.date(2021, 1, 1)) & (d['drone_date'] < datetime.date(2022, 1, 1))]
    plotfile = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_landsat_comparison_plots\2021_drone_landsat_comparison.png"
    make_comparison_plot(x, y, plotfile)


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


def makeDronePolygons(info, inputs, outputs, otherargs):
    poly = inputs.polygon[0]
    outputs.binary = np.array([poly*100]).astype(np.uint8)
    

def makeLandsatPolygons(info, inputs, outputs, otherargs):
    drone = inputs.drone[0]
    drone[drone == 255] = 0
    drone[drone < 100] = 0
    outputs.binary30m = np.array([drone]).astype(np.uint8)


def burn_polygons():
    droneDir = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi'
    for droneImage in glob.glob(os.path.join(droneDir, '*_ndvi.img')):
        polyDir = r"C:\Users\Adrian\Documents\fowlers_gap_ndvi\polygons"
        polygonFile = os.path.join(polyDir, os.path.basename(droneImage).replace('_ndvi.img', '.shp'))
        landsatImage = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\l8olre_p096r081_20160928_dbg_subset.img'
        
        # Burn polygon into drone pixels as (100, 0)
        binary = polygonFile.replace('.shp', '_binary.img')
        if os.path.exists(binary) is False:
            infiles = applier.FilenameAssociations()
            infiles.drone = droneImage
            infiles.polygon = polygonFile
            outfiles = applier.FilenameAssociations()
            outfiles.binary = binary
            otherargs = applier.OtherInputs()
            controls = applier.ApplierControls()
            controls.setStatsIgnore(255)
            controls.setCalcStats(True)
            applier.apply(makeDronePolygons, infiles, outfiles, otherArgs=otherargs, controls=controls)
        
        # Resample binary to landsat pixels as average and then make 30 m binary
        binary30m = polygonFile.replace('.shp', '_binary_30m.img')
        if os.path.exists(binary30m) is False:
            infiles = applier.FilenameAssociations()
            infiles.drone = binary
            infiles.landsat = landsatImage
            outfiles = applier.FilenameAssociations()
            outfiles.binary30m = binary30m
            otherargs = applier.OtherInputs()
            controls = applier.ApplierControls()
            controls.setStatsIgnore(0)
            controls.setCalcStats(True)
            controls.setReferenceImage(infiles.landsat)
            controls.setResampleMethod('average')
            applier.apply(makeLandsatPolygons, infiles, outfiles, otherArgs=otherargs, controls=controls)


def extractNDVI(info, inputs, outputs, otherargs):
    poly = inputs.poly[0]
    ndvi = inputs.ndvi[0]
    polysPresent = np.unique(poly[poly != 0])
    if len(polysPresent) > 0:
        uids = poly[poly != 0]
        ndviValues = ndvi[poly != 0]
        for i in range(uids.size):
            otherargs.pixels.append([uids[i], ndviValues[i]])


def extract_plant_ndvi():
    csvfile = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\plant_ndvi.csv'
    with open(csvfile, 'w') as f:
        f.write('Drone image,Plant,Mean NDVI,Stdev NDVI,Mean PV,Stdev PV,Mean NPV,Stdev NPV\n')
    
    shapeDir = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\plants'
    ndviDir = r'C:\Users\Adrian\Documents\fowlers_gap_ndvi\drone_ndvi'
    for polyfile in glob.glob(os.path.join(shapeDir, r'*.shp')):
        prefix = '_'.join(os.path.basename(polyfile).split('_')[0:-1])
        ndviImage = os.path.join(ndviDir, r'%s_multiband_ndvi.img'%(prefix))
        infiles = applier.FilenameAssociations()
        infiles.ndvi = ndviImage
        infiles.poly = polyfile
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        otherargs.pixels = []
        controls = applier.ApplierControls()
        controls.setBurnAttribute('Id')
        applier.apply(extractNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
        # Calculate statistics on pixels within polygons
        values = np.array(otherargs.pixels).astype(np.float32)
        
        Name = os.path.basename(polyfile)
        
        print(os.path.basename(polyfile))
        print(np.mean(values[:, 1]), np.std(values[:, 1]))
        
        for i in range(uids.size):
            siteID = int(uids[i])
            
            with open(csvfile, "a") as f:
                f.write('%s,%i,%i,%.4f,%.4f\n'%(Name, siteID, countValues[i],
                                                    meanNDVI[i], stdNDVI[i]))


#make_landsat_ndvi_images()
#compare_landsat_ndvi_pv()
#make_drone_ndvi_images()
#burn_polygons()
#scale_drone_to_landsat()
#plot_landsat_vs_drone()
#make_drone_pv_images()
extract_plant_ndvi()