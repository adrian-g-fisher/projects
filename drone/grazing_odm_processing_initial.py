"""
This processes the initial drone imagery captured across all grazing study sites
at Boolcoomatta, Fowlers Gap, and Witchelina.

Subsequent data will use a separate script, to ensure outputs are aligned to the
initial data by copying the LAZ file into the image directory and naming it
align.laz

outputs = ["project\\odm_dem\\dsm.tif",
           "project\\odm_dem\\dtm.tif",
           "project\\odm_georeferencing\\odm_georeferenced_model.laz",
           "project\\odm_orthophoto\\odm_orthophoto.tif"]
           
orthomosaics have 6 bands
    band 1 - Red
    band 2 - Green
    band 3 - Blue
    band 4 - NIR
    band 5 - Red-edge
    band 6 - Alpha
"""

import os
import sys
import shutil
import glob
import subprocess
from osgeo import gdal


def rename_bands(inTif):
    raster = gdal.Open(inTif, gdal.GA_Update)
    raster.GetRasterBand(1).SetDescription('Red')
    raster.GetRasterBand(2).SetDescription('Green')
    raster.GetRasterBand(3).SetDescription('Blue')
    raster.GetRasterBand(4).SetDescription('NIR')    
    raster.GetRasterBand(5).SetDescription('RedEdge')
    raster.GetRasterBand(5).SetDescription('Alpha')
    raster = None


def process_projects():
    for inDir in glob.glob("D:\\drone_multispec\\odm_initial\\*"):
        name = os.path.basename(inDir)
        logfile = os.path.join(inDir, "project\\odm.log")
        orthomosaic = os.path.join(inDir, "project\\odm_orthophoto\\odm_orthophoto.tif")
        if os.path.exists(orthomosaic) is False:
            print("Processing %s"%name)
            cmd = ["docker", "run", "--rm", "-v", "%s:/datasets"%inDir,
                   "--gpus", "all", "opendronemap/odm:gpu",
                   "--project-path", "/datasets", "project",
                   "--radiometric-calibration", "camera+sun",
                   "--pc-quality", "high", "--sfm-algorithm", "planar",
                   "--split", "800", "--split-overlap", "100"]
            with open(logfile, "a") as output:
                subprocess.run(cmd, stdout=output, stderr=output)
            
            # Remove the submodels folder to free space
            #shutil.rmtree(os.path.join(inDir, "project\\submodels"))
            
            # Move and rename TIF, LAZ, DTM, DSM and LOG to main directory
            
            # Remove all sub folders
            
            # Rename band names in orthomosaic tif
            #rename_bands(mosaic)
            
            sys.exit()
            

def copy_images(nameDirList):

    for name, srcDir in nameDirList:
        
        print(name)
        
        # Create project and images folders, and copy all images
        dstDir = os.path.join("D:\\drone_multispec\\odm_initial", name)
        if os.path.exists(dstDir) is False:
            os.mkdir(dstDir)
            os.mkdir(os.path.join(dstDir, "project"))
            os.mkdir(os.path.join(dstDir, "project\\images"))
        
            # Copy images and rename to ensure names are unique
            for imageDir in glob.glob(os.path.join(srcDir, "*PLAN")):
                for i in glob.glob(os.path.join(imageDir, "*.TIF")):
                    outDir = os.path.join(dstDir, "project\\images")
                    outImage = "%s_%s"%(os.path.basename(imageDir), os.path.basename(i))
                    outImage = os.path.join(outDir, outImage)
                    if os.path.exists(outImage) is False:
                        shutil.copy(i, outImage)

    
# Hard code all initial data
nameDirList = [["p4m_bc1_20230317", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLCON1"],
               ["p4m_bc2_20230316", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLCON2"],
               ["p4m_bc3_20230317", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLCON3"],
               ["p4m_be1_20230316", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLEX1"],
               ["p4m_be2_20230317", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLEX2"],
               ["p4m_be3_20230316", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLEX3"],
               ["p4m_wc1_20230314", "S:\\witchelina\\drone\\202303\\mosaics\\WITCHCON1"],
               ["p4m_wc2_20230313", "S:\\witchelina\\drone\\202303\\mosaics\\WITCHCON2"],
               ["p4m_we1_20230314", "S:\\witchelina\\drone\\202303\\mosaics\\WITCHEX1"],
               ["p4m_we2_20230313", "S:\\witchelina\\drone\\202303\\mosaics\\WITCHEX2"],
               ["p4m_wc3_20240305", "S:\\witchelina\\drone\\202403\\raw\\20240305\\witchcon3"],
               ["p4m_we3_20240304", "S:\\witchelina\\drone\\202403\\raw\\20240304\\witchex3"],
               ["p4m_fc1_20230318", "S:\\fowlers_gap\\imagery\\drone\\2023\\202303_exclosures\\mosaics\\conservation_control"],
               ["p4m_fe1_20230318", "S:\\fowlers_gap\\imagery\\drone\\2023\\202303_exclosures\\mosaics\\conservation_exclosure"],
               ["p4m_fc2_20230318", "S:\\fowlers_gap\\imagery\\drone\\2023\\202303_exclosures\\mosaics\\warrens_control"],
               ["p4m_fe2_20230318", "S:\\fowlers_gap\\imagery\\drone\\2023\\202303_exclosures\\mosaics\\warrens_exclosure"],
               ["p4m_fc4_20250315", "D:\\drone_multispec\\raw\\202503\\20250315\\p4m_fc4_20250315"],
               ["p4m_fc5_20250315", "D:\\drone_multispec\\raw\\202503\\20250315\\p4m_fc5_20250315"],
               ["p4m_fe4_20250315", "D:\\drone_multispec\\raw\\202503\\20250315\\p4m_fe4_20250315"],
               ["p4m_fp3_20250315", "D:\\drone_multispec\\raw\\202503\\20250315\\p4m_fp3_20250315"],
               ["p4m_fc3_20250316", "D:\\drone_multispec\\raw\\202503\\20250316\\p4m_fc3_20250316"],
               ["p4m_fe3_20250316", "D:\\drone_multispec\\raw\\202503\\20250316\\p4m_fe3_20250316"],
               ["p4m_fe5_20250316", "D:\\drone_multispec\\raw\\202503\\20250316\\p4m_fe5_20250316"],
               ["p4m_fp4_20250316", "D:\\drone_multispec\\raw\\202503\\20250316\\p4m_fp4_20250316"],
               ["p4m_fp5_20250316", "D:\\drone_multispec\\raw\\202503\\20250316\\p4m_fp5_20250316"]]

# Copy data
#copy_images(nameDirList)

# Process data
process_projects()
