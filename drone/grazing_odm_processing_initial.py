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
"""

import os
import sys
import shutil
import glob
import subprocess


def process_project(nameDirList):
    
    for name, srcDir in nameDirList:
    
        inDir = os.path.join("D:\\drone_multispec\\odm", name)
    
        logfile = os.path.join(inDir, "project\\odm.log")
        if os.path.exists(logfile):
            os.remove(logfile)
        print("Processing %s"%name)
        cmd = ("docker run --rm -v %s:/datasets "%inDir +
               "opendronemap/odm --project-path /datasets project --dtm --dsm " +
               "–-radiometric-calibration camera+sun --pc-quality high " +
               "--orthophoto-resolution 4 --dem-resolution 4 --smrf-threshold 0.1 " +
               "--split 400 --split-overlap 50")
        with open(logfile, "a") as output:
            subprocess.call(cmd, shell=True, stdout=output, stderr=output)


def copy_images(nameDirList):

    for name, srcDir in nameDirList:
        
        # Create project and images folders, and copy all images
        dstDir = os.path.join("D:\\drone_multispec\\odm", name)
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
                    cmd = r"cp %s %s"%(i, outImage)
                    os.system(cmd)
    
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
               ["p4m_fe2_20230318", "S:\\fowlers_gap\\imagery\\drone\\2023\\202303_exclosures\\mosaics\\warrens_exclosure"]]

# Copy data
copy_images(nameDirList)

# Process data
#process_project(nameDirList)
