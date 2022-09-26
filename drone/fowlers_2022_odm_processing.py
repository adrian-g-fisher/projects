import os
import sys
import shutil
import glob
import subprocess

#outputs = ["project\\odm_dem\\dsm.tif",
#           "project\\odm_dem\\dtm.tif",
#           "project\\odm_georeferencing\\odm_georeferenced_model.laz",
#           "project\\odm_orthophoto\\odm_orthophoto.tif"]
dirList = ["p4m_fg04_20220513"]

for name in dirList:
    
    # Process imagery
    inDir = os.path.join("C:\\Users\\Adrian\\Documents\\drone_imagery", name)
    
    # Create project and images folders, and move all images inside
    dem = os.path.join(inDir, "dtm_%s.tif"%name)
    if os.path.exists(dem) is False:
        if os.path.exists(os.path.join(inDir, "project")) is False:
            os.mkdir(os.path.join(inDir, "project"))
            os.mkdir(os.path.join(inDir, "project\\images"))
        for imageDir in glob.glob(os.path.join(inDir, "images\\*")):
            for i in glob.glob(os.path.join(imageDir, "*.TIF")):
                outDir = os.path.join(inDir, "project\\images")
                outImage = "%s_%s"%(os.path.basename(imageDir), os.path.basename(i))
                outImage = os.path.join(outDir, outImage)
                if os.path.exists(outImage) is False:
                    cmd = r"cp %s %s"%(i, outImage)
                    os.system(cmd)
                    
        numImages = len(glob.glob(os.path.join(inDir, "project\\images\\*.TIF")))
        
        # Run ODM
        logfile = os.path.join(inDir, "project\\odm.log")
        if os.path.exists(logfile):
            os.remove(logfile)
        print("Processing %s"%name)
        cmd = ("docker run --rm -v %s:/datasets "%inDir +
               "opendronemap/odm --project-path /datasets project --dtm --dsm " +
               "–-radiometric-calibration camera+sun")
        if numImages > 400:
            cmd += " --split 400 --split-overlap 50"
        #with open(logfile, "a") as output:
        #    subprocess.call(cmd, shell=True, stdout=output, stderr=output)
    
        print(cmd)
    
    # Organise results
    #dem = os.path.join(inDir, outputs[0])
    #if os.path.exists(dem) is True:
    #    inDir = os.path.join("C:\\Users\\Adrian\\Documents\\drone_imagery", name)
    #    for inData in outputs:
    #        inData = os.path.join(inDir, inData)
    #        if os.path.exists(inData):
    #            outData = os.path.join(inDir, os.path.basename(inData)[:-4] + "_" + name + os.path.basename(inData)[-4:])
    #            os.rename(inData, outData)
    #    if os.path.exists(os.path.join(inDir, "project\\images")):
    #        os.rename(os.path.join(inDir, "project\\images"), os.path.join(inDir, "images"))
    #    if os.path.exists(os.path.join(inDir, "project")):
    #        shutil.rmtree(os.path.join(inDir, "project"))  
    #    dem = os.path.join(inDir, "dtm_%s.tif"%name)
    #    hillshade = dem.replace("dtm_", "hillshade_")
    #    if os.path.exists(hillshade) is False:
    #        if os.path.exists(dem) is True:
    #            os.system("gdaldem hillshade %s %s"%(dem, hillshade))