"""
This copies the drone imagery captured across all grazing study sites
at Boolcoomatta, Fowlers Gap, and Witchelina onto the D drive.

To backup data use:

    conda activate rclone

    rclone copy --progress D:\grazing_study_drone_data S:\grazing_study_drone_data

"""

import os
import sys
import shutil
import glob


def transfer_images():
    
    srcDir = r"S:\grazing_study_drone_data\metashape_initial"
    dstDir = r"D:\grazing_study_drone_data\metashape_initial"
    
    for projSrc in glob.glob(os.path.join(srcDir, "*")):
        
        proj = os.path.basename(projSrc)
        print(proj)
        
        imageSrc = os.path.join(projSrc, "images")
        projDst = os.path.join(dstDir, proj)
        imageDst = os.path.join(projDst, "images")
        
        if os.path.exists(imageDst) is False:
            os.mkdir(imageDst)
            
        for inImage in glob.glob(os.path.join(imageSrc, "*.TIF")):
            outImage = os.path.join(imageDst, os.path.basename(inImage))
            if os.path.exists(outImage) is False:
                shutil.copy(inImage, outImage)

transfer_images()

sys.exit()


def copy_images(masterDir, nameDirList):

    for name, srcDir in nameDirList:
        
        print(name)
        
        # Create project and images folders, and copy all images
        dstDir = os.path.join(masterDir, name)
        if os.path.exists(dstDir) is False:
            osD
            os.mkdir(os.path.join(dstDir, "images"))
        
            # Copy images and rename to ensure names are unique
            for imageDir in glob.glob(os.path.join(srcDir, "*PLAN")):
                for i in glob.glob(os.path.join(imageDir, "*.TIF")):
                    outDir = os.path.join(dstDir, "images")
                    outImage = "%s_%s"%(os.path.basename(imageDir), os.path.basename(i))
                    outImage = os.path.join(outDir, outImage)
                    if os.path.exists(outImage) is False:
                        shutil.copy(i, outImage)

    
# Hard code all initial data
initialList = [["p4m_bc1_20230317", "S:\\boolcoomata\\drone\\202303\\mosaics\\BOOLCON1"],
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

# Subsequent data
subList = [["p4m_bc1_20240308", r"S:\boolcoomata\drone\202403\raw\20240308\p4m_bc1_20240308"],
           ["p4m_bc3_20240308", r"S:\boolcoomata\drone\202403\raw\20240308\p4m_bc3_20240308"],
           ["p4m_be1_20240308", r"S:\boolcoomata\drone\202403\raw\20240308\p4m_be1_20240308"],
           ["p4m_be3_20240308", r"S:\boolcoomata\drone\202403\raw\20240308\p4m_be3_20240308"],
           ["p4m_bc2_20240309", r"S:\boolcoomata\drone\202403\raw\20240309\p4m_bc2_20240309"],
           ["p4m_be2_20240309", r"S:\boolcoomata\drone\202403\raw\20240309\p4m_be2_20240309"],
           ["p4m_we2_20240305", r"S:\witchelina\drone\202403\raw\20240305\wicthex2"],
           ["p4m_wc2_20240305", r"S:\witchelina\drone\202403\raw\20240305\witchcon2"],
           ["p4m_we1_20240305", r"S:\witchelina\drone\202403\raw\20240305\witchex1"],
           ["p4m_wc1_20240306", r"S:\witchelina\drone\202403\raw\20240306\witchcon1"],
           ["p4m_fc1_20240310", r"S:\fowlers_gap\imagery\drone\2024\202403_exclosures\raw\20240310\concon"],
           ["p4m_fe1_20240310", r"S:\fowlers_gap\imagery\drone\2024\202403_exclosures\raw\20240310\conex"],
           ["p4m_fc2_20240310", r"S:\fowlers_gap\imagery\drone\2024\202403_exclosures\raw\20240310\warcon"],
           ["p4m_fe2_20240311", r"S:\fowlers_gap\imagery\drone\2024\202403_exclosures\raw\20240311\warex"],
           ["p4m_bc1_20250308", r"D:\drone_multispec\raw\202503\20250308\p4m_bc1_20250308"],
           ["p4m_be1_20250308", r"D:\drone_multispec\raw\202503\20250308\p4m_be1_20250308"],
           ["p4m_bc2_20250309", r"D:\drone_multispec\raw\202503\20250309\p4m_bc2_20250309"],
           ["p4m_bc3_20250309", r"D:\drone_multispec\raw\202503\20250309\p4m_bc3_20250309"],
           ["p4m_be2_20250309", r"D:\drone_multispec\raw\202503\20250309\p4m_be2_20250309"],
           ["p4m_be3_20250309", r"D:\drone_multispec\raw\202503\20250309\p4m_be3_20250309"],
           ["p4m_wc3_20250311", r"D:\drone_multispec\raw\202503\20250311\p4m_wc3_20250311"],
           ["p4m_we2_20250311", r"D:\drone_multispec\raw\202503\20250311\p4m_we2_20250311"],
           ["p4m_we3_20250311", r"D:\drone_multispec\raw\202503\20250311\p4m_we3_20250311"],
           ["p4m_wc1_20250312", r"D:\drone_multispec\raw\202503\20250312\p4m_wc1_20250312"],
           ["p4m_wc2_20250312", r"D:\drone_multispec\raw\202503\20250312\p4m_wc2_20250312"],
           ["p4m_we1_20250312", r"D:\drone_multispec\raw\202503\20250312\p4m_we1_20250312"],
           ["p4m_fc1_20250317", r"D:\drone_multispec\raw\202503\20250317\p4m_fc1_20250317"],
           ["p4m_fc2_20250317", r"D:\drone_multispec\raw\202503\20250317\p4m_fc2_20250317"],
           ["p4m_fe1_20250317", r"D:\drone_multispec\raw\202503\20250317\p4m_fe1_20250317"],
           ["p4m_fe2_20250317", r"D:\drone_multispec\raw\202503\20250317\p4m_fe2_20250317"]]

#copy_images(r"D:\drone_multispec\metashape_initial", initialList)
#copy_images(r"D:\drone_multispec\metashape_subsequent", subList)


def remove_images(mainDir):
    """
    Removes all images folders from D to free up space
    """
    for srcDir in glob.glob(os.path.join(mainDir, "*")):
        imageDir = os.path.join(srcDir, 'images')
        if os.path.exists(imageDir):
            print(imageDir)
            shutil.rmtree(imageDir)

#remove_images(r"D:\drone_multispec\metashape_initial")