import glob
import os

folderList = [r'C:\Users\Adrian\Documents\fowlers_game_cameras\kangaroo_shades\2c\DCIM',
              r'C:\Users\Adrian\Documents\fowlers_game_cameras\kangaroo_shades\2w\DCIM',
              r'C:\Users\Adrian\Documents\fowlers_game_cameras\kangaroo_shades\2s\DCIM']

for main in folderList:
    for subfolder in glob.glob(os.path.join(main, r'*')):
        dstDir = os.path.dirname(main)
        sub = os.path.basename(subfolder)
        for srcfile in glob.glob(os.path.join(subfolder, r'*.JPG')):
            dstfile = os.path.join(dstDir, '%s_%s'%(sub, os.path.basename(srcfile)))
            os.rename(srcfile, dstfile)