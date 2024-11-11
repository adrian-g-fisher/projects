#!/usr/bin/env python
"""

Makes a timelapse video from daily images

conda create -c conda-forge -n pheno numpy pillow matplotlib scipy moviepy

conda activate pheno

"""

import argparse
import glob
import os, sys
import numpy as np
from PIL import Image, ExifTags
import datetime
from moviepy.video.io import ImageSequenceClip


def main(baseDir, outDir, fps, imageDirName):
    """
    """
    for siteDir in glob.glob(os.path.join(baseDir, '*')):
        site = os.path.basename(siteDir)
        print(site)
        videofile = os.path.join(outDir, r'%s_daily_timelapse.mp4'%site)
        
        if os.path.exists(videofile) is False:
            
            # Get image list
            imageDir = os.path.join(siteDir, imageDirName)
            imageList = glob.glob(os.path.join(imageDir, '**/*.JPG'), recursive=True)
            imageList = np.array(imageList)

            # Get datetime list
            dates = []
            times = []
            for i in imageList:
                im = Image.open(i)
                exifdata = im.getexif()
                for tag_id in exifdata:
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == r'DateTime':
                        dt = exifdata.get(tag_id)
                        if isinstance(dt, bytes):
                            dt = dt.decode()
                (d, t) = dt.split(r' ')
                d = d.replace(r':', r'')
                d = datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]))
                t = datetime.datetime(year=d.year, month=d.month, day=d.day,
                                  hour=int(t[:2]), minute=int(t[3:5]), second=int(t[6:]))
                dates.append(d)
                times.append(t)
            dates = np.array(dates)
            times = np.array(times)
        
            # For each day select the image closest to 12:00
            selected_images = []
            for d in np.unique(dates):
                midday = datetime.datetime(year=d.year, month=d.month, day=d.day,
                                       hour=12, minute=0, second=0)
                day_times = times[dates == d]
                dif = np.array([abs((dtime - midday).total_seconds()) for dtime in day_times])
                selected_time = day_times[dif == min(dif)]
                selected_image = imageList[times == selected_time][0]
                selected_images.append(selected_image)
        
            # Make video
            clip = ImageSequenceClip.ImageSequenceClip(selected_images, fps=fps)
            clip.write_videofile(videofile, fps=fps)


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Makes a timelapse video from daily images"))
    p.add_argument("-i", "--inDir", dest="inDir", default=None,
                   help=("Input directory with subdirectories for each camera"))
    p.add_argument("-o", "--outDir", dest="outDir", default=None,
                   help=("Output directory for video files"))
    p.add_argument("-f", "--fps", dest="fps", default=4,
                   help=("Frames per second (default=4)"))
    p.add_argument("-m", "--imageDir", dest="imageDir", default='images',
                   help=("Name of directory with images (default='images')")) 
    cmdargs = p.parse_args()
    if (cmdargs.inDir is None and cmdargs.outDir is None):
        p.print_help()
        print("Must name input and output directories")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.inDir, cmdargs.outDir, float(cmdargs.fps), cmdargs.imageDir)