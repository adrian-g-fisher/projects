#!/usr/bin/env python
"""
conda create -c conda-forge -n pheno numpy pillow matplotlib scipy moviepy

conda activate pheno

Correct dates
20250702-20250923 Z:/LinkageGrazing/September2025/Witchelina/Phenocams/WX2PC
20250311-20250702 Z:/LinkageGrazing/Winter2025/Witchelina/Phenocams/WX2PC
20240926-20250311 Z:/LinkageGrazing/Autumn25/Witchelina/Phenocams/WX2PC
20240731-20240908 Z:/LinkageGrazing/Spring24/Phenocams/Witchelina/PCWX2

Incorrect dates :
20220131-20220528 Z:/LinkageGrazing/Winter24/Phenocams/Witchelina/PCWX2
20210719-20220131 Z:/LinkageGrazing/Witchelina/Spring23-Autumn24/Witchx2/100MEDIA
20210110-20210718 Z:/LinkageGrazing/Witchelina/witchx2/March-Sep23

The dates should be:
Winter24          20240403-20240730
Spring23-Autumn24 20230926-20240403
March-Sep23       20230320-20230925
"""

import argparse
import glob
import os, sys, shutil
import numpy as np
from PIL import Image, ExifTags, ImageDraw, ImageFont
import datetime
from moviepy.video.io import ImageSequenceClip


imageDirs = [r'Z:/LinkageGrazing/September2025/Witchelina/Phenocams/WX2PC',
             r'Z:/LinkageGrazing/Winter2025/Witchelina/Phenocams/WX2PC',
             r'Z:/LinkageGrazing/Autumn25/Witchelina/Phenocams/WX2PC',
             r'Z:/LinkageGrazing/Spring24/Phenocams/Witchelina/PCWX2',
             r'Z:/LinkageGrazing/Winter24/Phenocams/Witchelina/PCWX2',
             r'Z:/LinkageGrazing/Witchelina/Spring23-Autumn24/Witchx2/100MEDIA',
             r'Z:/LinkageGrazing/Witchelina/witchx2/March-Sep23']

def copy_midday_images():
    imageList = []
    for imageDir in imageDirs:
        imageList += glob.glob(os.path.join(imageDir, '*.JPG'))
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
    unique_dates = np.unique(dates)
    selected_images = []
    for d in unique_dates:
        midday = datetime.datetime(year=d.year, month=d.month, day=d.day,
                                   hour=12, minute=0, second=0)
        day_times = times[dates == d]
        dif = np.array([abs((dtime - midday).total_seconds()) for dtime in day_times])
        selected_time = day_times[dif == min(dif)]
        selected_image = imageList[times == selected_time][0]
        selected_images.append(selected_image)

    for i in range(len(selected_images)):
        srcImage = selected_images[i]
        dstImage = os.path.join(r'C:/Data/phenocams/wx2', '%s.JPG'%(unique_dates[i].strftime('%Y%m%d')))
        shutil.copy(srcImage, dstImage)
        print("Copied %i of %i"%(i+1, len(selected_images)))
    
    
def fix_image_dates():
    
    srcDir = r'C:/Data/phenocams/wx2'
    dstDir = r'C:/Data/phenocams/wx2_correctdates'
    srcList = glob.glob(os.path.join(srcDir, '*JPG'))
    dstList = []
    for srcImage in srcList:
        d = os.path.basename(srcImage).replace('.JPG', '')
        d = datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:]))
        if d < datetime.date(year=2022, month=5, day=29):
            dd = d + datetime.timedelta(days=794)
        else:
            dd = d
        dstImage = os.path.join(dstDir, '%s.JPG'%(dd.strftime('%Y%m%d')))
        im = Image.open(srcImage)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, im.height-180, im.width, im.height], fill="black")
        font = ImageFont.truetype(r'C:/Windows/Fonts/Arial/arial.ttf', 120)
        text = dd.strftime('%d-%m-%Y')
        position = (im.width - 650, im.height - 160)
        draw.text(position, text, font=font)
        im.save(dstImage)


def make_video():
    fps = 8
    videofile = r'C:/Data/phenocams/witchelina_ex2_daily_timelapse_8fps.mp4'
    srcDir = r'C:/Data/phenocams/wx2_correctdates'
    srcList = glob.glob(os.path.join(srcDir, '*JPG'))
    clip = ImageSequenceClip.ImageSequenceClip(srcList, fps=fps)
    clip.write_videofile(videofile, fps=fps)
    

#copy_midday_images()
#fix_image_dates()
make_video()