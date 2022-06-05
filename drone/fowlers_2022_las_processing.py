"""
This combines the LAZ files made from each image band into a single file. Then
it uses droneLasProcessing.py to make the DSM, DEM and CHM for each file.
"""

import os
import sys
import glob


projList = ["p4m_fg03_20220513", "p4m_fg04_20220513", "p4m_fg05_20220513",
            "p4m_fg08_20220512", "p4m_fg11_20220514", "p4m_fg13_20220515",
            "p4m_fg14_20220512", "p4m_fg15_20220515"]

w = [40, 40, 200, 200, 200, 40, 40, 40]

# Iterate over projects
baseDir = r"C:\Users\Adrian\Documents\drone_imagery\%s\pix4d\2_densification\point_cloud"
for i, p in enumerate(projList):
    inDir = baseDir%p
    lazList = glob.glob(os.path.join(inDir, "*.laz"))
    outDir = r"C:\Users\Adrian\Documents\drone_imagery\%s"%p
    outLaz = os.path.join(outDir, '%s_merged.laz'%p)

    # Combine las files using las2las
    cmd = r"las2las -merged -i %s -o %s"%(' '.join(lazList), outLaz)
    if os.path.exists(outLaz) is False:
        os.system(cmd)
    
    # Run droneLasProcessing.py to make dsm
    cmd = r"python droneLasProcessing.py -i %s -e 32754 -p %s -o %s -w %s"%(outLaz, p, outDir, w[i])
    print(p)
    os.system(cmd)