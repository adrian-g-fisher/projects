"""
Process drone data for grazing study using Agisoft Metashape. This was modified
from the agisoft script using parts of the tern script.
 https://github.com/agisoft-llc/metashape-scripts/blob/master/src/samples/general_workflow.py
 https://github.com/ternaustralia/drone_metashape/blob/main/metashape_proc.py

I also added a section to make the NIR band the master, as advised by Victoria
Inman.

To create the conda environment:
 - Download the installer
 - conda create -n metashape python=3.11
 - python -m pip install Metashape-2.2.0-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl
 - conda activate metashape
 
"""


import Metashape
import os
import sys
import time
import glob
import shutil


def process_project(projectDir, epsg):
    """
    Processes a single drone survey project using metashape.
    """
    project = os.path.basename(projectDir)
    image_folder = os.path.join(projectDir, "images")
    output_folder = os.path.join(projectDir, "outputs")
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    photos = glob.glob(os.path.join(image_folder, "*.TIF"))
    print('%s photos found'%len(photos))
    doc = Metashape.Document()
    doc.save(os.path.join(output_folder, '%s_project.psx'%project))
    chunk = doc.addChunk()
    chunk.addPhotos(photos, load_xmp_accuracy=True)
    doc.save()
    
    # Change master band to NIR rather than Blue
    new_master = chunk.sensors[4]
    for s in chunk.sensors:
        s.master = None
        s.fixed_rotation = False
    for s in chunk.sensors:
        s.master = new_master
        s.fixed_rotation = True
    
    # Calibrate photos
    chunk.calibrateReflectance(use_reflectance_panels=False, use_sun_sensor=True)
    doc.save()
    
    # Change to the projected coordinate system
    SOURCE_CRS = Metashape.CoordinateSystem("EPSG::4326")
    target_crs = Metashape.CoordinateSystem("EPSG::" + epsg)
    for camera in chunk.cameras:
        camera.reference.location = Metashape.CoordinateSystem.transform(camera.reference.location,
                                                                         SOURCE_CRS,
                                                                         target_crs)
    chunk.crs = target_crs
    
    # Process data
    chunk.matchPhotos(downscale=1, generic_preselection=False, reference_preselection=True,
                      reference_preselection_mode=Metashape.ReferencePreselectionSource)
    doc.save()
    chunk.alignCameras()
    doc.save()
    chunk.optimizeCameras()
    doc.save()
    chunk.buildDepthMaps(downscale=4)
    doc.save()
    chunk.buildPointCloud()
    doc.save()
    
    # When doing subsequant surveys, instead of buildModel, I should use:
    # chunk.importModel(path=model, crs=target_crs, format=Metashape.ModelFormatOBJ)
    
    chunk.buildModel(surface_type=Metashape.HeightField,
                     source_data=Metashape.PointCloudData,
                     face_count=Metashape.MediumFaceCount)
    doc.save()

    # Decimate and smooth mesh to use as orthorectification surface
    chunk.decimateModel(face_count=len(chunk.model.faces) / 2)
    chunk.smoothModel(50) # 'low': 50, 'medium': 100, 'high': 200
    chunk.exportModel(path=os.path.join(output_folder, '%s_model.obj'%project),
                      crs=target_crs, format=Metashape.ModelFormatOBJ)

    # Build and export orthomosaic
    chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ModelData,
                           refine_seamlines=True)
    doc.save()
    if chunk.orthomosaic:
        res_xy = round(chunk.orthomosaic.resolution, 2)
        compression = Metashape.ImageCompression()
        compression.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
        compression.tiff_big = True
        compression.tiff_tiled = True
        compression.tiff_overviews = True
        chunk.exportRaster(path=os.path.join(output_folder, '%s_mosaic.tif'%project),
                           resolution_x=res_xy, resolution_y=res_xy,
                           image_format=Metashape.ImageFormatTIFF,
                           save_alpha=False,
                           source_data=Metashape.OrthomosaicData,
                           image_compression=compression)
    
    # Export point cloud
    if chunk.point_cloud:
        chunk.exportPointCloud(os.path.join(output_folder, '%s_mosaic.laz'%project),
                               format=Metashape.PointCloudFormatLAZ,
                               source_data = Metashape.PointCloudData,
                               crs=chunk.crs)
    
    # Export report
    chunk.exportReport(os.path.join(output_folder, '%s_report.pdf'%project))
    
    # Delete project folder
    del doc
    projDir = os.path.join(output_folder, '%s_project.files'%project)
    shutil.rmtree(projDir)
    
    print('Processing finished')
    
    
# Hardcode
dirList = glob.glob(r"D:\drone_multispec\metashape_initial\*")
site2epsg = {'b': '32754', 'w': '32753', 'f': '32754'}

for projectDir in dirList:
    project = os.path.basename(projectDir)
    site = project.split('_')[1][0]
    epsg = site2epsg[site]
    
    # Only run if output not present
    outputDir = os.path.join(projectDir, 'outputs')
    mosaic = os.path.join(outputDir, '%s_mosaic.tif'%project)
    if os.path.exists(mosaic) is False:
        process_project(projectDir, epsg)

print('All processing finished')