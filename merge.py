import os
from osgeo import gdal

def merge_tiffs(input_dir, output_file):
    # Get all TIFF files in the directory
    tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

    # Use GDAL to create a Virtual Raster (VRT) to merge the images
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
    vrt = gdal.BuildVRT(os.path.join(input_dir, 'temp.vrt'), tiff_files, options=vrt_options)

    # Convert the VRT to the final GeoTIFF output
    gdal.Translate(output_file, vrt, format='GTiff')

    # Delete the temporary VRT file
    vrt = None
    os.remove(os.path.join(input_dir, 'temp.vrt'))

    print(f"Merged TIFF saved as {output_file}")
