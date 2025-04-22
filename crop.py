from osgeo import gdal, ogr
import os

def clip_raster_by_vector(input_raster, input_shapefile, output_dir):
    # Open the raster and vector layers
    raster = gdal.Open(input_raster)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = driver.Open(input_shapefile)
    layer = shapefile.GetLayer()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each feature to perform clipping
    for feature in layer:
        # Get the cutId of the vector feature
        cut_id = feature.GetField("cutId")

        # Create the output path
        output_path = os.path.join(output_dir, f"cut_{cut_id}.tif")

        # Get the bounding box of the vector
        geom = feature.GetGeometryRef()
        minX, maxX, minY, maxY = geom.GetEnvelope()

        # Set the pixel resolution for the output image, assuming each cell has an actual size of 0.000269 degrees
        pixel_size = 0.000269

        # Calculate the pixel dimensions of the output image
        x_res = int((maxX - minX) / pixel_size)
        y_res = int((maxY - minY) / pixel_size)

        # Adjust the output image dimensions to 256x256
        if x_res != 256 or y_res != 256:
            minX = maxX - (pixel_size * 256)
            maxY = minY + (pixel_size * 256)

        # Create the rasterized output path
        gdal.Warp(output_path, raster, format='GTiff', outputBounds=[minX, minY, maxX, maxY],
                  xRes=pixel_size, yRes=pixel_size, targetAlignedPixels=True, width=256, height=256,
                  cutlineDSName=input_shapefile, cutlineLayer=layer.GetName(), cutlineWhere=f"cutId = '{cut_id}'",
                  cropToCutline=True, copyMetadata=True, dstNodata=0)

    # Clean up
    raster = None
    shapefile = None

# Define input and output paths
input_raster = f"" # Input path for the national scale SAR image to be predicted
input_shapefile = "./shp/BGD_fishnet" # Fishnet shapefile for clipping 256x256 grid
output_dir = f"" # Output path for the clipped slices

# Call the function
clip_raster_by_vector(input_raster, input_shapefile, output_dir)
