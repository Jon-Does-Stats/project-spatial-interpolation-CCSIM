from contextlib import contextmanager
from rasterio import MemoryFile
from rasterio import Affine
from rasterio.enums import Resampling
from contextlib import contextmanager

####### FUNCTIONS ##########################################
@contextmanager
def rescale_raster(raster, scale=0.5, write_output=True):
    """resizes a raster image while maintaining its aspect ratio.

            If sim_surface is given a data source that is a raster,
            validator and visualizer plot the surface and automatically
            record values of interest.  This can be used if source raster
            is truly too large to evaluate.

            Parameters
            ----------
            raster: a raster from rasterio.open

            scale: float
                The desired percentage reduction of final image. default = 0.5,
                meaning the returned raster will be half the size.

            write_output: boolean
                if true, a copy of the rescaled raster will be saved to the
                ./output/ folder.  otherwise, the function will just pass
                the open raster.

            Returns
            ----------
            an open raster ready to have data read and saved.
    """


    geotransform = raster.transform

    if not geotransform is None:
        pixel_width = geotransform[0]
        rast_rotation_b = geotransform[1]
        origin_x = geotransform[2]
        rast_rotation_d = geotransform[3]
        pixel_height = geotransform[4]
        origin_y = geotransform[5]

    # rescale the metadata
    transform = Affine(pixel_width / scale, rast_rotation_b, origin_x, rast_rotation_d, pixel_height / scale, origin_y)

    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    this_driver = raster.profile.get("driver")
    profile.update(transform=transform, driver=this_driver, height=height, width=width)

    data = raster.read(
        out_shape=(int(raster.count), height, width),
        resampling=Resampling.cubic)

    if write_output:
        file_name = os.path.basename(fp_andreas)
        fp_out_root = "./output/"
        if not os.path.isdir(fp_out_root):
            os.mkdir(fp_out_root)
        fp_out = fp_out_root + "scaled_" + file_name
        with rasterio.open(fp_out, 'w', **profile) as output:
            output.write(data)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return