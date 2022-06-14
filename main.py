from libs.CCSIM_classes import sim_surface
import os

#### FILEPATHS FOR EXAMPLE DATA ####

# FLUORIDE LEVELS IN DRINKING WATER, SAN DIEGO SAMPLE SITES
fp_water = "./raw_data/san_diego_water_quality.csv"

# GROUNDWATER DEPTH MEASUREMENTS (FEET BELOW SURFACE) IN SOUTHERN CALIFORNIA
fp_wells = "./raw_data/awl_wells_CA_latest.shp"

# ELEVATION RASTER OF AREA EAST OF SALTON SEA.
fp_elevation = "./raw_data/n33_w116_1arc_v2.tif"

path = './raw_data'


#### WORKFLOW EXAMPLE ####
# 1) GIVE FILEPATH, AND COLUMN NAMES OF COORDINATES, SURFACE VALUES TO SIM_SURFACE()
# 2) ADD DESIRED SURFACES WITH METHOD ADD_SURFACE().  REQUIRES ARG "ALL", "IDW", "KRIGE", "GAM"
# 3) TO ADD SURFACE WITH CUSTOM PARAMETERS, FEED A SECOND ARGUMENT TO METHOD ADD_SURFACE().  SEE BELOW FOR DETAILS.
# 4) ONCE ALL DESIRED SURFACES ARE DEFINED, VALIDATE RESULTS WITH CALL TO METHOD VALIDATIOR()
# 5) AFTER VALIDATOR(), CALL VISUALIZER(<FILE OUTPUT NAME>).  RESULTS WILL BE IN OUTPUT FOLDER.

# NOTE: THERE'S NO CONVEINENT WAY TO ALTER PARAMETERS AFTER A SURFACE IS INITIALIZED.
# SUGGEST TO REINITIALIZE FROM SIM_SURFACE() TO CREATE A NEW SET OF SURFACES.


def CCSIM_basic_example():

    print("ADV. EXAMPLE 1: EXPLORE SINGLE UNDERLYING MODEL")
    obj_drinking_water = sim_surface(file_path=fp_water,
                                     surface_col_name="analyte_value",
                                     x_coords_name="lng",
                                     y_coords_name="lat")

    obj_drinking_water.add_surface("krige", [None, "linear"])
    obj_drinking_water.add_surface("krige", [None, "power"])
    obj_drinking_water.add_surface("krige", [None, "exponential"])
    obj_drinking_water.add_surface("krige", [[0.06,0.10], "linear"])
    obj_drinking_water.add_surface("krige", [[0.2,2.5,0], "power"])
    obj_drinking_water.add_surface("krige", [[0.02,0.8,0.05], "exponential"])
    obj_drinking_water.add_surface("krige", [None, "gaussian"])
    obj_drinking_water.add_surface("krige", [None, "spherical"])
    obj_drinking_water.add_surface("krige", [None, "hole-effect"])

    obj_drinking_water.validator()
    obj_drinking_water.visualizer("SD_drinking_water_surfaces")
    print("Done! Check output folder!")


    print("ADV. EXAMPLE 2: COMPARE DIFFERENT MODEL TYPES")
    obj_drinking_water = sim_surface(file_path=fp_water,
                                     surface_col_name="analyte_value",
                                     x_coords_name="lng",
                                     y_coords_name="lat")

    obj_drinking_water.add_surface("all")
    obj_drinking_water.add_surface("idw", [5, 1, 15])
    obj_drinking_water.add_surface("krige", [None, "exponential"])
    obj_drinking_water.add_surface("gam", [[5, 5], 2, None])
    obj_drinking_water.add_surface("idw", [7,0.5,20])
    obj_drinking_water.add_surface("krige", [None, "power"])
    obj_drinking_water.add_surface("gam", [[20, 20], 4, None])

    obj_drinking_water.validator()
    obj_drinking_water.visualizer()
    print("Done! Check output folder!")



def CCSIM_advanced_example():

    print("ADV. EXAMPLE 1: EXPLORE SINGLE UNDERLYING MODEL")
    obj_drinking_water = sim_surface(file_path=fp_water,
                                     surface_col_name="analyte_value",
                                     x_coords_name="lng",
                                     y_coords_name="lat")

    obj_drinking_water.add_surface("krige", [None, "linear"])
    obj_drinking_water.add_surface("krige", [None, "power"])
    obj_drinking_water.add_surface("krige", [None, "exponential"])
    obj_drinking_water.add_surface("krige", [[0.06,0.10], "linear"])
    obj_drinking_water.add_surface("krige", [[0.2,2.5,0], "power"])
    obj_drinking_water.add_surface("krige", [[0.02,0.8,0.05], "exponential"])
    obj_drinking_water.add_surface("krige", [None, "gaussian"])
    obj_drinking_water.add_surface("krige", [None, "spherical"])
    obj_drinking_water.add_surface("krige", [None, "hole-effect"])

    obj_drinking_water.validator()
    obj_drinking_water.visualizer("SD_drinking_water_surfaces")
    print("Done! Check output folder!")


    print("ADV. EXAMPLE 2: COMPARE DIFFERENT MODEL TYPES")
    obj_groundwater = sim_surface(file_path = fp_wells,
                                  surface_col_name = "DATA_VAL",
                                  x_coords_name = "DECLON",
                                  y_coords_name = "DECLAT")

    obj_groundwater.add_surface("all")
    obj_groundwater.add_surface("idw", [5, 1, 15])
    obj_groundwater.add_surface("krige", [None, "exponential"])
    obj_groundwater.add_surface("gam", [[5, 5], 2, None])
    obj_groundwater.add_surface("idw", [7,0.5,20])
    obj_groundwater.add_surface("krige", [None, "power"])
    obj_groundwater.add_surface("gam", [[20, 20], 4, None])

    obj_groundwater.validator()
    obj_groundwater.visualizer()
    print("Done! Check output folder!")


    print("ADV. EXAMPLE 3: ILLUSTRATE TRUTH/RASTER BEHAVIOR")
    obj_elevation = sim_surface(file_path = fp_elevation,
                                  surface_col_name = 1,
                                  x_coords_name = "x",
                                  y_coords_name = "y",
                                  sample_size = 100)

    obj_elevation.add_surface("idw")
    obj_elevation.add_surface("idw", [7,0.75,15])
    obj_elevation.add_surface("krige",[None, "power"])
    obj_elevation.add_surface("krige",[None, "gaussian"])
    obj_elevation.add_surface("krige",[None, "exponential"])
    obj_elevation.add_surface("gam",[[4, 4], 2, None])
    obj_elevation.add_surface("gam",[[9, 9], None, None])
    obj_elevation.add_surface("gam",[[15, 15], None, None])
    obj_elevation.validator()
    obj_elevation.visualizer("Elevation_surfaces")
    print("Done! Check output folder!")


def CCSIM_new_data_check():
    #### CHECK FOR NEW FILES ####
    files = os.listdir(path)

    default_files = ["awl_wells_CA_latest.dbf",
                     "awl_wells_CA_latest.prj",
                     "awl_wells_CA_latest.shp",
                     "awl_wells_CA_latest.shx",
                     "n33_w116_1arc_v2.tif",
                     "san_diego_water_quality.csv"]

    path1 = './raw_data'
    path2 = './raw_data/'

    files = os.listdir(path1)

    found = 0

    #### ANALYZE NEW FILES ####
    for f in files:
        if not f in default_files:
            found += 1
            file_path = path2 + f
            x_name = input("file detected: {}, What's Name of X (longitude) Coordinates Column?".format(f))
            y_name = input("file detected: {}, What's Name of Y (latitude) Coordinates Column?".format(f))
            surface_name = input("file detected: {}, What's Name of the Surface Value Column?".format(f))

            sim_surface(file_path=file_path,
                        surface_col_name=surface_name,
                        x_coords_name=x_name,
                        y_coords_name=y_name)
            sim_surface.add_surface("all")
            sim_surface.validator()
            sim_surface.visualizer()

    if found == 0:
        print("No external data sources found in data folder!")

CCSIM_new_data_check()

if __name__ == "__main__":
    print(sim_surface.__doc__)
    print(sim_surface.add_surface.__doc__)
    print(sim_surface.validator.__doc__)
    print(sim_surface.visualizer.__doc__)
    CCSIM_basic_example()
    # CCSIM_advanced_example
    # CCSIM_new_data_check()