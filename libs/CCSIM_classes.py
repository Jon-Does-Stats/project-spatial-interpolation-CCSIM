import os
from random import uniform
from time import perf_counter as timer
from time import strftime as str_time

import geopandas
import numpy as np
import pandas
import rasterio
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pygam import LinearGAM, te
from pykrige.ok import OrdinaryKriging
from rasterio.windows import Window
from scipy.spatial import cKDTree as KDTree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, max_error
from sklearn.model_selection import LeaveOneOut


class sim_surface:
    """The main platform for data import, on which to add models, validate, and visualize.
        assumes file path is to source data for interpolation. If initialized with
        raster data, sim_surface object assumes it to be a truth set, and a random
        set of points will be selected from which to interpolate.

        Parameters
        ----------
        file_path: str
            The file location of the csv, excel, or spatial data object.
        surface_col_name: str (vector data) int (raster data)
            The name of the column with the desired interpolated values.
            If raster, the band index of desired sampled and interpolated
            surface.
        x_coords_name: str
            The name of the column with the desired interpolated values.
         x_coords_name: str
            The name of the column with the x coordinates, or longitude values.
        y_coords_name: str
            The name of the column with the y coordinates, or latitude values.
        sample_size: int
            Raster only; the number of points to sample from a raster of truth
            , after which to interpolate from.  Defaults to 100.

        #### WORKFLOW EXAMPLE ####
        1)
            GIVE FILEPATH, AND COLUMN NAMES OF COORDINATES, SURFACE VALUES
            TO SIM_SURFACE().
        2)
            ADD DESIRED SURFACES WITH METHOD ADD_SURFACE().  REQUIRES
            ARG "ALL", "IDW", "KRIGE", "GAM".
        3)
            TO ADD SURFACE WITH CUSTOM PARAMETERS, FEED A SECOND ARGUMENT
            TO METHOD ADD_SURFACE().  SEE ADD_SURFACE DOCSTRING FOR MORE
            DETAILS.
        4)
            ONCE ALL DESIRED SURFACES ARE DEFINED, VALIDATE RESULTS WITH CALL
            TO METHOD VALIDATIOR().
        5)
            AFTER VALIDATOR(), CALL VISUALIZER(<FILE OUTPUT NAME>).
            RESULTS WILL BE IN OUTPUT FOLDER.
        """

    def __init__(self, file_path, surface_col_name, x_coords_name, y_coords_name, sample_size=100):

        ext = os.path.splitext(file_path)[1][1:]
        self.filename = os.path.splitext(file_path)[1][7:]
        self.filetype = ext
        self.model_list = []
        self.surface_list = []
        self.modeltype_list = []
        self.pred_surface_list = []
        self.true_surface_list = []
        self.left_out_list = []
        self.cv_metrics_list = []
        self.results_list = []

        if ext == "csv":
            dat = pandas.read_csv(file_path)
            x_coords_str = getattr(dat, x_coords_name)
            y_coords_str = getattr(dat, y_coords_name)
            dat = geopandas.GeoDataFrame(dat, geometry=geopandas.points_from_xy(x_coords_str, y_coords_str))
            surface_vals = dat.loc[:, surface_col_name]
            known_x_coords = dat.geometry.x
            known_y_coords = dat.geometry.y
            self.surface_vals = surface_vals.to_numpy()
            self.known_x_coords = known_x_coords.to_numpy()
            self.known_y_coords = known_y_coords.to_numpy()
            self.known_x_coords = self.known_x_coords % 360

        elif ext == "shp":
            dat = geopandas.read_file(file_path)
            surface_vals = dat.loc[:, surface_col_name]
            known_x_coords = dat.geometry.x
            known_y_coords = dat.geometry.y
            self.surface_vals = surface_vals.to_numpy()
            self.known_x_coords = known_x_coords.to_numpy()
            self.known_y_coords = known_y_coords.to_numpy()
            self.known_x_coords = self.known_x_coords % 360

        elif ext == "xls" or ext == "xlsx":
            dat = pandas.read_csv(file_path)
            x_coords_str = getattr(dat, x_coords_name)
            y_coords_str = getattr(dat, y_coords_name)
            dat = geopandas.GeoDataFrame(dat, geometry=geopandas.points_from_xy(x_coords_str, y_coords_str))
            surface_vals = dat.loc[:, surface_col_name]
            known_x_coords = dat.geometry.x
            known_y_coords = dat.geometry.y
            self.surface_vals = surface_vals.to_numpy()
            self.known_x_coords = known_x_coords.to_numpy()
            self.known_y_coords = known_y_coords.to_numpy()
            self.known_x_coords = self.known_x_coords % 360

        elif ext == "img" or ext == "jpg" or ext == "tif":
            print("Assuming Raster is True Surface, Sampling {} Points...".format(sample_size))
            with rasterio.open(file_path) as src:
                index = src.index  # actual function
                read = src.read  # actual function
                x_start = src.bounds[0]
                x_stop = src.bounds[2]
                y_start = src.bounds[1]
                y_stop = src.bounds[3]
                x = []
                y = []
                z = []
                row_offset = []
                col_offset = []
                self.surface_list.append(read(surface_col_name))
                self.truth_min = self.surface_list[0].min()
                self.truth_max = self.surface_list[0].max()
                self.truth_source_dims = self.surface_list[0].shape
                self.truth_sample_n = sample_size

                for i in range(sample_size):
                    xi, yi = uniform(x_start, x_stop), uniform(y_start, y_stop)
                    x.append(xi)
                    y.append(yi)
                    row_off, col_off = index(xi, yi)
                    row_offset.append(row_off)
                    col_offset.append(col_off)
                    window = Window(col_off, row_off, 1, 1)
                    dat = read(surface_col_name, window=window)
                    z.append(dat[0, 0])
            structure = {"x": x, "y": y, "z": z, "row_offset": row_offset, "col_offset": col_offset}
            df = pandas.DataFrame(data=structure)
            dat = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))
            surface_vals = dat.z
            known_x_coords = dat.geometry.x
            known_y_coords = dat.geometry.y
            self.surface_vals = surface_vals.to_numpy()
            self.known_x_coords = known_x_coords.to_numpy()
            self.known_y_coords = known_y_coords.to_numpy()
            self.known_x_coords = self.known_x_coords % 360
            self.model_list.append("truth")
            self.modeltype_list.append("truth")

        else:
            raise ValueError("Unsupported Filetype!")

        self.x_start = min(self.known_x_coords)
        self.x_stop = max(self.known_x_coords)
        self.y_start = min(self.known_y_coords)
        self.y_stop = max(self.known_y_coords)
        self.known_coords = np.array([self.known_x_coords, self.known_y_coords])
        self.known_coords = self.known_coords.transpose()
        self.unknown_coords = np.mgrid[self.x_start:self.x_stop:100j, self.y_start:self.y_stop:100j].reshape(2, -1).T
        self.unknown_coords_idx = np.mgrid[0:100, 0:100].reshape(2, -1).T
        self.unknown_x_coords = np.linspace(self.x_start, self.x_stop, 100)
        self.unknown_y_coords = np.linspace(self.y_start, self.y_stop, 100)
        self.x_range = self.x_stop - self.x_start
        self.y_range = self.y_stop - self.y_start
        self.x_start_pad = -0.05 * self.x_start
        self.x_stop_pad = 0.05 * self.x_stop
        self.y_start_pad = -0.05 * self.y_start
        self.y_stop_pad = 0.05 * self.y_stop
        self.fp = file_path

        assert len(self.known_coords) == len(self.surface_vals), "len(known_coords) %d != len(surface_vals) %d" % (
            len(self.known_coords), len(self.surface_vals))

    def add_surface(self, method, user_params=None):
        """Define an interpolated surface to add to the sim_surface model list.

                Parameters
                ----------
                method: str
                    Expects "idw", "krige", "gam", or "all".  Each surface can
                    be initalized with no further parameter input.

                    The arg "all" will initialize one of each surface with its
                    default parameter values listed below.

                user_params: a list object
                    if method = "idw"; [neighbors, penalty_term, leaf_size]
                        neighbors: integer
                            number of nearest points to each query point
                            from which to interpolate the surface.

                        penalty_term: float
                            higher values weight nearer points more, farther points less.

                        leaf_size: integer
                            KDtree dimension control. only affects computation speed.
                            lower \ higher values balance tree-traversal performance
                            with the leaf-level comparisons.

                    if method = "krige"; [model_params, variogram_model]
                        variogram_model: str
                            underlying variogram model. Accepts "linear", "power",
                            "gaussian", "exponential", "spherical", and "hole-effect".
                            Defaults to "linear".

                        model_params: a list object
                            if variogram_model = "linear" [slope, nugget]
                                slope: float
                                    scaling factor for distance decay.
                                nugget: float
                                    y intercept of variogram.  Eepresents the small-scale
                                    variability of the data.

                            if variogram_model = "power" [scale, exponent, nugget]
                                scale: float
                                    scaling factor for distance decay.
                                exponent: float
                                    exponent for power model. should be between 0 and 2.
                                nugget: float
                                    y intercept of variogram.  Eepresents the small-scale
                                    variability of the data.

                            if variogram_model = <otherwise> [sill, range, nugget]
                                sill: float
                                    the asymptotic maximum spatial variance at
                                    longest lags (distances)
                                range: float
                                    the distance at which the spatial variance has reached
                                    ~95% of the sill variance.
                                nugget: float
                                    y intercept of variogram.  Eepresents the small-scale
                                    variability of the data.

                    if method = "gam"; [numb_splines, spline_order, lambdas]
                        numb_splines: list object
                            number of basis functions from which to construct the spline
                            surface.  must be list object in form of [#,#].
                        spline_order:
                            order of the basis. 4 is a cubic spline basis. Must be >1.
                        lambdas:
                            single value. strength of smoothing penalty. Must be a positive float.
                            larger values enforce stronger smoothing.
                """

        supported_methods = pandas.DataFrame(["idw", "krige", "gam", "all"])
        check = supported_methods[supported_methods[0] == method]
        assert len(check) > 0, "Unsupported Surface Type!"

        if method == "idw" or method == "all":
            idw_model = IDW_model(known_coords=self.known_coords,
                                  known_x_coords=self.known_x_coords,
                                  known_y_coords=self.known_y_coords,
                                  surface_vals=self.surface_vals,
                                  unknown_coords=self.unknown_coords,
                                  unknown_x_coords=self.unknown_x_coords,
                                  unknown_y_coords=self.unknown_y_coords,
                                  idw_user_parameters=user_params)
            idw_surface = idw_model(self.unknown_coords)
            self.model_list.append(idw_model)
            self.surface_list.append(idw_surface)
            self.modeltype_list.append(idw_model.model_type)
            print("Interpolated Surface Via Inverse-Distance Weighting Created!")

        if method == "krige" or method == "all":
            krige_model = Krige_model(known_coords=self.known_coords,
                                      known_x_coords=self.known_x_coords,
                                      known_y_coords=self.known_y_coords,
                                      surface_vals=self.surface_vals,
                                      unknown_coords=self.unknown_coords,
                                      unknown_x_coords=self.unknown_x_coords,
                                      unknown_y_coords=self.unknown_y_coords,
                                      kr_user_parameters=user_params)
            krige_surface, self.krige_ss = krige_model(self.unknown_x_coords, self.unknown_y_coords, pred_type="grid")
            self.model_list.append(krige_model)
            self.surface_list.append(krige_surface)
            self.modeltype_list.append(krige_model.model_type)
            print("Interpolated Surface Via Kriging Created!")

        if method == "gam" or method == "all":
            gam_model = Gam_model(known_coords=self.known_coords,
                                  known_x_coords=self.known_x_coords,
                                  known_y_coords=self.known_y_coords,
                                  surface_vals=self.surface_vals,
                                  unknown_coords=self.unknown_coords,
                                  unknown_x_coords=self.unknown_x_coords,
                                  unknown_y_coords=self.unknown_y_coords,
                                  gam_user_parameters=user_params)
            gam_surface = gam_model(self.unknown_coords)
            self.model_list.append(gam_model)
            self.surface_list.append(gam_surface)
            self.modeltype_list.append(gam_model.model_type)
            print("Interpolated Surface Via GAM Created!")

    def validator(self):
        """perform leave-one-out cross-validation on all model surfaces.

            Returns
            ----------
            sim_surface_obj.results_list: list object
                stores the mean absolute error, the relative absolute error,
                the max error and the total time to complete the cross-
                validation process for each model.

            sim_surface_obj.pred_surface_list: list object
                the predicted value for each left out point for each
                surface model.

            sim_surface_obj.true_surface_list: list object
                the true value for each left out point for each surface
                model. for error checking. should match
                sim_surface_obj.surface_vals

            sim_surface_obj.left_out_list: list object
                the coordinates for each left out point for each surface model.
                for error checking. should match sim_surface_obj.known_coords
        """

        cv = LeaveOneOut()

        self.results_list = []
        total_models = len(self.model_list)
        model_counter = 0

        for model in self.model_list:
            print("Cross-validating Model {0} of {1}!".format((model_counter + 1), (total_models)))
            type = self.modeltype_list[model_counter]

            held_attrib1 = self.known_x_coords
            held_attrib2 = self.known_y_coords
            held_attrib3 = self.known_coords
            held_attrib4 = self.surface_vals

            if not type == "truth":
                begin = timer()
                held_attrib1 = model.known_x_coords
                held_attrib2 = model.known_y_coords
                held_attrib3 = model.known_coords
                held_attrib4 = model.surface_vals
                held_attrib5 = model.unknown_coords

            pred_surface = []
            true_surface = []
            left_out = []

            for in_sample, out_sample in cv.split(held_attrib3):
                known_coords_in, known_coords_out = held_attrib3[in_sample, :], held_attrib3[out_sample, :]
                x_coords_in, x_coords_out = held_attrib1[in_sample], held_attrib1[out_sample]
                y_coords_in, y_coords_out = held_attrib2[in_sample], held_attrib2[out_sample]
                z_vals_in, z_vals_out = held_attrib4[in_sample], held_attrib4[out_sample]

                if type == "truth":
                    pred_surface.append(0)
                    true_surface.append(0)
                    left_out.append(0)

                if type == "idw":
                    user_params = model.input_arg[0]
                    temp = IDW_model(known_coords=known_coords_in,
                                     known_x_coords=x_coords_in,
                                     known_y_coords=y_coords_in,
                                     surface_vals=z_vals_in,
                                     unknown_coords=known_coords_out,
                                     unknown_x_coords=x_coords_out,
                                     unknown_y_coords=y_coords_out,
                                     idw_user_parameters=user_params)

                    z_pred = temp(temp.unknown_coords, pred_type="points")
                    try:
                        z_vals = z_pred.get_values()
                    except:
                        try:
                            z_vals = z_pred.item()
                        except:
                            try:
                                z_vals = z_pred
                            except:
                                print("Weird values coming from a idw prediction!")
                    pred_surface.append(z_vals)

                    try:
                        z_vals_out_vals = z_vals_out.get_values()
                    except:
                        try:
                            z_vals_out_vals = z_vals_out.item()
                        except:
                            try:
                                z_vals_out_vals = z_vals_out
                            except:
                                print("Weird values coming from a idw input!")
                    true_surface.append(z_vals_out_vals)

                    left_out.append(model.unknown_coords)

                if type == "gam":
                    user_params = model.input_arg[0]
                    temp = Gam_model(known_coords=known_coords_in,
                                     known_x_coords=x_coords_in,
                                     known_y_coords=y_coords_in,
                                     surface_vals=z_vals_in,
                                     unknown_coords=known_coords_out,
                                     unknown_x_coords=x_coords_out,
                                     unknown_y_coords=y_coords_out,
                                     gam_user_parameters=user_params)
                    z_pred = temp(temp.unknown_coords, pred_type="points")
                    try:
                        z_vals = z_pred.get_values()
                    except:
                        try:
                            z_vals = z_pred.item()
                        except:
                            try:
                                z_vals = z_pred
                            except:
                                print("Weird values coming from a gam prediction!")

                    pred_surface.append(z_vals)

                    try:
                        z_vals_out_vals = z_vals_out.get_values()
                    except:
                        try:
                            z_vals_out_vals = z_vals_out.item()
                        except:
                            try:
                                z_vals_out_vals = z_vals_out
                            except:
                                print("Weird values coming from a gam input!")
                    true_surface.append(z_vals_out_vals)

                    left_out.append(model.unknown_coords)

                if type == "krige":
                    user_params = model.input_arg[0]
                    temp = Krige_model(known_coords=known_coords_in,
                                       known_x_coords=x_coords_in,
                                       known_y_coords=y_coords_in,
                                       surface_vals=z_vals_in,
                                       unknown_coords=known_coords_out,
                                       unknown_x_coords=x_coords_out,
                                       unknown_y_coords=y_coords_out,
                                       kr_user_parameters=user_params)
                    z_pred, ss = temp(temp.unknown_coords[:, 0], temp.unknown_coords[:, 1], pred_type="points")
                    try:
                        z_vals = z_pred.get_values()
                    except:
                        try:
                            z_vals = z_pred.item()
                        except:
                            try:
                                z_vals = z_pred
                            except:
                                print("Weird values coming from a krige prediction!")
                    pred_surface.append(z_vals)

                    try:
                        z_vals_out_vals = z_vals_out.get_values()
                    except:
                        try:
                            z_vals_out_vals = z_vals_out.item()
                        except:
                            try:
                                z_vals_out_vals = z_vals_out
                            except:
                                print("Weird values coming from a krige input!")
                    true_surface.append(z_vals_out_vals)

                    left_out.append(model.unknown_coords)

            if not type == "truth":
                model.known_x_coords = held_attrib1
                model.known_y_coords = held_attrib2
                model.known_coords = held_attrib3
                model.surface_vals = held_attrib4
                model.unknown_coords = held_attrib5
                if type == "idw":
                    model.tree = KDTree(model.known_coords, leafsize=model.leafsize)

            ############  BUILD PERFORMANCE AND METRIC LIST ###############
            if type == "truth":
                metric1 = self.truth_max
                metric2 = self.truth_min
                statement = "Max value within raster: {0},  Min value: {1}".format(metric1, metric2)
                self.cv_metrics_list.append([metric1, metric2])
            else:
                metric1 = mean_absolute_error(true_surface, pred_surface)
                metric2 = mean_absolute_percentage_error(true_surface, pred_surface)
                metric3 = max_error(true_surface, pred_surface)

                end = timer()
                metric4 = end - begin

                statement = "MAE: {0:1.4f},  MAE%: {1:1.4f},  Max Err: {2:1.4f},  CV time: {3:1.4f}".format(metric1,
                                                                                                            metric2,
                                                                                                            metric3,
                                                                                                            metric4)
                self.cv_metrics_list.append([metric1, metric2, metric3, metric4])

            ############  SAVE RESULTS OF STACK FOR EACH MODEL ###############
            self.pred_surface_list.append(pred_surface)
            self.true_surface_list.append(true_surface)
            self.left_out_list.append(left_out)
            self.results_list.append(statement)
            model_counter += 1
    print("Cross-validation complete!")


    def visualizer(self, output_filename = None):
        """visualizes the interpolated surfaces stored in sim_surface objects along with
            their parameter values and LOOCV performance metrics.

            accepts up to 9 models simultaneously; if more than 9 models loaded into
            sim_surface object, only the last 9 loaded will be displayed.

            If data source is raster, visualizer automatically reserves the upper right
            plot for an unalterated truth image.

            Parameters
            ----------
             output_filename: str
                The desired file name of the output visualization. Expects a string
                without file extension.  Defaults to auto-generated name including
                date and time.

            Returns
            ----------
            Saves png output in ./output/ folder.
            """


        file_name = str_time("%Y%m%d-%H%M%S") + ("_Analysis")

        if not output_filename is None:
            file_name = output_filename

        file_output_path = "./output/"+file_name+".png"

        plots_needed = len(self.model_list)
        last_slot_to_fill = 3 * plots_needed - 1

        if plots_needed <= 3:
            fig = plt.figure(figsize=(20, 6), constrained_layout=True)
            outer_grid = fig.add_gridspec(1, 3, wspace=0.0, hspace=0.0)
            plot_rows = 1
            title_list = [0, 3, 6]
            plot_list = [1, 4, 7]
            perf_list = [2, 5, 8]
        elif plots_needed <= 6:
            fig = plt.figure(figsize=(20, 10), constrained_layout=True)
            outer_grid = fig.add_gridspec(2, 3, wspace=0.0, hspace=0.0)
            plot_rows = 2
            title_list = [0, 3, 6, 9, 12, 15]
            plot_list = [1, 4, 7, 10, 13, 16]
            perf_list = [2, 5, 8, 11, 14, 17]
        else:
            fig = plt.figure(figsize=(20, 14), constrained_layout=True)
            outer_grid = fig.add_gridspec(3, 3, wspace=0.0, hspace=0.0)
            plot_rows = 3
            title_list = [0, 3, 6, 9, 12, 15, 18, 21, 24]
            plot_list = [1, 4, 7, 10, 13, 16, 19, 22, 25]
            perf_list = [2, 5, 8, 11, 14, 17, 20, 23, 26]

        slot_counter = 0
        for row in range(plot_rows):
            for col in range(3):
                inner_grid = outer_grid[row, col].subgridspec(3, 1, wspace=0, hspace=0, height_ratios=[0.15, 1, 0.05])
                axs = inner_grid.subplots()
                for ax in axs:
                    if slot_counter < (last_slot_to_fill + 1):
                        fig.add_subplot(ax)
                    slot_counter += 1

        all_axes = fig.get_axes()
        cax_list = []
        param_statement_list = []
        model_counter = 0

        for model_spec in self.modeltype_list:
            if model_spec == "truth":
                val1 = self.truth_source_dims
                val2 = self.truth_sample_n
                val3 = self.truth_min
                val4 = self.truth_max
                statement = "Original size: {0} x {1}, Sampled points: {2}".format(val1[0], val1[1], val2)
            if model_spec == "idw":
                val1 = self.model_list[model_counter].neighbors
                val2 = self.model_list[model_counter].p
                val3 = self.model_list[model_counter].tree.leafsize
                statement = "Neighbors: {0}, Penalty term: {1}, Leaf size: {2}".format(val1, val2, val3)
            if model_spec == "krige":
                if self.model_list[model_counter].model_obj.variogram_model == "linear":
                    val1 = self.model_list[model_counter].model_obj.variogram_model_parameters[0]
                    val2 = self.model_list[model_counter].model_obj.variogram_model_parameters[1]
                    statement = "Variogram Model: Linear, Slope = {0:1.4f}, Nugget = {1:1.4f}".format(val1, val2)
                elif self.model_list[model_counter].model_obj.variogram_model == "power":
                    val1 = self.model_list[model_counter].model_obj.variogram_model_parameters[0]
                    val2 = self.model_list[model_counter].model_obj.variogram_model_parameters[1]
                    val3 = self.model_list[model_counter].model_obj.variogram_model_parameters[2]
                    statement = "Variogram Model: Power, Scale: {0:1.4f}, Exponent: {1:1.4f}, Nugget = {2:1.4f}".format(
                        val1, val2, val3)
                else:
                    val1 = self.model_list[model_counter].model_obj.variogram_model
                    val2 = self.model_list[model_counter].model_obj.variogram_model_parameters[0]
                    val3 = self.model_list[model_counter].model_obj.variogram_model_parameters[1]
                    val4 = self.model_list[model_counter].model_obj.variogram_model_parameters[2]
                    statement = "Variogram Model: {0}, Scale: {1:1.4f}, Exponent: {2:1.4f}, Nugget = {3:1.4f}".format(
                        val1, val2, val3, val4)
            if model_spec == "gam":
                val1 = self.model_list[model_counter].model_obj.terms.info['terms'][0]['terms'][0]['n_splines']
                val2 = self.model_list[model_counter].model_obj.terms.info['terms'][0]['terms'][0]['spline_order']
                val3 = self.model_list[model_counter].model_obj.terms.info['terms'][0]['terms'][0]['lam'][0]
                statement = "Spline Number: {0}, Spline Order: {1}, Lambdas: {2}".format(val1, val2, val3)
            param_statement_list.append(statement)
            model_counter += 1

        slot_counter = 0
        plot_counter = 0
        model_counter = 0
        results_counter = 0

        for ax in all_axes:
            if slot_counter < (last_slot_to_fill + 1):
                if slot_counter in title_list:
                    for pos in ['top', "bottom", "left", "right"]:
                        ax.spines[pos].set_visible(False)
                    if self.modeltype_list[plot_counter] == "truth":
                        ax.text(0.5, 0, "ORIGINAL RASTER (TRUTH SET)", ha="center", size=12, fontweight="bold")
                    if self.modeltype_list[plot_counter] == "idw":
                        ax.text(0.5, 0, "INVERSE-DISTANCE WEIGHTING INTERPOLATION", ha="center", size=12,
                                fontweight="bold")
                    if self.modeltype_list[plot_counter] == "krige":
                        ax.text(0.5, 0, "KRIGING INTERPOLATION", ha="center", size=12, fontweight="bold")
                    if self.modeltype_list[plot_counter] == "gam":
                        ax.text(0.5, 0, "GAM/TENSOR SPLINE INTERPOLATION", ha="center", size=12, fontweight="bold")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    model_counter += 1

                if slot_counter in perf_list:
                    for pos in ['top', "bottom", "left", "right"]:
                        ax.spines[pos].set_visible(False)
                    ax.text(0.5, 0.25, self.results_list[results_counter], ha="center",
                            size=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    results_counter += 1

                if slot_counter in plot_list:
                    for pos in ['top', "bottom", "left", "right"]:
                        ax.spines[pos].set_visible(False)

                    ax.plot(self.known_x_coords, self.known_y_coords, "wx", markersize=3)
                    ax.invert_xaxis()

                    if self.modeltype_list[plot_counter] == "truth":
                        img = ax.imshow(self.surface_list[plot_counter],
                                        extent=[self.x_start, self.x_stop, self.y_start, self.y_stop],
                                        origin="upper")
                    else:
                        img = ax.imshow(self.surface_list[plot_counter].T,
                                        extent=[self.x_start, self.x_stop, self.y_start, self.y_stop],
                                        origin="lower")
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.10)
                    cax_list.append(cax)
                    plt.colorbar(img, cax=cax_list[plot_counter])
                    ax.set_title(param_statement_list[plot_counter], fontsize=8)

                    plot_counter += 1
                slot_counter += 1
            if slot_counter > (last_slot_to_fill):
                ax.set_axis_off()


        plt.savefig(file_output_path)
        print("Done! Graphic saved to output folder!")


class IDW_model:
    """custom IDW interpolator based on kdtree nearest neighbor algorithms
    all ingested parameters passed by add_surface, so this should be
    relatively invisible to end-users.
    """


    def __init__(self, known_coords, known_x_coords, known_y_coords, surface_vals, unknown_coords, unknown_x_coords,
                 unknown_y_coords, idw_user_parameters=None):

        self.input_arg = [idw_user_parameters]
        if idw_user_parameters is None:
            self.neighbors = 3
            self.p = 1
            self.leafsize = 10
        else:
            self.neighbors, self.p, self.leafsize = idw_user_parameters
        self.model_type = "idw"
        self.known_coords = known_coords
        self.known_x_coords = known_x_coords
        self.known_y_coords = known_y_coords
        self.surface_vals = surface_vals
        self.unknown_coords = unknown_coords
        self.unknown_x_coords = unknown_x_coords
        self.unknown_y_coords = unknown_y_coords
        self.tree = KDTree(self.known_coords, leafsize=self.leafsize)  # build the tree
        self.weight_n = 0
        self.weight_sum = None

    def __call__(self, unknown_coords, pred_type=None):

        unknown_coords = np.asarray(unknown_coords)

        dim = unknown_coords.ndim

        if dim == 1:
            unknown_coords = np.array([unknown_coords])

        if self.weight_sum is None:
            self.weight_sum = np.zeros(self.neighbors)

        self.distances, self.source_i = self.tree.query(unknown_coords, k=self.neighbors)

        interpol = np.zeros((len(self.distances),) + np.shape(self.surface_vals[0]))

        jinterpol = 0

        for dist, source in zip(self.distances, self.source_i):
            if self.neighbors == 1:
                prediction = self.surface_vals[source]

            elif dist[0] < 1e-10:
                prediction = self.surface_vals[source[0]]

            else:
                weight = 1 / (dist ** self.p)
                weight /= np.sum(weight)

                prediction = np.dot(weight, self.surface_vals[source])

                self.weight_n += 1
                self.weight_sum += weight

            interpol[jinterpol] = prediction

            jinterpol += 1

        if pred_type is None:
            interpol = interpol.reshape(100, 100).T
        return interpol if dim > 1 else interpol[0]

    def data_update_cv(self, known_coords_in, known_x_coords_in, known_y_coords_in, surface_vals_in, unknown_coords):
        self.known_coords = known_coords_in
        self.known_x_coords = known_x_coords_in
        self.known_y_coords = known_y_coords_in
        self.surface_vals = surface_vals_in
        self.unknown_coords = unknown_coords
        self.tree = KDTree(self.known_coords, leafsize=self.leafsize)


class Krige_model:
    """wrapper class object for ordinary kriging procedure from PyKrige.

    all ingested parameters passed by add_surface, so this should be
    relatively invisible to end-users.
    """

    def __init__(self, known_coords, known_x_coords, known_y_coords, surface_vals, unknown_coords, unknown_x_coords,
                 unknown_y_coords, kr_user_parameters=None):
        self.input_arg = [kr_user_parameters]
        self.my_variogram_model = "linear"
        self.my_variogram_params = None
        if kr_user_parameters is not None:
            if kr_user_parameters[0] is not None:
                self.my_variogram_params = kr_user_parameters[0]
            if kr_user_parameters[1] is not None:
                self.my_variogram_model = kr_user_parameters[1]
        self.model_type = "krige"
        self.known_coords = known_coords
        self.known_x_coords = known_x_coords
        self.known_y_coords = known_y_coords
        self.surface_vals = surface_vals
        self.unknown_coords = unknown_coords
        self.unknown_coords_x = unknown_x_coords
        self.unknown_coords_y = unknown_y_coords

        if self.my_variogram_params is None:
            self.model_obj = OrdinaryKriging(self.known_x_coords, self.known_y_coords, self.surface_vals,
                                             variogram_model=self.my_variogram_model, verbose=False,
                                             enable_plotting=False,
                                             coordinates_type="geographic")
        else:
            self.model_obj = OrdinaryKriging(self.known_x_coords, self.known_y_coords, self.surface_vals,
                                             variogram_parameters=self.my_variogram_params,
                                             variogram_model=self.my_variogram_model, verbose=False,
                                             enable_plotting=False,
                                             coordinates_type="geographic")

    # "grid" treat xpoints and ypoints as two arrays of x and y coordinates that define a grid.
    # points treats xpoints and ypoints as two arrays that provide coordiante pairs at which to solve the system.
    def __call__(self, unknown_coords_x, unknown_coords_y, pred_type="grid"):
        z_values, sigma_sq = self.model_obj.execute(pred_type, unknown_coords_x, unknown_coords_y)
        return z_values, sigma_sq

    def data_update_cv(self, known_coords_in, known_x_coords_in, known_y_coords_in, surface_vals_in, unknown_coords):
        self.known_coords = known_coords_in
        self.known_x_coords = known_x_coords_in
        self.known_y_coords = known_y_coords_in
        self.surface_vals = surface_vals_in
        self.unknown_coords = unknown_coords



class Gam_model:
    """wrapper class object for linear GAM procedure from pyGAM.

    all ingested parameters passed by add_surface, so this should be
    relatively invisible to end-users.
    """

    def __init__(self, known_coords, known_x_coords, known_y_coords, surface_vals, unknown_coords, unknown_x_coords,
                 unknown_y_coords, gam_user_parameters=None):
        self.input_arg = [gam_user_parameters]
        self.model_type = "gam"
        self.known_coords = known_coords
        self.known_x_coords = known_x_coords
        self.known_y_coords = known_y_coords
        self.surface_vals = surface_vals
        self.unknown_coords = unknown_coords
        self.unknown_x_coords = unknown_x_coords
        self.unknown_y_coords = unknown_y_coords
        self.my_numb_splines = 10
        self.my_spline_order = None
        self.my_lambdas = None

        if gam_user_parameters is not None:
            self.my_numb_splines = gam_user_parameters[0]
            self.my_spline_order = gam_user_parameters[1]
            self.my_lambdas = gam_user_parameters[2]

        if self.my_spline_order is None and self.my_lambdas is None:
            self.model_obj = LinearGAM(te(0, 1, n_splines=self.my_numb_splines)).fit(self.known_coords,
                                                                                     self.surface_vals)

        elif self.my_spline_order is not None and self.my_lambdas is None:
            self.model_obj = LinearGAM(te(0, 1, n_splines=self.my_numb_splines,
                                          spline_order=self.my_spline_order)).fit(self.known_coords, self.surface_vals)

        elif self.my_spline_order is None and self.my_lambdas is not None:
            self.model_obj = LinearGAM(te(0, 1, n_splines=self.my_numb_splines,
                                          lam=self.my_lambdas)).fit(self.known_coords, self.surface_vals)
        else:
            self.model_obj = LinearGAM(te(0, 1, n_splines=self.my_numb_splines,
                                          spline_order=self.my_spline_order,
                                          lam=self.my_lambdas)).fit(self.known_coords, self.surface_vals)

    def __call__(self, unknown_coords, pred_type=None):
        gam_surface = self.model_obj.predict(unknown_coords)
        if pred_type is None:
            gam_surface = gam_surface.reshape(100, 100).T
        return gam_surface

    def data_update_cv(self, known_coords_in, known_x_coords_in, known_y_coords_in, surface_vals_in, unknown_coords):
        self.known_coords = known_coords_in
        self.known_x_coords = known_x_coords_in
        self.known_y_coords = known_y_coords_in
        self.surface_vals = surface_vals_in
        self.unknown_coords = unknown_coords
