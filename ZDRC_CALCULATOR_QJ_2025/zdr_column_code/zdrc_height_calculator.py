
import numpy as np

# PROPRIETARY LIBRARY to load MeteoSwiss radarfiles directly
# import radlib
import scipy
from skimage.measure import label, regionprops

# load paths and dictionaries
from ZDRC_CALCULATOR_QJ_2025.util.util import load_paths_and_configs

# from shapely.geometry import Point, Polygon, MultiPolygon


paths, shared_dicts, zdrc_algorithm_config = load_paths_and_configs()


def calc_zdr_columns_snyder(
    zdr_composite: np.ndarray,
    hzt_grid: np.ndarray,
    zdr_db_threshold: float,
):
    """Calculates the ZDRC height using the algorithm following Snyder et al. (2015)
    Additonaly, the maximum ZDR within the ZDRC is saved for each 2D pixel.

    Args:
        grid (np.ndarray): _description_
        hzt_grid (np.ndarray): _description_
        zdr_db_threshold (float): _description_

    Returns:
        _type_: _description_
    """

    # Filtering out empty columns for faster calculations
    filter_frame_lut_p = np.nanmax(zdr_composite, axis=2)

    # calculate the grid elevation for HZT from meters
    # the composite grid has a vertical resolution of 200 m. level 0 starts at -100m
    hzt_grid_lvl_lut_p = np.floor((hzt_grid + 100) / 200)

    # create new grids to be filled
    zdr_column_height_grid = np.zeros((640, 710))
    max_zdr_value_in_column = np.zeros((640, 710))

    for y in range(0, 640):
        for x in range(0, 710):
            # filter empty columns
            if np.isnan(filter_frame_lut_p[y, x]):
                zdr_column_height_grid[y, x] = np.nan
                continue

            fz_height = int(hzt_grid_lvl_lut_p[y, x])
            column_height = 0
            h = fz_height
            for h in range(fz_height, 93):

                # Next check whether zdr is below the threshold
                if zdr_composite[y, x, h] < zdr_db_threshold:
                    break
                column_height = h  # if zdr is above threshold, set the column height to the current height

                # check if the zdr value is the highest in the column
                if zdr_composite[y, x, h] > max_zdr_value_in_column[y, x]:
                    max_zdr_value_in_column[y, x] = zdr_composite[y, x, h]

            # calculate the actual zdr column height
            if column_height != 0:
                zdr_column_height_grid[y, x] = column_height - fz_height + 1
            elif np.isnan(filter_frame_lut_p[y, x]):
                zdr_column_height_grid[y, x] = np.nan

    # convert from bin to meters
    zdr_column_height = (zdr_column_height_grid * 200) - 100

    return (
        zdr_column_height,
        max_zdr_value_in_column,
    )


def calculate_zdr_column_height(
    timestamp: str,
    zdr_composite: np.ndarray,
    zdr_config: str = "config_zdrc_publication_version",
):
    """Calculates ZDRC from a given zdr composite and applies chosen filters

    Args:
        timestamp (str): timestamp in format "211791432" (YYDOYHHMM) supports 2.5 minute interval data use xx:x2 for xx:x2:30 and xx:x7 for xx:x7:30 timesteps
        zdr_composite (np.ndarray): 3d ZDR grid
        zdr_config (str, optional): Configuration Dictionary which should be used. Defaults to "config_zdrc_publication_version".


    Returns:
        np.ndarray: 2D zdr column height field
        np.ndarray: 2D max zdr within zdrc field
    """

    algorithm_config = zdrc_algorithm_config[zdr_config]

    # get the appropriate HZT grid for the ZDR calculation
    # TODO Deal with different HZT endings/versions
    hzt_filename = f"HZT{timestamp[0:7]}000L.800"
    hzt_path = f"{paths['HZT_DIRECTORY']}/{hzt_filename}"
    try:
        # hzt = radlib.read_file(
        #     file=hzt_path,
        #     physic_value=True,
        # ).data

        # np.save(f"{paths['H0_DIRECTORY']}/{hzt_filename}_h0.npy", hzt)

        h0 = np.load(f"{paths['H0_DIRECTORY']}/{hzt_filename}_h0.npy")

    except FileNotFoundError as e:
        print(f"{timestamp}, HZT does not exist for timestamp")
        raise FileNotFoundError(e)

    (
        output_zdr_column_height,
        max_zdr_value_in_column,
    ) = calc_zdr_columns_snyder(
        zdr_composite,
        h0,
        algorithm_config["zdr_db_threshold"],
    )

    # filtering steps:

    if algorithm_config["apply_gaussian_filter"]:
        # applies gaussian smoothing on the raw zdr column height field
        # used in the original Snyder 2015 implementation but not in the ZDRC publication
        output_zdr_column_height = scipy.ndimage.filters.gaussian_filter(
            output_zdr_column_height,
            sigma=algorithm_config["gaussian_filter_sigma"],
            truncate=algorithm_config["gaussian_filter_truncate"],
        )

    if algorithm_config["apply_reflectivity_filter"]:
        maxecho_filename = f"CZC{timestamp}VL.801"
        czc_path = f"{paths['CZC_DIRECTORY']}/{maxecho_filename}"

        try:
            # ret_data = radlib.read_file(
            #     file=czc_path,
            #     physic_value=True,
            # ).data

            # np.save(
            #     f"{paths['MAXECHO_DIRECTORY']}/{hzt_filename}_maxecho.npy", ret_data
            # )

            maxecho = np.load(
                f"{paths['MAXECHO_DIRECTORY']}/{hzt_filename}_maxecho.npy"
            )

        except FileNotFoundError as e:
            print(
                f"{timestamp}, CZC does not exist for timestamp, not filtering by czc"
            )  # results in a more noisy field
            # raise FileNotFoundError(e)

        output_zdr_column_height = np.where(
            maxecho >= algorithm_config["reflectivity_threshold"],
            output_zdr_column_height,
            0,
        )

    if algorithm_config["min_zdr_column_height_filter"]:

        output_zdr_column_height = np.where(
            np.isnan(output_zdr_column_height)
            | (
                output_zdr_column_height
                >= algorithm_config["min_zdr_column_height_filter_threshold"]
            ),
            output_zdr_column_height,
            0,
        )

    if algorithm_config["apply_contiguity_filter"]:

        image = np.where(output_zdr_column_height > 0, 1, 0)
        labeled_image = label(label_image=image, connectivity=2)
        valid_pixels = np.zeros_like(output_zdr_column_height)

        for region in regionprops(labeled_image):
            if (
                region.area
                >= algorithm_config["contiguous_column_size_filter_pixel_threshold"]
            ):
                valid_pixels = np.where(labeled_image == region.label, 1, valid_pixels)

        output_zdr_column_height = valid_pixels * output_zdr_column_height

    return output_zdr_column_height, max_zdr_value_in_column
