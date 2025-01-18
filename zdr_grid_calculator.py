from re import A
from typing import List, Tuple

import gzip
import importlib.resources
import json
import cv2
import datetime

import pandas as pd
import numpy as np
import os

from shapely.geometry import Point, Polygon, MultiPolygon
from collections import defaultdict
from typing import List, Tuple
import statistics
import logging


# Create a logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(
    f"/users/maregger/PhD/slurm_logging/zdr_v2_calculation.log"
)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


import radlib
from radar_work_repository.data_io.get_archived_data import (
    build_zip_file_paths,
    unzip_radar_files,
)

from radar_work_repository.data_io.get_archived_data import (
    prepare_complete_radar_object,
    prepare_gridded_radar_data_from_zip,
    prepare_polar_moments_from_zip,
)

from radar_work_repository.util.miscellaneous_utility import (
    apply_gauss_filter,
    filter_by_reflectivity,
    apply_gauss_filter_with_spatial_mask,
    contiguous_pixels_in_grid_filter,
)

with importlib.resources.open_text("radar_work_repository.config", "config.json") as f:
    config = json.load(f)

with importlib.resources.open_text(
    "radar_work_repository.config", "shared_dictionaries.json"
) as f:
    config_dicts = json.load(f)


def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), "same") / n


# kernprof -l -v zdr_grid_calculator.py


def prep_rad_snyder(
    timestamp: str,
    radar: str,
    lut: pd.DataFrame,
    field: str,
    rhohv_threshold: float,
    rhohv_filter: bool = True,
    gatefilter: str = "moving_average",
    rad_range: str = "140_km",
    bias_correction: bool = True,
    bias_value: float = 0,
) -> List:
    """This function grids the polar radar data onto a cartesian 3D grid.

    Args:
        timestamp (str): timestamp in format "20210628154000"
        radar (str): short for selected radar e.g. "A"
        lut (pd.DataFrame): lookuptable for the selected radar
        field (str): the radar field which will be prepared
        rhohv_threshold (float): _description_
        rhohv_filter (bool, optional): _description_. Defaults to True.
        gatefilter (str, optional): filter function applied along the gates. Defaults to "moving_average".
        rad_range (str, optional): _description_. Defaults to "140_km".

    Returns:
        List: a dataframe containing all the cartesian pixels with
    """
    df_list = []
    missing_layers = 0
    # get the right file paths for the specific timestamp. Supports 2.5 min intervals
    # First Scan is layers 1,3,...,19
    # Second Scan is layers 2,4,...,20
    # To get intermediate scans we take the next timestamp for 1,3,...,19 and the previous timestamp for 2,4,6,...,20!
    if timestamp[-2] == "3":
        ts_a = datetime.datetime.strptime(
            timestamp, "%Y%m%d%H%M%S"
        ) - datetime.timedelta(minutes=2, seconds=30)
        ts_b = datetime.datetime.strptime(
            timestamp, "%Y%m%d%H%M%S"
        ) + datetime.timedelta(minutes=2, seconds=30)

        ts_a = datetime.datetime.strftime(ts_a, "%Y%m%d%H%M%S")
        ts_b = datetime.datetime.strftime(ts_b, "%Y%m%d%H%M%S")
    else:
        ts_a = ts_b = timestamp

    sweeps = range(1, 21)
    for radar_elevation in sweeps:  # Loop on all  sweeps

        if radar_elevation % 2 == 0:
            timestamp = ts_a
        else:
            timestamp = ts_b
        (
            zipfile_path_2,
            unzipped_file_path,
            filename_pattern_2,
        ) = build_zip_file_paths(
            timestamp, "ML", radar_elevation=radar_elevation, radar=radar
        )
        try:
            file_path_out2 = unzip_radar_files(
                zipfile_path_2, unzipped_file_path, filename_pattern_2
            )
        except (FileNotFoundError, UnicodeDecodeError) as e:
            logger.error(
                f"{timestamp}, {filename_pattern_2} for radar {radar} not found at {timestamp}. Error {e}"
            )
            missing_layers += 1

            if (radar_elevation < 3) | (missing_layers > 2):
                logger.error(
                    f"{timestamp}, More than 2 layers missing for radar {radar} at {timestamp}."
                )
                raise FileNotFoundError(e)

            if radar_elevation == 1:
                logger.error(
                    f"{timestamp}, Bottom Layer missing for radar {radar} at {timestamp}."
                )
                raise FileNotFoundError(e)
            continue
        try:
            ret_data = radlib.read_file(
                file=file_path_out2, physic_value=True, moment="ZDR"
            )

            if ret_data is not None:
                zdr = ret_data.data
            else:
                raise FileNotFoundError(
                    f"{timestamp}, zdr does not exist for timestamp"
                )

        except (AttributeError, UnicodeDecodeError) as e:
            logger.error(
                f"{timestamp}, No complete radar volume for {radar} and timestamp {timestamp} ({e})."
            )
            missing_layers += 1

            if (radar_elevation < 3) | (missing_layers > 2):
                logger.error(
                    f"{timestamp}, More than 2 layers missing for radar {radar} at {timestamp}."
                )
                raise FileNotFoundError(e)

            if radar_elevation == 1:
                logger.error(
                    f"{timestamp}, Bottom Layer missing for radar {radar} at {timestamp}."
                )
                raise FileNotFoundError(e)
            continue

        # offset_mean bias correction
        if bias_correction:
            zdr = zdr - bias_value

        # rhohv filtering
        try:
            if rhohv_filter:
                # rhohv = rad.fields["uncorrected_cross_correlation_ratio"]["data"]

                ret_data = radlib.read_file(
                    file=file_path_out2, physic_value=True, moment="RHO"
                )

                if ret_data is not None:
                    rhohv = ret_data.data
                else:
                    raise FileNotFoundError(
                        f"{timestamp}, zdr does not exist for timestamp"
                    )

                zdr = np.where(rhohv >= rhohv_threshold, zdr, np.nan)

            os.remove(file_path_out2)

        except AttributeError as e:
            logger.error(
                f"{timestamp}, No complete radar volume for rhohv in {radar} layer {i} at timestamp {timestamp} ({e})."
            )
            missing_layers += 1

        l_s = lut[lut["sweep"] == radar_elevation]

        # Range Limitation to 140km for quality and resolution
        if rad_range == "140_km":
            l_s = l_s.loc[l_s.rng <= 280]

        # moving average filtering after the prefilterings?
        try:
            if gatefilter == "moving_average_5":
                zdr_filtered = np.apply_along_axis(moving_average, 0, zdr, 5)
                l_s["data"] = zdr_filtered[l_s["az"], l_s["rng"]]

            elif gatefilter == "moving_average_3":
                zdr_filtered = np.apply_along_axis(moving_average, 0, zdr, 3)
                l_s["data"] = zdr_filtered[l_s["az"], l_s["rng"]]
            else:
                l_s["data"] = zdr[l_s["az"], l_s["rng"]]
        except IndexError as e:
            logger.error(
                f"{timestamp}, IndexError for ZDR_Filtering at {radar} at {timestamp}. {e}"
            )
            raise FileNotFoundError(e)
        df_list.append(l_s)

    return df_list


def offset_mean_bias_value(
    radar: str, timestamp: str, number_of_observations_threshold: int = 10000
) -> Tuple[float, bool]:
    """Returns the q50 zdr-snow bias value for a given radar and timestamp


    Args:
        radar (str): Radar name (e.g. "A")
        timestamp (str): timestamp in the format YYYYMMDDHHMMSS
        number_of_observations_threshold (int, optional): Minimum number of observations for valid zdr-snow value. Defaults to 5000.

    Returns:
        Tuple[float,bool]: Bias value and boolean if bias value is valid (invalid if there is no bias value for the given radar and timestamp)
    """
    date_plot = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S").date()

    # date_bias_value = date_plot + datetime.timedelta(days=1) Probably not necessary
    date_bias_value = date_plot

    bias_pickle = pd.read_pickle(f"/users/maregger/PhD/Output/bias_df.pkl")
    bias_df = bias_pickle[
        (bias_pickle["radar"] == radar)
        & (bias_pickle["Time"].dt.date == date_bias_value)
    ]

    if bias_df.empty:
        bias = 0
        valid_bias = False
    else:
        bias_df = bias_df[
            bias_df["mon_zdr_snow_L_NP"] > number_of_observations_threshold
        ]
        if bias_df.empty:
            bias = 0
        elif radar in ["W", "D", "P", "L"]:
            # we're using mon_zdr_snow_L_q50 which is the "raw" bias value
            # Correction is only happening if the bias is more than 0.2
            bias_snow = statistics.mean(list(bias_df["mon_zdr_snow_L_q50"]))

            if (bias_snow > 0.4) or (bias_snow < 0):
                bias = bias_snow - 0.2
            else:
                bias = 0

        else:
            # Albis has a default offset of 0.2
            bias_snow = statistics.mean(list(bias_df["mon_zdr_snow_L_q50"]))
            if (bias_snow < 0.2) or (bias_snow > 0.2):
                bias = bias_snow - 0.2
            else:
                bias = 0

        valid_bias = True
        # print(f"bias{bias}, valid: {valid_bias}")

    return bias, valid_bias


def calculate_3d_grid_snyder(
    rad_list: List[str],
    timestamp: str,
    field: str = "differential_reflectivity",
    rhohv_threshold: float = 0.8,
    rhohv_filter: bool = True,
    visibility_threshold: int = 50,
    gatefilter: str = "moving_average",
    lut: str = "new",
    rad_range: str = "140_km",
    bias_correction: bool = False,
    number_of_observations_threshold: int = 10000,
    calc_contributor_grid: bool = False,
):

    rad_df_list = []
    radar_name_list = []

    missing_radars = 0

    for rad in rad_list:
        # get LUT
        if lut == "old":
            lut_data = pd.read_pickle(
                f"{config['zdr_grid_calculator_paths']['LUT_DIRECTORY']}/lut_{config_dicts['radar_short_full_name_dict'][rad]}.p"
            )
        else:
            if visibility_threshold != 50:
                lut_data = pd.read_pickle(
                    f"{config['zdr_grid_calculator_paths']['LUT_DIRECTORY']}/lut_{config_dicts['radar_short_full_name_dict'][rad]}_new.p"
                )

                # applying visibility threshold
                lut_data.drop(
                    lut_data[lut_data["vis"] < visibility_threshold].index, inplace=True
                )

            else:
                lut_data = pd.read_pickle(
                    f"{config['zdr_grid_calculator_paths']['LUT_DIRECTORY']}/lut_{config_dicts['radar_short_full_name_dict'][rad]}_new_vis_50.p"
                )

        if bias_correction:
            # get bias value
            bias_value, valid_bias = offset_mean_bias_value(
                radar=rad,
                timestamp=timestamp,
                number_of_observations_threshold=number_of_observations_threshold,
            )
            if not valid_bias:
                logger.error(
                    f"{timestamp}, No valid bias value for radar {rad} and timestamp {timestamp}"
                )
                bias_value = 0
        else:
            bias_value = 0

        try:
            rad_df_list.extend(
                prep_rad_snyder(
                    timestamp=timestamp,
                    radar=rad,
                    rhohv_threshold=rhohv_threshold,
                    rhohv_filter=rhohv_filter,
                    lut=lut_data,
                    field=field,
                    gatefilter=gatefilter,
                    rad_range=rad_range,
                    bias_correction=bias_correction,
                    bias_value=bias_value,
                )
            )
        except FileNotFoundError as e:
            missing_radars += 1
            logger.error(
                f"{timestamp}, No complete radar volume for {rad} and timestamp {timestamp}"
            )

        if missing_radars > 2:
            raise FileNotFoundError("More than 2 radars are missing")

        radar_name_list.append(rad)

    # combining the results from all radars
    comb_df = pd.concat(rad_df_list)

    # fill info into a grid
    # comb_df = comb_df[["y", "x", "hgt", "data"]]

    # mask = comb_df["data"].isna()

    df_test = comb_df.loc[~comb_df["data"].isna()].groupby(["y", "x", "hgt"]).max()
    df_test.reset_index(inplace=True)

    output_zdr_grid = np.full((640, 710, 93, 2), -9999.99999999999999)

    output_zdr_grid[df_test["y"], df_test["x"], df_test["hgt"], 0] = df_test["data"]

    # df_nan = comb_df[mask]
    # output_zdr_grid[df_nan["y"], df_nan["x"], df_nan["hgt"], 1] = -9999
    output_zdr_grid = np.nanmax(output_zdr_grid, axis=3)

    output_zdr_grid[output_zdr_grid == -9999.99999999999999] = -9999

    # if calc_contributor_grid:
    #     # creating contributor_grid
    #     empty_grid = np.full((640, 710, 93, 6), -10000)
    #     # argmax a lot faster than nanargmax -> replace nan if possible

    #     rad_df_list = [pd.concat(x.tolist()) for x in np.array_split(rad_df_list,5)]

    #     for id, df in enumerate(rad_df_list):
    #         # df = comb_df_vis[comb_df_vis["rad"]==radar]
    #         # getting rid of fillna saves a lot of time -> is the -9999 missing a problem? -> we loose na information , can we get it back?

    #         df_red = df  # [["y", "x", "hgt", "data"]]
    #         mask = df_red["data"].isna()

    #         df_test = df_red[~mask].groupby(["y", "x", "hgt"]).max()
    #         df_test.reset_index(inplace=True)
    #         empty_grid[df_test["y"], df_test["x"], df_test["hgt"], id + 1] = df_test["data"]
    #         empty_grid[:, :, :, 0] = -10000
    #         # treat nan values -> needed to see "empty" beams (set them to -9999)
    #         df_nan = df_red[mask]
    #         empty_grid[df_nan["y"], df_nan["x"], df_nan["hgt"], id + 1] = -9999

    #     # argmax a lot faster than nanargmax -> replace nan -> speedup to 1/3 of the time
    #     contributor_grid = np.argmax(empty_grid, axis=3)

    #     rad_cont_id_dict = {"A": 10, "D": 20, "P": 30, "L": 40, "W": 50}

    #     for id, r in enumerate(radar_name_list):
    #         contributor_grid[contributor_grid == id + 1] = rad_cont_id_dict[r]
    #     contributor_grid = np.where(contributor_grid == 0, np.nan, contributor_grid)
    # else:
    #     contributor_grid = np.full((640, 710, 93, 6), -10000)

    # return output_zdr_grid, contributor_grid

    empty_grid = np.empty((640, 710, 93, 6))
    if calc_contributor_grid:
        rad_df_list = [pd.concat(x.tolist()) for x in np.array_split(rad_df_list, 5)]
        empty_grid[:] = np.nan
        for id, df in enumerate(rad_df_list):

            df = (
                df[["y", "x", "hgt", "data"]]
                .fillna(-9999)
                .groupby(["y", "x", "hgt"])
                .max()
            )
            df.reset_index(inplace=True)
            empty_grid[df["y"], df["x"], df["hgt"], id + 1] = df["data"]
        empty_grid[:, :, :, 0] = -10000
        contributor_grid = np.nanargmax(empty_grid, axis=3)

        rad_cont_id_dict = {"A": 10, "D": 20, "P": 30, "L": 40, "W": 50}

        for id, r in enumerate(radar_name_list):
            contributor_grid[contributor_grid == id + 1] = rad_cont_id_dict[r]
        contributor_grid = np.where(contributor_grid == 0, np.nan, contributor_grid)
    else:
        contributor_grid = empty_grid
    return output_zdr_grid, contributor_grid


def calc_zdr_columns_snyder(
    # TODO: make faster like in the HC implementation?
    grid,
    hzt_grid,
    zdr_db_threshold,
    min_value=None,
):
    zdr_grid_lut_p = grid.copy()

    # Filtering empty columns for faster calculations
    filter_frame_lut_p = np.nanmax(zdr_grid_lut_p, axis=2)

    # calculate the grid_level for HZT
    # hzt_grid_lvl_lut_p = np.flipud(np.floor((hzt_grid + 100) / 200))
    # Is the flipud needed? # I think not anymore
    hzt_grid_lvl_lut_p = np.floor((hzt_grid + 100) / 200)

    # create new grids to be filled
    column_grid_lut_p = np.zeros((640, 710))
    zdr_column_3d_lut_p = np.zeros((640, 710, 93))
    max_zdr_grid_lut_p = np.zeros((640, 710))
    max_zdr_height_grid_lut_p = np.zeros((640, 710))

    for y in range(0, 640):
        for x in range(0, 710):

            if np.isnan(filter_frame_lut_p[y, x]):
                column_grid_lut_p[y, x] = np.nan
                continue
            else:
                fz_height = int(hzt_grid_lvl_lut_p[y, x])
                rain_sum = 0
                rain_count = 0
                found = 0
                if min_value is not None:
                    fz_height = min_value
                h = fz_height
                for h in range(fz_height, 93):
                    # deal with empty gridcells -> ignore and continue
                    if np.isnan(zdr_grid_lut_p[y, x, h]):
                        zdr_column_3d_lut_p[y, x, h] = 1
                        continue
                    # Next check whether zdr is below the threshold
                    elif zdr_grid_lut_p[y, x, h] < zdr_db_threshold:
                        break
                    elif zdr_grid_lut_p[y, x, h] >= zdr_db_threshold:
                        found = 1
                        rain_count += 1
                        rain_sum = h
                        zdr_column_3d_lut_p[y, x, h] = 1

                        if zdr_grid_lut_p[y, x, h] > max_zdr_grid_lut_p[y, x]:
                            max_zdr_grid_lut_p[y, x] = zdr_grid_lut_p[y, x, h]
                            max_zdr_height_grid_lut_p[y, x] = h

                    else:
                        if found > 0:
                            break

                        # if zdr is over threshold we can increase the column height counter.
                if rain_sum != 0:
                    column_grid_lut_p[y, x] = rain_sum - fz_height + 1
                    zdr_column_3d_lut_p[y, x, h] = 1
                else:
                    column_grid_lut_p[y, x] = 0

                # if zdr_over_threshold_pixels == 0:
                #     column_grid_lut_p[y, x] = 0

                if column_grid_lut_p[y, x] == 0:
                    zdr_column_3d_lut_p[y, x, :] = 0
    return (
        column_grid_lut_p,
        zdr_column_3d_lut_p,
        max_zdr_grid_lut_p,
        max_zdr_height_grid_lut_p,
    )


def save_calculated_zdr_grids_minimal(
    fname_2d_files,
    fname_3d_files,
    file_path_storage,
    zdr_column_height,
    zdr_3d_grid,
    snyder_grid,
    zh_filtered_grid,
    contiguous_pixels_in_grid_filtered,
    gauss_filtered_masked_grid,
    fields_3d_available,
    max_zdr_value,
    max_zdr_value_elevation,
):

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_GRIDS/{fname_2d_files}", "w"
    )
    np.save(f, zdr_column_height)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_SNYDER/{fname_2d_files}", "w"
    )
    np.save(f, snyder_grid)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_ZH_FILTERED/{fname_2d_files}", "w"
    )
    np.save(f, zh_filtered_grid)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_CONTIGUITY_FILTERED/{fname_2d_files}",
        "w",
    )
    np.save(f, contiguous_pixels_in_grid_filtered)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_GAUSS_GRIDS/{fname_2d_files}", "w"
    )
    np.save(f, gauss_filtered_masked_grid)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_MAX_VALUE/{fname_2d_files}", "w"
    )
    np.save(f, max_zdr_value)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_MAX_VALUE_ELEVATION/{fname_2d_files}", "w"
    )
    np.save(f, max_zdr_value_elevation)
    f.close()

    if not fields_3d_available:
        print("storing 3d grids")
        f = gzip.GzipFile(
            f"{file_path_storage}/3D_GRIDS/3D_ZDR_GRIDS/{fname_3d_files}", "w"
        )
        np.save(f, zdr_3d_grid)
    f.close()


def get_previously_calculated_zdr_grids_minimal_2D(fname_2d_files, file_path_storage):

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_GRIDS/{fname_2d_files}", "r"
    )
    zdr_column_height = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_SNYDER/{fname_2d_files}", "r"
    )
    snyder_grid = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_ZH_FILTERED/{fname_2d_files}", "r"
    )
    zh_filtered_grid = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_CONTIGUITY_FILTERED/{fname_2d_files}",
        "r",
    )
    contiguous_pixels_in_grid_filtered = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_COLUMN_GAUSS_GRIDS/{fname_2d_files}", "r"
    )
    gauss_filtered_masked_grid = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_MAX_VALUE/{fname_2d_files}", "r"
    )
    max_zdr_value = np.load(f)
    f.close()

    f = gzip.GzipFile(
        f"{file_path_storage}/2D_GRIDS/ZDR_MAX_VALUE_ELEVATION/{fname_2d_files}", "r"
    )
    max_zdr_value_elevation = np.load(f)
    f.close()

    return (
        zdr_column_height,
        snyder_grid,
        zh_filtered_grid,
        contiguous_pixels_in_grid_filtered,
        gauss_filtered_masked_grid,
        max_zdr_value,
        max_zdr_value_elevation,
    )


def get_previously_calculated_zdr_grids_minimal_3D(fname_3d_files, file_path_storage):

    f = gzip.GzipFile(
        f"{file_path_storage}/3D_GRIDS/3D_ZDR_GRIDS/{fname_3d_files}", "r"
    )
    grid_3d = np.load(f)
    f.close
    return grid_3d


def get_appropriate_hzt_grid(timestamp, HZT_timestep):
    if HZT_timestep == "current":
        hzt_grid = prepare_gridded_radar_data_from_zip(
            product="HZT", timestamp=timestamp
        )
    else:
        timestamp_hzt_lut_p = timestamp[:8] + HZT_timestep + "00" + timestamp[12:]
        hzt_grid = prepare_gridded_radar_data_from_zip(
            product="HZT", timestamp=timestamp_hzt_lut_p
        )

    return hzt_grid


def create_zdr_column_grid_snyder(
    timestamp: str,
    config_2d: str = "v2.3",
    config_3d: str = "v2.1",
    recalculate_zdr_max=False,
):

    # load config
    with importlib.resources.open_text(
        "radar_work_repository.config", "zdr_algorithm_config.json"
    ) as f:
        config_dicts = json.load(f)

    config_2d_dict = config_dicts[f"2d_{config_2d}"]
    config_3d_dict = config_dicts[f"3d_{config_3d}"]

    # construct file name for storing all the fields
    fname_2d_files = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}.npy.gz"
    fname_3d_files = f"{timestamp}_config_3d_{config_3d}.npy.gz"

    file_path_storage = config["zdr_grid_calculator_paths"]["SNYDER_RESULT_PATH"]

    # try to load previously calculated results
    fields_3d_available = False
    fields_2d_available = False

    # initialize the fields
    zdr_3d_grid = np.zeros((1, 1, 1))
    contributor_grid = np.zeros((1, 1, 1))
    zdr_column_3d = np.zeros((1, 1, 1))
    zdr_column_height_levels = np.zeros((1, 1))
    zdr_column_height = np.zeros((1, 1))
    zdr_3d_grid = np.zeros((1, 1))
    snyder_grid = np.zeros((1, 1))
    zh_filtered_grid = np.zeros((1, 1))
    contiguous_pixels_in_grid_filtered = np.zeros((1, 1))
    gauss_filtered_masked_grid = np.zeros((1, 1))
    max_zdr_value = np.zeros((1, 1))
    max_zdr_value_elevation = np.zeros((1, 1))

    # get the appropriate HZT grid for the ZDR calculation
    hzt_grid = get_appropriate_hzt_grid(timestamp, config_3d_dict["HZT_timestep"])

    try:
        if config_3d_dict["recalculate"]:
            print("forced recalculation")
            raise FileNotFoundError
        else:

            if config_3d_dict["calculation_mode"] == "minimal":
                try:
                    zdr_3d_grid = get_previously_calculated_zdr_grids_minimal_3D(
                        fname_3d_files, file_path_storage
                    )
                    fields_3d_available = True
                except ValueError:
                    logger.error(
                        f"{timestamp}, 3D ZDR Grids not found for {timestamp} pickle error most likely"
                    )
                    raise FileNotFoundError
                if fields_3d_available:
                    print("3D fields available")

                if config_2d_dict["recalculate_2D"]:
                    print("forced recalculation of 2D fields")
                    fields_2d_available = False
                else:
                    try:
                        (
                            zdr_column_height,
                            snyder_grid,
                            zh_filtered_grid,
                            contiguous_pixels_in_grid_filtered,
                            gauss_filtered_masked_grid,
                            max_zdr_value,
                            max_zdr_value_elevation,
                        ) = get_previously_calculated_zdr_grids_minimal_2D(
                            fname_2d_files, file_path_storage
                        )
                        fields_2d_available = True
                        print("All fields available")
                        if recalculate_zdr_max:
                            print("recalculating max zdr values")
                            fields_2d_available = False
                            raise FileNotFoundError

                    except ValueError:
                        logger.error(
                            f"{timestamp}, 2D ZDR Grids not found for {timestamp} pickle error most likely"
                        )
                        raise FileNotFoundError

            elif config_3d_dict["calculation_mode"] == "complete":
                raise NotImplementedError(
                    "storing, loading of all fields not yet implemented"
                )

    except FileNotFoundError:
        if not fields_3d_available:
            print("(re)calculating 3D grids")
            zdr_3d_grid, contributor_grid = calculate_3d_grid_snyder(
                rad_list=config_3d_dict["rad_list"],
                timestamp=timestamp,
                field=config_3d_dict["field"],
                rhohv_filter=config_3d_dict["rhohv_filter"],
                rhohv_threshold=config_3d_dict["rhohv_threshold"],
                lut=config_3d_dict["lut"],
                visibility_threshold=config_3d_dict["visibility_threshold"],
                gatefilter=config_3d_dict["gatefilter"],
                bias_correction=config_3d_dict["bias_correction"],
                number_of_observations_threshold=config_3d_dict[
                    "number_of_observations_threshold"
                ],
                calc_contributor_grid=config_3d_dict["calc_contributor_grid"],
            )

    if not fields_2d_available:
        print("(re)calculating 2D grids")
        (
            zdr_column_height_levels,
            zdr_column_3d,
            max_zdr_value,
            max_zdr_value_elevation,
        ) = calc_zdr_columns_snyder(
            zdr_3d_grid,
            hzt_grid,
            config_2d_dict["zdr_db_threshold"],
        )
        zdr_column_height = zdr_column_height_levels * 200
        max_zdr_value_elevation = max_zdr_value_elevation * 200

    # producing 2d fields
    if not fields_2d_available:
        snyder_grid = apply_gauss_filter(zdr_column_height)

        zh_filtered_grid = filter_by_reflectivity(
            timestamp, config_2d_dict["reflectivity_threshold"], zdr_column_height
        )

        # v3
        contiguous_pixels_in_grid_filtered = contiguous_pixels_in_grid_filter(
            grid=zh_filtered_grid,
            minimum_threshold=config_2d_dict["min_zdr_column_height"],
            size_threshold=config_2d_dict["min_zdr_column_pixel_area"],
            connectivity=config_2d_dict["pixel_connectivity"],
        )
        gauss_filtered_masked_grid = apply_gauss_filter_with_spatial_mask(
            contiguous_pixels_in_grid_filtered
        )

    if config_3d_dict["save"]:
        if config_3d_dict["calculation_mode"] == "minimal":
            save_calculated_zdr_grids_minimal(
                fname_2d_files=fname_2d_files,
                fname_3d_files=fname_3d_files,
                file_path_storage=file_path_storage,
                zdr_column_height=zdr_column_height,
                zdr_3d_grid=zdr_3d_grid,
                snyder_grid=snyder_grid,
                zh_filtered_grid=zh_filtered_grid,
                contiguous_pixels_in_grid_filtered=contiguous_pixels_in_grid_filtered,
                gauss_filtered_masked_grid=gauss_filtered_masked_grid,
                fields_3d_available=fields_3d_available,
                max_zdr_value=max_zdr_value,
                max_zdr_value_elevation=max_zdr_value_elevation,
            )
        elif config_3d_dict["calculation_mode"] == "complete":
            raise NotImplementedError(
                "storing, loading of all fields not yet implemented"
            )
            save_calculated_zdr_grids_complete(
                fname,
                fname_3d_files,
                file_path_storage,
                zdr_3d_grid,
                zdr_column_3d,
                zdr_column_height,
                contributor_grid,
                fields_3d_available=fields_3d_available,
            )

    print(np.unique(contiguous_pixels_in_grid_filtered, return_counts=True))

    return (
        zdr_3d_grid,
        zdr_column_height,
        snyder_grid,
        zh_filtered_grid,
        contiguous_pixels_in_grid_filtered,
        gauss_filtered_masked_grid,
        zdr_column_3d,
        hzt_grid,
        contributor_grid,
        max_zdr_value,
        max_zdr_value_elevation,
    )


def zdr_3d_field_loader(timestamp: str, config_3d: str = "v2.1"):

    # load config
    with importlib.resources.open_text(
        "radar_work_repository.config", "zdr_algorithm_config.json"
    ) as f:
        config_dicts = json.load(f)

    config_3d_dict = config_dicts[f"3d_{config_3d}"]

    # construct file name for storing all the fields
    fname_3d_files = f"{timestamp}_config_3d_{config_3d}.npy.gz"

    file_path_storage = config["zdr_grid_calculator_paths"]["SNYDER_RESULT_PATH"]

    # try to load previously calculated results
    fields_3d_available = False

    # initialize the fields
    zdr_3d_grid = np.zeros((1, 1, 1))

    try:
        if config_3d_dict["recalculate"]:
            print("forced recalculation")
            raise FileNotFoundError
        else:

            if config_3d_dict["calculation_mode"] == "minimal":
                try:
                    zdr_3d_grid = get_previously_calculated_zdr_grids_minimal_3D(
                        fname_3d_files, file_path_storage
                    )
                    fields_3d_available = True
                except ValueError:
                    logger.error(
                        f"{timestamp}, 3D ZDR Grids not found for {timestamp} pickle error most likely"
                    )
                    raise FileNotFoundError
                # if fields_3d_available:
                # print("3D fields available")
    except FileNotFoundError:
        if not fields_3d_available:
            print("(re)calculating 3D grids")
            zdr_3d_grid, contributor_grid = calculate_3d_grid_snyder(
                rad_list=config_3d_dict["rad_list"],
                timestamp=timestamp,
                field=config_3d_dict["field"],
                rhohv_filter=config_3d_dict["rhohv_filter"],
                rhohv_threshold=config_3d_dict["rhohv_threshold"],
                lut=config_3d_dict["lut"],
                visibility_threshold=config_3d_dict["visibility_threshold"],
                gatefilter=config_3d_dict["gatefilter"],
                bias_correction=config_3d_dict["bias_correction"],
                number_of_observations_threshold=config_3d_dict[
                    "number_of_observations_threshold"
                ],
                calc_contributor_grid=config_3d_dict["calc_contributor_grid"],
            )
    return zdr_3d_grid


# def create_zdr_column_grid_snyder_filter_test_version(
#     timestamp: str, config_2d: str = "v1", config_3d: str = "v1",
# ):

#     # load config
#     with importlib.resources.open_text(
#         "radar_work_repository.config", "zdr_algorithm_config.json"
#     ) as f:
#         config_dicts = json.load(f)

#     config_2d_dict = config_dicts[f"2d_{config_2d}"]
#     config_3d_dict = config_dicts[f"3d_{config_3d}"]

#     # construct file name for storing all the fields
#     fname_2d_files = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}.npy.gz"
#     fname_3d_files = f"{timestamp}_config_3d_{config_3d}.npy.gz"

#     file_path_storage = config["zdr_grid_calculator_paths"]["SNYDER_RESULT_PATH"]

#     # try to load previously calculated results
#     fields_3d_available = False

#     # initialize the fields
#     zdr_3d_grid = np.zeros((1, 1, 1))
#     contributor_grid = np.zeros((1, 1, 1))
#     zdr_column_3d = np.zeros((1, 1, 1))
#     zdr_column_height_levels = np.zeros((1, 1))

#     # get the appropriate HZT grid for the ZDR calculation
#     hzt_grid = get_appropriate_hzt_grid(timestamp, config_3d_dict["HZT_timestep"])

#     try:
#         if config_3d_dict["recalculate"]:
#             print("forced recalculation")
#             raise FileNotFoundError
#         else:

#             if config_3d_dict["calculation_mode"] == "minimal":
#                 (
#                     zdr_column_height,
#                     zdr_3d_grid,
#                 ) = get_previously_calculated_zdr_grids_minimal(
#                     fname_2d_files, fname_3d_files, file_path_storage
#                 )

#             elif config_3d_dict["calculation_mode"] == "complete":
#                 raise NotImplementedError(
#                     "storing, loading of all fields not yet implemented"
#                 )
#                 (
#                     zdr_3d_grid,
#                     contributor_grid,
#                 ) = get_previously_calculated_zdr_grids_complete_3D_fields(
#                     fname, fname_3d_files, file_path_storage
#                 )
#                 # raises the FileNotFoundError if the 3d grid is not available, else tries to load rest of fields
#                 fields_3d_available = True
#                 if fields_3d_available:
#                     print("3D fields available")
#                 (
#                     zdr_column_height,
#                     zdr_column_3d,
#                 ) = get_previously_calculated_zdr_grids_complete_2D_fields(
#                     fname, file_path_storage
#                 )
#                 print("All fields available")

#     except FileNotFoundError:
#         print("(re)calculating grids")

#         if not fields_3d_available:
#             zdr_3d_grid, contributor_grid = calculate_3d_grid_snyder(
#                 rad_list=config_3d_dict["rad_list"],
#                 timestamp=timestamp,
#                 field=config_3d_dict["field"],
#                 rhohv_filter=config_3d_dict["rhohv_filter"],
#                 rhohv_threshold=config_3d_dict["rhohv_threshold"],
#                 lut=config_3d_dict["lut"],
#                 visibility_threshold=config_3d_dict["visibility_threshold"],
#                 gatefilter=config_3d_dict["gatefilter"],
#                 bias_correction=config_3d_dict["bias_correction"],
#                 number_of_observations_threshold=config_3d_dict[
#                     "number_of_observations_threshold"
#                 ],
#                 calc_contributor_grid=config_3d_dict["calc_contributor_grid"],
#             )

#         zdr_column_height_levels, zdr_column_3d = calc_zdr_columns_snyder(
#             zdr_3d_grid, hzt_grid, config_2d_dict["zdr_db_threshold"],
#         )
#         zdr_column_height = zdr_column_height_levels * 200

#     # TODO: gauss_grid & Filtering
#     # creating gauss_grid

#     # v1
#     gauss_grid_v1 = apply_gauss_filter(zdr_column_height)
#     zh_filtered_grid_v1 = filter_by_reflectivity(
#         timestamp, config_2d_dict["reflectivity_threshold"], gauss_grid_v1
#     )
#     output_v1 = contiguous_pixels_in_grid_filter(
#         grid=zh_filtered_grid_v1,
#         minimum_threshold=config_2d_dict["min_zdr_column_height"],
#         size_threshold=config_2d_dict["min_zdr_column_pixel_area"],
#         connectivity=config_2d_dict["pixel_connectivity"],
#     )

#     # v2

#     zh_filtered_grid_v2 = filter_by_reflectivity(
#         timestamp, config_2d_dict["reflectivity_threshold"], zdr_column_height
#     )

#     gauss_grid_v2 = apply_gauss_filter(zh_filtered_grid_v2)
#     output_v2 = contiguous_pixels_in_grid_filter(
#         grid=gauss_grid_v2,
#         minimum_threshold=config_2d_dict["min_zdr_column_height"],
#         size_threshold=config_2d_dict["min_zdr_column_pixel_area"],
#         connectivity=config_2d_dict["pixel_connectivity"],
#     )

#     # v3
#     contiguous_pixels_in_grid_filtered_v3 = contiguous_pixels_in_grid_filter(
#         grid=zh_filtered_grid_v2,
#         minimum_threshold=config_2d_dict["min_zdr_column_height"],
#         size_threshold=config_2d_dict["min_zdr_column_pixel_area"],
#         connectivity=config_2d_dict["pixel_connectivity"],
#     )
#     output_v3 = apply_gauss_filter(contiguous_pixels_in_grid_filtered_v3)

#     if config_3d_dict["save"]:
#         if config_3d_dict["calculation_mode"] == "minimal":
#             save_calculated_zdr_grids_minimal(
#                 fname_2d_files=fname_2d_files,
#                 fname_3d_files=fname_3d_files,
#                 file_path_storage=file_path_storage,
#                 zdr_column_height=zdr_column_height,
#                 zdr_3d_grid=zdr_3d_grid,
#             )
#         elif config_3d_dict["calculation_mode"] == "complete":
#             raise NotImplementedError(
#                 "storing, loading of all fields not yet implemented"
#             )
#             save_calculated_zdr_grids_complete(
#                 fname,
#                 fname_3d_files,
#                 file_path_storage,
#                 zdr_3d_grid,
#                 zdr_column_3d,
#                 zdr_column_height,
#                 contributor_grid,
#                 fields_3d_available=fields_3d_available,
#             )

#     return (
#         zdr_3d_grid,
#         zdr_column_height,
#         zdr_column_3d,
#         hzt_grid,
#         gauss_grid_v1,
#         contributor_grid,
#         output_v1,
#         output_v2,
#         output_v3,
#         zh_filtered_grid_v2,
#         contiguous_pixels_in_grid_filtered_v3,
#     )


def mask_to_polygons(mask, min_area=0.5):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # plt.imshow(mask)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours:  # and cv2.contourArea(cnt) >= 0.1:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[
                    c[:, 0, :]
                    for c in cnt_children.get(idx, [])
                    # if cv2.contourArea(c) >= 0.5
                ],
            )
            all_polygons.append(poly)

    # for idx, contour in enumerate(contours):
    #     if idx not in child_contours:
    #         epsilon = 0.01# * cv2.arcLength(contour, True)
    #         approx = cv2.approxPolyDP(contour, epsilon, True)
    #         if len(approx) < 3:
    #             continue
    #         polygon = Polygon(approx.reshape(-1, 2))
    #         all_polygons.append(polygon)

    all_polygons = MultiPolygon(all_polygons)

    return all_polygons


def find_zdr_column_polygons(data: np.ndarray, min_area: float = 1):
    masked = data * np.load("/users/maregger/PhD/masks/swiss_borders_50km_raster.npy")

    masked = masked.repeat(2, axis=0).repeat(2, axis=1)

    zdr_polys = mask_to_polygons(np.flipud(masked.astype("uint8")))
    # coordinates of the polygons must be brought to swiss coordinates
    p_plot = [
        Polygon(
            # zip(
            #     [1000 * (x + 256) for x in list(p.exterior.coords.xy[0])],
            #     [1000 * (y - 159) for y in list(p.exterior.coords.xy[1])],
            # )
            zip(
                [500 * (x + 511.5) for x in list(p.exterior.coords.xy[0])],
                [500 * (y - 318.5) for y in list(p.exterior.coords.xy[1])],
            )
        )
        for p in zdr_polys.geoms
    ]

    return p_plot
