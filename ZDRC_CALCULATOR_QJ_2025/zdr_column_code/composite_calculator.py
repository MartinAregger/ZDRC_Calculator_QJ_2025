"""This script contains all the functions needed to calculate the ZDR composite based on data
from the swiss weather radar network. The data is in a proprietary format which means that this
script is not usable as is!
"""

import datetime
import os
from typing import List

import numpy as np
import pandas as pd

# PROPRIETARY LIBRARY to load MeteoSwiss radarfiles directly
# import radlib

# Add Metranetlib Path
if os.environ.get("METRANETLIB_PATH") is None:
    os.environ["METRANETLIB_PATH"] = "/users/maregger/PhD/lib/metranet/"


# load paths and dictionaries
from ZDRC_CALCULATOR_QJ_2025.util.util import load_paths_and_configs

paths, shared_dicts, zdrc_algorithm_config = load_paths_and_configs()


def moving_average(a, n=3):
    """
    Calculates the moving average of a given array.

    Parameters:
    a (array-like): The input array.
    n (int, optional): The number of elements to include in the moving average window. Default is 3.

    Returns:
    array-like: The moving average of the input array.

    """
    return np.convolve(a, np.ones(n), "same") / n


def polar_to_cartesian_single_radar(
    timestamp: str = "211791432",
    radar: str = "A",
    rhohv_filter: bool = True,
    rhohv_threshold: float = 0.8,
    moving_average_filter: bool = True,
    moving_average_window: int = 3,
    min_rad_range: int = None,
    max_rad_range: int = 140,
    bias_correction: bool = False,
    bias_value: float = 0,
) -> List:
    """This function grids the polar radar data onto the MCH radar cartesian 3D grid with 1x1x0.2 km resolution for an individual radar.

    Args:
        timestamp (str): timestamp in format "211791432" (YYDOYHHMM) supports 2.5 minute interval data use xx:x2 for xx:x2:30 and xx:x7 for xx:x7:30 timesteps
        radar (str): short for selected radar e.g. "A"
        rhohv_filter (bool, optional): Whether to apply the RhoHV filter. Defaults to True.
        rhohv_threshold (float, optional): RhoHV filter threshold, everything smaller than this value is removed. Defaults to 0.8.
        moving_average_filter (bool, optional): Whether to apply the moving average filter along the beam. Defaults to True.
        moving_average_window (int, optional): Moving average filter window size. Defaults to 3.
        min_rad_range (int, optional): Minimum distance from radar for gates to be used in km. Defaults to None.
        max_rad_range (int, optional): Maximum distance from radar for gates to be used in km. Defaults to 140.
        bias_correction (bool, optional): Whether to apply the bias correction value. Defaults to False.
        bias_value (float, optional): Bias correction value. Defaults to 0.

    Returns:
        List: a dataframe containing all the cartesian pixels with ZDR values
    """

    df_list = []
    # later in the code we are checking for missing layers, currently a max of 2 missing layers is allowed
    # Exception: we always require the lowest layer to be present
    missing_layers = 0

    # prepare the lookup table for the conversion to cartesian coordinates
    lut = pd.read_pickle(
        f"{paths['LUT_DIRECTORY']}/lut_{shared_dicts['radar_short_full_name_dict'][radar]}_new_vis_50.p"
    )

    # get the right file paths to build a full radar volume for the specific timestamp. Supports 2.5 min intervals
    date_format = "%y%j%H%M"
    if timestamp[-1] in ["2", "7"]:
        ts_a = datetime.datetime.strptime(timestamp, date_format) - datetime.timedelta(
            minutes=2
        )
        ts_b = datetime.datetime.strptime(timestamp, date_format) + datetime.timedelta(
            minutes=3
        )

        ts_a = datetime.datetime.strftime(ts_a, date_format)
        ts_b = datetime.datetime.strftime(ts_b, date_format)
    else:
        ts_a = ts_b = timestamp

    sweeps = range(1, 21)
    for radar_elevation in sweeps:  # Loop on all  sweeps

        if radar_elevation % 2 == 0:
            timestamp = ts_a
        else:
            timestamp = ts_b

        # reading in the data for individual radar sweeps
        mlx_filename = f"ML{radar}{timestamp}0U.0{str(radar_elevation).zfill(2)}"
        mlx_path = f"{paths['MLX_DIRECTORY']}/{mlx_filename}"

        try:
            # zdr = radlib.read_file(file=mlx_path, physic_value=True, moment="ZDR").data
            # np.save(f"{paths['POLAR_DATA_DIRECTORY']}/{mlx_filename}_ZDR.npy", zdr)

            zdr = np.load(f"{paths['POLAR_DATA_DIRECTORY']}/{mlx_filename}_ZDR.npy")

        # Treatment of missing layers, safer to keep it but not tragic if it is not removed
        except (FileNotFoundError, UnicodeDecodeError) as e:
            missing_layers += 1
            if radar_elevation < 2:
                print(f"Bottom Layer missing for radar {radar} at {timestamp}.")
                raise FileNotFoundError(e)
            elif missing_layers > 2:
                print(
                    f"{timestamp}, More than 2 layers missing for radar {radar} at {timestamp}."
                )
                raise FileNotFoundError(e)
            continue

        # ZDR Bias correction
        if bias_correction:
            zdr = zdr - bias_value

        # rhohv filtering
        try:
            if rhohv_filter:
                # rhohv = radlib.read_file(
                #     file=mlx_path, physic_value=True, moment="RHO"
                # ).data
                # np.save(
                #     f"{paths['POLAR_DATA_DIRECTORY']}/{mlx_filename}_RHO.npy",
                #     rhohv,
                # )
                rhohv = np.load(
                    f"{paths['POLAR_DATA_DIRECTORY']}/{mlx_filename}_RHO.npy"
                )

                # zdr filtering with rhohv
                zdr = np.where(rhohv >= rhohv_threshold, zdr, np.nan)

        except AttributeError as e:
            print(
                f"{timestamp}, No complete radar volume for rhohv in {radar} layer {radar_elevation} at timestamp {timestamp} ({e})."
            )
            missing_layers += 1

        working_lut = lut[lut["sweep"] == radar_elevation]

        # Range Limitation to 140km for quality and resolution
        if max_rad_range is not None:
            working_lut = working_lut.loc[working_lut.rng <= 2 * max_rad_range]
        if min_rad_range is not None:
            working_lut = working_lut.loc[working_lut.rng >= 2 * min_rad_range]

        # apply moving average filter (smoothing of the ZDR beam)
        try:
            if moving_average_filter == True:
                zdr_filtered = np.apply_along_axis(
                    moving_average, 0, zdr, moving_average_window
                )
                working_lut["data"] = zdr_filtered[
                    working_lut["az"], working_lut["rng"]
                ]

                working_lut["data"] = zdr[working_lut["az"], working_lut["rng"]]
        except IndexError as e:
            print(
                f"{timestamp}, IndexError for ZDR_Filtering at {radar} at {timestamp}. {e}"
            )
            raise FileNotFoundError(e)
        df_list.append(working_lut)

    return df_list  # list of dataframes, one dataframe for each radar elevation


def polar_to_cartesian_composite(
    timestamp: str = "211791432",
    zdr_config: str = "config_zdrc_publication_version",
) -> np.ndarray:
    """Builds the complete ZDR composite from all selected radars.
    See publication for details.

    Args:
        rad_list (List): list of all radars which will be used for the composite in shortform naming.
        timestamp (str): timestamp in format "211791432" (YYDOYHHMM) supports 2.5 minute interval data use xx:x2 for xx:x2:30 and xx:x7 for xx:x7:30 timesteps
        rhohv_filter (bool, optional): Whether to apply the RhoHV filter. Defaults to True.
        rhohv_threshold (float, optional): RhoHV filter threshold, everything smaller than this value is removed. Defaults to 0.8.
        moving_average_filter (bool, optional): Whether to apply the moving average filter along the beam. Defaults to True.
        moving_average_window (int, optional): Moving average filter window size. Defaults to 3.
        min_rad_range (int, optional): Minimum distance from radar for gates to be used in km. Defaults to None.
        max_rad_range (int, optional): Maximum distance from radar for gates to be used in km. Defaults to 140.
        bias_correction (bool, optional): Whether to apply the bias correction value. Defaults to False.
        bias_value (float, optional): Bias correction value. Defaults to 0.
        maximum_number_of_allowed_missing_radars (int): Maximum number of incomplete radar volumes which are allowed


    Returns:
        np.ndarray: 3D array of the ZDR Composite
    """

    # loading configuration
    algorithm_config = zdrc_algorithm_config[zdr_config]

    rad_df_list = []
    radar_name_list = []

    missing_radars = 0

    for rad in algorithm_config["rad_list"]:
        # if bias_correction:
        # could be reimplemented for bias correction
        bias_value = 0

        try:
            rad_df_list.extend(
                polar_to_cartesian_single_radar(
                    timestamp=timestamp,
                    radar=rad,
                    rhohv_filter=algorithm_config["rhohv_filter"],
                    rhohv_threshold=algorithm_config["rhohv_threshold"],
                    moving_average_filter=algorithm_config["moving_average_filter"],
                    moving_average_window=algorithm_config["moving_average_window"],
                    min_rad_range=algorithm_config["min_radar_range_km"],
                    max_rad_range=algorithm_config["max_radar_range_km"],
                    bias_correction=algorithm_config["bias_correction"],
                    bias_value=bias_value,
                )
            )
        except FileNotFoundError as e:
            missing_radars += 1
            print(
                f"{timestamp}, No sufficiently complete radar volume for {rad} and timestamp {timestamp}"
            )
        if (
            missing_radars
            > algorithm_config["maximum_number_of_allowed_missing_radars"]
        ):
            raise FileNotFoundError(f"More than allowed number of radars are missing")

        radar_name_list.append(rad)

    # combining the results from all radars
    comb_df = pd.concat(rad_df_list)

    # Here we are actually combining the radar data into a single grid with the maximum value for each pixel
    df_test = comb_df.loc[~comb_df["data"].isna()].groupby(["y", "x", "hgt"]).max()
    df_test.reset_index(inplace=True)

    # Write the pandas dataframe to a 3D numpy array
    output_zdr_grid = np.full((640, 710, 93, 2), -9999)
    output_zdr_grid[df_test["y"], df_test["x"], df_test["hgt"], 0] = df_test["data"]
    output_zdr_grid = np.nanmax(output_zdr_grid, axis=3)

    return output_zdr_grid  # 3d zdr grid
