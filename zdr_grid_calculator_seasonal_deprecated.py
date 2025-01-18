from typing import List, Tuple
import scipy
import gzip
import importlib.resources
import json
import cv2

import pandas as pd
import numpy as np

import radlib
from radar_work_repository.data_io.get_archived_data import (
    build_zip_file_paths,
    unzip_radar_files,
)
from radar_work_repository.zdr_column_code.zdr_grid_calculator import (
    offset_mean_bias_value,
    mask_to_polygons,
    calc_zdr_columns_snyder,
    get_appropriate_hzt_grid,
)


from radar_work_repository.data_io.get_archived_data import (
    prepare_polar_moments_from_zip,
)

with importlib.resources.open_text("radar_work_repository.config", "config.json") as f:
    config = json.load(f)

with importlib.resources.open_text(
    "radar_work_repository.config", "shared_dictionaries.json"
) as f:
    config_dicts = json.load(f)


def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), "same") / n


# TODO: Refactor zdr-column code such that 3d file is saved separately and loaded separately since it is zdr threshold independent


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
    for i in range(1, 21):  # Loop on all  sweeps
        (
            zipfile_path_2,
            unzipped_file_path,
            filename_pattern_2,
        ) = build_zip_file_paths(timestamp, "ML", radar_elevation=i, radar=radar)
        file_path_out2 = unzip_radar_files(
            zipfile_path_2, unzipped_file_path, filename_pattern_2
        )
        ret_data = radlib.read_file(
            file=file_path_out2, physic_value=True, moment="ZDR"
        )
        
        if ret_data is not None:
            zdr = ret_data.data
        else: 
            raise FileNotFoundError(f"{timestamp}, zdr does not exist for timestamp")

        if bias_correction:
            zdr = zdr - bias_value

        if rhohv_filter:
            # rhohv = rad.fields["uncorrected_cross_correlation_ratio"]["data"]
            ret_data = radlib.read_file(
                file=file_path_out2, physic_value=True, moment="RHO"
            )
            
            if ret_data is not None:
                rhohv = ret_data.data
            else: 
                raise FileNotFoundError(f"{timestamp}, rhohv does not exist for timestamp")
            
            zdr = np.where(rhohv >= rhohv_threshold, zdr, np.nan)

        l_s = lut[lut["sweep"] == i]

        if rad_range == "140_km":
            l_s.drop(l_s[l_s["rng"] > 280].index, inplace=True)

        if gatefilter == "moving_average":
            zdr_filtered = np.apply_along_axis(moving_average, 0, zdr)
            l_s["data"] = zdr_filtered[l_s["az"], l_s["rng"]]
        else:
            l_s["data"] = zdr[l_s["az"], l_s["rng"]]

        df_list.append(l_s)

    return df_list


def calculate_3d_grid_snyder(
    rad_list,
    timestamp,
    field="differential_reflectivity",
    rhohv_threshold=0.8,
    rhohv_filter=True,
    visibility_threshold=50,
    gatefilter="moving_average",
    lut="old",
    rad_range="140_km",
):

    rad_df_list = []
    radar_name_list = []

    for rad in rad_list:
        if lut == "old":
            lut_albis = pd.read_pickle(
                f"/users/maregger/PhD/lut/lut_{config_dicts['radar_short_full_name_dict'][rad]}.p"
            )
        else:
            lut_albis = pd.read_pickle(
                f"/users/maregger/PhD/lut/lut_{config_dicts['radar_short_full_name_dict'][rad]}_new.p"
            )
        rad_df_list.extend(
            prep_rad_snyder(
                timestamp=timestamp,
                radar=rad,
                rhohv_threshold=rhohv_threshold,
                rhohv_filter=rhohv_filter,
                lut=lut_albis,
                field=field,
                gatefilter=gatefilter,
                rad_range=rad_range,
            )
        )
        radar_name_list.append(rad)

    comb_df = pd.concat(rad_df_list)
    comb_df_vis = comb_df[comb_df["vis"] >= visibility_threshold]
    comb_df_group = (
        comb_df_vis[["y", "x", "hgt", "data"]]
        .fillna(-9999)
        .groupby(["y", "x", "hgt"])
        .max()
    )
    comb_df_group.reset_index(inplace=True)
    output_zdr_grid = np.empty((640, 710, 93))
    output_zdr_grid[:] = np.nan
    output_zdr_grid[
        comb_df_group["y"], comb_df_group["x"], comb_df_group["hgt"]
    ] = comb_df_group["data"]

    return output_zdr_grid


# @profile


def calculate_3d_grid_snyder_fast(
    rad_list: List[str],
    timestamp: str,
    field: str = "differential_reflectivity",
    rhohv_threshold: float = 0.8,
    rhohv_filter: bool = True,
    visibility_threshold: int = 50,
    gatefilter: str = "moving_average",
    lut: str = "old",
    rad_range: str = "140_km",
    bias_correction: bool = False,
):

    rad_df_list = []
    radar_name_list = []

    for rad in rad_list:
        # get LUT
        if lut == "old":
            lut_data = pd.read_pickle(
                f"/users/maregger/PhD/lut/lut_{config_dicts['radar_short_full_name_dict'][rad]}.p"
            )
        else:
            lut_data = pd.read_pickle(
                f"/users/maregger/PhD/lut/lut_{config_dicts['radar_short_full_name_dict'][rad]}_new.p"
            )

        if bias_correction:
            # get bias value
            bias_value, valid_bias = offset_mean_bias_value(
                radar=rad, timestamp=timestamp
            )
            if not valid_bias:
                raise ValueError(
                    f"No valid bias value for radar {rad} and timestamp {timestamp}"
                )
        else:
            bias_value = 0

        # applying visibility threshold
        lut_data.drop(
            lut_data[lut_data["vis"] < visibility_threshold].index, inplace=True
        )

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
        radar_name_list.append(rad)

    # combining the results from all radars
    comb_df = pd.concat(rad_df_list)
    del rad_df_list
    # fill info into a grid
    comb_df = comb_df[["y", "x", "hgt", "data"]]

    mask = comb_df["data"].isna()

    df_test = comb_df[~mask].groupby(["y", "x", "hgt"]).max()
    df_test.reset_index(inplace=True)

    output_zdr_grid = np.full((640, 710, 93, 2), np.nan)

    output_zdr_grid[df_test["y"], df_test["x"], df_test["hgt"], 0] = df_test["data"]

    df_nan = comb_df[mask]
    output_zdr_grid[df_nan["y"], df_nan["x"], df_nan["hgt"], 1] = -9999
    output_zdr_grid = np.nanmax(output_zdr_grid, axis=3)

    return output_zdr_grid


# @profile
def save_calculated_zdr_grids(
    fname, file_path_storage, gauss_grid, zdr_column_height,
):

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GAUSS_GRIDS/{fname}", "w")
    np.save(f, gauss_grid)
    f.close()

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GRIDS/{fname}", "w")
    np.save(f, zdr_column_height)
    f.close()


def get_previously_calculated_zdr_grids(fname, file_path_storage):

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GAUSS_GRIDS/{fname}", "r")
    gauss_grid = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GRIDS/{fname}", "r")
    column_grid = np.load(f)
    f.close
    return (
        gauss_grid,
        column_grid,
    )


# @profile
def create_zdr_column_grid_snyder(
    timestamp: str,
    field: str = "differential_reflectivity",
    lut: str = "new",
    visibility_threshold: int = 50,
    gatefilter: str = "moving_average",
    HZT_timestep: str = "current",
    zdr_db_threshold: float = 1,
    reflectivity_threshold: int = 0,
    rhohv_threshold: float = 0.8,
    rhohv_filter: bool = True,
    min_value=None,
    rad_list: List[str] = ["D", "L", "P", "W", "A"],
    radar_range_limit: str = "140_km",
    save: bool = True,
    recalculate: bool = False,
    bias_correction: bool = False,
):

    # get the appropriate HZT grid for the ZDR calculation
    hzt_grid = get_appropriate_hzt_grid(timestamp, HZT_timestep)

    # construct file name for storing all the fields
    fname = f"{timestamp}_snyderapproach_old_lut_{lut}_zdrthreshold_{zdr_db_threshold}_rhohvthreshold_{rhohv_threshold}_visibilitythreshold_{visibility_threshold}_reflectivitythreshold_{reflectivity_threshold}_radars_{''.join(rad_list)}_radrangelimit_{radar_range_limit}_gatefilter_{gatefilter}_biascorrection_{bias_correction}.np"

    file_path_storage = config["zdr_grid_calculator_paths"]["SNYDER_RESULT_PATH"]

    # try to load previously calculated results
    try:
        if recalculate:
            raise FileNotFoundError
        (
            gauss_grid,
            zdr_column_height,  # TODO Make sure to create these grids in all versions!
        ) = get_previously_calculated_zdr_grids(fname, file_path_storage)
    # TODO File Raising seems not functional
    except FileNotFoundError:
        print("(re)calculating grids")

        zdr_3d_grid = calculate_3d_grid_snyder_fast(
            rad_list=rad_list,
            timestamp=timestamp,
            field=field,
            rhohv_filter=rhohv_filter,
            rhohv_threshold=rhohv_threshold,
            lut=lut,
            visibility_threshold=visibility_threshold,
            gatefilter=gatefilter,
            bias_correction=bias_correction,
        )
        zdr_column_height, _ = calc_zdr_columns_snyder(
            zdr_3d_grid,
            timestamp,
            hzt_grid,
            zdr_db_threshold,
            min_value=min_value, # type: ignore
            reflectivity_threshold=reflectivity_threshold, # type: ignore
        )
        gauss_grid = scipy.ndimage.filters.gaussian_filter(
            zdr_column_height, sigma=1, truncate=(((3 - 1) / 2) - 0.5 / 1)
        )  # truncate for the window size of 3
        # TODO check if there is a faster method for saving -> it takes quite a bit of time
    if save:
        save_calculated_zdr_grids(
            fname, file_path_storage, gauss_grid, zdr_column_height,
        )

    return (
        200 * zdr_column_height,
        200 * gauss_grid,
    )


if __name__ == "__main__":
    (gauss_grid_new_lut, contributor_grid_new_lut,) = create_zdr_column_grid_snyder(
        timestamp="20210621145000",
        lut="new",
        recalculate=True,
        # rad_list=["A"],
    )

