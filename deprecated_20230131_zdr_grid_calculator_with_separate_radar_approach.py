import scipy
import gzip
import importlib.resources
import json
import cv2
import datetime

import pandas as pd
import numpy as np

from shapely.geometry import Point, Polygon, MultiPolygon
from collections import defaultdict
from typing import List, Tuple
import statistics

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

with importlib.resources.open_text("radar_work_repository.config", "config.json") as f:
    config = json.load(f)

with importlib.resources.open_text(
    "radar_work_repository.config", "shared_dictionaries.json"
) as f:
    config_dicts = json.load(f)


def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), "same") / n


def prep_rad_snyder(
    timestamp: str,
    radar: str,
    lut: pd.DataFrame,
    field: str,
    rhohv_threshold: float,
    rhohv_filter: bool = True,
    gatefilter: str = "moving_average",
    rad_range: str = "140_km",
) -> pd.DataFrame:
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
        pd.DataFrame: a dataframe containing all the cartesian pixels with 
    """
    df_list = []
    for i in range(1, 21):  # Loop on all  sweeps
        # Read file and extract field
        rad = prepare_polar_moments_from_zip(
            timestamp=timestamp, radar=radar, radar_elevation=i
        )

        zdr = rad.fields[field]["data"]

        # rhohv filtering
        if rhohv_filter:
            rhohv = rad.fields["uncorrected_cross_correlation_ratio"]["data"]
            zdr = np.where(rhohv.data >= rhohv_threshold, zdr.data, np.nan)

        l_s = lut[lut["sweep"] == i]

        if rad_range == "140_km":
            l_s = l_s[l_s["rng"] <= 280]

        if gatefilter == "moving_average":
            zdr_filtered = np.apply_along_axis(moving_average, 0, zdr)
            l_s["data"] = zdr_filtered[l_s["az"], l_s["rng"]]
        else:
            l_s["data"] = zdr[l_s["az"], l_s["rng"]]
        df_list.append(l_s)
    return pd.concat(df_list)


# kernprof -l -v zdr_grid_calculator.py


def calculate_3d_grid_snyder(
    rad_list,
    timestamp,
    snyder_approach,
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
        rad_df_list.append(
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

    if snyder_approach == "old":
        # creating contributor_grid
        empty_grid = np.empty((640, 710, 93, 6))
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

    elif snyder_approach == "sep":
        contributor_grid = np.empty((640, 710, 93))
        contributor_grid = radar_name_list[0]
    else:
        raise ValueError("No valid ZDR calculation method selected.")

    return output_zdr_grid, contributor_grid


def offset_mean_bias_value(
    radar: str, timestamp: str, number_of_observations_threshold: int = 5000
) -> Tuple[float, bool]:
    """Returns the q50 zdr-snow bias value for a given radar and timestamp

    Args:
        radar (str): Radar name (e.g. "A")
        timestamp (str): timestamp in the format YYYYMMDDHHMMSS
        number_of_observations_threshold (int, optional): Minimum number of observations for valid zdr-snow value. Defaults to 5000.

    Returns:
        Tuple[float,bool]: Bias value and boolean if bias value is valid (invalid if there is no bias value for the given radar and timestamp)
    """
    date = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S").date()
    bias_pickle = pd.read_pickle(f"/users/maregger/PhD/Output/bias_df.pkl")
    bias_df = bias_pickle[
        (bias_pickle["radar"] == radar) & (bias_pickle["Time"].dt.date == date)
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
        else:
            bias_snow = statistics.mean(list(bias_df["mon_zdr_snow_L_q50"]))
            bias = bias_snow - 0.2
        valid_bias = True
    return bias, valid_bias


def calculate_3d_grid_snyder_fast(
    rad_list: List[str],
    timestamp: str,
    snyder_approach: str,
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
            lut_albis = pd.read_pickle(
                f"/users/maregger/PhD/lut/lut_{config_dicts['radar_short_full_name_dict'][rad]}.p"
            )
        else:
            lut_albis = pd.read_pickle(
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

        rad_df_list.append(
            prep_rad_snyder_fast(
                timestamp=timestamp,
                radar=rad,
                rhohv_threshold=rhohv_threshold,
                rhohv_filter=rhohv_filter,
                lut=lut_albis,
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

    # applying visibility threshold
    comb_df_vis = comb_df[comb_df["vis"] >= visibility_threshold]

    # fill info into a grid
    df_red = comb_df_vis[["y", "x", "hgt", "data"]]

    mask = df_red["data"].isna()

    df_test = df_red[~mask].groupby(["y", "x", "hgt"]).max()
    df_test.reset_index(inplace=True)

    output_zdr_grid = np.full((640, 710, 93, 2), np.nan)

    output_zdr_grid[df_test["y"], df_test["x"], df_test["hgt"], 0] = df_test["data"]

    df_nan = df_red[mask]

    output_zdr_grid[df_nan["y"], df_nan["x"], df_nan["hgt"], 1] = -9999
    output_zdr_grid = np.nanmax(output_zdr_grid, axis=3)

    if snyder_approach == "old":
        # creating contributor_grid
        empty_grid = np.full((640, 710, 93, 6), -10000)
        # argmax a lot faster than nanargmax -> replace nan if possible

        for id, df in enumerate(rad_df_list):
            # df = comb_df_vis[comb_df_vis["rad"]==radar]
            # getting rid of fillna saves a lot of time -> is the -9999 missing a problem? -> we loose na information , can we get it back?

            df_red = df  # [["y", "x", "hgt", "data"]]
            mask = df_red["data"].isna()

            df_test = df[~mask].groupby(["y", "x", "hgt"]).max()
            df_test.reset_index(inplace=True)
            empty_grid[df_test["y"], df_test["x"], df_test["hgt"], id + 1] = df_test[
                "data"
            ]
            # treat nan values -> needed to see "empty" beams (set them to -9999)

            df_nan = df_red[mask]
            empty_grid[df_nan["y"], df_nan["x"], df_nan["hgt"], id + 1] = -9999

        # argmax a lot faster than nanargmax -> replace nan -> speedup to 1/3 of the time
        contributor_grid = np.argmax(empty_grid, axis=3)

        rad_cont_id_dict = {"A": 10, "D": 20, "P": 30, "L": 40, "W": 50}

        for id, r in enumerate(radar_name_list):
            contributor_grid[contributor_grid == id + 1] = rad_cont_id_dict[r]
        contributor_grid = np.where(contributor_grid == 0, np.nan, contributor_grid)

    elif snyder_approach == "sep":
        contributor_grid = np.empty((640, 710, 93))
        contributor_grid = radar_name_list[0]
    else:
        raise ValueError("No valid ZDR calculation method selected.")

    return output_zdr_grid, contributor_grid


def calc_zdr_columns_snyder(
    grid, timestamp, hzt_grid, zdr_db_threshold, reflectivity_threshold, min_value=None,
):
    zdr_grid_lut_p = grid
    czc_grid_lut_p = prepare_gridded_radar_data_from_zip(
        product="CZC", timestamp=timestamp
    )

    # Filtering empty columns for faster calculations
    filter_frame_lut_p = np.nanmax(zdr_grid_lut_p, axis=2)
    filter_frame_lut_p = np.where(
        czc_grid_lut_p >= reflectivity_threshold, filter_frame_lut_p, np.nan
    )

    # calculate the grid_level for HZT
    hzt_grid_lvl_lut_p = np.flipud(np.floor((hzt_grid + 100) / 200))

    # create new grids to be filled
    column_grid_lut_p = np.zeros((640, 710))
    zdr_column_3d_lut_p = np.zeros((640, 710, 93))

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
    return column_grid_lut_p, zdr_column_3d_lut_p


def save_calculated_zdr_grids(
    fname,
    file_path_storage,
    zdr_3d_grid,
    zdr_column_3d,
    gauss_grid,
    zdr_column_height,
    contributor_grid,
    zdr_nr_radars,
    zdr_col_3d_nr_radars,
):
    f = gzip.GzipFile(f"{file_path_storage}/3D_ZDR_GRIDS/{fname}", "w")
    np.save(f, zdr_3d_grid)
    f.close()

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_3D_GRIDS/{fname}", "w")
    np.save(f, zdr_column_3d)
    f.close()

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GAUSS_GRIDS/{fname}", "w")
    np.save(f, gauss_grid)
    f.close()

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GRIDS/{fname}", "w")
    np.save(f, zdr_column_height)
    f.close()

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_CONTRIBUTOR_GRIDS/{fname}", "w")
    np.save(f, contributor_grid)
    f.close

    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_RADAR_COUNT_GRIDS/{fname}", "w")
    np.save(f, zdr_nr_radars)
    f.close

    f = gzip.GzipFile(f"{file_path_storage}/3D_ZDR_COLUMN_RADAR_COUNT/{fname}", "w")
    np.save(f, zdr_col_3d_nr_radars)
    f.close


def get_previously_calculated_zdr_grids(fname, file_path_storage):
    f = gzip.GzipFile(f"{file_path_storage}/3D_ZDR_GRIDS/{fname}", "r")
    grid_3d = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_3D_GRIDS/{fname}", "r")
    zdr_column_3d = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GAUSS_GRIDS/{fname}", "r")
    gauss_grid = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_GRIDS/{fname}", "r")
    column_grid = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_CONTRIBUTOR_GRIDS/{fname}", "r")
    contributor_grid = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/ZDR_COLUMN_RADAR_COUNT_GRIDS/{fname}", "r")
    zdr_nr_radars = np.load(f)
    f.close
    f = gzip.GzipFile(f"{file_path_storage}/3D_ZDR_COLUMN_RADAR_COUNT/{fname}", "r")
    zdr_col_3d_nr_radars = np.load(f)
    return (
        grid_3d,
        zdr_column_3d,
        gauss_grid,
        column_grid,
        contributor_grid,
        zdr_nr_radars,
        zdr_col_3d_nr_radars,
    )


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
    field: str = "differential_reflectivity",
    snyder_approach: str = "old",
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
    if bias_correction:
        fname = f"{timestamp}_snyderapproach_{snyder_approach}_lut_{lut}_zdrthreshold_{zdr_db_threshold}_rhohvthreshold_{rhohv_threshold}_visibilitythreshold_{visibility_threshold}_reflectivitythreshold_{reflectivity_threshold}_radars_{''.join(rad_list)}_radrangelimit_{radar_range_limit}_gatefilter_{gatefilter}_biascorrection_{bias_correction}.np"
    else:
        fname = f"{timestamp}_snyderapproach_{snyder_approach}_lut_{lut}_zdrthreshold_{zdr_db_threshold}_rhohvthreshold_{rhohv_threshold}_visibilitythreshold_{visibility_threshold}_reflectivitythreshold_{reflectivity_threshold}_radars_{''.join(rad_list)}_radrangelimit_{radar_range_limit}_gatefilter_{gatefilter}.np"

    file_path_storage = config["zdr_grid_calculator_paths"]["SNYDER_RESULT_PATH"]

    # try to load previously calculated results
    try:
        if recalculate:
            raise FileNotFoundError
        (
            zdr_3d_grid,
            zdr_column_3d,
            gauss_grid,
            zdr_column_height,
            contributor_grid,
            zdr_nr_radars,
            zdr_col_3d_nr_radars,  # TODO Make sure to create these grids in all versions!
        ) = get_previously_calculated_zdr_grids(fname, file_path_storage)
    # TODO File Raising seems not functional
    except FileNotFoundError:
        print("(re)calculating grids")

        if snyder_approach == "old":
            zdr_3d_grid, contributor_grid = calculate_3d_grid_snyder_fast(
                rad_list=rad_list,
                timestamp=timestamp,
                field=field,
                snyder_approach=snyder_approach,
                rhohv_filter=rhohv_filter,
                rhohv_threshold=rhohv_threshold,
                lut=lut,
                visibility_threshold=visibility_threshold,
                gatefilter=gatefilter,
                bias_correction=bias_correction,
            )
            zdr_column_height, zdr_column_3d = calc_zdr_columns_snyder(
                zdr_3d_grid,
                timestamp,
                hzt_grid,
                zdr_db_threshold,
                min_value=min_value,
                reflectivity_threshold=reflectivity_threshold,
            )

            zdr_nr_radars = np.zeros((640, 710))
            zdr_col_3d_nr_radars = np.zeros((640, 710, 93))

        elif snyder_approach == "sep":
            zdr_column_height_grid_list = []
            zdr_column_3d_grid_list = []
            zdr_grid_3d_list = []
            for radar in rad_list:
                # TODO create sep contributor grids?
                grid_3d_sep_rad, contributor_grid = calculate_3d_grid_snyder_fast(
                    [radar],
                    timestamp=timestamp,
                    field=field,
                    snyder_approach=snyder_approach,
                    rhohv_filter=rhohv_filter,
                    rhohv_threshold=rhohv_threshold,
                    lut=lut,
                    visibility_threshold=visibility_threshold,
                    gatefilter=gatefilter,
                )
                column_grid_sep_rad, zdr_column_3d_sep_rad = calc_zdr_columns_snyder(
                    grid_3d_sep_rad,
                    timestamp,
                    hzt_grid,
                    zdr_db_threshold,
                    min_value=min_value,
                    reflectivity_threshold=reflectivity_threshold,
                )
                zdr_column_height_grid_list.append(column_grid_sep_rad)
                zdr_column_3d_grid_list.append(zdr_column_3d_sep_rad)
                zdr_grid_3d_list.append(grid_3d_sep_rad)

            # combined zdr columns
            # 2d max height
            zdr_column_height = np.nanmax(np.stack(zdr_column_height_grid_list), axis=0)
            # 2d nr of radars
            zdr_nr_radars = np.nansum(
                np.where(np.stack(zdr_column_height_grid_list) > 0, 1, 0), axis=0
            )

            # 3d zdr value grid

            zdr_3d_grid = np.nanmax(np.stack(zdr_grid_3d_list), axis=0)

            # 3d zdr column grid

            zdr_column_3d = np.nanmax(np.stack(zdr_column_3d_grid_list), axis=0)

            # 3d nr of radars
            zdr_col_3d_nr_radars = np.nansum(np.stack(zdr_column_3d_grid_list), axis=0)

            contributor_grid = np.empty((640, 710, 93))
        else:
            raise ValueError("No valid zdr_calculation method selected")

        gauss_grid = scipy.ndimage.filters.gaussian_filter(
            zdr_column_height, sigma=1, truncate=(((3 - 1) / 2) - 0.5 / 1)
        )  # truncate for the window size of 3
        # TODO check if there is a faster method for saving -> it takes quite a bit of time
        if save:
            save_calculated_zdr_grids(
                fname,
                file_path_storage,
                zdr_3d_grid,
                zdr_column_3d,
                gauss_grid,
                zdr_column_height,
                contributor_grid,
                zdr_nr_radars,
                zdr_col_3d_nr_radars,
            )

    return (
        zdr_3d_grid,
        200 * zdr_column_height,
        zdr_column_3d,
        hzt_grid,
        200 * gauss_grid,
        contributor_grid,
        zdr_nr_radars,
        zdr_col_3d_nr_radars,
    )


# TODO Refactor zdr poly detection


def mask_to_polygons(mask, min_area=0.5):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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
        if idx not in child_contours and cv2.contourArea(cnt) >= 0.1:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[
                    c[:, 0, :]
                    for c in cnt_children.get(idx, [])
                    if cv2.contourArea(c) >= 0.5
                ],
            )
            all_polygons.append(poly)

    all_polygons = [p for p in all_polygons if p.area >= min_area]

    # all_polygons = MultiPolygon(all_polygons)

    return all_polygons


def find_zdr_column_polygons(
    data: np.ndarray, min_height: int = 400, min_area: float = 1
):
    masked = np.where(data > min_height, 1, 0) * np.load(
        "/users/maregger/PhD/masks/swiss_borders_50km_raster.npy"
    )

    zdr_polys = mask_to_polygons(np.flipud(masked.astype("uint8")), min_area=min_area)
    # coordinates of the polygons must be brought to swiss coordinates
    p_plot = [
        Polygon(
            zip(
                [1000 * (x + 256) for x in list(p.exterior.coords.xy[0])],
                [1000 * (y - 159) for y in list(p.exterior.coords.xy[1])],
            )
        )
        for p in zdr_polys
    ]

    return p_plot


def prep_rad_snyder_fast(
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
) -> pd.DataFrame:  # List:
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
        pd.DataFrame: a dataframe containing all the cartesian pixels with 
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
        zdr = radlib.read_file(
            file=file_path_out2, physic_value=True, moment="ZDR"
        ).data

        # offset_mean bias correction
        if bias_correction:
            zdr = zdr - bias_value

        # rhohv filtering
        if rhohv_filter:
            # rhohv = rad.fields["uncorrected_cross_correlation_ratio"]["data"]
            rhohv = radlib.read_file(
                file=file_path_out2, physic_value=True, moment="RHO"
            ).data
            zdr = np.where(rhohv >= rhohv_threshold, zdr, np.nan)

        l_s = lut[lut["sweep"] == i]

        if rad_range == "140_km":
            l_s = l_s[l_s["rng"] <= 280]

        if gatefilter == "moving_average":
            zdr_filtered = np.apply_along_axis(moving_average, 0, zdr)
            l_s["data"] = zdr_filtered[l_s["az"], l_s["rng"]]
        else:
            l_s["data"] = zdr[l_s["az"], l_s["rng"]]

        df_list.append(l_s)

    return pd.concat(df_list)


if __name__ == "__main__":
    (
        zdr_3d_grid_new_lut,
        zdr_column_height_new_lut,
        zdr_column_3d,
        hzt_grid_new_lut,
        gauss_grid_new_lut,
        contributor_grid_new_lut,
        zdr_nr_radars_new_lut,
        zdr_col_3d_nr_radars_new_lut,
    ) = create_zdr_column_grid_snyder(
        timestamp="20210621145000",
        snyder_approach="old",
        lut="new",
        recalculate=True,
        # rad_list=["A"],
    )

