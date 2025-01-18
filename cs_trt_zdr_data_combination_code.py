"""This module contains code to combine the information from cs, trt, and zdr data.
"""

from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
import importlib.resources
import json
import statistics
import rasterio
import rasterio.mask
import pyproj
import datetime
import geopandas as gpd
import gc
from typing import List

from shapely.geometry import Point, Polygon
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from shapely.ops import transform
import warnings
import math
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(f"/users/maregger/PhD/slurm_logging/zdr_trt_combination.log")
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from radar_work_repository.zdr_column_code.zdr_grid_calculator import (
    find_zdr_column_polygons,
    create_zdr_column_grid_snyder,
)

from radar_work_repository.data_io.get_archived_data import (
    prepare_gridded_radar_data_from_zip,
    prepare_trt_from_zip,
)

from radar_work_repository.util.miscellaneous_utility import (
    mask_polygon_in_radar_grid
)

with importlib.resources.open_text("radar_work_repository.config", "config.json") as f:
    config = json.load(f)

# load config
with importlib.resources.open_text(
    "radar_work_repository.config", "zdr_algorithm_config.json"
) as f:
    config_dicts = json.load(f)

warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "All-NaN slice encountered")



def get_zdr_column_poly_stat_list(poly, grid, max_zdr_value, max_zdr_value_elevation, hzt_grid_lvl, name_prefix, id):
    poly_area = poly.area
    poly_circumference = poly.length
    poly_centroid_x = poly.centroid.xy[0][0]
    poly_centroid_y = poly.centroid.xy[1][0]
    x, y = poly.minimum_rotated_rectangle.exterior.coords.xy
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )
    poly_box_min_edge_length = min(edge_length)
    poly_box_max_edge_length = max(edge_length)

    grid_mask = mask_polygon_in_radar_grid(poly)

    #we use the elevation above hzt
    zdr_max_distance = max_zdr_value_elevation-hzt_grid_lvl
    
    grid_out = grid * grid_mask
    max_zdr_value_out = max_zdr_value * grid_mask
    max_zdr_value_elevation_out = zdr_max_distance * grid_mask
    hzt_grid_lvl_out = hzt_grid_lvl * grid_mask
    
    masked_grid = np.where(grid_out > 0, grid_out, np.nan)
    masked_max_zdr_value = np.where(grid_out > 0, max_zdr_value_out, np.nan)
    masked_max_zdr_value_elevation = np.where(grid_out > 0, max_zdr_value_elevation_out, np.nan)
    masked_hzt_grid_lvl = np.where(grid_out > 0, hzt_grid_lvl_out, np.nan)

    flattened_mask_poly = masked_grid.flatten()
    flattened_mask_max_zdr_value = masked_max_zdr_value.flatten()
    flattened_mask_max_zdr_value_elevation = masked_max_zdr_value_elevation.flatten() 
    flattened_masked_hzt_grid_lvl = masked_hzt_grid_lvl.flatten()
    
    
    cell_value_list = flattened_mask_poly[
        np.logical_not(np.isnan(flattened_mask_poly))
    ].tolist()
    
    max_zdr_value_list = flattened_mask_max_zdr_value[
        np.logical_not(np.isnan(flattened_mask_max_zdr_value))
    ].tolist()
    
    max_zdr_value_elevation_list = flattened_mask_max_zdr_value_elevation[
        np.logical_not(np.isnan(flattened_mask_max_zdr_value_elevation))
    ].tolist()
    
    hzt_grid_lvl_list = flattened_masked_hzt_grid_lvl[
        np.logical_not(np.isnan(flattened_masked_hzt_grid_lvl))
    ].tolist()

    poly_area_by_masked_pixels = len(cell_value_list)

    if len(cell_value_list) != 0:
        cell_mean = statistics.mean(cell_value_list)
        cell_median = statistics.median(cell_value_list)
        cell_min = min(cell_value_list)
        cell_max = max(cell_value_list)
        cell_p75 = np.percentile(cell_value_list, 75)
        cell_p90 = np.percentile(cell_value_list, 90)
        cell_p95 = np.percentile(cell_value_list, 95)
        cell_p99 = np.percentile(cell_value_list, 99)
        poly_volume = sum([x/1000 for x in cell_value_list])
        
        cell_max_zdr_max = max(max_zdr_value_list)
        cell_max_zdr_median = statistics.median(max_zdr_value_list)
        
        cell_max_zdr_elevation_max = max(max_zdr_value_elevation_list)
        cell_max_zdr_elevation_median = statistics.median(max_zdr_value_elevation_list)
        
        hzt_grid_lvl_max = max(hzt_grid_lvl_list)
        hzt_grid_lvl_median = statistics.median(hzt_grid_lvl_list)

        values = [
            id,
            poly,
            poly_area / 1000000,  # transform from m^2 to km^2
            poly_area_by_masked_pixels,
            poly_circumference,
            poly_centroid_x,
            poly_centroid_y,
            poly_box_max_edge_length,
            poly_box_min_edge_length,
            cell_value_list,
            cell_mean,
            cell_median,
            cell_min,
            cell_max,
            cell_p75,
            cell_p90,
            cell_p95,
            cell_p99,
            poly_volume,
            
            cell_max_zdr_max,
            cell_max_zdr_median,
            max_zdr_value_list,
            
            cell_max_zdr_elevation_max,
            cell_max_zdr_elevation_median,
            max_zdr_value_elevation_list,
            
            hzt_grid_lvl_max,
            hzt_grid_lvl_median,
            hzt_grid_lvl_list
        ]

        names = [
            "ZDR_id",
            "poly",
            "poly_area",
            "poly_area_by_masked_pixels",
            "poly_circumference",
            "poly_centroid_x",
            "poly_centroid_y",
            "poly_box_max_edge_length",
            "poly_box_min_edge_length",
            "cell_value_list",
            "cell_mean",
            "cell_median",
            "cell_min",
            "cell_max",
            "cell_p75",
            "cell_p90",
            "cell_p95",
            "cell_p99",
            "poly_volume",
            
            "cell_max_zdr_max",
            "cell_max_zdr_median",
            "max_zdr_value_list",
            
            "cell_max_zdr_elevation_max",
            "cell_max_zdr_elevation_median",
            "max_zdr_value_elevation_list",
            
            "hzt_grid_lvl_max",
            "hzt_grid_lvl_median",
            "hzt_grid_lvl_list"
        ]

        names = [name_prefix + n for n in names]
        return (pd.DataFrame([values], columns=names), masked_grid)
    return (pd.DataFrame(), masked_grid)





# TODO: Adapt


def check_if_hail_for_cell_in_df(c_df, id):
    s_df = c_df[c_df["id"] == id]
    tiny = small = large = huge = False
    if s_df[(s_df[">0-5 mm_5"] != 0) | (s_df[">0-15 mm_5"] != 0)].empty == False:
        tiny = True
    if s_df[(s_df[">0-15 mm_5"] != 0) | (s_df["5-15 mm_5"] != 0)].empty == False:
        small = True
    if s_df[(s_df["15-27 mm_5"] != 0) | (s_df["27-32 mm_5"] != 0)].empty == False:
        large = True
    if (
        s_df[
            (s_df["27-37 mm_5"] != 0)
            | (s_df[">32 mm_5"] != 0)
            | (s_df["37-55 mm_5"] != 0)
            | (s_df[">55 mm_5"] != 0)
        ].empty
        == False
    ):
        huge = True
    return (tiny | small | large | huge), tiny, small, large, huge




def prepare_zdr_column_polygons_with_stats(
    timestamp: str, config_2d: str, config_3d: str
):
    """This function creates a pandas dataframe with all the zdr_columns for a given timestamp.
    The dataframe is further 

    Args:
        timestamp (str): _description_
        zdr_db_threshold (float, optional): _description_. Defaults to 1.0.
        zh_threshold (int, optional): _description_. Defaults to 20.
        min_zdr_column_height (int, optional): _description_. Defaults to 600.
        min_zdr_column_area (float, optional): _description_. Defaults to 1.
    """
    (
        _,
        _,
        _,
        _,
        contiguous_pixels_in_grid_filtered,
        _,
        _,
        _,
        _,
        max_zdr_value, 
        max_zdr_value_elevation
    ) = create_zdr_column_grid_snyder(
        timestamp=timestamp, config_2d=config_2d, config_3d=config_3d,
    )

    p_mask = find_zdr_column_polygons(data=contiguous_pixels_in_grid_filtered)

    df_list = []

    hzt = prepare_gridded_radar_data_from_zip(timestamp=timestamp, product="HZT")
    hzt_grid_lvl = np.floor((hzt + 100) / 200)*200
    
    for i, poly in enumerate(p_mask):
        p_stat = get_zdr_column_poly_stat_list(
            poly=poly,
            grid=contiguous_pixels_in_grid_filtered,
            max_zdr_value=max_zdr_value,
            max_zdr_value_elevation=max_zdr_value_elevation,
            hzt_grid_lvl=hzt_grid_lvl,
            name_prefix="",
            id=f"{timestamp}_{i}",
        )[0]
        df_list.append(p_stat)

    if len(df_list) != 0:
        out_df = pd.concat(df_list)
    else:
        out_df = pd.DataFrame()

    zdr_pickle_path = config["cs_zdr_trt_combination_paths"]["ZDR_COLUMN_POLYGONS_PATH"]
    fname = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}"
    out_df.to_pickle(f"{zdr_pickle_path}/{fname}.pickle")
    return out_df


def get_population_density(x, y):
    arr = np.load("/users/maregger/PhD/data/Population_Density_CH_on_Radar_Grid.npy")

    x_ind = math.floor(x / 1000) - 255
    y_ind = math.floor(y / 1000) + 160
    density = np.flipud(arr)[y_ind, x_ind]
    return density


def create_zdr_trt_cs_merge_data(timestamp: str, config_2d: str, config_3d: str):

    config_2d_dict = config_dicts[f"2d_{config_2d}"]
    config_3d_dict = config_dicts[f"3d_{config_3d}"]

    zdr_pickle_path = config["cs_zdr_trt_combination_paths"]["ZDR_COLUMN_POLYGONS_PATH"]
    fname = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}"

    any_data_defects = False
    try:
        if config_3d_dict["recalculate"] | config_2d_dict["recalculate_2D"]:
            raise FileNotFoundError("recalculating")
        zdr_df = pd.read_pickle(f"{zdr_pickle_path}/{fname}.pickle")
    except FileNotFoundError:
        _ = prepare_zdr_column_polygons_with_stats(
            timestamp=timestamp, config_2d=config_2d, config_3d=config_3d,
        )
        zdr_df = pd.read_pickle(f"{zdr_pickle_path}/{fname}.pickle")

    try:
        trt_df = prepare_trt_from_zip(timestamp=timestamp)
    except FileNotFoundError:
        trt_df = pd.DataFrame()

    buffer_50km_filter_poly = gpd.read_file(
        f"{config['base_plotting_paths']['BACKGROUND_SHAPEFILES_PATH']}/Swiss_Borders_50km.shp"
    )["geometry"][0]
    try:
        if config_2d_dict["recalculate_cs_counts"]:
            raise FileNotFoundError
        else:
            trt_cs_info = pd.read_pickle(
                f"{config['cs_zdr_trt_combination_paths']['CS_COUNTS_PER_TRT_CELL_PATH']}cs_counts_per_trt_cell_{timestamp}.pickle"
            )
    except FileNotFoundError:
        print("(re)calculating cs_counts_list")
        calculate_cs_in_trt_frame(timestamp)
        trt_cs_info = pd.read_pickle(
            f"{config['cs_zdr_trt_combination_paths']['CS_COUNTS_PER_TRT_CELL_PATH']}cs_counts_per_trt_cell_{timestamp}.pickle"
        )
    try:
        poh = prepare_gridded_radar_data_from_zip(timestamp=timestamp, product="BZC")
    except FileNotFoundError:
        print("no BZC data found, adding empty array")
        poh = np.zeros((640, 710))
        any_data_defects = True

    try:
        meshs = prepare_gridded_radar_data_from_zip(timestamp=timestamp, product="MZC")
    except FileNotFoundError:
        print("no MZC data found, adding empty array")
        meshs = np.zeros((640, 710))
        any_data_defects = True

    if (not trt_df.empty) & (not zdr_df.empty):
        # TRT poly prep
        crs_data = pyproj.CRS("EPSG:4326")
        crs_shapefiles = pyproj.CRS("EPSG:21781")
        project = pyproj.Transformer.from_crs(
            crs_data, crs_shapefiles, always_xy=True
        ).transform

        trt_poly_list = list(trt_df["poly_zip"])
        for p in trt_poly_list:
            print(Polygon(p).centroid)

        trt_poly_list = [transform(project, Polygon(x)) for x in trt_poly_list]

        for p in trt_poly_list:
            print(p.centroid)

        trt_poly_ids = list(trt_df["traj_ID"])

        # zdr poly prep
        zdr_poly_list = list(zdr_df["poly"])
        zdr_poly_id = list(zdr_df["ZDR_id"])
        zdr_poly_area = list(zdr_df["poly_area"])
        zdr_poly_area_pixels = list(zdr_df["poly_area_by_masked_pixels"])
        zdr_poly_max = list(zdr_df["cell_max"])
        zdr_poly_p95 = list(zdr_df["cell_p95"])
        zdr_poly_mean = list(zdr_df["cell_mean"])
        zdr_poly_median = list(zdr_df["cell_median"])
        zdr_poly_centroid_xy = list(zip(zdr_df["poly_centroid_x"], zdr_df["poly_centroid_y"]))

        # zdr - trt merge
        trt_intersect_list = []

        for trt_poly, trt_id in zip(trt_poly_list, trt_poly_ids):
            for zdr_data in zip(
                zdr_poly_list,
                zdr_poly_id,
                zdr_poly_area,
                zdr_poly_area_pixels,
                zdr_poly_max,
                zdr_poly_p95,
                zdr_poly_mean,
                zdr_poly_median,
                zdr_poly_centroid_xy
            ):
                trt_intersect_list.append(
                    (
                        trt_id,
                        trt_poly.intersects(zdr_data[0]),
                        buffer_50km_filter_poly.contains(trt_poly),
                        zdr_data[1],
                        zdr_data[2],
                        zdr_data[3],
                        zdr_data[4],
                        zdr_data[5],
                        zdr_data[6],
                        zdr_data[7],
                        zdr_data[8],
                        #TODO: zdr_location_statistics
                    )
                )

        comp_df = pd.DataFrame(
            trt_intersect_list,
            columns=[
                "id",
                "zdr_intersection",
                "trt_in_50km_buffer",
                "ZDR_id",
                "zdr_poly_area",
                "zdr_poly_area_pixels",
                "zdr_poly_max",
                "zdr_poly_p95",
                "zdr_poly_mean",
                "zdr_poly_median",
                "zdr_poly_centroid_xy"
            ],
        )
        res_df = comp_df.groupby("id").sum()[["zdr_intersection", "trt_in_50km_buffer"]]
        res_df["zdr_intersection_bool"] = res_df["zdr_intersection"] > 0
        res_df["trt_in_50km_buffer"] = res_df["trt_in_50km_buffer"] > 0

        comp_df_only_intersecting = comp_df[comp_df["zdr_intersection"]]
        for col in (
            "ZDR_id",
            "zdr_poly_area",
            "zdr_poly_area_pixels",
            "zdr_poly_max",
            "zdr_poly_p95",
            "zdr_poly_mean",
            "zdr_poly_median",
            "zdr_poly_centroid_xy"
        ):
            zdr_list_df = comp_df_only_intersecting.groupby("id")[col].apply(list)
            res_df = res_df.merge(zdr_list_df, how="left", left_on="id", right_on="id")

        res_df.reset_index(inplace=True)

        # merge POH/MESHS info
        poh_meshs_trt = []
        for trt_poly, trt_id in zip(trt_poly_list, trt_poly_ids):
            grid_mask = mask_polygon_in_radar_grid(poly=trt_poly)

            meshs_masked = np.where(meshs * grid_mask > 0, meshs * grid_mask, np.nan)
            poh_masked = np.where(poh * grid_mask > 0, poh * grid_mask, np.nan)

            poh_max = np.nanmax(poh_masked)
            poh_median = np.nanmedian(poh_masked)
            poh_mean = np.nanmean(poh_masked)
            poh_bool = ~np.isnan(poh_max)
            poh_80_bool = poh_max >= 80
            meshs_max = np.nanmax(meshs_masked)
            meshs_median = np.nanmedian(meshs_masked)
            meshs_mean = np.nanmean(meshs_masked)
            meshs_bool = ~np.isnan(meshs_max)
            trt_poly_centroid_x = trt_poly.centroid.x
            trt_poly_centroid_y = trt_poly.centroid.y

            try:
                pop_density = get_population_density(
                    trt_poly_centroid_x, trt_poly_centroid_y
                )
            except IndexError:
                pop_density = -7777
                logger.error(
                    f"{timestamp}, {trt_id} TRT Polygon is defective at {timestamp}."
                )
                any_data_defects = True

            poh_meshs_trt.append(
                (
                    trt_id,
                    (trt_poly_centroid_x, trt_poly_centroid_y),
                    pop_density,
                    poh_max,
                    poh_median,
                    poh_mean,
                    poh_bool,
                    poh_80_bool,
                    meshs_max,
                    meshs_mean,
                    meshs_median,
                    meshs_bool,
                )
            )
        poh_meshs_df = pd.DataFrame(
            poh_meshs_trt,
            columns=[
                "id",
                "trt_poly_centroid_coords_xy",
                "population_density",
                "poh_max",
                "poh_median",
                "poh_mean",
                "poh_bool",
                "poh_80_bool",
                "meshs_max",
                "meshs_mean",
                "meshs_median",
                "meshs_bool",
            ],
        )
        res_df = res_df.merge(poh_meshs_df, how="left", left_on="id", right_on="id")
        # merge cs info
        # trt_cs_info["0-5mm"] = trt_cs_info[">0-5 mm_5"] + trt_cs_info[">0-15 mm_5"]  # type: ignore
        # trt_cs_info["5-15mm"] = trt_cs_info[">0-15 mm_5"] + trt_cs_info["5-15 mm_5"]  # type: ignore
        # trt_cs_info["15-32mm"] = trt_cs_info["15-27 mm_5"] + trt_cs_info["27-32 mm_5"]  # type: ignore
        # trt_cs_info[">32mm"] = trt_cs_info["27-37 mm_5"] + trt_cs_info[">32 mm_5"] + trt_cs_info["37-55 mm_5"] + trt_cs_info[">55 mm_5"]  # type: ignore

        trt_cs_info["<10"] = trt_cs_info["2.5_5"] + trt_cs_info["6.5_5"]
        trt_cs_info[">15"] = (
            trt_cs_info["23_5"]
            + trt_cs_info["32_5"]
            + trt_cs_info["43_5"]
            + trt_cs_info["50_5"]
            + trt_cs_info["68_5"]
        )
        trt_cs_info[">32"] = (
            trt_cs_info["32_5"]
            + trt_cs_info["43_5"]
            + trt_cs_info["50_5"]
            + trt_cs_info["68_5"]
        )
        trt_cs_info["Any_Hail"] = (
            trt_cs_info["2.5_5"]
            + trt_cs_info["6.5_5"]
            + trt_cs_info["23_5"]
            + trt_cs_info["32_5"]
            + trt_cs_info["43_5"]
            + trt_cs_info["50_5"]
            + trt_cs_info["68_5"]
        )

        # Minimum 1 report of hail in the following 5 minutes
        trt_cs_info["<10_bool"] = trt_cs_info["<10"] > 0
        trt_cs_info[">15_bool"] = trt_cs_info[">15"] > 0
        trt_cs_info[">32_bool"] = trt_cs_info[">32"] > 0
        trt_cs_info["Any_Hail_bool"] = trt_cs_info["Any_Hail"] > 0

        # Minimum 3 reports of hail in the following 5 minutes
        trt_cs_info["<10_bool_min_3"] = trt_cs_info["<10"] > 2
        trt_cs_info[">15_bool_min_3"] = trt_cs_info[">15"] > 2
        trt_cs_info[">32_bool_min_3"] = trt_cs_info[">32"] > 2
        trt_cs_info["Any_Hail_bool_min_3"] = trt_cs_info["Any_Hail"] > 2

        out_df = res_df.merge(trt_cs_info, how="inner", on=["id"])  # type: ignore
        out_df["timestamp"] = timestamp
    else:
        out_df = pd.DataFrame(columns=["id", "timestamp"])

    out_df["data_defects"] = any_data_defects
    return out_df, any_data_defects


def create_zdr_trt_cs_merge_delay_dfs(
    timestamp: str, config_2d: str, config_3d: str,
):
    defects = False

    base_df, data_defects = create_zdr_trt_cs_merge_data(
        timestamp=timestamp, config_2d=config_2d, config_3d=config_3d,
    )
    if data_defects:
        defects = True
    timestamp_dt = datetime.datetime.strptime(str(timestamp), "%Y%m%d%H%M%S")
    for d in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

        timestamp_d = datetime.datetime.strftime(
            timestamp_dt - datetime.timedelta(minutes=d), "%Y%m%d%H%M%S"
        )

        df, data_defects = create_zdr_trt_cs_merge_data(
            timestamp=timestamp_d, config_2d=config_2d, config_3d=config_3d,
        )
        df = df.drop("timestamp", axis=1)

        base_df = base_df.merge(
            df, how="left", on=["id"], suffixes=["", f"_d{d}"],
        )  # type: ignore
    if data_defects:
        defects = True

    zdr_pickle_path = config["cs_zdr_trt_combination_paths"]["ZDR_TRT_CS_COMB_PATH"]
    # fname = f"{timestamp}_snyder_old_lut_{lut}_zdrt_{zdr_db_threshold}_rhohvt_{rhohv_threshold}_reflectivityt_{zh_threshold}_radars_{''.join(rad_list)}_radrangelimit_140_gatefilter_{gatefilter}_biascorrection_{bias_correction}_biascorrection_number_{number_of_observations_threshold}_zhthreshold_{zh_threshold}_columnheightmin_{min_zdr_column_height}_columnareamin_{min_zdr_column_area}"
    fname = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}"
    base_df["poh_meshs_defects"] = defects
    base_df.to_pickle(f"{zdr_pickle_path}/{fname}.pickle")
    gc.collect()
    return base_df



def merge_zdr_trt_polygons(timestamp,config_2d,config_3d,recalculate=False):

    zdr_merge_pickle_path = config["cs_zdr_trt_combination_paths"]["ZDR_TRT_COMB_PATH"]
    trt_merge_pickle_path = config["cs_zdr_trt_combination_paths"]["TRT_ZDR_COMB_PATH"]
    zdr_no_trt_pickle_path= config["cs_zdr_trt_combination_paths"]["ZDR_WITHOUT_TRT_POLYS_PATH"]
    fname = f"{timestamp}_config_3d_{config_3d}_2d_{config_2d}"
    
    try:
        if recalculate:
            raise FileNotFoundError
        
        zdr_res_df = pd.read_pickle(f"{zdr_merge_pickle_path}/{fname}.pkl")
        trt_res_df = pd.read_pickle(f"{trt_merge_pickle_path}/{fname}.pkl")
        print("Combination Files found")
        
    except FileNotFoundError:
        print("Combination Files not found, creating them")
        # ZDR Polygons
        try: 
            zdr_df = prepare_zdr_column_polygons_with_stats(timestamp=timestamp,config_2d=config_2d,config_3d=config_3d)
        except FileNotFoundError:
            logger.error(f"Error in {timestamp} no ZDR data available")
            data_defects = True
            zdr_df = pd.DataFrame()

        # TRT Polygons 
        try:
            trt_df = prepare_trt_from_zip(timestamp=timestamp)
        except FileNotFoundError:
            trt_df = pd.DataFrame()

        # distance filter
        buffer_50km_filter_poly = gpd.read_file(
                f"{config['base_plotting_paths']['BACKGROUND_SHAPEFILES_PATH']}/Swiss_Borders_50km.shp"
            )["geometry"][0]

        # finished data preparation, now we merge

        if (not trt_df.empty) & (not zdr_df.empty):
            # TRT poly prep
            crs_data = pyproj.CRS("EPSG:4326")
            crs_shapefiles = pyproj.CRS("EPSG:21781")
            project = pyproj.Transformer.from_crs(
                crs_data, crs_shapefiles, always_xy=True
            ).transform

            trt_poly_list = list(trt_df["poly_zip"])

            trt_poly_list = [transform(project, Polygon(x)) for x in trt_poly_list]

            trt_poly_ids = list(trt_df["traj_ID"])

            # zdr poly prep
            zdr_poly_list = list(zdr_df["poly"])
            zdr_poly_id = list(zdr_df["ZDR_id"])
            zdr_poly_area = list(zdr_df["poly_area"])
            zdr_poly_area_pixels = list(zdr_df["poly_area_by_masked_pixels"])
            zdr_poly_max = list(zdr_df["cell_max"])
            zdr_poly_p95 = list(zdr_df["cell_p95"])
            zdr_poly_mean = list(zdr_df["cell_mean"])
            zdr_poly_median = list(zdr_df["cell_median"])
            zdr_poly_centroid_xy = list(zip(zdr_df["poly_centroid_x"], zdr_df["poly_centroid_y"]))
            zdr_poly_cell_value_list = list(zdr_df["cell_value_list"])
            zdr_poly_volume = list(zdr_df["poly_volume"])
            
            cell_max_zdr_max = list(zdr_df["cell_max_zdr_max"])
            cell_max_zdr_median = list(zdr_df["cell_max_zdr_median"])
            max_zdr_value_list = list(zdr_df["max_zdr_value_list"])
            
            cell_max_zdr_elevation_max = list(zdr_df["cell_max_zdr_elevation_max"])
            cell_max_zdr_elevation_median = list(zdr_df["cell_max_zdr_elevation_median"])
            max_zdr_value_elevation_list = list(zdr_df["max_zdr_value_elevation_list"])
            
            hzt_grid_lvl_max = list(zdr_df["hzt_grid_lvl_max"])
            hzt_grid_lvl_median = list(zdr_df["hzt_grid_lvl_median"])
            hzt_grid_lvl_list = list(zdr_df["hzt_grid_lvl_list"])
            
            
            
            
            

            # zdr - trt merge
            trt_intersect_list = []
            zdr_polys_with_intersect = []

            for trt_poly, trt_id in zip(trt_poly_list, trt_poly_ids):
                for zdr_data in zip(
                    zdr_poly_list,
                    zdr_poly_id,
                    zdr_poly_area,
                    zdr_poly_area_pixels,
                    zdr_poly_max,
                    zdr_poly_p95,
                    zdr_poly_mean,
                    zdr_poly_median,
                    zdr_poly_centroid_xy,
                    zdr_poly_cell_value_list,
                    zdr_poly_volume,
                    cell_max_zdr_max,
                    cell_max_zdr_median,
                    max_zdr_value_list,
                    cell_max_zdr_elevation_max,
                    cell_max_zdr_elevation_median,
                    max_zdr_value_elevation_list,
                    hzt_grid_lvl_max,
                    hzt_grid_lvl_median,
                    hzt_grid_lvl_list
                ):
     
                    trt_zdr_intersect = trt_poly.intersects(zdr_data[0])
                
                    trt_intersect_list.append(
                        (
                            trt_id,
                            trt_zdr_intersect,
                            buffer_50km_filter_poly.contains(trt_poly),
                            zdr_data[0],
                            zdr_data[1],
                            zdr_data[2],
                            zdr_data[3],
                            zdr_data[4],
                            zdr_data[5],
                            zdr_data[6],
                            zdr_data[7],
                            zdr_data[8],
                            zdr_data[9],
                            zdr_data[10],
                            zdr_data[11],
                            zdr_data[12],
                            zdr_data[13],
                            zdr_data[14],
                            zdr_data[15],
                            zdr_data[16],
                            zdr_data[17],
                            zdr_data[18],
                            zdr_data[19],
                        )
                    )
                    if trt_zdr_intersect:
                        zdr_polys_with_intersect.append((zdr_data[1],trt_id))
                    

            # postprocessing trt combination dataframe
            trt_comp_df = pd.DataFrame(
                trt_intersect_list,
                columns=[
                    "id",
                    "zdr_intersection",
                    "trt_in_50km_buffer",
                    "zdr_poly",
                    "ZDR_id",
                    "zdr_poly_area",
                    "zdr_poly_area_pixels",
                    "zdr_poly_max",
                    "zdr_poly_p95",
                    "zdr_poly_mean",
                    "zdr_poly_median",
                    "zdr_poly_centroid_xy",
                    "zdr_poly_cell_value_list",
                    "zdr_poly_volume",
                    "cell_max_zdr_max",
                    "cell_max_zdr_median",
                    "max_zdr_value_list",
                    
                    "cell_max_zdr_elevation_max",
                    "cell_max_zdr_elevation_median",
                    "max_zdr_value_elevation_list",
                    
                    "hzt_grid_lvl_max",
                    "hzt_grid_lvl_median",
                    "hzt_grid_lvl_list"
                ],
            )
            trt_res_df = trt_comp_df[["id","zdr_intersection", "trt_in_50km_buffer"]].groupby("id").sum()[["zdr_intersection", "trt_in_50km_buffer"]]
            trt_res_df["zdr_intersection_bool"] = trt_res_df["zdr_intersection"] > 0
            trt_res_df["trt_in_50km_buffer"] = trt_res_df["trt_in_50km_buffer"] > 0

            comp_df_only_intersecting = trt_comp_df[trt_comp_df["zdr_intersection"]]
            for col in (
                "zdr_poly",
                "ZDR_id",
                "zdr_poly_area",
                "zdr_poly_area_pixels",
                "zdr_poly_max",
                "zdr_poly_p95",
                "zdr_poly_mean",
                "zdr_poly_median",
                "zdr_poly_centroid_xy",
                "zdr_poly_cell_value_list",
                "zdr_poly_volume",
                "cell_max_zdr_max",
                "cell_max_zdr_median",
                "max_zdr_value_list",
                
                "cell_max_zdr_elevation_max",
                "cell_max_zdr_elevation_median",
                "max_zdr_value_elevation_list",
                
                "hzt_grid_lvl_max",
                "hzt_grid_lvl_median",
                "hzt_grid_lvl_list"
            ):
                trt_list_df = comp_df_only_intersecting.groupby("id")[col].apply(list)
                trt_res_df = trt_res_df.merge(trt_list_df, how="left", left_on="id", right_on="id")

            trt_res_df.reset_index(inplace=True)
            
            #postprocessing zdr combination dataframe
            zdr_comp_df = pd.DataFrame(
                zdr_polys_with_intersect,
                columns=[
                    "ZDR_id",
                    "trt_id"
                ],
            )
            zdr_list_df = zdr_comp_df.groupby("ZDR_id")["trt_id"].apply(list)
            zdr_res_df = zdr_df.merge(zdr_list_df, how="left", left_on="ZDR_id", right_on="ZDR_id")
            
            
        else:
            trt_res_df = pd.DataFrame(columns = ["id"])
            zdr_res_df = zdr_df
            zdr_res_df["trt_id"] = np.nan
            
        zdr_res_df["timestamp"] = timestamp
        trt_res_df["timestamp"] = timestamp

        zdr_res_df.to_pickle(f"{zdr_merge_pickle_path}/{fname}.pkl")
        trt_res_df.to_pickle(f"{trt_merge_pickle_path}/{fname}.pkl")       
        
        
        #TODO:
        #ZDR_WITHOUT_TRT_POLYS_PATH
        zdr_without_trt = zdr_res_df[(zdr_res_df["trt_id"].isna())]
        if not zdr_without_trt.empty:
            zdr_without_trt.to_pickle(f"{zdr_no_trt_pickle_path}/{fname}.pkl")

    return zdr_res_df, trt_res_df, zdr_without_trt

