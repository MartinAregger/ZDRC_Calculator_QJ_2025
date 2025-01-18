from radar_work_repository.zdr_column_code.zdr_grid_calculator import (
    create_zdr_column_grid_snyder,
)
import logging
import gc


if __name__ == "__main__":

    # logging.basicConfig(
    #     filename=f"/users/maregger/PhD/slurm_logging/test.log", level=logging.ERROR, filemode="a"
    # )

    logging.basicConfig(
        filename=f"/users/maregger/PhD/slurm_logging/zdr_log.log",
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.ERROR,
        filemode="a",
    )

    timestamp = "20210604112500"  # "Missing zipfile"
    # timestamp = "20210125000000" # Missing Bias value

    try:
        zdr_db_threshold = 1.5
        save = True
        recalculate = True
        bias_correction = True
        calculation_mode = "minimal"
        calc_contributor_grid = False
        rad_list = ["A", "D", "L", "P", "W"]
        field = "differential_reflectivity"
        rhohv_threshold = 0.8
        rhohv_filter = True
        lut = "new"
        visibility_threshold = 50
        gatefilter = "moving_average"
        number_of_observations_threshold = 10000

        create_zdr_column_grid_snyder(
            timestamp=timestamp,
            save=save,
            recalculate=recalculate,
            bias_correction=bias_correction,
            calculation_mode=calculation_mode,
            calc_contributor_grid=calc_contributor_grid,
            rad_list=rad_list,
            field=field,
            rhohv_threshold=rhohv_threshold,
            rhohv_filter=rhohv_filter,
            lut=lut,
            visibility_threshold=visibility_threshold,
            gatefilter=gatefilter,
            number_of_observations_threshold=number_of_observations_threshold,
            zdr_db_threshold=zdr_db_threshold,
        )
    except Exception as e:
        logging.error(f"{timestamp},{e}")

    gc.collect()
