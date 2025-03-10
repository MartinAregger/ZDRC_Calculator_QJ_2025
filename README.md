This repository contains the python code snippet used to calculate Differential reflectivity columns (ZDRC) in the Research Article "Differential reflectivity columns and hail – linking C-band radar-based estimated column characteristics to crowdsourced hail observations in Switzerland" submitted to the Quarterly Journal of the Royal Meteorological Society 

- The radar data from the swiss weather radar network is provided by MeteoSwiss in a proprietary format which requires the use of a proprietary library to read them. Due to this, we have converted data for one example timestep into .npy files which you can look at in the "Testfiles_npy" folder. All data is available upon request from meteosvizzera@meteoswiss.ch.

- The ZDRC computation within the "ZDRC_Caluclation.py" script is functional and there is an example jupyter notebook to show it's use for the given timestep only.

- final_analysis_dataframes: Two dataframes containing the aggregated data used to create the analysis shown in the paper. The dataframes contain the ZDRC characteristics for each TRT-cell used in the paper. "aggregated_storm_types..." contains the aggregated storm classes (SmHS,MedHS,LgHS,SevHS,NoHS,AnyHS). "all_storm_types..." contains the storms for each hail report size as in Fig. 13 of the paper.
