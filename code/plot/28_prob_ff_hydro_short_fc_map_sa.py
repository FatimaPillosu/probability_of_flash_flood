import os
import sys
from datetime import datetime, timedelta
import metview as mv

######################################################################################
# CODE DESCRIPTION
# 28_prob_ff_hydro_short_fc_map_sa.py created the map plots for the probability of flash floods.

# Usage: python3 28_prob_ff_hydro_short_fc_map_sa.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# the_date (date, in YYYYmmdd format): date to plot.
# lt (positive integer, in days): lead time to consider.
#                                                     - if lt=0 -> reanalysis
#                                                     - if lt>0 -> forecasts 
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.Considered values are:
#                                                         - [22,-130,52,-60] USA
#                                                         - [25, 100, 40, 125] China
#                                                         - [35, -10, 44, 4] Spain
# model_name (string): name of the model to train.
# loss_func_list (list of strings): type of loss function considered. Valid values are:
#                                                           - bce: no weights applied to loss function.
#                                                           - weighted_bce: wheight applied to loss function.
# eval_metric_list (list of strings): evaluation metric for the data-driven models. Valid values are:
#                                                           - auc: area under the roc curve.
#                                                           - auprc: area under the precion-recall curve.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the reanalysis/forecasts of poff.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

######################################################################################
# INPUT PARAMETERS
the_date = datetime(2021,9,1) 
mask_domain = [22,-130,52,-60]
sa_name_list = ["east_rep_east_domain", "east_rep_full_domain", "west_rep_west_domain", "west_rep_full_domain", "red_10/iter_0", "red_50/iter_0", "red_90/iter_0"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in_short_fc = "data/processed/20_prob_ff_hydro_short_fc_compute_poff_sa"
dir_out = "data/plot/28_prob_ff_hydro_short_fc_map_sa"
######################################################################################


# Setting date parameters
the_data_s = the_date
the_data_f = the_date + timedelta(hours = 24)
DayVS = the_data_s.strftime("%d")
MonthVS = the_data_s.strftime("%B")
YearVS = the_data_s.strftime("%Y")
TimeVS = the_data_s.strftime("%H")
DayVF = the_data_f.strftime("%d")
MonthVF = the_data_f.strftime("%B")
YearVF = the_data_f.strftime("%Y")
TimeVF = the_data_f.strftime("%H")

# Creating the map plots for poff
for sa_name in sa_name_list:

            print(f"\nCreating map plots for poff for sa: {sa_name}")
            
            # Creating the output directory
            dir_out_temp = f"{git_repo}/{dir_out}/{sa_name}"
            os.makedirs(dir_out_temp, exist_ok=True)

            # Reading poff
            
            print(f" - Creating map plots for poff (reanalysis) on {the_date.strftime("%Y")}")
            file_in = f"{git_repo}/{dir_in_short_fc}/{sa_name}/poff_{the_data_f.strftime("%Y%m%d")}_00.grib"
            poff = mv.read(file_in)
                  
            # Plotting poff
            coastlines = mv.mcoast(
                  map_coastline_colour = "charcoal",
                  map_coastline_thickness = 2,
                  map_coastline_resolution = "full",
                  map_coastline_sea_shade = "on",
                  map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
                  map_boundaries = "on",
                  map_boundaries_colour = "charcoal",
                  map_boundaries_thickness = 2,
                  map_administrative_boundaries = "on",
                  map_administrative_boundaries_countries_list = "usa",
                  map_administrative_boundaries_style = "solid",
                  map_administrative_boundaries_thickness = 2,
                  map_administrative_boundaries_colour = "charcoal",
                  map_grid_latitude_increment = 10,
                  map_grid_longitude_increment = 20,
                  map_label_right = "off",
                  map_label_top = "off",
                  map_label_colour = "charcoal",
                  map_grid_thickness = 1,
                  map_grid_colour = "charcoal",
                  map_label_height = 5
                  )

            geo_view = mv.geoview(
                  map_projection = "epsg:3857",
                  map_area_definition = "corners",
                  area = mask_domain,
                  coastlines = coastlines
                  )

            contouring = mv.mcont(
                  legend = "on", 
                  contour = "off", 
                  contour_level_selection_type = "level_list",
                  contour_level_list = [0, 1, 3, 5, 7, 10, 25, 50, 100],
                  contour_label = "off",
                  contour_shade = "on",
                  contour_shade_technique = "grid_shading",
                  contour_shade_colour_method = "list",
                  contour_shade_colour_list = [
                        "white", # 0 - 1
                        "rgb(0.55,0.55,0.55)", # 1 - 3
                        "rgb(0.9583,0.579,0.7686)", # 3 - 5
                        "rgb(1,0,0.498)", # 5 - 7
                        "rgb(0.451,0.6392,1)", # 7 - 10
                        "rgb(0.1451,0,1)", # 10 - 25
                        "rgb(0.9606,0.8478,0.5374)", # 25 - 50
                        "rgb(0.749,0.5765,0.07451)"] # 50 - 100
                        )

            legend = mv.mlegend(
                  legend_text_colour = "charcoal",
                  legend_text_font_size = 5,
                  )

            title_plot1 = f"Probability (%) of having a flash flood report in each grid-box ({sa_name})"
            title_plot2 = "VT: " + DayVS + " " + MonthVS + " " + YearVS + " " + TimeVS + " UTC - " + DayVF + " " + MonthVF + " " + YearVF + " " + TimeVF  + " UTC"          
            title = mv.mtext(
                  text_line_count = 3,
                  text_line_1 = title_plot1,
                  text_line_2 = title_plot2,
                  text_line_3 = " ",
                  text_colour = "charcoal",
                  text_font_size = 8
                  )

            # Saving the plot
            file_out = f"{dir_out_temp}/poff_{the_data_f.strftime("%Y%m%d")}_00.grib"
            png = mv.png_output(output_width = 5000, output_name = file_out)
            mv.setoutput(png)
            mv.plot(geo_view, poff, contouring, legend, title)