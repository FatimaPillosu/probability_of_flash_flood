import os
import sys
from datetime import datetime, timedelta
import metview as mv


FileIN = "/home/mofp/vol_ecpoint_dev/mofp/phd/probability_of_flash_flood/data/processed/16_prob_ff_hydro_short_fc_compute_poff/2021/poff_20210902_00.grib"
#FileIN = "/home/mofp/vol_ecpoint_dev/mofp/phd/probability_of_flash_flood/data/processed/16_prob_ff_hydro_short_fc_compute_poff/2024/poff_20241030_00.grib"
Prob_AccRepFF = mv.read(FileIN)
      
      
TheDateTime_Start = datetime(2021, 9, 1, 0) 
TheDateTime_Final = datetime(2021, 9, 2, 0) 

# Defining the plot titles
ValidityDateS = TheDateTime_Start
DayVS = ValidityDateS.strftime("%d")
MonthVS = ValidityDateS.strftime("%B")
YearVS = ValidityDateS.strftime("%Y")
TimeVS = ValidityDateS.strftime("%H")
ValidityDateF = TheDateTime_Final
DayVF = ValidityDateF.strftime("%d")
MonthVF = ValidityDateF.strftime("%B")
YearVF = ValidityDateF.strftime("%Y")
TimeVF = ValidityDateF.strftime("%H")

title_plot1 = "Probability (%) of having a flash flood report in each grid-box"
title_plot2 = "Model: XGBoost"
title_plot3 = "VT: " + DayVS + " " + MonthVS + " " + YearVS + " " + TimeVS + " UTC - " + DayVF + " " + MonthVF + " " + YearVF + " " + TimeVF  + " UTC"          

# Plotting the probabilities
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
      map_label_height = 0.7
      )
      
contouring = mv.mcont(
      legend = "on", 
      contour = "off", 
      contour_level_selection_type = "level_list",
      contour_level_list = [0, 1, 2, 3, 5, 10, 100],
      contour_label = "off",
      contour_shade = "on",
      contour_shade_technique = "grid_shading",
      contour_shade_colour_method = "list",
      contour_shade_colour_list = [
            "white", # 0 - 1
            "rgb(0.55,0.55,0.55)", # 1 - 2
            "rgb(0.45,0.45,0.45)", # 2 - 3
            "rgb(1,0,0.498)", # 3 - 5
            "rgb(0.1451,0,1)", # 5 - 10
            "rgb(0.749,0.5765,0.07451)" # 10 - 100
            ]
      )

legend = mv.mlegend(
      legend_text_colour = "charcoal",
      legend_text_font_size = 0.5,
      )

title = mv.mtext(
      text_line_count = 4,
      text_line_1 = title_plot1,
      text_line_2 = title_plot2,
      text_line_3 = title_plot3,
      text_line_4 = " ",
      text_colour = "charcoal",
      text_font_size = 0.7
      )

# Saving the plot
# MainDirOUT = Git_Repo + "/" + DirOUT
# if not os.path.exists(MainDirOUT):
#       os.makedirs(MainDirOUT)
# FileOUT = MainDirOUT + "/Prob_AccRepFF_" +  TheDateTime_Final.strftime("%Y%m%d") + "_" + TheDateTime_Final.strftime("%H")
# png = mv.png_output(output_width = 5000, output_name = FileOUT)
# mv.setoutput(png)
mv.plot(coastlines, Prob_AccRepFF, contouring, legend, title)