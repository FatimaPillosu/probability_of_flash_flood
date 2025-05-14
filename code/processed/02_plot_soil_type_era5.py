import os
import metview as mv

#################################################################
# CODE DESCRIPTION
# 02_plot_soil_type_era5.py plots the soil type from ERA5 (at 31 km resolution). 
# This plot is used only for publication purposes. 
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_out (string): relative path of the directory containing the soil type plot.

# INPUT PARAMETERS
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_1km/mask.grib"
dir_out = "data/plot/02_soil_type_era5"
#################################################################


# Retrieving the soil type
print("Retrieving the soil type from ERA5 ...")
slt = mv.retrieve(
    {"class" : "ea",
     "stream" : "oper", 
     "type" : "an", 
     "expver" : "1", 
     "levtype" : "sfc",
     "param" : "43.128",
     "date" : "1940-01-01",
     "time" : "00:00:00"
    })

# Reading  the USA mask
print("Reading the USA domain ...")
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Selecting the soil type values within the considered domain
print("Selecting the soil type values within the domain ...")
slt_mask = (mask == 1) * slt

# Plotting the soil type values within the considered domain
print("Plotting the soil type values within the domain ...")

coastlines = mv.mcoast(
    map_coastline_colour = "charcoal",
    map_coastline_thickness = 2,
    map_coastline_resolution = "full",
    map_coastline_sea_shade = "on",
    map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
    map_boundaries = "on",
    map_boundaries_colour = "charcoal",
    map_boundaries_thickness = 2,
    map_grid_latitude_increment = 10,
    map_grid_longitude_increment = 20,
    map_label_right = "off",
    map_label_top = "off",
    map_label_colour = "charcoal",
    map_grid_thickness = 1,
    map_grid_colour = "charcoal",
    map_label_height = 0.7
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
    contour_level_list = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_technique = "grid_shading",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = ["rgb(1,0.8863,0.6588)", "rgb(0.4039,0.0549,0.003922)", "rgb(0,0.9961,0.01176)", "rgb(0.298,0.3843,0)", "cyan", "charcoal", "blue"])

legend = mv.mlegend(
    legend_text_colour = "charcoal",
    legend_text_font_size = 0.7,
    legend_display_type = "disjoint",
    legend_text_composition = "user_text_only",
    legend_user_lines = ["coarse","medium","medium fine","fine","very fine","organic","tropical organic"],
    legend_entry_text_width = 50.00,
    )

title = mv.mtext(
    text_line_count = 2,
    text_line_1 = "Soil Type",
    text_line_2 = " ",
    text_colour = "charcoal",
    text_font_size = 0.75
    )

# Saving the plot
print("Saving the map plot ...")
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)
FileOUT = dir_out_temp + "/soil_type" 
png = mv.png_output(output_width = 5000, output_name = FileOUT)
mv.setoutput(png)
mv.plot(geo_view, slt_mask, contouring, legend, title)