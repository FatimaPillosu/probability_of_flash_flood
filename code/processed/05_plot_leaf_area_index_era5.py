import os
from datetime import datetime, timedelta
import metview as mv

####################################################################
# CODE DESCRIPTION
# 05_plot_leaf_area_index_era5.py plots the leaf area index from ERA5 (at 31 km 
# resolution).
# Runtime: the code takes up to 20 minutes.

# INPUT PARAMETERS DESCRIPTION
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the leaf area index files.
# dir_out (string): relative path of the directory containing the leaf area index plots.

# INPUT PARAMETERS
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
dir_in = "data/raw/reanalysis/era5/lai"
dir_out = "data/plot/05_leaf_area_index_era5"
####################################################################


# Plotting the leaf area index for different days in the year
print()
print("Plotting the leaf area index for:")

date_s = datetime(2020,1,1)
date_f = datetime(2020,12,31)
the_date = date_s
while the_date <= date_f:

    print(" - " + the_date.strftime("%m-%d"))

    # Reading the leaf area index
    lai = mv.read(git_repo + "/" + dir_in + "/lai_" + the_date.strftime("%m%d") + ".grib")

    # Masking the US domain
    mask = mv.read(git_repo + "/" + file_in_mask)
    mask = mv.bitmap(mask,0) # bitmap the values outside the domain
    lai_mask = (mask == 1) * lai

    # Plotting the leaf area index for the US domain
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
        contour_level_list = [-0.05, 0.05, 0.2, 0.6, 1, 1.5, 2, 3, 4, 5, 6 , 7],
        contour_label = "off",
        contour_shade = "on",
        contour_shade_method = "area_fill",
        contour_shade_colour_method = "list",
        contour_shade_colour_list = [
            "rgb(1,0.9216,0.8431)", # -0.05 - 0.05
            "rgb(0.902,0.902,0.8)", # 0.05 - 0.2
            "rgb(0.7804,0.851,0.7412)", # 0.2 - 0.6
            "rgb(0.549,0.8,0.502)", # 0.6 - 1
            "rgb(0,1,0)", # 1 - 1.5
            "rgb(0.09804,0.8196,0.09804)", # 1.5 - 2
            "rgb(0.1098,0.651,0.1098)", # 2 - 3
            "rgb(0.09804,0.502,0.298)", # 3- 4
            "rgb(0.06667,0.349,0.2078)",  # 4 - 5
            "rgb(0,0.2,0.09804)", # 5 - 6
            "black"  # 6 - 7
            ])

    legend = mv.mlegend(
        legend_text_colour = "charcoal",
        legend_text_font_size = 0.5,
        )

    title = mv.mtext(
        text_line_count = 2,
        text_line_1 = "Leaf Area Index [m^2 / m^2] - " + the_date.strftime("%m-%d"),
        text_line_2 = " ",
        text_colour = "charcoal",
        text_font_size = 0.75
        )

    # Saving the plot
    dir_out_temp = git_repo + "/" + dir_out
    if not os.path.exists(dir_out_temp):
        os.makedirs(dir_out_temp)
    file_out = dir_out_temp + "/lai_" + the_date.strftime("%m%d")
    png = mv.png_output(output_width = 5000, output_name = file_out)
    mv.setoutput(png)
    mv.plot(geo_view, lai_mask, contouring, legend, title)

    the_date = the_date + timedelta(days=1)
