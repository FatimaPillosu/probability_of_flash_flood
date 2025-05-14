import os
import numpy as np
import metview as mv

#########################################################################
# CODE DESCRIPTION
# 06_plot_era5_ecpoint_tp_24h_climate_1991_2020.py plots the 24-hourly rainfall 
# climatology, computed from 1991 to 2020, from ERA5-ecPoint, and for different return 
# periods.
# Runtime: the script takes up to 30 seconds.

# INPUT PARAMETERS DESCRIPTION
# rp_list (list of integers): list of the return periods.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the 24-hourly rainfall climatology.
# dir_out (string): relative path of the directory containing the rainfall climatology's plots.

# INPUT PARAMETERS
rp_list = [1, 2, 5, 10, 20, 50, 100]
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
dir_in = "data/raw/reanalysis/era5_ecpoint/tp_24h_climate_1991_2020"
dir_out = "data/plot/06_era5_ecpoint_tp_24h_climate_1991_2020"
#########################################################################

# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)

# Reading the rainfall climatology and the return periods
climate = mv.read(git_repo + "/" + dir_in + "/climate_rp.grib")
rp_computed = np.load(git_repo + "/" + dir_in + "/rp.npy")

# Select the percentiles to plot
print()
print("Creating and saving the map plot of the 12-hourly rainfall climatology, computed between 1991 and 2020, from ERA5-ecPoint, for the: ")
for ind_rp in range(len(rp_list)):

    # Select the considered return period and the corresponding climatology
    rp_2_plot = rp_list[ind_rp]
    print(" - " + str(rp_2_plot) + "-year return period")
    
    # Select the considered return period and the corresponding climatology
    ind_climate_rp = np.where(rp_computed == rp_2_plot)[0]
    climate_rp = climate[ind_climate_rp]
    
    # Selecting the grid-points within the considered domain
    climate_rp_mask = ((mask == 0) * -9999) + ((mask == 1) * climate_rp)

    # Plot the rainfall climatology
    coastlines = mv.mcoast(
        map_coastline_colour = "charcoal",
        map_coastline_thickness = 2,
        map_coastline_resolution = "full",
        map_coastline_sea_shade = "on",
        map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
        map_boundaries = "on",
        map_boundaries_colour = "charcoal",
        map_boundaries_thickness = 4,
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
        contour_level_list = [0,0.5,2,5,10,20,30,40,50,60,80,100,125,150,200,300,500,5000],
        contour_label = "off",
        contour_shade = "on",
        contour_shade_colour_method = "list",
        contour_shade_method = "area_fill",
        contour_shade_colour_list = ["white","RGB(0.75,0.95,0.93)","RGB(0.45,0.93,0.78)","RGB(0.07,0.85,0.61)","RGB(0.53,0.8,0.13)","RGB(0.6,0.91,0.057)","RGB(0.9,1,0.4)","RGB(0.89,0.89,0.066)","RGB(1,0.73,0.0039)","RGB(1,0.49,0.0039)","red","RGB(0.85,0.0039,1)","RGB(0.63,0.0073,0.92)","RGB(0.37,0.29,0.91)","RGB(0.04,0.04,0.84)","RGB(0.042,0.042,0.43)","RGB(0.45,0.45,0.45)"]
        )

    legend = mv.mlegend(
        legend_text_colour = "charcoal",
        legend_text_font_size = 0.5,
        )

    title = mv.mtext(
        text_line_count = 3,
        text_line_1 = "Climatology (between 1991 and 2020) from ERA5-ecPoint for 12-hourly rainfall",
        text_line_2 = str(rp_2_plot) + "-year return period)",
        text_line_3 =" ",
        text_colour = "charcoal",
        text_font_size = 0.75
        )

    # Saving the maps
    dir_out_temp = git_repo + "/" + dir_out
    if not os.path.exists(dir_out_temp):
        os.makedirs(dir_out_temp)
    file_out = dir_out_temp + "/" + str(rp_2_plot) + "_year_rp"
    png = mv.png_output(output_width = 5000, output_name = file_out)
    mv.setoutput(png)
    mv.plot(coastlines, climate_rp_mask, geo_view, contouring, legend, title)