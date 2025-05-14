import os
import metview as mv

############################################################################################################
# CODE DESCRIPTION
# 03_plot_sdfor_era5.py plots the standard deviation of the filtered sub-gridscale orography from ERA5 (at 31 km resolution).
# Runtime: negligible.

# INPUT PARAMETERS DESCRIPTION
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# file_in (string): relative path of the file containing the standard deviation of the filtered sub-grid scale orography.
# dir_out (string): relative path of the directory containing the plot for the standard deviation of the filtered sub-grid scale orography.

# INPUT PARAMETERS
mask_domain = [22,-130,52,-60]
git_repo = "/ec/vol/ecpoint_dev/mofp/papers_2_write/PoFF_USA"
file_in_mask = "data/raw/mask/usa_era5/mask.grib"
file_in = "data/raw/reanalysis/era5/orography/sdfor.grib"
dir_out = "data/plot/03_sdfor_era5"
############################################################################################################


# Retrieving the standard deviation of the filtered sub-gridscale orography
print("Reading the standard deviation of the filtered sub-gridscale orography...")
sdfor = mv.read(git_repo + "/" + file_in) * 1

# Reading the USA mask
print("Reading the USA domain ...")
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Selecting the standard deviation of the filtered sub-gridscale orography values within the considered domain
print("Selecting the standard deviation of the filtered sub-gridscale orography values within the considered domain...")
sdfor_mask = (mask == 1) * sdfor

# Plotting the standard deviation of the filtered sub-gridscale orography values within the considered domain
print("Plotting the standard deviation of the filtered sub-gridscale orography values within the considered domain...")

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
    contour_level_list = [0, 10, 25, 50, 75, 100, 250, 500, 1000, 5000],
    contour_label = "off",
    contour_shade = "on",
    contour_shade_method = "area_fill",
    contour_shade_colour_method = "list",
    contour_shade_colour_list = [
        "rgb(0.0477,0.3209,0.2253)", # 0-10
        "rgb(0,0.451,0.2941)", # 10-25
        "rgb(0.02191,0.7232,0.4777)", # 25-50
        "rgb(0.3929,0.8071,0.6621)", # 50-75
        "rgb(0.6554,0.9211,0.8237)", # 75-100
        "rgb(0.9367,0.9484,0.7143)", # 100-250
        "rgb(0.926,0.6198,0.007324)", # 250-500
        "rgb(0.5832,0.4009,0.03645)", # 500 - 1000,
        "rgb(0.8118,0.8118,0.8118)"  # 1000 - 5000
        ])

legend = mv.mlegend(
    legend_text_colour = "charcoal",
    legend_text_font_size = 0.5,
    )

title = mv.mtext(
    text_line_count = 2,
    text_line_1 = "Standard deviation of the filtered sub-gridscale orography [metres]",
    text_line_2 = " ",
    text_colour = "charcoal",
    text_font_size = 0.75
    )

# Saving the plot
print("Saving the map plot ...")
dir_out_temp = git_repo + "/" + dir_out
if not os.path.exists(dir_out_temp):
    os.makedirs(dir_out_temp)
file_out = dir_out_temp + "/sdfor" 
png = mv.png_output(output_width = 5000, output_name = file_out)
mv.setoutput(png)
mv.plot(geo_view, sdfor_mask, contouring, legend, title)