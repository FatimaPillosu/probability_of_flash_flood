import metview as mv

def plot_poff(poff, title):

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
            map_administrative_boundaries_colour = "blue",
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
            contour_level_list = [0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 25, 50, 100],
            contour_label = "off",
            contour_shade = "on",
            contour_shade_method = "area_fill",
            contour_shade_colour_method = "list",
            contour_shade_colour_list = ["white", # 0 - 0.5
                                                                  # grey area
                                                                  "rgb(0.9,0.9,0.9)", # 0.5 - 1
                                                                  "rgb(0.7,0.7,0.7)", # 1 - 2
                                                                  "rgb(0.5,0.5,0.5)", # 2 - 3
                                                                  # blue area
                                                                  "rgb(0.5572,0.7529,0.9487)", # 3 - 5
                                                                  "rgb(0,0.498,1)", # 5 - 7.5
                                                                  "rgb(0.03969,0.349,0.6584)", # 7.5 - 10 
                                                                  # pink area
                                                                  "rgb(0.9624,0.5199,0.7412)", # 10- 15
                                                                  "rgb(1,0,0.498)", # 15 - 25
                                                                  "rgb(0.6053,0.03783,0.3216)", # 25 - 50
                                                                  # cream area
                                                                  "rgb(1,0.8863,0.6706)" # 50 - 100
                                                                  ]
            )

      legend = mv.mlegend(
            legend_text_colour = "charcoal",
            legend_text_font_size = 0.5,
            )

      title = mv.mtext(
            text_line_count = 2,
            text_line_1 = title,
            text_line_2 = " ",
            text_colour = "charcoal",
            text_font_size = 0.75
            )

      # Saving the plot
      # dir_out_temp = git_repo + "/" + dir_out
      # if not os.path.exists(dir_out_temp):
      #     os.makedirs(dir_out_temp)
      # file_out = dir_out_temp + "/orog" 
      # png = mv.png_output(output_width = 5000, output_name = file_out)
      # mv.setoutput(png)
      mv.plot(coastlines, poff, contouring, legend, title)

