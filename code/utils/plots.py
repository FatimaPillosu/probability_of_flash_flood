import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator

##################################
def roc_curve_ci(rp, rp_colour, hr, far, aroc, fb, alpha, file_out):

      # Read the hit rate, false alarm rate, and area under the roc curve for the original dataset
      hr_real = hr[0,:]
      far_real = far[0,:]
      aroc_real = aroc[0]
      fb_real = fb[0,:]

      # Read the hit rate, false alarm rate, and area under the roc curve for the bootstrapped dataset
      hr_bs = hr[1:,:]
      far_bs = far[1:,:]
      aroc_bs = aroc[1:]

      # Compute confidence intervals for the roc curve and the aroc
      cl_lower = (100 - alpha) / 2
      hr_ci_lower = np.percentile(hr_bs, cl_lower, axis = 0)
      far_ci_lower = np.percentile(far_bs, cl_lower, axis = 0)
      aroc_ci_lower = np.percentile(aroc_bs, cl_lower)

      cl_upper = (100 + alpha) / 2 
      hr_ci_upper = np.percentile(hr_bs, cl_upper, axis = 0)
      far_ci_upper = np.percentile(far_bs, cl_upper, axis = 0)
      aroc_ci_upper = np.percentile(aroc_bs, cl_upper)

      x_poly = np.concatenate([far_ci_upper, far_ci_lower[::-1]])
      y_poly = np.concatenate([hr_ci_upper, hr_ci_lower[::-1]])
      
      # Determine the probability threshold for which fb ~ 1 and
      ind_fb_1 = np.where(fb_real >= 1)[0][-1]
      fb_1 = fb_real[ind_fb_1]
      fb_1_x = far_real[ind_fb_1]
      fb_1_y = hr_real[ind_fb_1]

      # Determine the fb for the smallest probability threshold of ~ 1%
      ind_prob_1 = 1
      fb_prob_1 = fb_real[ind_prob_1]
      fb_prob_1_x = far_real[ind_prob_1]
      fb_prob_1_y = hr_real[ind_prob_1]

      # Create the plot of the roc curve with frequency bias and aroc
      fig, ax = plt.subplots(figsize=(5, 5))
      ax.plot(far_real, hr_real, "-o", markersize=2, color=rp_colour, lw = 1, label = f"AROC = {aroc_real:.3f}")
      ax.fill(x_poly, y_poly, color=rp_colour, alpha=0.2, label=f"{alpha}% confidence interval\nAROC ({aroc_ci_lower:.3f} , {aroc_ci_upper:.3f})", edgecolor='none')
      plt.plot(fb_prob_1_x, fb_prob_1_y, "s", markersize=6, color = "#333333")
      plt.plot(fb_1_x, fb_1_y, "D", markersize=6, color = "#333333")
      plt.plot([-0.02,1.02], [-0.02,1.02],  "--", color="darkgrey", lw = 1)

      ax.text(0.04, 0.85, f'       FB[prob<=23%] = 28.00\nFB[prob<=23%] = 28.00\nFB[prob<=23%] = 28.00', color = "white", bbox=dict(facecolor='white', edgecolor='#333333', boxstyle='square, pad=0.4', linewidth=0.5))

      ax.text(0.06, 0.92, f" ■ FB[prob<={ind_prob_1}%] = {fb_prob_1:.2f}", fontsize=10, color='#333333')
      ax.text(0.06, 0.86, f" ◆ FB[prob<={ind_fb_1}%] = {fb_1:.2f}", fontsize=10, color='#333333')
      
      ax.set_title(f"tp > {rp}-year return period", color='#333333', fontweight='bold', pad = 30, fontsize = 10)
      ax.set_xlabel("False Alarm Rate [-]", color='#333333')
      ax.set_ylabel("Hit Rate [-] ", color='#333333')
      legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize = 10)
      for text in legend.get_texts():
            text.set_color('#333333')
      
      plt.xlim([-0.02,1.02])
      plt.ylim([-0.02,1.02])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()


#####################################################
def reliability_diagram_ci(rp, rp_colour, prob_fc, freq_obs, sharpness, alpha, file_out):

      # Read the probabilities for the forecasts and the observational frequencies for the original dataset
      prob_fc_real = prob_fc[0,:]
      freq_obs_real = freq_obs[0,:]
      sharpness_real = sharpness[0,:]

      # Read the probabilities for the forecasts and the observational frequencies for the bootstrapped dataset
      prob_fc_bs = prob_fc[1:,:] 
      freq_obs_bs = freq_obs[1:,:]

      # Compute confidence intervals for the reliability diagram
      cl_lower = (100 - alpha) / 2
      prob_fc_ci_lower = np.percentile(prob_fc_bs, cl_lower, axis = 0)
      freq_obs_ci_lower = np.percentile(freq_obs_bs, cl_lower, axis = 0)

      cl_upper = (100 + alpha) / 2 
      prob_fc_ci_upper = np.percentile(prob_fc_bs, cl_upper, axis = 0)
      freq_obs_ci_upper = np.percentile(freq_obs_bs, cl_upper, axis = 0)

      x_poly = np.concatenate([prob_fc_ci_upper, prob_fc_ci_lower[::-1]])
      y_poly = np.concatenate([freq_obs_ci_upper, freq_obs_ci_lower[::-1]])

      # Create the plot for the reliability diagram (main plot)
      fig, ax = plt.subplots(figsize=(5, 5))
      ax.plot(prob_fc_real, freq_obs_real, color=rp_colour, lw = 1, label = "Reliability diagram")
      ax.fill(x_poly, y_poly, color=rp_colour, alpha=0.2, edgecolor='none', label = f"{alpha}% confidence interval")
      ax.plot([-1,101], [-1,101],  "--", color="darkgrey", lw = 1)
      
      ax.set_title(f"tp > {rp}-year return period", color='#333333', fontweight='bold', pad = 25, fontsize = 10)
      ax.set_xlabel("Forecast Probability [%]", color='#333333')
      ax.set_ylabel("Observed Frequency [%] ", color='#333333')
      legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False, fontsize = 10)
      for text in legend.get_texts():
            text.set_color('#333333')
      
      plt.xlim([-1,101])
      plt.ylim([-1,101])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)
     
      # Create the plot for sharpness (inset plot)
      inset_ax = fig.add_axes([0.22, 0.615, 0.35, 0.23])
      inset_ax.plot(np.arange(len(sharpness_real)), sharpness_real, color=rp_colour, lw = 1)

      inset_ax.set_title("Sharpness", fontsize=8, fontweight='bold', pad=3)
      inset_ax.set_xlabel("Forecast Probability [%]", fontsize=8, labelpad=2)
      inset_ax.set_ylabel("Absolute Frequency", fontsize=8, labelpad=1) 
      legend = inset_ax.legend(fontsize = 8, loc = "upper center")
      legend.get_frame().set_linewidth(0) 
      legend.get_frame().set_edgecolor('none')
      
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_scientific(True)
      formatter.set_powerlimits((0, 3))
      inset_ax.yaxis.set_major_formatter(formatter)
      inset_ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
      inset_ax.tick_params(axis='both', labelsize=8)
      inset_ax.yaxis.get_offset_text().set_fontsize(8)
      inset_ax.set_xlim(-1, 101)
      inset_ax.set_ylim(1e0, 10 ** np.ceil(np.log10(np.sum(sharpness_real)))) 
      ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
      ax.yaxis.set_major_formatter(ScalarFormatter())
      ax.yaxis.set_minor_formatter(plt.NullFormatter())

      inset_ax.spines['bottom'].set_color('darkgrey')
      inset_ax.spines['left'].set_color('darkgrey')
      inset_ax.spines['top'].set_color('darkgrey')
      inset_ax.spines['right'].set_color('darkgrey')
      inset_ax.tick_params(axis='x', colors='#333333') 
      inset_ax.tick_params(axis='y', colors='#333333')  
      inset_ax.set_yscale('log') 
      inset_ax.grid(True, color='gainsboro', linewidth=0.3)

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()


##############################
def aroc_ci(rp_list, rp_colour_list, aroc, alpha, file_out):

      fig, ax = plt.subplots(figsize=(5, 5))

      for ind_rp, rp in enumerate(rp_list):

            aroc_rp = aroc[:, :, ind_rp]
            rp_colour = rp_colour_list[ind_rp]

            # Extracting the original aroc values
            aroc_original = aroc_rp[:,0]
            m = aroc_original.shape[0]

            # Computing the confidence intervals
            cl_lower = (100 - alpha) / 2
            cl_upper = (100 + alpha) / 2 
            aroc_ci_lower = np.percentile(aroc_rp[:,1:], cl_lower, axis=1)
            aroc_ci_upper = np.percentile(aroc_rp[:,1:], cl_upper, axis=1)

            # Creating the plot
            ax.plot(np.arange(m), aroc_original, color=rp_colour, lw = 1)
            ax.fill_between(np.arange(m), aroc_ci_lower, aroc_ci_upper, color=rp_colour, alpha=0.2, edgecolor='none')
      
      ax.plot([-0.1, m + 0.1], [0.5, 0.5], "--", color="#333333", lw = 1)

      ax.set_xlabel("Lead Times [Days]", color='#333333')
      ax.set_ylabel("AROC [-] ", color='#333333')

      plt.xlim([-0.1, m - 1 + 0.1])
      plt.ylim([0.4,1])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()


##############################
def fb_ci(rp_list, rp_colour_list, fb, alpha, file_out):

      fig, ax = plt.subplots(figsize=(5, 5))

      for ind_rp, rp in enumerate(rp_list):

            fb_rp = fb[:, :, ind_rp]
            rp_colour = rp_colour_list[ind_rp]

            # Extracting the original aroc values
            fb_original = fb_rp[:,0]
            m = fb_original.shape[0]

            # Computing the confidence intervals
            cl_lower = (100 - alpha) / 2
            cl_upper = (100 + alpha) / 2 
            fb_ci_lower = np.percentile(fb_rp[:,1:], cl_lower, axis=1)
            fb_ci_upper = np.percentile(fb_rp[:,1:], cl_upper, axis=1)

            # Creating the plot
            ax.plot(np.arange(m), fb_original, color=rp_colour, lw = 1)
            ax.fill_between(np.arange(m), fb_ci_lower, fb_ci_upper, color=rp_colour, alpha=0.2, edgecolor='none')
      
      ax.plot([-0.1, m + 0.1], [1, 1], "--", color="#333333", lw = 1)

      ax.set_xlabel("Lead Times [Days]", color='#333333')
      ax.set_ylabel("FB [-] ", color='#333333')

      plt.xlim([-0.1, m - 1 + 0.1])
      plt.ylim([0,80])
      ax.spines['bottom'].set_color('darkgrey')
      ax.spines['left'].set_color('darkgrey')
      ax.spines['top'].set_color('darkgrey')
      ax.spines['right'].set_color('darkgrey')
      ax.tick_params(axis='x', colors='#333333') 
      ax.tick_params(axis='y', colors='#333333')  
      ax.grid(True, color='gainsboro', linewidth=0.3)

      # Saving the plot
      plt.savefig(file_out, dpi = 1000)
      plt.close()
















































