import numpy as np


#######################################
def reliability_diagram_bs(obs_bs, prob_bs):

      prob_bins = np.arange(0, 100 + 1, 1)
      mean_prob_fc = []
      mean_freq_obs = []
      sharpness = []
      
      for ind in range(len(prob_bins) - 1):

            bin_i = prob_bins[ind]
            bin_j = prob_bins[ind + 1]
            ind_bin = ( (prob_bs >= bin_i) & (prob_bs < bin_j) ).astype(float)
            ind_bin[ ind_bin == 0 ] = np.nan

            mean_prob_fc.append( np.nan_to_num( np.nanmean(prob_bs * ind_bin, axis = 0), nan = (bin_i + bin_j)/2 ) )
            mean_freq_obs.append( np.nanmean(obs_bs * ind_bin, axis = 0) )
            sharpness.append( np.nansum(ind_bin, axis = 0) )
            
      mean_freq_obs = np.nan_to_num(np.array(mean_freq_obs), nan=0)
      sharpness = np.nan_to_num(np.array(sharpness), nan=0)

      return np.array(mean_prob_fc), mean_freq_obs, sharpness


#########################################################
def contingency_table_probabilistic_bs(obs_bs, prob_bs, num_em):

      h_bs = []
      fa_bs = []
      m_bs = []
      cn_bs = []
      prob_nwp_list = np.arange(0, num_em) / num_em * 100
      for prob_nwp in prob_nwp_list:
            h_bs.append( ( (prob_bs >= prob_nwp) & (obs_bs == 1) ).sum(axis = 0) )
            fa_bs.append( ( (prob_bs >= prob_nwp) & (obs_bs == 0) ).sum(axis = 0) )
            m_bs.append( ( (prob_bs < prob_nwp) & (obs_bs == 1) ).sum(axis = 0) )
            cn_bs.append( ( (prob_bs < prob_nwp) & (obs_bs == 0) ).sum(axis = 0) )
      
      return np.array(h_bs), np.array(fa_bs), np.array(m_bs), np.array(cn_bs)


########################
def frequency_bias(h, fa, m):
      fb = (h + fa) / (h + m)
      return fb


###############
def hit_rate(h, m):
      hr = h / (h + m)
      return hr


#######################
def false_alarm_rate(fa, cn):
      far = fa / (fa + cn)
      return far


#######################
def aroc_trapezium(hr, far):

      dim = hr.ndim
      
      if dim == 1: # for 1-d arrays (e.g.., for original datasets)
            
            hr_i = hr[:-1]
            hr_j = hr[1:]
            far_i = far[:-1,]
            far_j = far[1:]
            aroc = np.abs( np.sum( ((hr_i + hr_j) * (far_j - far_i)) / 2 ) )
      
      else: # for 2-d arrays (e.g.., for bootstrapped datasets)
            
            hr_i = hr[:-1, :]
            hr_j = hr[1:, :]
            far_i = far[:-1, :]
            far_j = far[1:, :]
            aroc = np.abs( np.sum( ((hr_i + hr_j) * (far_j - far_i)) / 2, axis = 0 ) )
      
      return aroc