import numpy as np


#######################################
def reliability_diagram(obs, prob):

      prob_bins = np.arange(0, 100 + 1, 1)
      
      mean_prob_fc = []
      mean_freq_obs = []
      sharpness = []

      for ind_prob in range(len(prob_bins) - 1):

            prob_i = prob_bins[ind_prob]
            prob_j = prob_bins[ind_prob + 1]
            ind_prob_bin = np.where( (prob>= prob_i) & (prob < prob_j) )[0]
            num_elements = len(ind_prob_bin)

            if num_elements == 0:
                  mean_prob_fc.append( (prob_i + prob_j) / 2 )
                  mean_freq_obs.append(0)
            else:
                  mean_prob_fc.append( np.mean(prob[ind_prob_bin]) )
                  mean_freq_obs.append( np.mean(obs[ind_prob_bin]) )
            sharpness.append( num_elements )
            
      return np.array(mean_prob_fc), np.array(mean_freq_obs), np.array(sharpness)
      

#########################################################
def contingency_table_probabilistic(obs, prob, num_em):

      dim = obs.ndim
      prob_nwp_list = np.arange(0, num_em) / num_em * 100

      if dim == 1: # for 1-d arrays (e.g.., for original datasets)

            yes_event_obs = (obs == 1)[:, None]
            non_event_obs = (obs == 0)[:, None]

            yes_event_fc = prob[:, None] >= prob_nwp_list  # shape: (n_samples, n_thresholds)
            non_event_fc = ~yes_event_fc

            h = np.sum(yes_event_fc & yes_event_obs, axis=0)
            fa = np.sum(yes_event_fc & non_event_obs, axis=0)
            m = np.sum(non_event_fc & yes_event_obs, axis=0)
            cn = np.sum(non_event_fc & non_event_obs, axis=0)

      else: # for 2-d arrays (e.g.., for bootstrapped datasets)

            h = []
            fa = []
            m = []
            cn = []

            yes_event_obs = obs == 1
            non_event_obs = obs == 0

            for prob_nwp in prob_nwp_list: # need to mantain the for loop due to memory issues

                  yes_event_fc = prob >= prob_nwp
                  non_event_fc = ~yes_event_fc

                  h.append(np.sum(yes_event_fc & yes_event_obs, axis = 0))
                  fa.append(np.sum(yes_event_fc & non_event_obs, axis = 0))
                  m.append(np.sum(non_event_fc & yes_event_obs, axis = 0))
                  cn.append(np.sum(non_event_fc & non_event_obs, axis = 0))
            
            h = np.array(h)
            fa = np.array(fa)
            m = np.array(m)
            cn = np.array(cn)

      return h, fa, m, cn


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