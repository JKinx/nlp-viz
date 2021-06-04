import numpy as np

class TrainingLog(object):
  def __init__(self, config):
    self.model_name = config.model_name

    self.log = {}
    for n in config.log_info:
      self.log[n] = []
    return 


  def update(self, output_dict):
    """Update the log"""
    for l in self.log: 
      if(l in output_dict): self.log[l].append(output_dict[l])
    return

  def print(self):
    """Print out the log"""
    s = ""
    # for l in self.log: s += "%s: mean = %.4g, var = %.4g " %\
    #   (l, np.average(self.log[l]), np.var(self.log[l]))
    for l in self.log: s += "%s %.4g\t" % (l, np.average(self.log[l]))
    print(s)
    print("")
    return 

  def reset(self):
    """Reset the log"""
    for l in self.log: 
        self.log[l] = []
    return