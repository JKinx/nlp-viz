import torch 
import numpy as np 

from torch import nn 
from torch.optim import Adam, SGD, RMSprop
from torch.nn.utils.clip_grad import clip_grad_norm_

from .latent_temp_crf_ar import LatentTemplateCRFAR
from .ftmodel import FTModel
from transformers import AdamW

class LatentTemplateCRFARModel(FTModel):
  def __init__(self, config):
    super().__init__()

    self.model = LatentTemplateCRFAR(config)
    
    tapas_params = [el[1] for el in self.model.named_parameters() \
                       if "tapas" in el[0]]
    other_params = [el[1] for el in self.model.named_parameters() \
                       if "tapas" not in el[0]]
    
    self.tapas_optimizer = AdamW(tapas_params, lr=5e-5)
    self.other_optimizer = Adam(other_params, lr=config.learning_rate)

    self.dataset = config.dataset

    self.max_grad_norm = config.max_grad_norm

    self.dataset = config.dataset
    self.device = config.device
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params):
    model = self.model
    sentences = torch.from_numpy(batch['sentences']).to(self.device)

    data_dict = {}
    for key in batch:
      data_dict[key] = torch.from_numpy(batch[key]).to(self.device)

    model.zero_grad()
    loss, out_dict = model(
      data_dict=data_dict,
      tau=schedule_params['tau'], 
      x_lambd=schedule_params['x_lambd']
      )

    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.tapas_optimizer.step()
    self.other_optimizer.step()

    out_dict['tau'] = schedule_params['tau']
    out_dict['x_lambd'] = schedule_params['x_lambd']
    return out_dict

  def valid_step(self, template_manager, batch, n_iter, ei, bi, 
    mode='dev', dataset=None, schedule_params=None):
    """Single batch validation"""
    data_dict = {}
    for key in batch:
      data_dict[key] = torch.from_numpy(batch[key]).to(self.device)

    with torch.no_grad():
      out_dict = model.infer(data_dict)
    
    return out_dict