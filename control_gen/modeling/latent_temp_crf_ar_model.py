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
    
    self.grad_accum = config.grad_accum
    self.iter_count = 0

    self.dataset = config.dataset

    self.max_grad_norm = config.max_grad_norm

    self.dataset = config.dataset
    self.device = config.device
    return 

  def train_step(self, batch, n_iter, ei, bi, schedule_params):
    model = self.model

    data_dict = {}
    for key in batch:
      try:
        data_dict[key] = torch.from_numpy(batch[key]).to(self.device)
      except:
        data_dict[key] = batch[key]
    
    if self.iter_count == 0:
        self.tapas_optimizer.zero_grad()
        self.other_optimizer.zero_grad()
    
    loss, out_dict = model(
      data_dict=data_dict,
      tau=schedule_params['tau'], 
      x_lambd=schedule_params['x_lambd'],
      z_beta= schedule_params['z_beta'],
      bi = bi
      )

    loss.backward()
    clip_grad_norm_(model.parameters(), self.max_grad_norm)
    self.iter_count += 1
    
    if self.iter_count % self.grad_accum == 0:
        self.tapas_optimizer.step()
        self.other_optimizer.step()
        self.tapas_optimizer.zero_grad()
        self.other_optimizer.zero_grad()

    out_dict['tau'] = schedule_params['tau']
    out_dict['x_lambd'] = schedule_params['x_lambd']
    out_dict['z_beta'] = schedule_params['z_beta']
    return out_dict

  def valid_step(self, template_manager, batch, n_iter, ei, bi, 
    mode='dev', dataset=None, schedule_params=None):
    """Single batch validation"""
    model = self.model
    
    data_dict = {}
    for key in batch:
      try:
        data_dict[key] = torch.from_numpy(batch[key]).to(self.device)
      except:
        data_dict[key] = batch[key]

    with torch.no_grad():
      out_dict = model.infer(data_dict)
    
    return out_dict