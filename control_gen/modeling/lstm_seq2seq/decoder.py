

import torch

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from .. import torch_model_utils as tmu 

def attention(query, memory, mem_mask, device, no_softmax=False):
  """The attention function, Transformer style, scaled dot product
  
  Args:
    query: the query vector, shape = [batch_size, state_size]
    memory: the memory, shape = [batch_size, max_mem_len, state_size]
    mem_mask: the memory mask, shape = [batch_size, max_mem_len]. 1 = not masked
      0 = masked

  Returns:
    context_vec: the context vector, shape = [batch_size, state_size]
    attn_dist: the attention distribution, shape = [batch_size, max_mem_len]
  """
  state_size = query.shape[-1]
  batch_size = query.shape[0]
  max_mem_len = memory.shape[1]

  memory_ = memory.transpose(2, 1) # [B, M, S] -> [B, S, M]
  query_ = query.unsqueeze(1) # [B, 1, S]
  
  attn_weights = torch.bmm(query_, memory_) 
  attn_weights /= torch.sqrt(torch.Tensor([state_size]).to(device)) # [B, 1, M]
  attn_weights = attn_weights.view(batch_size, max_mem_len) # [B, M]

  if(mem_mask is not None):
    attn_weights = attn_weights.masked_fill(mem_mask == 0, -1e9)
  
  attn_dist = F.softmax(attn_weights, -1)
  attn_dist = attn_dist.unsqueeze(2)

  context_vec = attn_dist * memory
  context_vec = context_vec.sum(1) # [B, S]

  if no_softmax:
    return context_vec, attn_weights
  return context_vec, attn_dist.squeeze(2)

class Attention(nn.Module):
  """Simple scaled product attention"""
  def __init__(self, q_state_size, m_state_size, embedding_size):
    super(Attention, self).__init__()

    self.query_proj = nn.Linear(q_state_size, m_state_size)
    self.attn_proj = nn.Linear(m_state_size, embedding_size)
    return 

  def forward(self, query, memory, mem_mask=None, no_softmax=False):
    """
    Args:
      query: size=[batch, state_size]
      memory: size=[batch, mem_len, state_size]
      mem_mask: size=[batch, mem_len]

    Returns:
      context_vec: size=[batch, state_size]
      attn_dist: size=[batch, mem_len]
    """
    # map the memory and the query to the same space 
    # print(query.shape)
    device = query.device
    query = self.query_proj(query)
    context_vec, attn_dist = attention(query, memory, mem_mask, device, no_softmax)
    context_vec = self.attn_proj(context_vec)
    return context_vec, attn_dist

class LSTMDecoder(nn.Module):
  """The attentive LSTM decoder"""

  def __init__(self, config, mi_dec=False):
    super(LSTMDecoder, self).__init__()

    self.state_size = config.state_size
    self.attn_entropy = 0.0
    self.device = config.device
    self.vocab_size = config.vocab_size
    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.max_dec_len = config.max_y_len + 1

    if(config.lstm_layers == 1): dropout = 0.
    else: dropout = config.dropout

    self.cell = nn.LSTM(input_size=config.embedding_size, 
                        hidden_size=self.state_size,
                        num_layers=config.lstm_layers,
                        dropout=dropout)

    self.attention = Attention(
      config.state_size, config.tapas_state_size + config.embedding_size, config.embedding_size)

    self.dropout = nn.Dropout(config.dropout)
    self.attn_cont_proj = nn.Linear(
      2 * config.embedding_size, config.embedding_size)
    self.output_proj = nn.Linear(self.state_size, config.vocab_size)
    return 

  def forward(self, inp, state, memory=None, mem_mask=None):
    """
    Args: 
      state = (h, c)
        h: type = torch.tensor(Float)
           shape = [num_layers, batch, hidden_state]
        c: type = torch.tensor(Float)
           shape = [num_layers, batch, hidden_state]
    """
    inp = self.dropout(inp)
    device = inp.device
    query = state[0][0] # use the bottom layer output as query, as in GNMT
    context_vec = None
    if(memory is not None):
      context_vec, attn_dist = self.attention(query, memory, mem_mask)

    if(context_vec is not None):
      inp = self.attn_cont_proj(torch.cat([inp, context_vec], dim=1))
    out, state = self.cell(inp.unsqueeze(0), state)

    out = self.dropout(out)
    return out, state
