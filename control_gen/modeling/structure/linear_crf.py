"""A implementation of Linear-chain CRF inference algorithms, including:

* Viterbi, relaxed Viterbi
* Perturb and MAP sampling, and its relaxed version 
* Forward algorithm
* Entropy 
* Forward Filtering Backward Sampling, and it Gumbelized version

"""

import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .. import torch_model_utils as tmu

from ..torch_struct import LinearChainCRF as LCRF

class LinearChainCRF(nn.Module):
  """Implemention of the linear chain CRF, since we only need the forward, 
  relaxed sampling, and entropy here, we emit other inference algorithms like 
  forward backward, evaluation, and viterbi"""

  def __init__(self, config):
    super(LinearChainCRF, self).__init__()
    return 

  def calculate_all_scores(self, emission_scores, transition_scores):
    """Mix the transition and emission scores

    Args:
      emission_scores: type=torch.Tensor(float), 
        size=[batch, max_len, num_class]
      transition_scores: type=torch.Tensor(float), 
        size=[batch, num_class, num_class]

    Returns:
      scores: size=[batch, len, num_class, num_class]
      scores = log phi(batch, x_t, y_{t-1}, y_t)
    """
    label_size = emission_scores.size(2)
    batch_size = emission_scores.size(0)
    seq_len = emission_scores.size(1)

    # scores[batch, t, C, C] = log_potential(t, from y_{t-1}, to y_t)
    scores = transition_scores.view(batch_size, 1, label_size, label_size)\
      .expand(batch_size, seq_len, label_size, label_size) + \
      emission_scores.view(batch_size, seq_len, 1, label_size)\
      .expand(batch_size, seq_len, label_size, label_size)

    return scores

  def forward_score(self, emission_scores, transition_scores, seq_lens):
    """The forward algorithm
    
    score = log(potential)

    Args:
      emission_scores: size=[batch, max_len, label_size]
      transition_scores: type=torch.Tensor(float), 
        size=[batch, num_class, num_class]

    Returns:
      alpha: size=[batch, max_len, label_size]
      Z: size=[batch]
    """
    label_size = emission_scores.size(2)
    device = emission_scores.device
    all_scores = self.calculate_all_scores(emission_scores,
                                          transition_scores)

    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)
    alpha = torch.zeros(batch_size, seq_len, label_size).to(device)

    # the first position of all labels = 
    # (the transition from start - > all labels) + current emission.
    alpha[:, 0, :] = emission_scores[:, 0, :]

    for word_idx in range(1, seq_len):
      # batch_size, label_size, label_size
      before_log_sum_exp = alpha[:, word_idx - 1, :]\
        .view(batch_size, label_size, 1)\
        .expand(batch_size, label_size, label_size)\
        + all_scores[:, word_idx, :, :]
      alpha[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 1)

    # batch_size x label_size
    last_alpha = torch.gather(alpha, 1, seq_lens.view(batch_size, 1, 1)\
      .expand(batch_size, 1, label_size) - 1)\
        .view(batch_size, label_size)

    # assuming transition to end has potential 1   
    # last_alpha.shape=batch_size
    last_alpha = torch.logsumexp(
      last_alpha.view(batch_size, label_size, 1), 1).view(batch_size)
    log_Z = last_alpha
    return alpha, log_Z

  def backward_score(self, emission_scores, transition_scores, seq_lens):
    """backward algorithm"""
    label_size = emission_scores.size(2)
    device = emission_scores.device
    all_scores = self.calculate_all_scores(emission_scores, transition_scores)

    batch_size = all_scores.size(0)
    seq_len = all_scores.size(1)

    # beta[T] initialized as 0
    beta = torch.zeros(batch_size, seq_len, label_size).to(device)

    # beta stored in reverse order
    # all score at i: phi(from class at L - i - 1, to class at L - i)
    all_scores = tmu.reverse_sequence(all_scores, seq_lens)
    for word_idx in range(1, seq_len):
      # beta[t + 1]: batch_size, t + 1, to label_size
      # indexing tricky here !! and different than the forward algo
      beta_t_ = beta[:, word_idx - 1, :]\
        .view(batch_size, 1, label_size)\
        .expand(batch_size, label_size, label_size)\

      # all_scores[t]: batch, from_state t-1, to state t
      before_log_sum_exp = beta_t_ + all_scores[:, word_idx - 1, :, :]
      beta[:, word_idx, :] = torch.logsumexp(before_log_sum_exp, 2)

    # reverse beta:
    beta = tmu.reverse_sequence(beta, seq_lens)
    # set the first beta to emission
    # beta[:, 0] = emission_scores[:, 0]
    return beta

  def rsample(self, emission_scores, transition_scores, seq_lens, tau, 
    return_switching=False, return_prob=False):
    """Reparameterized CRF sampling, a Gumbelized version of the 
    Forward-Filtering Backward-Sampling algorithm

    TODO: an autograd based implementation 
    requires to redefine the backward function over a relaxed-sampling semiring
    
    Args:
      emission_scores: type=torch.tensor(float), 
        size=[batch, max_len, num_class]
      seq_lens: type=torch.tensor(int), size=[batch]
      tau: type=float, anneal strength

    Returns
      sample: size=[batch, max_len]
      relaxed_sample: size=[batch, max_len, num_class]
    """
    all_scores = self.calculate_all_scores(emission_scores, transition_scores)
    alpha, log_Z = self.forward_score(emission_scores, transition_scores, seq_lens)

    batch_size = emission_scores.size(0)
    max_len = emission_scores.size(1)
    num_class = emission_scores.size(2)
    device = emission_scores.device

    # backward sampling, reverse the sequence
    relaxed_sample_rev = torch.zeros(batch_size, max_len, num_class).to(device)
    sample_prob = torch.zeros(batch_size, max_len).to(device)
    sample_rev = torch.zeros(batch_size, max_len).type(torch.long).to(device)
    alpha_rev = tmu.reverse_sequence(alpha, seq_lens).to(device)
    all_scores_rev = tmu.reverse_sequence(all_scores, seq_lens).to(device)
    
    # w.shape=[batch, num_class]
    w = alpha_rev[:, 0, :].clone()
    w -= log_Z.view(batch_size, -1)
    p = w.exp()
    if(return_switching):
      switching = 0.

    # DEBUG, to show exp(w) gives a valid distribution
    # p(y_T = k | x) = exp(w)
    # print(0)
    # print(torch.exp(w)[0])
    # print(torch.exp(w)[0].sum())
    
    relaxed_sample_rev[:, 0] = tmu.reparameterize_gumbel(w, tau)
    sample_rev[:, 0] = relaxed_sample_rev[:, 0].argmax(dim=-1)
    sample_prob[:, 0] = tmu.batch_index_select(p, sample_rev[:, 0]).flatten()
    mask = tmu.length_to_mask(seq_lens, max_len).type(torch.float)
    prev_p = p
    for i in range(1, max_len):
      # y_after_to_current[j, k] = log_potential(y_{t - 1} = k, y_t = j, x_t)
      # size=[batch, num_class, num_class]
      y_after_to_current = all_scores_rev[:, i-1].transpose(1, 2)
      # w.size=[batch, num_class]
      w = tmu.batch_index_select(y_after_to_current, sample_rev[:, i-1])
      w_base = tmu.batch_index_select(alpha_rev[:, i-1], sample_rev[:, i-1])
      w = w + alpha_rev[:, i] - w_base.view(batch_size, 1)
      p = F.softmax(w, dim=-1)
      if(return_switching):
        switching += (tmu.js_divergence(p, prev_p) * mask[:, i]).sum()
      prev_p = p

      # DEBUG: to show exp(w) gives a valid distribution
      # p(y_{t - 1} = j | y_t = k, x) = exp(w)
      # print(i)
      # print(torch.exp(w)[0])
      # print(torch.exp(w)[0].sum())
      
      relaxed_sample_rev[:, i] = tmu.reparameterize_gumbel(w, tau)
      sample_rev[:, i] = relaxed_sample_rev[:, i].argmax(dim=-1)
      sample_prob[:, i] = tmu.batch_index_select(p, sample_rev[:, i]).flatten()

    sample = tmu.reverse_sequence(sample_rev, seq_lens)
    relaxed_sample = tmu.reverse_sequence(relaxed_sample_rev, seq_lens)
    sample_prob = tmu.reverse_sequence(sample_prob, seq_lens)
    sample_prob = sample_prob.masked_fill(mask == 0, 1.)
    sample_log_prob_stepwise = (sample_prob + 1e-10).log()
    sample_log_prob = sample_log_prob_stepwise.sum(dim=1)

    ret = [sample, relaxed_sample]
    if(return_switching): 
      switching /= (mask.sum(dim=-1) - 1).sum()
      ret.append(switching)
    if(return_prob):
      ret.extend([sample_log_prob, sample_log_prob_stepwise])
    return ret

  def entropy(self, emission_scores, transition_scores, seq_lens):
    """The entropy of the CRF, another DP algorithm. See the write up
    
    Args:
      emission_scores:
      seq_lens:

    Returns:
      H_total: the entropy, type=torch.Tensor(float), size=[batch]
    """

    all_scores = self.calculate_all_scores(emission_scores, transition_scores)
    alpha, log_Z = self.forward_score(emission_scores, transition_scores, seq_lens)

    batch_size = emission_scores.size(0)
    max_len = emission_scores.size(1)
    num_class = emission_scores.size(2)
    device = emission_scores.device

    H = torch.zeros(batch_size, max_len, num_class).to(device)
    for t in range(max_len - 1):
      # log_w.shape = [batch, from_class, to_class]
      log_w = all_scores[:, t+1, :, :] +\
        alpha[:, t, :].view(batch_size, num_class, 1) -\
        alpha[:, t+1, :].view(batch_size, 1, num_class)
      w = log_w.exp()
      # DEBUG
      # print(t)
      # print(w) # expect all 1 tensors
      H[:, t+1, :] = torch.sum(
        w * (H[:, t, :].view(batch_size, num_class, 1) - log_w), dim=1)
    
    last_alpha = tmu.gather_last(alpha, seq_lens)
    H_last = tmu.gather_last(H, seq_lens)
    log_p_T = last_alpha - log_Z.view(batch_size, 1)
    p_T = log_p_T.exp()
    # DEBUG
    # print('finally')
    # print(p_T) # expect tensors sum to 1
    H_total = p_T * (H_last - log_p_T)
    H_total = H_total.sum(dim = -1)
    return H_total

#   def rsample(self, emission_scores, transition_scores, seq_lens, tau):
#     all_scores = self.calculate_all_scores(emission_scores, transition_scores)
#     dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
#     return dist.gumbel_crf(tau).sum(-1)

#   def entropy(self, emission_scores, transition_scores, seq_lens):
#     all_scores = self.calculate_all_scores(emission_scores, transition_scores)
#     dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
#     return dist.entropy
  
  def marginals(self, emission_scores, transition_scores, seq_lens):
    all_scores = self.calculate_all_scores(emission_scores, transition_scores)
    dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
    return dist.marginals

  def argmax(self, emission_scores, transition_scores, seq_lens):
    all_scores = self.calculate_all_scores(emission_scores, transition_scores)
    dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
    return dist.argmax.max(-1)[0].argmax(-1)