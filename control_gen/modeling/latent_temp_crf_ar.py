"""Latent template CRF, autoregressive version"""

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Uniform

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder, Attention
# from .structure.linear_crf import LinearChainCRF
from .structure import LinearChainCRF
from . import torch_model_utils as tmu
import operator

from .torch_struct import LinearChainCRF as LC

from .fst import make_fst
from .beamtree import BeamTree

class LatentTemplateCRFAR(nn.Module):
  """The latent template CRF autoregressive version, table to text setting"""

  def __init__(self, config, embeddings=None):
    super().__init__()
    self.config = config
    self.device = config.device

    self.z_beta = config.z_beta
    self.z_overlap_logits = config.z_overlap_logits
    self.z_sample_method = config.z_sample_method
    self.gumbel_st = config.gumbel_st
    self.use_src_info = config.use_src_info
    self.use_copy = config.use_copy
    self.num_sample = config.num_sample

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.seg_id = config.seg_id

    self.vocab_size = config.vocab_size
    self.latent_vocab_size = config.latent_vocab_size 

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len
    self.max_bow_len = config.max_bow_len

    self.z_pred_strategy = config.z_pred_strategy
    self.x_pred_strategy = config.x_pred_strategy

    ## Model parameters
    # emb
    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    if(embeddings is not None): 
      self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    self.z_embeddings = nn.Embedding(
      config.latent_vocab_size, config.embedding_size)
    # enc
    self.q_encoder = LSTMEncoder(config)
    # latent 
    self.z_crf_proj = nn.Linear(config.state_size, config.latent_vocab_size)
    self.z_crf = LinearChainCRF(config)
    # dec 
    self.p_dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_decoder = LSTMDecoder(config)
    # copy 
    self.p_copy_attn = Attention(
      config.state_size, config.state_size, config.state_size)
    self.p_copy_g = nn.Linear(config.state_size, 1)
    # dec z proj
    self.p_z_proj = nn.Linear(config.state_size, config.latent_vocab_size)
    self.p_z_intermediate = nn.Linear(2 * config.state_size, config.state_size)
    
    # posterior regularization
    self.pr = config.pr
    self.pr_inc_lambd = config.pr_inc_lambd
    self.pr_exc_lambd = config.pr_exc_lambd
    self.num_pr_constraints = config.num_pr_constraints
    
    # constraints for beam search
    if "_ndend_" in config.word2id:
        self.locked_zs = [0,1,2]
        self.key_ys = {0 : config.word2id["_ndend_"],
                        1 : config.word2id["_odend_"],
                        2 : config.word2id["_mend_"]}
    else:
        self.locked_zs = []
    return 

  def init_state(self, s):
    batch_size = s.shape[0]
    init_state_h = self.p_dec_init_state_proj_h(s)
    init_state_h = init_state_h.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_h = init_state_h.transpose(0, 1).contiguous()
    init_state_c = self.p_dec_init_state_proj_c(s)
    init_state_c = init_state_c.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_c = init_state_c.transpose(0, 1).contiguous()
    return (init_state_h, init_state_c)

  def encode_kv(self, keys, vals):
    """Encode the key-valud table"""
    kv_mask = keys != self.pad_id 
    keys_emb = self.embeddings(keys)
    vals_emb = self.embeddings(vals)
    kv_emb = keys_emb + vals_emb # [batch, mem_len, state_size]

    kv_mask_ = kv_mask.type(torch.float)
    kv_enc = kv_emb * kv_mask_.unsqueeze(-1)
    # kv_enc.shape = [batch, embedding_size]
    kv_enc = kv_enc.sum(dim=1) / kv_mask_.sum(dim=1, keepdim=True)
    return kv_emb, kv_enc, kv_mask
    
  def forward(self, keys, vals, 
    sentences, sent_lens, sent_full, tau, x_lambd,  return_grad=False, zcs=None):
    """Forward pass, first run the inference network, then run the decoder
    
    Args:
      keys: torch.tensor(torch.long), size=[batch, max_mem_len]
      vals: torch.tensor(torch.long), size=[batch, max_mem_len]
      sentences: torch.tensor(torch.long), size=[batch, sent_len]
      sent_lens: torch.tensor(torch.long), size=[batch]
      tau: gumbel temperature, anneal from 1 to 0.01
      x_lambd: decoder coefficient for the word in, controll how 'autogressive'
       the model is, anneal from 0 to 1 

    Returns:
      loss: torch.float, the total loss 
      out_dict: dict(), output dict  
      out_dict['inspect']: dict(), training process inspection
    """
    out_dict = {}
    inspect = {}
    batch_size = sentences.size(0)
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()
        
    # PR
    if self.pr:
      out_dict["sent_mask"] = sent_mask
      out_dict["z_scores"] = z_emission_scores
        
      # get pr loss
      pr_inc_loss, pr_exc_loss = self.compute_pr(z_emission_scores, zcs[:, :max_len], 
                                sent_mask, sent_lens)
      out_dict["pr_inc_val"] = tmu.to_np(pr_inc_loss)
      out_dict["pr_exc_val"] = tmu.to_np(pr_exc_loss)
      out_dict["pr_inc_loss"] = self.pr_inc_lambd * tmu.to_np(pr_inc_loss)
      out_dict["pr_exc_loss"] = self.pr_exc_lambd * tmu.to_np(pr_exc_loss)
      loss -= self.pr_inc_lambd * pr_inc_loss + self.pr_exc_lambd * pr_exc_loss
        
    
    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    loss += self.z_beta * ent_z
    out_dict['ent_z'] = tmu.to_np(ent_z)
    out_dict['ent_z_loss'] = self.z_beta * tmu.to_np(ent_z)

    # reparameterized sampling
    if(self.z_sample_method == 'gumbel_ffbs'):
      z_sample_ids, z_sample, _ = self.z_crf.rsample(
        z_emission_scores, sent_lens, tau, return_switching=True)
    elif(self.z_sample_method == 'pm'):
      z_sample_ids, z_sample = self.z_crf.pmsample(
        z_emission_scores, sent_lens, tau)
    else:
      raise NotImplementedError(
        'z_sample_method %s not implemented!' % self.z_sample_method)

    z_sample_max, _ = z_sample.max(dim=-1)
    z_sample_max = z_sample_max.masked_fill(~sent_mask, 0)
    inspect['z_sample_max'] = (z_sample_max.sum() / sent_mask.sum()).item()
    out_dict['z_sample_max'] = inspect['z_sample_max']

    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 
    z_sample_ids_out = z_sample_ids.masked_fill(~sent_mask, -1)
    out_dict['z_sample_ids'] = tmu.to_np(z_sample_ids_out)
    inspect['z_sample_ids'] = tmu.to_np(z_sample_ids_out)
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids,
      self.z_embeddings, self.gumbel_st)

    # decoding
    if self.config.dataset == "dateSet":
        sentences = sent_full
    elif sent.config.dataset == "e2e":
        sentences = sentences
    else:
        raise NotImplementedError
        
    sentences = sentences[:, : max_len]
    p_log_prob, _, p_log_prob_x, p_log_prob_z, z_acc, _ = self.decode_train(
      z_sample_ids, z_sample_emb, sent_lens,
      keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['z_acc'] = z_acc.item()
    loss += p_log_prob

    # # turn maximization to minimization
    loss = -loss

    if(return_grad):
      self.zero_grad()
      g = torch.autograd.grad(
        loss, z_emission_scores, retain_graph=True)[0]
      g_mean = g.mean(0)
      g_std = g.std(0)
      g_r =\
        g_std.log() - g_mean.abs().log()
      out_dict['g_mean'] =\
        g_mean.abs().log().mean().item()
      out_dict['g_std'] = g_std.log().mean().item()
      out_dict['g_r'] = g_r.mean().item()   

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    return loss, out_dict

  def compute_pr(self, emission_scores, zcs, sent_mask, sent_lens):
    # edge potentials
    all_scores = self.z_crf.calculate_all_scores(emission_scores)
    
    # Linear Chain CRF
    dist = LC(all_scores.transpose(3,2), (sent_lens + 1).float())
    
    # marginals : [batch, max_len, state_size]
    marginals = dist.marginals.sum(-1)
    rel_marginals = marginals[:, :, :self.num_pr_constraints]
    
    # filters for the constrained z states
    filters = torch.arange(self.num_pr_constraints).view(1,1,self.num_pr_constraints).to(self.device)
    
    # check if state is used
    check = zcs.unsqueeze(-1) == filters
    
    # computer loss
    # {1 - q(z = sigma(f) | x, y)} if f is used else {q(z = sigma(f) | x, y)}
    loss = (check.float() - rel_marginals).abs() * sent_mask.unsqueeze(-1).float()
    inc_loss = (1 - rel_marginals).abs() * check.float() * sent_mask.unsqueeze(-1).float()
    exc_loss = rel_marginals * (1 - check.float()) * sent_mask.unsqueeze(-1).float()
    
    # take average
    final_inc_loss = inc_loss.sum() / check.float().sum()
    final_exc_loss = exc_loss.sum() / (sent_lens.sum() - check.float().sum())
    
    return final_inc_loss, final_exc_loss

  def decode_train(self, 
    z_sample_ids, z_sample_emb, sent_lens,
    mem, mem_emb, mem_enc, mem_mask, sentences, x_lambd):
    """Train the decoder/ generative model. Same as 
    Li and Rush 20. Posterior Control of Blackbox Generation

    Args:

    Returns:
    """
    inspect = {}

    device = z_sample_ids.device
    state_size = self.state_size
    batch_size = sentences.size(0)

    dec_inputs, dec_targets_x, dec_targets_z = self.prepare_dec_io(
      z_sample_ids, z_sample_emb, sentences, x_lambd)
    max_len = dec_inputs.size(1)

    if(self.use_src_info):
      state = self.init_state(mem_enc)
    else: 
      state = self.init_state(mem_enc)
      state = [(state[0] * 0).detach(), (state[1] * 0).detach()]

    dec_inputs = dec_inputs.transpose(1, 0) # [T, B, S]
    dec_targets_x = dec_targets_x.transpose(1, 0) # [T, B]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_sample_emb = z_sample_emb[:, 1:].transpose(1, 0) # start from z[1]

    log_prob_x, log_prob_z, dec_outputs, z_pred = [], [], [], []

    for i in range(max_len): 
      if(self.use_src_info):
        dec_out, state = self.p_decoder(
          dec_inputs[i], state, mem_emb, mem_mask)
      else: 
        dec_out, state = self.p_decoder(
          dec_inputs[i], state)
      dec_out = dec_out[0]

      # predict z 
      z_logits = self.p_z_proj(dec_out)
      z_pred.append(z_logits.argmax(dim=-1))
      log_prob_z_i = -F.cross_entropy(
        z_logits, dec_targets_z[i], reduction='none')
      log_prob_z.append(log_prob_z_i)

      # predict x based on z 
      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_sample_emb[i]], dim=1))
      x_logits = self.p_decoder.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      if(self.use_copy): 
        _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        x_logits = (out_prob + 1e-10).log()

      log_prob_x_i = -F.cross_entropy(
        x_logits, dec_targets_x[i], reduction='none')
      log_prob_x.append(log_prob_x_i)
      
      dec_outputs.append(x_logits.argmax(dim=-1))

    # loss
    log_prob_x = torch.stack(log_prob_x).transpose(1, 0) # [B, T]
    log_prob_x = tmu.mask_by_length(log_prob_x, sent_lens)
    log_prob_z = torch.stack(log_prob_z).transpose(1, 0)
    log_prob_z = tmu.mask_by_length(log_prob_z, sent_lens)
    log_prob_step = log_prob_x + log_prob_z # stepwise reward

    log_prob_x_casewise = log_prob_x.sum(1)
    log_prob_x = log_prob_x.sum() / sent_lens.sum()
    log_prob_z_casewise = log_prob_z.sum(1)
    log_prob_z = log_prob_z.sum() / sent_lens.sum()
    log_prob_casewise = log_prob_x_casewise + log_prob_z_casewise
    log_prob = log_prob_x + log_prob_z

    # acc 
    z_pred = torch.stack(z_pred).transpose(1, 0) # [B, T]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_positive = tmu.mask_by_length(z_pred == dec_targets_z, sent_lens).sum() 
    z_acc = z_positive / sent_lens.sum()
    
    return (
      log_prob, log_prob_casewise, log_prob_x, log_prob_z, z_acc, log_prob_step)

  def infer(self, keys, vals):
    """Latent template inference step

    Args:
      keys: size=[batch, mem_len]
      vals: size=[batch, mem_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]

    Returns:
      out_dict
    """
    out_dict = {}
    batch_size = keys.size(0)
    mem_len = keys.size(1)
    state_size = self.state_size
    device = keys.device

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # decoding 
    predictions_x, predictions_z = self.decode_infer(vals, kv_emb, kv_enc, kv_mask)
    out_dict['predictions'] = tmu.to_np(predictions_x)
    out_dict['predictions_z'] = tmu.to_np(predictions_z)
    pred_lens_ = tmu.seq_ends(predictions_x, self.end_id) + 1
    out_dict['pred_lens'] = tmu.to_np(pred_lens_)
    return out_dict

  def decode_infer(self, mem, mem_emb, mem_enc, mem_mask):
    """Inference

    Args:
      mem: torch.Tensor(), size=[batch, mem_len]
      mem_emb: torch.Tensor(), size=[batch, mem_len, state_size]
      mem_enc: torch.Tensor(), size=[batch, state_size]
      mem_mask: torch.Tensor(), size=[batch, mem_len]
    
    Returns:
      predictions_x: torch.Tensor(int), size=[batch, max_dec_len]
      predictions_z: torch.Tensor(int), size=[batch, max_dec_len]
    """
    
    batch_size = mem.size(0)
    device = mem.device

    predictions_x, predictions_z = [], []
    inp = self.embeddings(
      torch.zeros(batch_size).to(device).long() + self.start_id)
    # assume use_src_info=True
    state = self.init_state(mem_enc)
    for i in range(self.max_dec_len): 
      # assume use_src_info=True
      dec_out, state = self.p_decoder(inp, state, mem_emb, mem_mask)
      dec_out = dec_out[0]

      # predict z 
      z_logits = self.p_z_proj(dec_out)
      if(self.z_pred_strategy == 'greedy'):
        z = z_logits.argmax(-1)
      elif(self.z_pred_strategy == 'sampling'):
        pass # TBC
      else: raise NotImplementedError(
        'Error z decode strategy %s' % self.z_pred_strategy)

      # predict x based on z 
      z_emb = self.z_embeddings(z)
      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_emb], dim=1))
      x_logits = self.p_decoder.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        x_logits = (out_prob + 1e-10).log()

      if(self.x_pred_strategy == 'greedy'):
        x = x_logits.argmax(-1)
      elif(self.x_pred_strategy == 'sampling'):
        pass # TBC
      else: raise NotImplementedError(
        'Error x decode strategy %s' % self.x_pred_strategy)

      inp = z_emb + self.embeddings(x)

      predictions_x.append(x)
      predictions_z.append(z)

    predictions_x = torch.stack(predictions_x).transpose(1, 0)
    predictions_z = torch.stack(predictions_z).transpose(1, 0)
    return predictions_x, predictions_z

  def posterior_infer(self, keys, vals, sentences, sent_lens):
    """Find the argmax for z given x and y
    
    Args:
      keys: torch.tensor(torch.long), size=[batch, max_mem_len]
      vals: torch.tensor(torch.long), size=[batch, max_mem_len]
      sentences: torch.tensor(torch.long), size=[batch, sent_len]
      sent_lens: torch.tensor(torch.long), size=[batch]

    Returns:
      argmax x for each sample
    """
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()
    
    out = self.z_crf.argmax(z_emission_scores, sent_lens).tolist()
    out_list = [out[i][:sent_lens[i].item()] for i in range(len(out))]
    return out_list

  def posterior_marginals(self, keys, vals, sentences, sent_lens):
    """Find the marginals for z given x and y
    
    Args:
      keys: torch.tensor(torch.long), size=[batch, max_mem_len]
      vals: torch.tensor(torch.long), size=[batch, max_mem_len]
      sentences: torch.tensor(torch.long), size=[batch, sent_len]
      sent_lens: torch.tensor(torch.long), size=[batch]

    Returns:
      argmax x for each sample
    """
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()
    
    return self.z_crf.marginals(z_emission_scores, sent_lens)
  
  def infer2(self, keys, vals, templates, return_bt =  False):
    """Latent template inference step

    Args:
      keys: size=[batch, mem_len]
      vals: size=[batch, mem_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]

    Returns:
      out_dict
    """
    out_dict = {}
    batch_size = keys.size(0)
    mem_len = keys.size(1)
    state_size = self.state_size
    device = keys.device

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # decoding 
    pred_y, pred_z, pred_score, beam_trees, bts, bid = self.decode_infer2(vals, kv_emb,
      kv_enc, kv_mask, templates, return_bt)
    out_dict['pred_y'] = pred_y
    out_dict['pred_z'] = pred_z
    out_dict['pred_score'] = pred_score
    out_dict["beam_trees"] = beam_trees
    out_dict["bts"] = bts
    out_dict["regex_alignment"] = bid
    return out_dict

  def decode_infer2(self, mem, mem_emb, mem_enc, mem_mask, templates, return_bt):    
    batch_size = mem.size(0)
    
    decoded_batch = []
    decoded_score = []
    decoded_states = []
    decoded_bid = []
    beam_trees = []
    bts = []


    predictions_x, predictions_z = [], []
    inp = self.embeddings(
      torch.zeros(batch_size).to(self.device).long() + self.start_id)
    
    # assume use_src_info=True
    state = self.init_state(mem_enc)
    
    beams_batch = []
    
    for idx in range(batch_size):
      inp_i = inp[idx:idx + 1]
      state_i = (state[0][:, idx:idx + 1, :].contiguous(), state[1][:, idx:idx + 1, :].contiguous())
      mem_emb_i = mem_emb[idx:idx+1]
      mem_mask_i = mem_mask[idx:idx+1]
      mem_i = mem[idx:idx+1]
    
      template = templates[idx]

      beam_tree = BeamTree(inp_i, state_i, mem_emb_i, mem_mask_i, mem_i, template)
      bs_init = beam_tree.init_bs_init(return_bt)
    
      endnodes = self.beam_search(bs_init, beam_tree)

      beam_trees.append(beam_tree)
      bts.append(beam_tree.get_bt(bs_init))

      utterances_w, utterances_s, scores_result, utterances_b = self.endnodes_to_utterances(
        endnodes)
        
      decoded_batch.append(utterances_w)
      decoded_states.append(utterances_s)
      decoded_score.append(scores_result)
      decoded_bid.append(utterances_b)
    
    pred_y = []
    pred_z = []
    pred_score = []
    bid = []
    for batch_idx in range(len(decoded_batch)):
        if decoded_batch[batch_idx] == []:
            pred_y.append([])
            pred_z.append([])
            pred_score.append(-float("inf"))
            bid([])
        else:
            pred_y.append(decoded_batch[batch_idx][0][1:-1])
            pred_z.append(decoded_states[batch_idx][0][1:-1])
            pred_score.append(decoded_score[batch_idx][0])
            bid.append(decoded_bid[batch_idx][0][1:-1])
            
    return pred_y, pred_z, pred_score, beam_trees, bts, bid

  def beam_tree_act(self, bs_init, beam_tree):
    endnodes = self.beam_search(bs_init, beam_tree) 
    utterances_w, utterances_s, scores_result, bids = self.endnodes_to_utterances(
        endnodes)

    out_dict = {}
    
    pre_z = bs_init["prev_zs"][2:] + [bs_init["z_id"]]
    pre_y = bs_init["prev_ys"][2:] + [bs_init["y_id"]]
    if utterances_s[0] == []:
      out_dict['pred_y'] = pre_y
      out_dict['pred_z'] = pre_z
      out_dict['pred_score'] = - bs_init["logp"]
      out_dict["bt"] = []
      out_dict["regex_alignment"] = - bs_init["fs_idx"][0][1]
    else:
      out_dict['pred_y'] = utterances_w[0][1:-1]
      out_dict['pred_z'] = utterances_s[0][1:-1]
      out_dict['pred_score'] = scores_result[0]
      out_dict["bt"] = beam_tree.get_bt(bs_init)
      out_dict["regex_alignment"] = bids[0][1:-1]
    
    return out_dict

  def probe_bst(self, inp, h, mem_emb, mem_mask, mem):
    num_z = self.latent_vocab_size

    dec_out, decoder_hidden = self.p_decoder(inp, h, mem_emb, mem_mask)   
    dec_out = dec_out[0]

    # get log_probs for all z
    z_logits = self.p_z_proj(dec_out)
    
    state_logp = torch.log_softmax(z_logits, dim=-1).view(num_z, -1)

    # embed all z
    all_z = torch.arange(num_z).to(self.device)
    z_emb = self.z_embeddings(all_z)

    # get y probs
    dec_out = dec_out.expand(num_z, -1)
    dec_intermediate = self.p_z_intermediate(torch.cat([dec_out, z_emb], dim=-1))
    x_logits = self.p_decoder.output_proj(dec_intermediate)
    lm_prob = F.softmax(x_logits, dim=-1)

    mem_emb = mem_emb.expand(num_z, -1, -1)
    mem_mask = mem_mask.expand(num_z, -1)
    mem = mem.expand(num_z, -1)

    if(self.use_copy):
      _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
      copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
      copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

      out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
      x_logits = (out_prob + 1e-10).log()

    temp_wordlp = torch.log_softmax(x_logits, dim=-1)
    log_prob_w, indexes_w = torch.topk(temp_wordlp, 5)

    probs = (state_logp + log_prob_w).exp().tolist()
    ys = indexes_w.tolist()

    return ys, probs

  def init_bst(self, bs_init, beam_tree, fss, mem_emb, mem_mask, mem):
    if bs_init["fs_idx"][0] == 0:
        node = {'h': bs_init["h"], 'inp': bs_init["inp"], 'prevNode': None, 
                'word_id': -1, "ys" : bs_init["prev_ys"] + [-1],
                'state_id': -1, "zs" : bs_init["prev_zs"] + [-1],
                'logp': bs_init["logp"], 'leng': bs_init["leng"] + 1, 
                "fs_idx" : 0, "bids" : list(bs_init["fs_idx"][1])}
        if bs_init["bt"]:
          beam_tree.update_node(bs_init, node)
        fss[0]["nodes"] = [(-node['logp'], node)]
        return 

    h = bs_init["h"]
    inp = bs_init["inp"]

    # decoder states
    dec_out, decoder_hidden = self.p_decoder(inp, h, mem_emb, mem_mask)
    dec_out = dec_out[0]

    z_logits = self.p_z_proj(dec_out)

    # shape: bsz X state_vocab
    state_logp = torch.log_softmax(z_logits, dim=-1)

    # get the chosen z
    state_id = bs_init["z_id"]
    log_ps = state_logp[0,state_id].item()
      
    z_emb = self.z_embeddings(torch.tensor([state_id]).to(self.device))
    dec_intermediate = self.p_z_intermediate(torch.cat([dec_out, z_emb], dim=1))
    x_logits = self.p_decoder.output_proj(dec_intermediate)
    lm_prob = F.softmax(x_logits, dim=-1)
      
    if(self.use_copy):
      _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
      copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
      copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

      out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
      x_logits = (out_prob + 1e-10).log()

    temp_wordlp = torch.log_softmax(x_logits, dim=-1)  

    word_id = bs_init["y_id"]
    log_pw = temp_wordlp[0, word_id].item()
    
    inp = z_emb + self.embeddings(torch.tensor([word_id]).to(self.device))
    node = {"h" : decoder_hidden, "inp" : inp, "prevNode" : None, 
            "word_id" : word_id, "ys" : bs_init["prev_ys"] + [word_id],
            "state_id" : state_id, "zs" : bs_init["prev_zs"] + [state_id],
            "leng" : bs_init["leng"] + 1, 
            "logp" : bs_init["logp"] + log_ps + log_pw,
            "fs_idx" : bs_init["fs_idx"]}
    
    for fs_idx in bs_init["fs_idx"]:
      node["fs_idx"] = fs_idx[0]
      node["bids"] = list(fs_idx[1])
      if bs_init["bt"]:
        beam_tree.update_node(bs_init, node)
      fss[fs_idx[0]]["nodes"] = [(-node['logp'], node)] 

  def beam_search(self, bs_init, beam_tree):
    mem_emb = beam_tree.mem_emb
    mem_mask = beam_tree.mem_mask
    mem = beam_tree.mem
    template = beam_tree.template

    # beam vars
    beam_width = 2 # for y
    beam_w_ = 2 # for z
    topk = 5
    #     max_len = self.max_dec_len
    max_len = 38
    window_size = 3
    
    endnodes = []
    
    # initialize fstates
    fs_dict = make_fst(template, self.config._dataset)
    num_fs = fs_dict["counter"] + 1
    fss = fs_dict["fss"]
  
    t = 0
    break_flag = False
    finished = False

    bt = bs_init["bt"]
    self.init_bst(bs_init, beam_tree, fss, mem_emb, mem_mask, mem)
    
    while not finished:
      if t>= max_len: break
      t += 1
          
      for fs_idx in range(num_fs):
        # current state
        fs = fss[fs_idx]
        # nodes coming from preceding fsm states
        prev_nodes = []
        prev_fstate_count = 0   

        # nodes generated in this step       
        nextnodes = []
        
        # collect nodes
        for prev_fs in fs["prev"]:
          prev_fstate_count += 1
          prev_nodes += fss[prev_fs]["nodes"]
        prev_nodes = sorted(prev_nodes, key=lambda x: x[0])
        
        for elem in range(min(len(prev_nodes), beam_width * prev_fstate_count)):
          score_top, n_top = prev_nodes[elem]
          
          if n_top["word_id"] == self.end_id and n_top["prevNode"] != None:
            endnodes.append((score, n_top))
          
            if len(endnodes) >= number_required:
              break
            else:
              continue
          
          # decoder states
          dec_out, decoder_hidden = self.p_decoder(n_top['inp'].contiguous(), 
            n_top['h'], mem_emb, mem_mask)
          dec_out = dec_out[0]

          if bt:
            beam_tree.update_hidden(n_top, decoder_hidden)
          
          z_logits = self.p_z_proj(dec_out)
          
          # shape: bsz X state_vocab
          state_logp = torch.log_softmax(z_logits, dim=-1)
          
          # z is being controlled
          if fs["yz"] == "z":
            if fs["type"] == "neg":
              state_logp[:, fs["val"]] = float("-Inf")
              log_prob, indexes = torch.topk(state_logp, beam_w_)
            elif fs["val"] == -1:
              log_prob, indexes = torch.topk(state_logp, beam_w_)
            else:
              state_id = fs["val"]
              log_prob = state_logp[:,state_id:state_id+1]
              indexes = torch.tensor([[[state_id]]]).to(self.device)
          # y is being controlled. keep all z
          else:
            log_prob, indexes = torch.topk(state_logp, state_logp.shape[1]) 

          for new_k in range(len(indexes.view(-1))):
            decoded_ts = indexes.view(-1)[new_k]
            log_ps = log_prob[0][new_k].item()
              
            z_emb = self.z_embeddings(torch.tensor([decoded_ts]).to(self.device))
            dec_intermediate = self.p_z_intermediate(torch.cat([dec_out, z_emb], dim=1))
            x_logits = self.p_decoder.output_proj(dec_intermediate)
            lm_prob = F.softmax(x_logits, dim=-1)
              
            if(self.use_copy):
              _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
              copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
              copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

              out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
              x_logits = (out_prob + 1e-10).log()
            
            # make eos invalid if not exit state
            if not fs["exit"]:
              x_logits[:,self.end_id] = float("-Inf")
              
            temp_wordlp = torch.log_softmax(x_logits, dim=-1)
            
            if fs["yz"] == "z":
              # we require the end state to not have any loops and force eos
              # + ? * not allowed
              if fs["exit"]:
                log_prob_w = temp_wordlp[:,self.end_id:self.end_id+1]
                indexes_w = torch.tensor([[[self.end_id]]]).to(self.device)
              else:
                log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)
            # if y is controlled, force the word
            else:
              word_id = fs["val"]
              log_prob_w = temp_wordlp[:,word_id:word_id+1]
              indexes_w = torch.tensor([[[word_id]]]).to(self.device)
            
            temp_ww = []
            temp_pp = []  
              
            for elem1, elem2 in zip(indexes_w.cpu().view(-1), log_prob_w.cpu().view(-1)):
              temp_ww.append(elem1)
              temp_pp.append(elem2)

              if len(temp_ww) >= beam_width:
                break
          
            for new_k_w in range(len(temp_ww)):
              decoded_tw = temp_ww[new_k_w].view(-1)[0]
              log_pw = temp_pp[new_k_w].item()
    
              inp = z_emb + self.embeddings(torch.tensor([decoded_tw]).to(self.device))
              
              word_id = decoded_tw.item()
              state_id = decoded_ts.item()
              node = {'h': decoder_hidden, 'inp': inp, 'prevNode': {}, 
                      'word_id': word_id, 'ys' : n_top["ys"] + [word_id],
                      'state_id': state_id, 'zs' : n_top["zs"] + [state_id],
                      'logp': n_top['logp'] + log_ps + log_pw, 
                      'leng': n_top['leng'] + 1, "fs_idx" : fs_idx,
                      "bids" : n_top["bids"] + [fs["bid"]]}
              if bt:
                beam_tree.update_node(bs_init, node)

              score = -node['logp']
              
              if node['word_id'] == self.end_id and node['prevNode'] != None:
                endnodes.append((score, node))
                continue
              if node["leng"] >= max_len and fs["exit"]:
                endnodes.append((score, node))
                continue
              else:
                nextnodes.append((score, node))
      
        next_nodes = []
        for i in range(len(nextnodes)):
          score, nn_ = nextnodes[i]
          next_nodes.append((score, nn_))

        next_nodes = sorted(next_nodes, key=lambda x: x[0])

        fss[fs_idx]["next_nodes"] = next_nodes
          
      for fs_idx in range(num_fs):
        fss[fs_idx]["nodes"] = fss[fs_idx]["next_nodes"]
        
    return endnodes

  def endnodes_to_utterances(self, endnodes):
    utterances_w = []
    utterances_s = []
    scores_result = []
    utterances_b = []
  
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
      utterance_w = n["ys"][1:]
      utterance_s = n["zs"][1:]
      utterance_b = n["bids"][1:] 
      utterances_w.append(utterance_w)
      utterances_s.append(utterance_s)
      utterances_b.append(utterance_b)
      scores_result.append(score)

    return utterances_w, utterances_s, scores_result, utterances_b

  def prepare_dec_io(self, 
    z_sample_ids, z_sample_emb, sentences, x_lambd):
    """Prepare the decoder output g based on the inferred z from the CRF 

    Args:
      x_lambd: word dropout ratio. 1 = all dropped

    Returns:
      dec_inputs
      dec_targets_x
      dec_targets_z
    """
    batch_size = sentences.size(0)
    max_len = sentences.size(1)
    device = sentences.device

    sent_emb = self.embeddings(sentences)
    z_sample_emb[:, 0] *= 0. # mask out z[0]

    # word dropout ratio = x_lambd. 0 = no dropout, 1 = all drop out
    m = Uniform(0., 1.)
    mask = m.sample([batch_size, max_len]).to(device)
    mask = (mask > x_lambd).float().unsqueeze(2)

    dec_inputs = z_sample_emb + sent_emb * mask
    dec_inputs = dec_inputs[:, :-1]

    dec_targets_x = sentences[:, 1:]
    dec_targets_z = z_sample_ids[:, 1:]
    return dec_inputs, dec_targets_x, dec_targets_z
