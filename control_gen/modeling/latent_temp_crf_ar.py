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
from transformers import TapasModel

# torch.autograd.set_detect_anomaly(True)

class LatentTemplateCRFAR(nn.Module):
  """The latent template CRF autoregressive version, table to text setting"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.device = config.device

    self.max_y_len = config.max_y_len
    self.max_x_len = config.max_x_len

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.seg_id = config.seg_id

    self.vocab_size = config.vocab_size

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size

    ## Model parameters
    # emb
    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)

    # tapas model
    self.tapas = TapasModel.from_pretrained('google/tapas-base')
    tapas_extended_vocab_size = 30542
    self.tapas.resize_token_embeddings(tapas_extended_vocab_size)

    # latent 
    self.z_crf = LinearChainCRF(config)

    # dec 
    self.p_dec_init_state_proj_h = nn.Linear(
      config.state_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.state_size, config.lstm_layers * config.state_size)
    self.p_decoder = LSTMDecoder(config)

    # copy 
    self.p_copy_attn = Attention(
      config.state_size, config.state_size, config.state_size)
    self.p_copy_g = nn.Linear(config.state_size, 1)
    
    # table_projection
    self.x_projection = nn.Linear(config.tapas_state_size + config.embedding_size, config.state_size)

    # dec z proj
    self.z_logits_attention = Attention(
      config.state_size, config.state_size, config.state_size)

    self.p_z_intermediate = nn.Linear(2 * config.state_size, config.state_size)
    
    # posterior regularization
    self.pr = config.pr
    self.pr_inc_lambd = config.pr_inc_lambd
    self.pr_exc_lambd = config.pr_exc_lambd
    
    # crf_attn
    self.crf_y_header = nn.Linear(config.tapas_state_size, config.tapas_state_size)
    self.crf_header_self = nn.Linear(config.tapas_state_size, config.tapas_state_size)
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

  def encode_x(self, input_ids, attention_masks, token_type_ids, tables):
    outputs = self.tapas(input_ids, attention_masks, token_type_ids)
    outputs = outputs.last_hidden_state[:,2:]
    table_embed = self.embeddings(tables)
    outputs = torch.cat([outputs, table_embed], dim=-1)
    outputs = self.x_projection(outputs)
    return outputs

  def embed_z(self, z_ids, z_embeddings):
    # z_ids : batch_size X seq_len
    # z_embeddings = encoded_x : batch_size X vocab_size X state_size

    batch_size = z_ids.shape[0]
    z_len = z_ids.shape[1]
    z_vocab_size = z_embeddings.shape[1]

    z_one_hot = tmu.ind_to_one_hot(z_ids.view(-1), z_vocab_size).float()
    z_one_hot = z_one_hot.view(batch_size, z_len, z_vocab_size)

    z_embed = torch.bmm(z_one_hot, z_embeddings)

    return z_embed

  def get_crf_scores(self, input_ids, attention_mask, token_type_ids, 
    y_header_mask, header_self_mask):
    outputs = self.tapas(input_ids, attention_mask, token_type_ids).last_hidden_state    
    
    y_header = torch.bmm(
        self.crf_y_header(outputs[:,:self.max_y_len+1]),
        outputs[:,self.max_y_len+2:].transpose(1,2)
    )
    y_header = y_header.masked_fill(y_header_mask == 0, -1e9)
    
    header_self = torch.bmm(
        self.crf_header_self(outputs[:,self.max_y_len+2:]),
        outputs[:,self.max_y_len+2:].transpose(1,2)
    )
    header_self = header_self.masked_fill(header_self_mask == 0, -1e9)
    
    return header_self, y_header
    
  def forward(self, data_dict, tau, x_lambd, z_beta, bi=None):
    out_dict = {}

    sentences = data_dict["sentences"]
    sent_lens = data_dict["sent_lens"]

    batch_size = sentences.size(0)
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    max_len = sent_lens.max().item()
    sent_mask = sent_mask[:, :max_len]

    z_transition_scores, z_emission_scores = self.get_crf_scores(
      data_dict["xy_input_id_lst"],
      data_dict["xy_attn_mask_lst"],
      data_dict["xy_token_type_id_lst"],
      data_dict["y_header_mask_lst"],
      data_dict["header_self_mask_lst"])

    z_emission_scores = z_emission_scores[:, :max_len]
        
    # PR
    if self.pr:
      zcs = data_dict["zcs"]
      max_z = data_dict["max_z_lst"]
      pr_inc_loss, pr_exc_loss = self.compute_pr(z_emission_scores,
        z_transition_scores, zcs[:, :max_len], sent_mask, sent_lens,
        data_dict["max_z_lst"])

      out_dict["pr_inc_val"] = tmu.to_np(pr_inc_loss)
      out_dict["pr_exc_val"] = tmu.to_np(pr_exc_loss)
      out_dict["pr_inc_loss"] = self.pr_inc_lambd * tmu.to_np(pr_inc_loss)
      out_dict["pr_exc_loss"] = self.pr_exc_lambd * tmu.to_np(pr_exc_loss)
      loss -= self.pr_inc_lambd * pr_inc_loss + self.pr_exc_lambd * pr_exc_loss
        
    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, z_transition_scores,
      sent_lens).mean()
    
    #e2
#     ent_z = self.z_crf.entropy2(z_emission_scores, z_transition_scores,
#       sent_lens).mean()
    
#     loss += z_beta * ent_z
#     out_dict['ent_z'] = tmu.to_np(ent_z)
#     out_dict['ent_z_loss'] = z_beta * tmu.to_np(ent_z)

#     ent_weight = 0.025 * (1 - z_beta) + 0.001
#     ent_weight = 0.025 * z_beta
    ent_weight = 0.025
    loss += ent_weight * ent_z
    out_dict['ent_weight'] = ent_weight
    out_dict['ent_z'] = tmu.to_np(ent_z)
    out_dict['ent_z_loss'] = ent_weight * tmu.to_np(ent_z)
    
     # encode table
    encoded_x = self.encode_x(
      data_dict["x_input_id_lst"],
      data_dict["x_attn_mask_lst"],
      data_dict["x_token_type_id_lst"],
      data_dict["tables"],
      )
    
    z_sample_ids, z_sample, _ = self.z_crf.rsample(
        z_emission_scores, z_transition_scores, sent_lens, tau,
        return_switching=True)
    
    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 

    # embed z using the encoded table
    z_embed = self.embed_z(z_sample_ids, encoded_x)

    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids, z_embed)
    
#     # r2 
#     z_sample = self.z_crf.rsample2(
#         z_emission_scores, z_transition_scores, sent_lens, tau)
    
#     z_sample_ids = z_sample.argmax(-1)
#     # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
#     z_sample_ids.masked_fill_(~sent_mask, 0) 
    
#     z_sample_emb = torch.bmm(z_sample, encoded_x)
    
    if bi is not None and bi % 200 == 0:
        print("batch : " + str(bi), flush=True)
        print(z_sample_ids[:2])
        print(zcs[:2, :max_len])

    sentences = sentences[:, :max_len]
    p_log_prob, p_log_prob_x, p_log_prob_z, z_acc, _ = self.decode_train(
      z_sample_ids, z_sample_emb, sent_lens, sentences, data_dict["tables"],
      x_lambd, encoded_x, data_dict["header_mask_lst"], 
      data_dict["x_mask_lst"], z_beta)

    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['z_acc'] = z_acc.item()
    loss += p_log_prob

    # turn maximization to minimization
    loss = -loss 

    out_dict['loss'] = tmu.to_np(loss)
    return loss, out_dict

  def compute_pr(self, emission_scores, transition_scores, zcs, sent_mask, 
    sent_lens, max_z):
    max_num_contraints = max_z.max().item() + 1

    # edge potentials
    all_scores = self.z_crf.calculate_all_scores(emission_scores, 
      transition_scores)
    
    # Linear Chain CRF
    dist = LC(all_scores.transpose(3,2), (sent_lens + 1).float())
    
    # marginals : [batch, max_len, state_size]
    marginals = dist.marginals.sum(-1)
    rel_marginals = marginals[:, :, :max_num_contraints]
    
    # filters for the constrained z states
    filters = torch.arange(max_num_contraints).view(1,1,max_num_contraints).to(self.device)
    
    # check if state is used
    check = zcs.unsqueeze(-1) == filters

    # is the z valid? dynamic, so not same for all
    is_valid_z = filters <= max_z.unsqueeze(-1).unsqueeze(-1)
    
    # computer loss
    # {1 - q(z = sigma(f) | x, y)} if f is used else {q(z = sigma(f) | x, y)}
    loss = (check.float() - rel_marginals).abs() * sent_mask.unsqueeze(-1).float() * is_valid_z.float()
    inc_loss = (1 - rel_marginals).abs() * check.float() * sent_mask.unsqueeze(-1).float() * is_valid_z.float()
    exc_loss = rel_marginals * (1 - check.float()) * sent_mask.unsqueeze(-1).float() * is_valid_z.float()
    
    # take average
    final_inc_loss = inc_loss.sum() / check.float().sum()
    final_exc_loss = exc_loss.sum() / (sent_lens.sum() - check.float().sum())
    
    return final_inc_loss, final_exc_loss

  def decode_train(self, z_sample_ids, z_sample_emb, sent_lens,
    sentences, tables, x_lambd, encoded_xs, header_masks, x_masks, z_beta):
    device = z_sample_ids.device
    state_size = self.state_size
    batch_size = sentences.size(0)

    dec_inputs, dec_targets_x, dec_targets_z = self.prepare_dec_io(
      z_sample_ids, z_sample_emb, sentences, x_lambd)
    max_len = dec_inputs.size(1)

    # average of table encoding
#     mem_enc = encoded_xs * x_masks.unsqueeze(-1).float()
#     mem_enc = mem_enc.sum(dim=1) / x_masks.sum(dim=1, keepdim=True)

    mem_enc = encoded_xs * header_masks.unsqueeze(-1).float()
    mem_enc = mem_enc.sum(dim=1) / header_masks.sum(dim=1, keepdim=True)

    state = self.init_state(mem_enc)
    
    dec_inputs = dec_inputs.transpose(1, 0) # [T, B, S]
    dec_targets_x = dec_targets_x.transpose(1, 0) # [T, B]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_sample_emb = z_sample_emb[:, 1:].transpose(1, 0) # start from z[1]

    log_prob_x, log_prob_z, dec_outputs, z_pred = [], [], [], []

    for i in range(max_len): 
#       dec_out, state = self.p_decoder(
#           dec_inputs[i], state, encoded_xs, x_masks)
      dec_out, state = self.p_decoder(
          dec_inputs[i], state, encoded_xs, header_masks)

      dec_out = dec_out[0]

      # predict z 
      _, z_logits = self.z_logits_attention(dec_out, encoded_xs, header_masks)
      z_logits = (z_logits + 1e-10).log()
      z_pred.append(z_logits.argmax(dim=-1))
        
      log_prob_z_i = -F.cross_entropy(
        z_logits, dec_targets_z[i], reduction='none')
      log_prob_z.append(log_prob_z_i)

      # predict x based on z 
      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_sample_emb[i]], dim=1))
      x_logits = self.p_decoder.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      _, copy_dist = self.p_copy_attn(dec_intermediate, encoded_xs, x_masks)
      copy_prob = tmu.batch_index_put(copy_dist, tables, self.vocab_size)
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

    log_prob_x = log_prob_x.sum() / sent_lens.sum()
    log_prob_z = log_prob_z.sum() / sent_lens.sum()
    z_beta = 1
    log_prob = log_prob_x + z_beta * log_prob_z

    # acc 
    z_pred = torch.stack(z_pred).transpose(1, 0) # [B, T]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_positive = tmu.mask_by_length(z_pred == dec_targets_z, sent_lens).sum() 
    z_acc = z_positive / sent_lens.sum()
    
    return (
      log_prob, log_prob_x, log_prob_z, z_acc, log_prob_step)

  def infer(self, data_dict):
    out_dict = {}
    
    encoded_x = self.encode_x(
      data_dict["x_input_id_lst"],
      data_dict["x_attn_mask_lst"],
      data_dict["x_token_type_id_lst"],
      data_dict["tables"],
      )

    # decoding 
    predictions_x, predictions_z = self.decode_infer(
      data_dict["tables"], encoded_x, 
      data_dict["header_mask_lst"],
      data_dict["x_mask_lst"]
      )

    out_dict['predictions'] = tmu.to_np(predictions_x)
    out_dict['predictions_z'] = tmu.to_np(predictions_z)
    pred_lens_ = tmu.seq_ends(predictions_x, self.end_id) + 1
    out_dict['pred_lens'] = tmu.to_np(pred_lens_)
    return out_dict

  def decode_infer(self, tables, encoded_xs, header_masks, x_masks):    
    batch_size = tables.size(0)
    device = tables.device

    predictions_x, predictions_z = [], []

    inp = self.embeddings(
      torch.zeros(batch_size).to(device).long() + self.start_id)
    
    # average of table encoding
#     mem_enc = encoded_xs * x_masks.unsqueeze(-1).float()
#     mem_enc = mem_enc.sum(dim=1) / x_masks.sum(dim=1, keepdim=True)
#     state = self.init_state(mem_enc)
    mem_enc = encoded_xs * header_masks.unsqueeze(-1).float()
    mem_enc = mem_enc.sum(dim=1) / header_masks.sum(dim=1, keepdim=True)
    state = self.init_state(mem_enc)

    for i in range(self.max_y_len+1): 
#       dec_out, state = self.p_decoder(inp, state, encoded_xs, x_masks)
#       dec_out = dec_out[0]
      dec_out, state = self.p_decoder(inp, state, encoded_xs, header_masks)
      dec_out = dec_out[0]

      # predict z 
      _, z_logits = self.z_logits_attention(dec_out, encoded_xs, header_masks)
      z_logits = (z_logits + 1e-10).log()

      z = z_logits.argmax(-1)

      # predict x based on z 
      z_emb = self.embed_z(z.unsqueeze(-1), encoded_xs)
      z_emb = z_emb.squeeze(1)

      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_emb], dim=1))
      x_logits = self.p_decoder.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      _, copy_dist = self.p_copy_attn(dec_intermediate, encoded_xs, x_masks)
      copy_prob = tmu.batch_index_put(copy_dist, tables, self.vocab_size)
      copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

      out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
      x_logits = (out_prob + 1e-10).log()

      x = x_logits.argmax(-1)

      inp = z_emb + self.embeddings(x)

      predictions_x.append(x)
      predictions_z.append(z)

    predictions_x = torch.stack(predictions_x).transpose(1, 0)
    predictions_z = torch.stack(predictions_z).transpose(1, 0)
    return predictions_x, predictions_z

  def posterior_infer(self, data_dict):
    sent_lens = data_dict["sent_lens"]
    device = sent_lens.device

    max_len = sent_lens.max().item()

    z_transition_scores, z_emission_scores = self.get_crf_scores(
      data_dict["xy_input_id_lst"],
      data_dict["xy_attn_mask_lst"],
      data_dict["xy_token_type_id_lst"],
      data_dict["y_header_mask_lst"],
      data_dict["header_self_mask_lst"])

    z_emission_scores = z_emission_scores[:, :max_len]
    
    out = self.z_crf.argmax(z_emission_scores, z_transition_scores, 
                            sent_lens).tolist()
    out_list = [out[i][:sent_lens[i].item()] for i in range(len(out))]
    return out_list

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
