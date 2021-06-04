"""d: Dynamic with z"""

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
    
#     #e2
#     ent_z = self.z_crf.entropy2(z_emission_scores, z_transition_scores,
#       sent_lens).mean()
    
    ent_weight = z_beta
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

#     rel_marginals = marginals[:, :, 10:max_num_contraints+10]
#     zcs = zcs - 10
    
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
    mem_enc = encoded_xs * x_masks.unsqueeze(-1).float()
    mem_enc = mem_enc.sum(dim=1) / x_masks.sum(dim=1, keepdim=True)

    state = self.init_state(mem_enc)
    
    dec_inputs = dec_inputs.transpose(1, 0) # [T, B, S]
    dec_targets_x = dec_targets_x.transpose(1, 0) # [T, B]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_sample_emb = z_sample_emb[:, 1:].transpose(1, 0) # start from z[1]

    log_prob_x, log_prob_z, dec_outputs, z_pred = [], [], [], []

    for i in range(max_len): 
      dec_out, state = self.p_decoder(
          dec_inputs[i], state, encoded_xs, x_masks)

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
    mem_enc = encoded_xs * x_masks.unsqueeze(-1).float()
    mem_enc = mem_enc.sum(dim=1) / x_masks.sum(dim=1, keepdim=True)
    state = self.init_state(mem_enc)

    for i in range(self.max_y_len+1): 
      dec_out, state = self.p_decoder(inp, state, encoded_xs, x_masks)
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

  def infer2(self, data_dict, templates, return_bt =  False):
    out_dict = {}
    
    encoded_x = self.encode_x(
      data_dict["x_input_id_lst"],
      data_dict["x_attn_mask_lst"],
      data_dict["x_token_type_id_lst"],
      data_dict["tables"],
      )


    # decoding 
    pred_y, pred_z, pred_score, beam_trees, bts, bid, pw, ps = self.decode_infer2(
      data_dict["tables"], encoded_x, 
      data_dict["header_mask_lst"],
      data_dict["x_mask_lst"],
      templates, 
      return_bt)

    out_dict['pred_y'] = pred_y
    out_dict['pred_z'] = pred_z
    out_dict['pred_score'] = pred_score
    out_dict["beam_trees"] = beam_trees
    out_dict["bts"] = bts
    out_dict["regex_alignment"] = bid
    out_dict["pw"] = pw
    out_dict["ps"] = ps
    return out_dict

  def decode_infer2(self, tables, encoded_xs, header_masks, x_masks, templates, return_bt):    
    batch_size = tables.size(0)
    device = tables.device
    
    decoded_batch = []
    decoded_score = []
    decoded_states = []
    decoded_bid = []
    decoded_pw = []
    decoded_ps = []
    beam_trees = []
    bts = []


    predictions_x, predictions_z = [], []
    inp = self.embeddings(
      torch.zeros(batch_size).to(self.device).long() + self.start_id)

    mem_enc = encoded_xs * x_masks.unsqueeze(-1).float()
    mem_enc = mem_enc.sum(dim=1) / x_masks.sum(dim=1, keepdim=True)
    state = self.init_state(mem_enc)
    
    # assume use_src_info=True
    state = self.init_state(mem_enc)

    mem_emb = encoded_xs
    mem_mask = x_masks
    mem = tables
    
    beams_batch = []
    
    for idx in range(batch_size):
      inp_i = inp[idx:idx + 1]
      state_i = (state[0][:, idx:idx + 1, :].contiguous(), state[1][:, idx:idx + 1, :].contiguous())
      mem_emb_i = mem_emb[idx:idx+1]
      mem_mask_i = mem_mask[idx:idx+1]
      mem_i = mem[idx:idx+1]
    
      template = templates[idx]

      beam_tree = BeamTree(inp_i, state_i, mem_emb_i, mem_mask_i, mem_i, template, header_masks[idx:idx+1])
      bs_init = beam_tree.init_bs_init(return_bt)
    
      endnodes = self.beam_search(bs_init, beam_tree)

      beam_trees.append(beam_tree)
      bts.append(beam_tree.get_bt(bs_init))

      utterances_w, utterances_s, scores_result, utterances_b, \
       utterances_pw, utterances_ps = self.endnodes_to_utterances(
        endnodes)
        
      decoded_batch.append(utterances_w)
      decoded_states.append(utterances_s)
      decoded_score.append(scores_result)
      decoded_bid.append(utterances_b)
      decoded_pw.append(utterances_pw)
      decoded_ps.append(utterances_ps)
    
    pred_y = []
    pred_z = []
    pred_score = []
    bid = []
    pw = []
    ps = []
    for batch_idx in range(len(decoded_batch)):
        if decoded_batch[batch_idx] == []:
            pred_y.append([])
            pred_z.append([])
            pred_score.append(-float("inf"))
            bid([])
        else:
#             pred_y.append(decoded_batch[batch_idx][0][1:-1])
#             pred_z.append(decoded_states[batch_idx][0][1:-1])
#             pred_score.append(decoded_score[batch_idx][0])
#             bid.append(decoded_bid[batch_idx][0][1:-1])
            pred_y.append(decoded_batch[batch_idx][0:15])
            pred_z.append(decoded_states[batch_idx][0:15])
            pred_score.append(decoded_score[batch_idx][0:15])
            bid.append(decoded_bid[batch_idx][0:15])
            pw.append(decoded_pw[batch_idx][0:15])
            ps.append(decoded_ps[batch_idx][0:15])
            
    return pred_y, pred_z, pred_score, beam_trees, bts, bid, pw, ps

  def init_bst(self, bs_init, beam_tree, fss, mem_emb, mem_mask, mem, header_mask):
    if bs_init["fs_idx"][0] == 0:
        node = {'h': bs_init["h"], 'inp': bs_init["inp"], 'prevNode': None, 
                'word_id': -1, "ys" : bs_init["prev_ys"] + [-1],
                'state_id': -1, "zs" : bs_init["prev_zs"] + [-1],
                'logp': bs_init["logp"], 'leng': bs_init["leng"] + 1, 
                "fs_idx" : 0, "bids" : list(bs_init["fs_idx"][1]),
                "log_ps" : bs_init["prev_ys"] + [-1],
                "log_pw" : bs_init["prev_ys"] + [-1]}
        if bs_init["bt"]:
          beam_tree.update_node(bs_init, node)
        fss[0]["nodes"] = [(-node['logp'], node)]
        return 

  def beam_search(self, bs_init, beam_tree):
    mem_emb = beam_tree.mem_emb
    mem_mask = beam_tree.mem_mask
    header_mask = beam_tree.header_mask
    mem = beam_tree.mem
    template = beam_tree.template
    
    # beam vars
    beam_width = 2 # for y
    beam_w_ = 2 # for z
    topk = 5
    #     max_len = self.max_dec_len
    max_len = 50
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
    self.init_bst(bs_init, beam_tree, fss, mem_emb, mem_mask, mem, 
      header_mask)
    
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
          
          _, z_logits = self.z_logits_attention(dec_out, mem_emb, header_mask)
          z_logits = (z_logits + 1e-10).log()

          z = z_logits.argmax(-1)
          
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
              
            z_emb = self.embed_z(torch.tensor([decoded_ts]).to(self.device).unsqueeze(-1), mem_emb)
            z_emb = z_emb.squeeze(1)
            dec_intermediate = self.p_z_intermediate(torch.cat([dec_out, z_emb], dim=1))
            x_logits = self.p_decoder.output_proj(dec_intermediate)
            lm_prob = F.softmax(x_logits, dim=-1)
              
            _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
            copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
            copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

            out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
            x_logits = (out_prob + 1e-10).log()
            
            # make eos invalid if not exit state
            if not fs["exit"]:
              x_logits[:,self.end_id] = float("-Inf")
              
            temp_wordlp = torch.log_softmax(x_logits, dim=-1)
            
#             print(temp_wordlp[:,self.end_id])
            
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
                      "bids" : n_top["bids"] + [fs["bid"]],
                      "log_pw" : n_top["log_pw"] + [log_pw],
                      "log_ps" : n_top["log_ps"] + [log_ps]}
                
              node1 = {'h': decoder_hidden, 'inp': inp, 'prevNode': {}, 
                      'word_id': word_id, 'ys' : n_top["ys"] + [word_id],
                      'state_id': state_id, 'zs' : n_top["zs"] + [state_id],
                      'logp': n_top['logp'] + log_pw, 
                      'leng': n_top['leng'] + 1, "fs_idx" : fs_idx,
                      "bids" : n_top["bids"] + [fs["bid"]],
                      "log_pw" : n_top["log_pw"] + [log_pw],
                      "log_ps" : n_top["log_ps"] + [log_ps]}
              
              node = node1
    
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
    utterances_pw = []
    utterances_ps = []
  
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
      utterance_w = n["ys"][1:]
      utterance_s = n["zs"][1:]
      utterance_b = n["bids"][1:] 
      utterance_pw = n["log_pw"][1:] 
      utterance_ps = n["log_ps"][1:]
      utterances_w.append(utterance_w)
      utterances_s.append(utterance_s)
      utterances_b.append(utterance_b)
      utterances_pw.append(utterance_pw)
      utterances_ps.append(utterance_ps)
      scores_result.append(score)

    return utterances_w, utterances_s, scores_result, utterances_b, \
            utterances_pw, utterances_ps

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
    
    x_lambd = 0

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
