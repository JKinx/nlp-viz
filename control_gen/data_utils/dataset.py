
import csv
import numpy as np 

from nltk import word_tokenize
from tqdm import tqdm

from collections import defaultdict
from .dataset_base import DatasetBase
from . import nlp_pipeline as nlpp

import pickle

from num2words import num2words
from copy import deepcopy as dc

def normalize_set(dset, word2id, max_sent_len, max_mem_len, word2id_zcs):
  """Normalize the train/ dev/ test set
  
  Args:
    dset: the output of the `read_data` function
    word2id: the output of the `build_vocab` function
    max_sent_len: an integer
    max_mem_len: an integer
  """
  data_sents = [d[1] for d in dset]
  data_templates = [d[2] for d in dset]
  
  # z_constraints
  data_zcs = [d[3] for d in dset]

  set_keys, set_vals = [], []
  for tb, _, _, _ in dset:
    keys = [k for k, _ in tb]
    set_keys.append(keys)
    vals = [v for _, v in tb]
    set_vals.append(vals)
  set_keys, set_lens = nlpp.normalize(
    set_keys, word2id, max_mem_len, add_start_end=False)
  set_vals, _ = nlpp.normalize(
    set_vals, word2id, max_mem_len, add_start_end=False)
  sentences, sent_lens = nlpp.normalize(data_sents, word2id, max_sent_len)
  templates, _ = nlpp.normalize(data_templates, word2id, max_sent_len)
    
  # z_constraints
  zcs, _ = nlpp.normalize(data_zcs, word2id_zcs, max_sent_len)

  return set_keys, set_vals, set_lens, sentences, templates, sent_lens, zcs

def normalize_kv(data_kvs, word2id, max_mem_len):
    set_keys, set_vals = [], []
    for tb in data_kvs:
        keys = [k for k, _ in tb]
        set_keys.append(keys)
        vals = [v for _, v in tb]
        set_vals.append(vals)
    set_keys, set_lens = nlpp.normalize(
    set_keys, word2id, max_mem_len, add_start_end=False)
    set_vals, _ = nlpp.normalize(
    set_vals, word2id, max_mem_len, add_start_end=False)
    return set_keys, set_vals, set_lens

def normalize_sent(data_sents, word2id, max_sent_len):
    return nlpp.normalize(data_sents, word2id, max_sent_len, add_start_end=False)

def read_data(dpath):
  """Read the raw e2e data
  
  Args:
    dpath: path to the .csv file

  Returns:
    dataset: a list of (tb, s, st) triple
      tb = the table, a list of (k, v) tuple
        k = the key 
        v = the value
      s = the sentence
      st = the sentence template 
      all characters are changed to lower
  """
  print('reading %s' % dpath)

  data_raw = pickle.load(open(dpath, "rb"))

  dataset = []

  for data_id in range(len(data_raw["xs"])):
    x = data_raw["xs"][data_id]

    t = []
    for k in x:
      v = x[k]
      for i, vi in enumerate(v.split()):
        t.append(('_' + k.replace(' ', '_') + '_' + str(i), vi))
    
    s = data_raw["ys"][data_id].split(" ")

    zc = data_raw["zs"][data_id].split(" ")

    dataset.append((t, s, dc(s), zc))
  print("%d cases" % len(dataset))
  return dataset, data_raw["tapas_input_ids"], data_raw["tapas_attn_mask"], \
          data_raw["tapas_token_type_ids"], data_raw["y_lengths"]

def prepare_inference(keys, vals, sents, pad_id=0):
  """Prepare the data format for inference
  
  Args:
    key:
    vals: 
    sents:

  Returns: 
    keys_inf:
    vals_inf:
    references: 
  """
  keys_inf, vals_inf, mem_lens, references = [], [], [], []
    
  i = 0
  for k, v, s in zip(keys, vals, sents):
    if(i == 0):
      r = [s[1:]]
      prev_k = k
      prev_v = v
      i += 1
    else:        
      keys_inf.append(prev_k)
      vals_inf.append(prev_v)
      mem_lens.append(np.sum(np.array(prev_k) != pad_id))
      references.append(r)

      r = [s[1:]]
      prev_k = k
      prev_v = v

  keys_inf.append(prev_k)
  vals_inf.append(prev_v)
  mem_lens.append(np.sum(prev_k != pad_id))
  references.append(r)
  return keys_inf, vals_inf, mem_lens, references

class Dataset(DatasetBase):

  def __init__(self, config):
    super(Dataset, self).__init__()
    
    self.model_name = config.model_name
    self.data_path = config.data_path
    self.word2id = config.word2id
    self.id2word = config.id2word
    self.pad_id = config.pad_id
    self.key2id = {}
    self.id2key = {}
    self.val2id = {}
    self.id2val = {}
    self.max_sent_len = config.max_sent_len
    self.max_mem_len = config.max_mem_len
    self.max_bow_len = config.max_bow_len
    self.latent_vocab_size = config.latent_vocab_size

    # model configuration 
    self.auto_regressive = config.auto_regressive

    self._dataset = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    self._reset_ptr = {"train": False, "dev": False, "test": False}
    
    self.num_pr_constraints = config.num_pr_constraints
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  @property
  def key_vocab_size(self): return len(self.key2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname]['keys'])

  def num_batches(self, setname, batch_size):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  def _update_ptr(self, setname, batch_size):
    if(self._reset_ptr[setname]):
      ptr = 0
      self._reset_ptr[setname] = False
    else: 
      ptr = self._ptr[setname]
      ptr += batch_size
      if(ptr + batch_size >= self.dataset_size(setname)):
        self._reset_ptr[setname] = True
    self._ptr[setname] = ptr
    return 

  def build(self):
    """Build the dataset"""
    max_sent_len = self.max_sent_len
    max_mem_len = self.max_mem_len

    ## read the sentences
    trainset = read_data(self.data_path['train'])
    devset = read_data(self.data_path['dev'])
    testset = read_data(self.data_path['test'])

    ## build vocabulary 
    train_sents = [t[1] for t in trainset[0]]
    word2id, id2word, _ = nlpp.build_vocab(train_sents, 
      word2id=self.word2id, id2word=self.id2word, vocab_size_threshold=1)
    train_keys = []
    train_vals = []
    for tb, _, _, _ in trainset[0]:
      keys = [k for k, _ in tb]
      train_keys.extend(keys)
      vals = [v for _, v in tb]
      train_vals.extend(vals)

    if "dateSet" in self.data_path["train"]:
        train_keys += ["_day_", "_month_", "_year_"]
    # join key vocab and the word vocab, but keep a seperate key dict
    self.word2id, self.id2word, self.key2id, self.id2key =\
      nlpp.extend_vocab_with_keys(word2id, id2word, train_keys)
    
    # join vals vocab and the word vocab, but keep a seperate key dict
    self.word2id, self.id2word, self.val2id, self.id2val =\
      nlpp.extend_vocab_with_keys(self.word2id, self.id2word, train_vals)
    
    word2id_zcs = {"-1" : self.num_pr_constraints, 
                   '_PAD': self.num_pr_constraints, '_GOO': self.num_pr_constraints, 
                   '_EOS': self.num_pr_constraints, '_UNK': self.num_pr_constraints, 
                   '_SEG': self.num_pr_constraints}
    for z_id in range(self.num_pr_constraints):
      word2id_zcs[str(z_id)] = z_id

    ## normalize the dataset 
    (train_keys, train_vals, train_mem_lens, train_sentences, train_templates, 
      train_sent_lens, train_zcs) = normalize_set(
        trainset[0], self.word2id, max_sent_len, max_mem_len, word2id_zcs)
    train_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in train_sentences]
    (dev_keys, dev_vals, dev_mem_lens, dev_sentences, dev_templates, 
      dev_sent_lens, dev_zcs) = normalize_set(
        devset[0], self.word2id, max_sent_len, max_mem_len, word2id_zcs)
    dev_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in dev_sentences]
    (test_keys, test_vals, test_mem_lens, test_sentences, test_templates, 
      test_sent_lens, test_zcs) = normalize_set(
        testset[0], self.word2id, max_sent_len, max_mem_len, word2id_zcs)
    test_bow = [nlpp.sent_to_bow(s, self.max_bow_len) for s in test_sentences]
    
    print("train_len %i" % len(train_keys))
    print("dev_len %i" % len(dev_keys))
    print("test_len %i" % len(test_keys))

    ## Prepare the inference format
    dev_keys_inf, dev_vals_inf, dev_lens_inf, dev_references =\
      prepare_inference(dev_keys, dev_vals, dev_sentences)
    print('%d processed dev cases' % len(dev_keys_inf))
    test_keys_inf, test_vals_inf, test_lens_inf, test_references =\
      prepare_inference(test_keys, test_vals, test_sentences)
    print('%d processed test cases' % len(test_keys_inf))

    ## finalize
    self._dataset = { "train": { 
                        'sentences': train_sentences,
                        'sent_bow': train_bow, 
                        'templates': train_templates,
                        'sent_lens': train_sent_lens,
                        'keys': train_keys,
                        'vals': train_vals,
                        'mem_lens': train_mem_lens,
                        'zcs' : train_zcs,
                        't_input_ids' : trainset[1],
                        't_attn_mask' : trainset[2],
                        't_token_type_ids' : trainset[3]}, 
                      "dev_casewise": { 
                        'sentences': dev_sentences,
                        'sent_bow': dev_bow, 
                        'templates': dev_templates,
                        'sent_lens': dev_sent_lens,
                        'keys': dev_keys,
                        'vals': dev_vals,
                        'mem_lens': dev_mem_lens,
                        'zcs' : dev_zcs,
                        't_input_ids' : devset[1],
                        't_attn_mask' : devset[2],
                        't_token_type_ids' : devset[3]}, 
                      'dev': {
                        'keys': dev_keys_inf,
                        'vals': dev_vals_inf,
                        'mem_lens': dev_lens_inf, 
                        'references': dev_references,
                        'zcs' : dev_zcs,
                        't_input_ids' : devset[1],
                        't_attn_mask' : devset[2],
                        't_token_type_ids' : devset[3]},
                      "test_casewise": { 
                        'sentences': test_sentences,
                        'sent_bow': test_bow, 
                        'templates': test_templates,
                        'sent_lens': test_sent_lens,
                        'keys': test_keys,
                        'vals': test_vals,
                        'mem_lens': test_mem_lens,
                        'zcs' : test_zcs,
                        't_input_ids' : testset[1],
                        't_attn_mask' : testset[2],
                        't_token_type_ids' : testset[3]},
                      'test': {
                        'keys': test_keys_inf,
                        'vals': test_vals_inf,
                        'mem_lens': test_lens_inf, 
                        'references': test_references,
                        'zcs' : test_zcs,
                        't_input_ids' : testset[1],
                        't_attn_mask' : testset[2],
                        't_token_type_ids' : testset[3]}
                      }
    return 

  def batch_kv(self, kv_list):
    keys, vals, mem_lens = normalize_kv(kv_list, self.word2id,
                                                self.max_mem_len)
    kv_batch = {"keys" : np.array(keys),
                "vals" : np.array(vals),
                "mem_lens" : np.array(mem_lens)}
    
    return kv_batch

  def batch_sent(self, sent_list):
    sentences, sent_lens = normalize_sent(sent_list, self.word2id,
                                                self.max_sent_len)
    kv_batch = {"sentences" : np.array(sentences),
                "sent_lens" : np.array(sent_lens)}
    
    return kv_batch

  def next_batch_train(self, setname, ptr, batch_size):
    sentences = self._dataset[setname]['sentences'][ptr: ptr + batch_size]
    sent_bow = self._dataset[setname]['sent_bow'][ptr: ptr + batch_size]
    templates = self._dataset[setname]['templates'][ptr: ptr + batch_size]
    sent_lens = self._dataset[setname]['sent_lens'][ptr: ptr + batch_size]
    keys = self._dataset[setname]['keys'][ptr: ptr + batch_size]
    vals = self._dataset[setname]['vals'][ptr: ptr + batch_size]
    mem_lens = self._dataset[setname]['mem_lens'][ptr: ptr + batch_size]
    zcs = self._dataset[setname]['zcs'][ptr: ptr + batch_size]
    t_input_ids = self._dataset[setname]['t_input_ids'][ptr: ptr + batch_size]
    t_attn_mask = self._dataset[setname]['t_attn_mask'][ptr: ptr + batch_size]
    t_token_type_ids = self._dataset[setname]['t_token_type_ids'][ptr: ptr + batch_size]

    batch = {'sentences': np.array(sentences),
             'sent_bow': np.array(sent_bow),
             'sent_dlex': np.array(templates), # delexicalized sentences
             'sent_lens': np.array(sent_lens), # sent_len + _GOO 
             'keys': np.array(keys),
             'vals': np.array(vals),
             'mem_lens': np.array(mem_lens),
             'zcs': np.array(zcs),
             't_input_ids' : np.array(t_input_ids),
             't_attn_mask' : np.array(t_attn_mask),
             't_token_type_ids' : np.array(t_token_type_ids)
             }
    return batch

  def next_batch_infer(self, setname, ptr, batch_size):
    keys = self._dataset[setname]['keys'][ptr: ptr + batch_size]
    vals = self._dataset[setname]['vals'][ptr: ptr + batch_size]
    mem_lens = self._dataset[setname]['mem_lens'][ptr: ptr + batch_size]
    references = self._dataset[setname]['references'][ptr: ptr + batch_size]
    zcs = self._dataset[setname]['zcs'][ptr: ptr + batch_size]
    t_input_ids = self._dataset[setname]['t_input_ids'][ptr: ptr + batch_size]
    t_attn_mask = self._dataset[setname]['t_attn_mask'][ptr: ptr + batch_size]
    t_token_type_ids = self._dataset[setname]['t_token_type_ids'][ptr: ptr + batch_size]

    batch = {'keys': np.array(keys),
             'vals': np.array(vals),
             'mem_lens': np.array(mem_lens), 
             'references': references,
             'zcs': np.array(zcs),
             't_input_ids' : np.array(t_input_ids),
             't_attn_mask' : np.array(t_attn_mask),
             't_token_type_ids' : np.array(t_token_type_ids)
             }
    return batch

  def next_batch(self, setname, batch_size):
    """Get next batch 
    
    Args:
      setname: 'train', 'dev', or 'test'
      batch_size: an integer

    Returns:
      batch: type=dict
      batch['sentences']
      batch['templates']
      batch['sent_lens']
      batch['keys']
      batch['vals']
      batch['mem_lens']
    """
    sname = {"train" : "train", "dev":"dev","test":"test",
            "dev2":"dev", "test2":"test"}
    ptr = self._ptr[sname[setname]]

    if(setname in ['train', 'dev2', 'test2']):
      sname2 = {"train" : "train", 
               "dev2":"dev_casewise",
               "test2":"test_casewise"}  
      batch = self.next_batch_train(sname2[setname], ptr, batch_size)
    else:
      batch = self.next_batch_infer(setname, ptr, batch_size)
    
    self._update_ptr(sname[setname], batch_size)
    return batch

  def decode_sent(self, sent, sent_len=-1, prob=None, add_eos=True):
    """Decode the sentence from id to string"""
    s_out = []
    is_break = False
    for wi, wid in enumerate(sent[:sent_len]):
      if(is_break): break
      w = self.id2word[wid]
      if(w == "_EOS"): 
        is_break = True
      s_out.append(w)
      if(prob is not None): s_out.append("(%.3f) " % prob[wi])
    if(add_eos == False): s_out = s_out[:-1]
    return " ".join(s_out)

  def post_process_sentence(self, keys, vals, sent):
    """Post processing single sentence"""
    sent_ = np.zeros_like(sent)
    for wi, w in enumerate(sent):
      for k, v in zip(keys, vals):
        if(w == k): 
          w = v
          break
      sent_[wi] = w
    return sent_
  
  def post_process(self, batch, out_dict):
    """Post processing the prediction, substitute keys with values"""
    if('predictions' in out_dict):
      predictions = out_dict['predictions']
      predictions_ = np.zeros_like(predictions)
      for bi in range(predictions.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        predictions_[bi] = self.post_process_sentence(
          keys_bi, vals_bi, predictions[bi])
      out_dict['predictions'] = predictions_

    if('post_predictions' in out_dict):
      post_predictions = out_dict['post_predictions']
      post_predictions_ = np.zeros_like(post_predictions)
      for bi in range(post_predictions.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        post_predictions_[bi] = self.post_process_sentence(
          keys_bi, vals_bi, post_predictions[bi])
      out_dict['post_predictions'] = post_predictions_

    if('predictions_all' in out_dict):
      predictions_all = out_dict['predictions_all']
      predictions_all_ = np.zeros_like(predictions_all)
      for bi in range(predictions_all.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        for si in range(predictions_all.shape[1]):
          predictions_all_[bi][si] = self.post_process_sentence(
            keys_bi, vals_bi, predictions_all[bi][si])
      out_dict['predictions_all'] = predictions_all_

    if('post_predictions_all' in out_dict):
      predictions_all = out_dict['post_predictions_all']
      predictions_all_ = np.zeros_like(predictions_all)
      for bi in range(predictions_all.shape[0]):
        keys_bi = batch['keys'][bi]
        vals_bi = batch['vals'][bi]
        for si in range(predictions_all.shape[1]):
          predictions_all_[bi][si] = self.post_process_sentence(
            keys_bi, vals_bi, predictions_all[bi][si])
      out_dict['post_predictions_all'] = predictions_all_
    return 
