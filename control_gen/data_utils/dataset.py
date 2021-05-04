
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

def normalize_set(dset, word2id, max_y_len, max_x_len, word2id_zcs):
  # x table
  data_xs = [d[0] for d in dset]
  xs, _ = nlpp.normalize(
    data_xs, word2id, max_x_len, add_start_end=False)

  # y text
  data_ys = [d[1] for d in dset]
  ys, y_lens = nlpp.normalize(data_ys, word2id, max_y_len)
  
  # z constraints
  data_zcs = [d[2] for d in dset]
  zcs, _ = nlpp.normalize(data_zs, word2id_zcs, max_y_len)

  return xs, ys, y_lens, zcs


def read_data(dpath):
  print('reading %s' % dpath)

  data_raw = pickle.load(open(dpath, "rb"))

  dataset = []

  for data_id in range(len(data_raw["x_lst"])):
    x = data_raw["x_lst"][data_id]
    y = data_raw["y_lst"][data_id]
    zc = data_raw["z_lst"][data_id]
    dataset.append((x, y, zc))

  print("%d cases" % len(dataset))
  
  return dataset, data_raw


class Dataset(DatasetBase):

  def __init__(self, config):
    super(Dataset, self).__init__()
    
    self.model_name = config.model_name
    self.data_path = config.data_path
    self.word2id = config.word2id
    self.id2word = config.id2word
    self.pad_id = config.pad_id
    # +1 is for how nlpp.normalize counts length
    self.max_y_len = config.max_y_len + 1
    self.max_x_len = config.max_x_len 

    self._dataset = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    self._reset_ptr = {"train": False, "dev": False, "test": False}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname]['sentences'])

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

    ## read the sentences
    trainset = read_data(self.data_path['train'])
    devset = read_data(self.data_path['dev'])
    testset = read_data(self.data_path['test'])

    ## build vocabulary 
    train_sents_tables = []
    for data in trainset:
      train_sents_tables.append(data[0])
      train_sents_tables.append(data[1])

    self.word2id, self.id2word, _ = nlpp.build_vocab(train_sents_tables, 
      word2id=self.word2id, id2word=self.id2word, vocab_size_threshold=1)
    
    self.word2id_zcs = {"-1" : -1, '_PAD': -1, '_GOO': -1, 
                        '_EOS': -1, '_UNK': -1, '_SEG': -1}
    for z_id in range(self.max_x_len):
      self.word2id_zcs[str(z_id)] = z_id

    ## normalize the dataset 
    (train_tables, train_sentences, train_sent_lens, 
      train_zcs) = normalize_set(trainset[0], self.word2id, self.max_y_len, 
                                 self.max_x_len, self.word2id_zcs)
    (dev_tables, dev_sentences, dev_sent_lens, 
      dev_zcs) = normalize_set(devset[0], self.word2id, self.max_y_len, 
                                 self.max_x_len, self.word2id_zcs)

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


    self._dataset = { "train": trainset[1], 
                      "dev": devset[1],
                      "test" : testset[1] 
                    }
    self._dataset["train"].update({
      "tables" : train_tables,
      "sentences" : train_sentences,
      "sent_lens" : train_sent_lens,
      "zcs" : train_zcs
      })
    self._dataset["dev"].update({
      "tables" : dev_tables,
      "sentences" : dev_sentences,
      "sent_lens" : dev_sent_lens,
      "zcs" : dev_zcs
      })
    self._dataset["test"].update({
      "tables" : test_tables,
      "sentences" : test_sentences,
      "sent_lens" : test_sent_lens,
      "zcs" : test_zcs
      })
    return 

  def next_batch_train(self, setname, ptr, batch_size):
    batch = {}
    for key in self._dataset["setname"]:
      value = self._dataset[setname][key][ptr: ptr + batch_size]
      batch[key] = np.array(value)
    return batch

  def next_batch_infer(self, setname, ptr, batch_size):
    batch = self.next_batch_train(setname, ptr, batch_size)
    batch["references"] = batch["sentences"][:,1:]
    return batch

  def next_batch(self, setname, batch_size):
    ptr = self._ptr[setname]

    if(setname == 'train'):
        batch = self.next_batch_train(setname, ptr, batch_size)
    else:
        batch = self.next_batch_infer(setname, ptr, batch_size)
    
    self._update_ptr(setname, batch_size)
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
  
  def post_process(self, batch, out_dict):
    return 
