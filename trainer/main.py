import argparse
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys 
import shutil
import torch

from datetime import datetime

from control_gen.modeling import torch_model_utils as tmu

from controller import Controller
from control_gen.config import Config

from control_gen.modeling import LatentTemplateCRFARModel

from control_gen.data_utils import Dataset

import pickle

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def define_argument(config):
  ## add commandline arguments, initialized by the default configuration
  parser = argparse.ArgumentParser()

  # general 
  parser.add_argument(
    "--model_name", default=config.model_name, type=str)
  parser.add_argument(
    "--model_version", default=config.model_version, type=str)
  parser.add_argument(
    "--dataset", default=config.dataset, type=str)
    
  # dataset len
  parser.add_argument(
    "--max_dec_len", default=config.max_dec_len, type=int)
  parser.add_argument(
    "--max_bow_len", default=config.max_bow_len, type=int)
  parser.add_argument(
    "--max_mem_len", default=config.max_mem_len, type=int)

  # train
  parser.add_argument(
    "--is_test", type=str2bool, 
    nargs='?', const=True, default=config.is_test)
  parser.add_argument(
    "--test_validate", type=str2bool, 
    nargs='?', const=True, default=config.test_validate)
  parser.add_argument(
    "--use_tensorboard", type=str2bool, 
    nargs='?', const=True, default=config.use_tensorboard)
  parser.add_argument(
    "--write_full_predictions", type=str2bool, 
    nargs='?', const=True, default=config.write_full_predictions)
  parser.add_argument(
    "--device", default=config.device, type=str)
  parser.add_argument(
    "--gpu_id", default=config.gpu_id, type=str)
  parser.add_argument(
    "--start_epoch", default=config.start_epoch, type=int)
  parser.add_argument(
    "--validate_start_epoch", default=config.validate_start_epoch, type=int)
  parser.add_argument(
    "--validation_criteria", 
    default=config.validation_criteria, type=str)
  parser.add_argument(
    "--num_epoch", default=config.num_epoch, type=int)
  parser.add_argument(
    "--batch_size_train", default=config.batch_size_train, type=int)
  parser.add_argument(
    "--batch_size_eval", default=config.batch_size_eval, type=int)
  parser.add_argument(
    "--print_interval", default=config.print_interval, type=int)
  parser.add_argument(
    "--save_ckpt", type=str2bool, 
    nargs='?', const=True, default=config.save_ckpt)
  parser.add_argument(
    "--save_temp", type=str2bool, 
    nargs='?', const=True, default=config.save_temp)

  # optimization
  parser.add_argument(
    "--learning_rate", default=config.learning_rate, type=float)
  parser.add_argument(
    "--enc_learning_rate", default=config.enc_learning_rate, type=float)
  parser.add_argument(
    "--dec_learning_rate", default=config.dec_learning_rate, type=float)
  parser.add_argument(
    "--bow_lambd", default=config.bow_lambd, type=float)
  parser.add_argument(
    "--bow_beta", default=config.bow_beta, type=float)
  parser.add_argument(
    "--bow_gamma", default=config.bow_gamma, type=float)
  parser.add_argument(
    "--bow_deterministic", type=str2bool, 
    nargs='?', const=True, default=config.bow_deterministic)
  parser.add_argument(
    "--z_sample_method", default=config.z_sample_method, type=str)
  parser.add_argument(
    "--z_beta", default=config.z_beta, type=float)
  parser.add_argument(
    "--z_overlap_logits", type=str2bool, 
    nargs='?', const=True, default=config.z_overlap_logits)  
  parser.add_argument(
    "--z_tau_final", default=config.z_tau_final, type=float)
  parser.add_argument(
    "--tau_anneal_epoch", type=int, default=config.tau_anneal_epoch)  
  parser.add_argument(
    "--gumbel_st", type=str2bool, 
    nargs='?', const=True, default=config.gumbel_st)  
  parser.add_argument(
    "--dec_adaptive", type=str2bool, 
    nargs='?', const=True, default=config.dec_adaptive)  
  parser.add_argument(
    "--auto_regressive", type=str2bool, 
    nargs='?', const=True, default=config.auto_regressive)  
  parser.add_argument(
    "--use_copy", type=str2bool, 
    nargs='?', const=True, default=config.use_copy)  
  parser.add_argument(
    "--use_src_info", type=str2bool, 
    nargs='?', const=True, default=config.use_src_info) 
  parser.add_argument(
    "--x_lambd_start_epoch", default=config.x_lambd_start_epoch, type=int)
  parser.add_argument(
    "--x_lambd_anneal_epoch", default=config.x_lambd_anneal_epoch, type=int)
  parser.add_argument(
    "--temp_rank_strategy", default=config.temp_rank_strategy, type=str)
  parser.add_argument(
    "--z_pred_strategy", default=config.z_pred_strategy, type=str)
  parser.add_argument(
    "--x_pred_strategy", default=config.x_pred_strategy, type=str)
  parser.add_argument(
    "--decode_strategy", default=config.decode_strategy, type=str)
  parser.add_argument(
    "--sampling_topk_k", default=config.sampling_topk_k, type=int)
  parser.add_argument(
    "--sampling_topp_gap", default=config.sampling_topp_gap, type=float)

  # model 
  parser.add_argument(
    "--load_ckpt", type=str2bool, 
    nargs='?', const=True, default=config.is_test)
  parser.add_argument(
    "--all_pretrained_path", type=str, 
    default=config.all_pretrained_path)
  parser.add_argument(
    "--lstm_layers", default=config.lstm_layers, type=int)
  parser.add_argument(
    "--state_size", default=config.state_size, type=int)
  parser.add_argument(
    "--dropout", default=config.dropout, type=float)
  parser.add_argument(
    "--copy_decoder", type=str2bool, 
    nargs='?', const=True, default=config.copy_decoder)  
  parser.add_argument(
    "--latent_vocab_size", default=config.latent_vocab_size, type=int)
  parser.add_argument(
    "--num_sample", default=config.num_sample, type=int)
  parser.add_argument(
    "--sample_strategy", default=config.sample_strategy, type=str)
  parser.add_argument(
    "--stepwise_score", type=str2bool, 
    nargs='?', const=True, default=config.stepwise_score)

  # pr
    
  parser.add_argument(
    "--pr", type=str2bool, 
    nargs='?', const=True)
  parser.add_argument(
    "--num_pr_constraints", default=config.num_pr_constraints, type=int)
  parser.add_argument(
    "--pr_inc_lambd", default=config.pr_inc_lambd, type=float)
  parser.add_argument(
    "--pr_exc_lambd", default=config.pr_inc_lambd, type=float)
  

  args = parser.parse_args()
  return args

def set_argument(config, args):
  """Set the commandline argument

  Because I see many different convensions of passing arguments (w. commandline,
  .py file, .json file etc.) Here I try to find a clean command line convension
  
  Argument convension:
    * All default value of commandline arguments are from config.py 
    * So instead of using commandline arguments, you can also modify config.py 
    * Instead of using commandline switching, all boolean values are explicitly
      set as 'True' or 'False'
    * The arguments passed through commandline will overwrite the default 
      arguments from config.py 
    * The final arguments are printed out
  """

  ## overwrite the default configuration  
  config.overwrite(args)
  config.embedding_size = config.state_size

  if(config.test_validate): 
    config.use_tensorboard = False
    config.validate_start_epoch = 0

  ## build model saving path 
  model = config.model_name + "_" + config.model_version
  output_path = config.output_path + model 
  model_path = config.model_path + model
  tensorboard_path = config.tensorboard_path + model + '_'
  tensorboard_path += datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
  if(config.is_test == False):
    # Training mode, create directory for storing model and outputs
    print('model path: %s' % model_path)
    if(os.path.exists(model_path)):
      print("model %s already existed" % model)
      print('removing existing cache directories')
      print('removing %s' % model_path)
      shutil.rmtree(model_path)
      os.mkdir(model_path)
      if(os.path.exists(output_path)): 
        print('removing %s' % output_path)
        shutil.rmtree(output_path)
      os.mkdir(output_path)
      if(config.use_tensorboard):
        for p in os.listdir(config.tensorboard_path):
          if(p.startswith(model)): 
            try:
              shutil.rmtree(config.tensorboard_path + p)
            except:
              print('cannot remove %s, pass' % (config.tensorboard_path + p))
        os.mkdir(tensorboard_path)
    else:
      os.mkdir(model_path)
      os.mkdir(output_path)
  else: pass # test mode, do not create any directory 
  config.model_path = model_path + '/'
  config.output_path = output_path + '/'
  config.tensorboard_path = tensorboard_path + '/'
  
#   config.data_path = {
#     'train': config.data_root + config.dataset + '/trainset.csv', 
#     'dev': config.data_root + config.dataset + '/devset.csv', 
#     'test': config.data_root + config.dataset + '/testset_w_refs.csv',
#     }
  
#   config.data_path = {
#     'train': config.data_root + config.dataset + '/trainset.pkl', 
#     'dev': config.data_root + config.dataset + '/devset.pkl', 
#     'test': config.data_root + config.dataset + '/testset.pkl',
#     }
 
  config.data_path = {
    'train': config.data_root + config.dataset + '/trainset_notable.pkl', 
    'dev': config.data_root + config.dataset + '/devset_notable.pkl', 
    'test': config.data_root + config.dataset + '/testset_notable.pkl',
    }
  
  config.write_arguments()

  ## set gpu 
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
  os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

  ## print out the final configuration
  config.print_arguments()
  return config

def main():
  # arguments
  config = Config()
  args = define_argument(config)
  config = set_argument(config, args)
  
  # dataset
  dataset = Dataset(config)
  dataset.build()
  config.key_vocab_size = dataset.key_vocab_size
  config.vocab_size = dataset.vocab_size
    
  # debug
  with open(config.output_path + 'id2word.txt', 'w') as fd:
    for i in dataset.id2word: fd.write('%d %s\n' % (i, dataset.id2word[i]))

  # model 
  model = LatentTemplateCRFARModel(config) 
#   tmu.print_params(model)

  # controller
  controller = Controller(config, model, dataset)

  if(config.is_test == False):
    if(config.load_ckpt):
      print('Loading model from: %s' % config.all_pretrained_path)
      model.load_state_dict(torch.load(config.all_pretrained_path))
    model.to(config.device)
    controller.train(model, dataset)
  else:
    print('Loading model from: %s' % config.all_pretrained_path)
    checkpoint = torch.load(config.all_pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    ckpt_e = int(config.all_pretrained_path.split('_')[-1][1:])
    print('restore from checkpoint at epoch %d' % ckpt_e)
    controller.test_model(model, dataset, ckpt_e)
  return 

if __name__ == '__main__':
  main()
