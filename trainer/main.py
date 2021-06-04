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
import wandb

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
    "--max_x_len", default=config.max_x_len, type=int)
  parser.add_argument(
    "--max_y_len", default=config.max_y_len, type=int)

  # train
  parser.add_argument(
    "--is_test", type=str2bool, 
    nargs='?', const=True, default=config.is_test)
  parser.add_argument(
    "--test_validate", type=str2bool, 
    nargs='?', const=True, default=config.test_validate)
  parser.add_argument(
    "--use_wandb", type=str2bool, 
    nargs='?', const=True, default=config.use_wandb)
  parser.add_argument(
    "--device", default=config.device, type=str)
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
    "--grad_accum", default=config.grad_accum, type=int)

  # optimization
  parser.add_argument(
    "--learning_rate", default=config.learning_rate, type=float)

  parser.add_argument(
    "--z_beta", default=config.z_beta, type=float) 
  parser.add_argument(
    "--z_beta_anneal", type=str2bool, 
    nargs='?', const=True, default=config.z_beta_anneal)
  parser.add_argument(
    "--z_beta_init", default=config.z_beta_init, type=float)
  parser.add_argument(
    "--z_beta_final", default=config.z_beta_final, type=float)
  parser.add_argument(
    "--z_beta_anneal_epoch", type=int, default=config.z_beta_anneal_epoch) 
    
  parser.add_argument(
    "--z_tau_final", default=config.z_tau_final, type=float)
  parser.add_argument(
    "--tau_anneal_epoch", type=int, default=config.tau_anneal_epoch)  
  parser.add_argument(
    "--x_lambd_start_epoch", default=config.x_lambd_start_epoch, type=int)
  parser.add_argument(
    "--x_lambd_anneal_epoch", default=config.x_lambd_anneal_epoch, type=int)

  # model 
  parser.add_argument(
    "--lstm_layers", default=config.lstm_layers, type=int)
  parser.add_argument(
    "--state_size", default=config.state_size, type=int)
  parser.add_argument(
    "--dropout", default=config.dropout, type=float)

  # pr
  parser.add_argument(
    "--pr", type=str2bool, 
    nargs='?', const=True)
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
    config.validate_start_epoch = 0

  ## build model saving path 
  model = config.model_name + "_" + config.model_version
  model_path = config.model_path + model
  print('model path: %s' % model_path)
  if(os.path.exists(model_path)):
    print("model %s already existed" % model)
    print('removing existing cache directories')
    print('removing %s' % model_path)
    shutil.rmtree(model_path)
    os.mkdir(model_path)
  else:
    os.mkdir(model_path)

  config.model_path = model_path + '/'
  
  config.data_path = {
    'train': config.data_root + 'e2e/trainset_dynamic_clean.pkl', 
    'dev': config.data_root + 'e2e/devset_dynamic_clean.pkl', 
    'test': config.data_root + 'e2e/testset_dynamic_clean.pkl',
    }
    
  config.group = model
  return config

wandb_config_list = ["dataset", "batch_size_train", "grad_accum", "learning_rate",\
                      "z_beta_anneal", "z_beta", "z_beta_init", "z_beta_final", \
                      "z_beta_anneal_epoch", "pr", "pr_inc_lambd", "pr_exc_lambd"]
def main():
  # arguments
  config = Config()
  args = define_argument(config)
  config = set_argument(config, args)
    
#   config.device = "cpu"
    
  for key in wandb_config_list:
    print(key + " : %s" % (str(config.__dict__[key])), flush=True)
  
#   dataset = Dataset(config)
#   dataset.build()
#   pickle.dump(dataset, open("../data/e2e/dynamic_data_c.pkl", "wb"))
#   sdsd

  dataset = pickle.load(open("../data/e2e/" + config.dataset + ".pkl", "rb"))
    
  config.vocab_size = dataset.vocab_size
  
  if config.use_wandb:
    wandb_config = {}
    for el in wandb_config_list:
        wandb_config[el] = config.__dict__[el]
    wandb.init(project="dynamic", group = config.group, config=wandb_config, reinit=True)

  # model 
  model = LatentTemplateCRFARModel(config) 
    
  # controller
  controller = Controller(config, model, dataset)
  
  model.to(config.device)
  controller.train(model, dataset)
  return 

if __name__ == '__main__':
  main()
