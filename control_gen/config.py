class Config:

  def __init__(self):
    self.model_name = 'test_model' 
    self.model_version = 'test'
    self.dataset = 'test_dataset'

    self.output_path = '../outputs/'
    self.tensorboard_path = 'tensorboard/'
    self.model_path = '../models/'

    ## Dataset 
    self.data_root = '../data/'
    self.data_path = ""
    
    self.max_y_len = 50
    self.max_x_len = 50
    
    self.pad_id = 0
    self.start_id = 1
    self.end_id = 2
    self.unk_id = 3
    self.seg_id = 4

    self.word2id = {'_PAD': self.pad_id, '_GOO': self.start_id, 
      '_EOS': self.end_id, '_UNK': self.unk_id, '_SEG': self.seg_id}
    self.id2word = {self.pad_id: '_PAD', self.start_id: '_GOO', 
      self.end_id: '_EOS', self.unk_id: '_UNK', self.seg_id: '_SEG'}

    self.key_vocab_size = -1
    self.vocab_size = -1

    ## Controller 
    # general
    self.is_test = False
    self.test_validate = False
    self.use_tensorboard = False
    self.write_full_predictions = False
    self.device = 'cuda'
    self.gpu_id = '0'
    self.start_epoch = 0
    self.validate_start_epoch = 8
    self.num_epoch = 30
    self.batch_size_train = 500
    self.batch_size_eval = 100
    self.print_interval = 200 
    self.load_ckpt = False
    self.save_ckpt = False # if save checkpoints
    self.all_pretrained_path = ''
    self.save_temp = False

    # logging info during training 
    self.log_info = [
        'loss', 
        'tau', 'x_lambd', 
        'p_log_prob', 'p_log_prob_x', 'p_log_prob_z', 'z_acc',  
        'ent_z', 'ent_z_loss', 'pr_inc_val', 'pr_inc_loss', 'pr_exc_val', 'pr_exc_loss'
        ]

    # scores to be reported during validation 
    self.validation_scores = [
        'ent_z', 'elbo', 'marginal', 'p_log_prob', 'z_sample_log_prob', 'ppl', 
        'p_log_prob_x', 'p_log_prob_z', 'z_acc'
        ]

    # validation criteria for different models 
    self.validation_criteria = 'marginal'

    # optimization
    self.learning_rate = 1e-4

    # latent z
    self.latent_vocab_size = 50

    self.y_beta = 0. # KLD for y 
    self.bow_beta = 0. # BOW KLD/ entropy regularization
    self.bow_lambd = 1.0 # BOW loss
    self.bow_gamma = 1.0 # stepwise bow loss

    self.z_lambd = 1.0 # learning signal scaling
    self.z_beta = 0.01 # entropy regularization 

    # Anneal tau 
    self.z_tau_init = 1.0
    self.z_tau_final = 0.01
    self.tau_anneal_epoch = 40

    # anneal word dropout
    self.x_lambd_start_epoch = 10
    self.x_lambd_anneal_epoch = 2
    
    # pr 
    self.pr = False
    self.pr_inc_lambd = None
    self.pr_exc_lambd = None

    self.max_grad_norm = 1.
    self.p_max_grad_norm = 1.
    self.q_max_grad_norm = 5.

    ## model
    # general 
    self.lstm_layers = 1 # 2 for LM on PTB
    self.lstm_bidirectional = True
    self.embedding_size = -1 # the same as state size
    self.state_size = 300 # 650 for LM on PTB
    self.dropout = 0.2 # 0.5 for LM on PTB
    self.tapas_state_size = 768

    # latent template
    self.gumbel_tau = 1.0

  def overwrite(self, args):
    args = vars(args)
    for v in args: setattr(self, v, args[v])
    return 

  def write_arguments(self):
    """Write the arguments to log file"""
    args = vars(self)
    with open(self.output_path + 'arguments.txt', 'w') as fd:
      fd.write('%s_%s\n' % (self.model_name, self.model_version))
      for k in args:
        fd.write('%s: %s\n' % (k, str(args[k])))
    return 

  def print_arguments(self):
    """Print the argument to commandline"""
    args = vars(self)
    print('%s_%s' % (self.model_name, self.model_version))
    for k in args:
      print('%s: %s' % (k, str(args[k])))
    return 
