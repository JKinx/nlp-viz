# from modeling.latent_temp_crf_ar_model import LatentTemplateCRFARModel
from .modeling import LatentTemplateCRFARModel
from .data_utils.dateSet_helpers import *
from .data_utils.e2e_helpers import *
from .data_utils.helpers import *
import torch
import pickle

class ControlGen:
    def __init__(self, model_path="", device="cpu", data="dateSet"):
        loaded = torch.load(model_path)
        self.config = loaded["config"]
        self.config.device = device
        self.data = data

        self.dataset = loaded["dataset"]
        
        self.config._dataset = self.dataset

        self.model = LatentTemplateCRFARModel(self.config)
        self.model.load_state_dict(loaded["model_state_dict"])
        self.model.to(self.config.device)
        self.model.eval()
        self.cid2alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        
        del loaded

    def get_yz_batched(self, x_list, template_list=None):
        batch_size = len(x_list)

        if template_list is None:
            if self.data == "dateSet":
                template_list = [".+." for _ in range(batch_size)]
            elif self.data == "e2e":
                template_list = [".+T." for _ in range(batch_size)]
        
        if self.data == "dateSet":
            kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        elif self.data == "e2e":
            kv_list = [e2e_dict_to_kvs(x) for x in x_list]
            
        x_batch = self.dataset.batch_kv(kv_list)

        keys = torch.from_numpy(x_batch['keys']).to(self.config.device).long()
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device).long()

        out_dict = self.model.model.infer2(keys, vals, template_list)
        
        pred_y, pred_z = decode_yz(self.dataset, 
                                  out_dict["pred_y"],
                                  out_dict["pred_z"])
        pred_bt = decode_bt(self.dataset, out_dict["bts"])
        
        out_list = []
        for i in range(batch_size):
            out = {"y" : pred_y[i], 
                   "z" : pred_z[i], 
                   "score" : out_dict["pred_score"][i],
                   "bt_object" : out_dict["beam_trees"][i],
                   "bt_graph" : pred_bt[i],
                   "y_raw" : out_dict["pred_y"][i],
                   "z_raw" : out_dict["pred_z"][i],
                   "regex_alignment" : out_dict["regex_alignment"][i]
                  }
            out_list.append(out)

        return out_list
    
    def get_yz(self, x, template=None):
        if template is None:
            template_list = None
        else:
            template_list = [template + "."]
        return self.get_yz_batched([x], template_list)[0]
    
    def get_z_batched(self, x_list, y_list):
        batch_size = len(x_list)

        if self.data == "dateSet":
            kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        elif self.data == "e2e":
            kv_list = [e2e_dict_to_kvs(x) for x in x_list]
        else:
            raise NotImplementedError("Dataset not supported")
            
        x_batch = self.dataset.batch_kv(kv_list)
        keys = torch.from_numpy(x_batch['keys']).to(self.config.device).long()
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device).long()
            
        y_batch = self.dataset.batch_sent(y_list)
        
        sentences = torch.from_numpy(y_batch['sentences']).to(self.config.device).long()
        if self.data == "e2e":
            sentences = e2e_dlex(sentences, keys, vals).long()
        elif self.data == "dateSet":
            sentences = dateSet_dlex(sentences, self.dataset).long()
            
        sent_lens = torch.from_numpy(y_batch['sent_lens']).to(self.config.device)

        out = self.model.model.posterior_infer(keys, vals, 
                        sentences, sent_lens)
        
        alpha_out = []
        for zi in out:
            alpha_out.append([self.cid2alpha[zid] for zid in zi])
        
        return alpha_out

    def get_z(self, x, y):
        return self.get_z_batched([x],[y])[0]
    
    def transfer_style(self, x0, y0, x):
        template0 = self.get_z(x0, y0)
        
        template = ""
        
        if self.data == "dateSet":
            specials = [0,1,2]
        elif self.data == "e2e":
            specials = list(range(8))
        specials = [self.cid2alpha[special] for special in specials]
            
        i = 0
        while i < len(template0):
            state = template0[i]
            if state in specials:
                template += str(state) + "+"
                while template0[i] == state:
                    i += 1
            else:
                template += str(state)
                i += 1
                
        template += "."

        out = self.get_yz(x, template)
        return out["y"]
    
    def decode_out(self, pred_y, pred_z):
        batch_size = len(pred_y)
        pred_y, pred_z = decode_yz(self.dataset, pred_y, pred_z)
        out_list = []
        for i in range(batch_size):
            out = {"y" : pred_y[i], 
                   "z" : pred_z[i]}
            out_list.append(out)
        return out_list
    
    def bt_probe(self, bt, key):
        inp, h, mem_emb, mem_mask, mem = bt.get_prob_init(key)
        probe_out = self.model.model.probe_bst(inp, h, mem_emb, mem_mask, mem)
        
        num_z = len(probe_out[0])
        options = []
        for i in range(num_z):
            for j in range(5):
                option = {"z_raw" : i,
                          "z" : self.cid2alpha[i],
                          "y_raw" : probe_out[0][i][j], 
                          "score" : probe_out[1][i][j]}
                option["y"] = self.dataset.id2word[option["y_raw"]]
                options.append(option)
                
        out_dict = {"decoded" : options, "scores" : probe_out[1], "indices" : probe_out[0]}
        
        return out_dict
    
    def bt_act(self, bt, key, y, z):
        bs_init = bt.get_bs_init(key, z, y)
        
        out_dict = self.model.model.beam_tree_act(bs_init, bt)
        
        pred_y, pred_z = decode_yz(self.dataset, 
                                   [out_dict["pred_y"]],
                                   [out_dict["pred_z"]])
        pred_bt = decode_bt(self.dataset, [out_dict["bt"]])
        
        out = {"y" : pred_y[0], 
               "z" : pred_z[0], 
               "score" : out_dict["pred_score"],
               "bt_graph" : pred_bt[0],
               "y_raw" : out_dict["pred_y"],
               "z_raw" : out_dict["pred_z"],
               "bt_object" : bt,
               "regex_alignment" : out_dict["regex_alignment"]}
        
        return out

