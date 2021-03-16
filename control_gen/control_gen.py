# from modeling.latent_temp_crf_ar_model import LatentTemplateCRFARModel
from .modeling import LatentTemplateCRFARModel
from .data_utils.dateSet_helpers import *
from .data_utils.e2e_helpers import *
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
        
        if self.data == "dateSet":
            pred_y, pred_z = dateSet_decode_out(self.dataset, out_dict["pred_y"], out_dict["pred_z"])
        elif self.data == "e2e":
            pred_y, pred_z = e2e_decode_out(self.dataset, out_dict["pred_y"], out_dict["pred_z"])
        
        out_list = []
        for i in range(batch_size):
            alpha_zi = [self.cid2alpha[zid] for zid in pred_z[i]]
            out = {"y" : pred_y[i], 
                   "z" : alpha_zi, 
                   "score" : out_dict["pred_score"][i]}
            out_list.append(out)

        return out_list
    
    def get_yz(self, x, template=None):
        if template is None:
            template_list = None
        else:
            template_list = [template]
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
        
        if self.data == "dateSet":
            pred_y, pred_z = dateSet_decode_out(self.dataset, pred_y, pred_z)
        elif self.data == "e2e":
            pred_y, pred_z = e2e_decode_out(self.dataset, pred_y, pred_z)
        
        out_list = []
        for i in range(batch_size):
            out = {"y" : pred_y[i], 
                   "z" : pred_z[i]}
            out_list.append(out)
        return out_list
    
    def ibeam_init(self, x_list):
        batch_size = len(x_list)

        kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        x_batch = self.dataset.batch_kv(kv_list)

        keys = torch.from_numpy(x_batch['keys']).to(self.config.device).long()
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device).long()

        return self.model.model.ibeam_init(keys, vals)
    
    def ibeam_act(self, ibeam_data, ibeam_key):
        return self.model.model.ibeam_act(ibeam_data, ibeam_key)
        
