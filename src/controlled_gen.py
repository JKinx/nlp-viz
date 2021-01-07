from modeling.latent_temp_crf_ar_model import LatentTemplateCRFARModel
from data_utils.dateSet_helpers import *
import torch
import pickle

class ControlledGen:
    def __init__(self, model_name="dateSet", device="cpu"):
        loaded = torch.load("../dateSet/models/" + model_name)
        self.config = loaded["config"]
        self.config.device = device

        self.dataset = loaded["dataset"]

        self.model = LatentTemplateCRFARModel(self.config)
        self.model.load_state_dict(loaded["model_state_dict"])
        
        
#         loaded = pickle.load(open("../dateSet/best", "rb"))
#         self.config = loaded["config"]
#         self.config.device = device

#         self.dataset = loaded["dataset"]

#         self.model = LatentTemplateCRFARModel(self.config)
#         self.model.load_state_dict(loaded["model"])
        self.model.to(self.config.device)
        self.model.eval()
        del loaded
        
    def get_unfiltered_yz_batched(self, x_list, template_list=None):
        batch_size = len(x_list)

        if template_list is None:
            template_list = [[-1] for _ in range(batch_size)]

        kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        x_batch = self.dataset.batch_kv(kv_list)

        keys = torch.from_numpy(x_batch['keys']).to(self.config.device)
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device)

        out_dict = self.model.model.infer2(keys, vals, template_list)

        return out_dict

    def get_yz_batched(self, x_list, template_list=None):
        batch_size = len(x_list)

        if template_list is None:
            template_list = [[-1] for _ in range(batch_size)]

        kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        x_batch = self.dataset.batch_kv(kv_list)

        keys = torch.from_numpy(x_batch['keys']).to(self.config.device)
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device)

        out_dict = self.model.model.infer2(keys, vals, template_list)

        pred_y, pred_z = dateSet_decode_out(self.dataset, out_dict["pred_y"], out_dict["pred_z"])

        out_list = []
        for i in range(batch_size):
            out = {"y" : pred_y[i], 
                   "z" : pred_z[i], 
                   "score" : out_dict["pred_score"][i]}
            out_list.append(out)

        return out_list
    
    def get_yz(self, x, template=None):
        if template is None:
            template_list = None
        else:
            template_list = [template]
        return self.get_yz_batched([x], template_list)[0]
    
    def get_unfiltered_z_batched(self, x_list, y_list):
        batch_size = len(x_list)

        kv_list = [dateSet_tuple_to_kvs(x) for x in x_list]
        x_batch = self.dataset.batch_kv(kv_list)
        keys = torch.from_numpy(x_batch['keys']).to(self.config.device)
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device)

#         prepped_y_list = [dateSet_prep_sent(y) for y in y_list]
        prepped_y_list = y_list
        y_batch = self.dataset.batch_sent(prepped_y_list)
        sentences = torch.from_numpy(y_batch['sentences']).to(self.config.device)
        sent_lens = torch.from_numpy(y_batch['sent_lens']).to(self.config.device)

        out = self.model.model.posterior_infer(keys, vals, 
                        sentences, sent_lens)
        
        return out, prepped_y_list

    def get_z_batched(self, x_list, y_list):
        batch_size = len(x_list)
        
        out, prepped_y_list = self.get_unfiltered_z_batched(x_list, y_list)
        
        out_list = []
        for i in range(batch_size):
            _, z = dateSet_filter_yz(prepped_y_list[i], out[i])
            out_list.append(z)

        return out_list

    def get_z(self, x, y):
        return self.get_z_batched([x],[y])[0]
    
    def transfer_style(self, x0, y0, x):
        template0 = self.get_unfiltered_z_batched([x0], [y0])[0][0]
        template0[-2] = 3
        
        template = []
        
        specials = [0,1,2]
        i = 0
        while i < len(template0):
            state = template0[i]
            
            if state in specials:
                template.append("(")
                template.append(state)
                template.append(")")
                while template0[i] == state:
                    i += 1
            else:
                template.append(state)
                i += 1
                
        template.append("_")
        
        out = self.get_yz(x, template)
        return out["y"]
    
    def decode_out(self, pred_y, pred_z):
        batch_size = len(pred_y)
        pred_y, pred_z = dateSet_decode_out(self.dataset, pred_y, pred_z)
        out_list = []
        for i in range(batch_size):
            out = {"y" : pred_y[i], 
                   "z" : pred_z[i]}
            out_list.append(out)
        return out_list
    
    def decode_out_unfiltered(self, pred_y, pred_z):
        batch_size = len(pred_y)
        pred_y, pred_z = dateSet_decode_out_unfiltered(self.dataset, pred_y, pred_z)
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

        keys = torch.from_numpy(x_batch['keys']).to(self.config.device)
        vals = torch.from_numpy(x_batch['vals']).to(self.config.device)

        return self.model.model.ibeam_init(keys, vals)
    
    def ibeam_act(self, ibeam_data, ibeam_key):
        return self.model.model.ibeam_act(ibeam_data, ibeam_key)
        
