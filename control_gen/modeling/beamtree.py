from collections import defaultdict

class BeamTree():
    "Object class for beam tree"

    def __init__(self, inp, state, mem_emb, mem_mask, mem, template):
        self.mem_emb = mem_emb
        self.mem_mask = mem_mask
        self.mem = mem
        self.template = template

        self.init_key = self.get_key([-2], [-2])
        self.inp = {(-2, -2) : inp}
        self.h = {self.init_key : state}

        self.fs_idx_lst = defaultdict(set)

        self.logp = {self.init_key : 0}

        self.bts = defaultdict(list)

    def init_bs_init(self, return_bt = False):
        bs_init = {"h" : self.h[self.init_key],
                   "inp" : self.inp[(-2,-2)],
                   "logp" : self.logp[self.init_key],
                   "prev_zs" : [-2],
                   "prev_ys" : [-2],
                   "fs_idx" : (0,[-2, -1]),
                   "leng" : -1,
                   "key" : self.get_key([-2,-1], [-2,-1]),
                   "bt" :  return_bt
                   }
        return bs_init

    def get_bs_init(self, key, z, y, return_bt = False):
        bs_init = {"bt" :  return_bt}
        zs, ys = self.reverse_key(key)

        h_key = self.get_key(zs[:-2], ys[:-2])
        bs_init["h"] = self.h[h_key]

        inp_key = (zs[-2], ys[-2])
        bs_init["inp"] = self.inp[inp_key]

        prev_key = self.get_key(zs[:-1], ys[:-1])
        bs_init["logp"] = self.logp[prev_key]

        bs_init["prev_ys"] = ys[:-1]
        bs_init["prev_zs"] = zs[:-1]

        bs_init["leng"] = len(zs) - 3

        bs_init["fs_idx"] = list(self.fs_idx_lst[key])

        bs_init["key"] = self.get_key(zs[:-1] + [z], ys[:-1] + [y])
        bs_init["z_id"] = z
        bs_init["y_id"] = y

        return bs_init

    def get_prob_init(self, key):
        zs, ys = self.reverse_key(key)

        h_key = self.get_key(zs[:-2], ys[:-2])
        h = self.h[h_key]

        inp_key = inp_key = (zs[-2], ys[-2])
        inp = self.inp[inp_key]

        return inp, h, self.mem_emb, self.mem_mask, self.mem 

    def get_bt(self, bs_init):
        pre_z = bs_init["prev_zs"] + [bs_init["z_id"]]
        pre_y = bs_init["prev_ys"] + [bs_init["y_id"]]

        pre_tree = []
        for i in range(2, len(pre_y)):
            prev_key = self.get_key(pre_z[:i], pre_y[:i])
            key = self.get_key(pre_z[:i+1], pre_y[:i+1])

            connection = {"key" : key, "prev_key" : prev_key,
                      "z_raw" : pre_z[i], "y_raw" : pre_y[i], 
                      "logp" : self.logp[key]}
            pre_tree.append([connection])

        return pre_tree + self.bts[bs_init["key"]]

    def update_hidden(self, node, decoder_hidden):
        key = self.get_key(node["zs"], node["ys"])

        try:
            h = self.h[key]
            assert h == decoder_hidden
        except:
            self.h[key] = decoder_hidden

    def update_node(self, bs_init, node):
        # update inp_cache
        z = node["state_id"]
        y = node["word_id"]
        try:
            inp = self.inp[(z,y)]
            assert inp == node["inp"]
        except:
            self.inp[(z,y)] = node["inp"]

        key = self.get_key(node["zs"], node["ys"])
        
        self.fs_idx_lst[key].add((node["fs_idx"], tuple(node["bids"])))

        try:
            logp = self.logp[key]
            assert logp == node["logp"] 
        except:
            self.logp[key] = node["logp"]

        self.update_bts(bs_init, node["zs"], node["ys"], node["logp"])

    def update_bts(self, bs_init, zs, ys, logp):
        bt_key = bs_init["key"]

        key = self.get_key(zs, ys)
        prev_key = self.get_key(zs[:-1], ys[:-1])

        t = len(zs) - len(bt_key[0])
        
        if t == 0:
            return

        if len(self.bts[bt_key]) < t:
            assert t - len(self.bts[bt_key]) == 1
            self.bts[bt_key].append([])

        connection = {"key" : key, "prev_key" : prev_key,
                      "z_raw" : zs[-1], "y_raw" : ys[-1], "logp" : logp}

        self.bts[bt_key][t-1].append(connection)

    def get_key(self, zs, ys):
        return (tuple(zs), tuple(ys))

    def reverse_key(self, key):
        return list(key[0]), list(key[1])
        
