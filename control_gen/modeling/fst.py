from copy import deepcopy as dc

def get_type(s):
    if s[0] == "!":
        return "neg", s[1:]
    elif s[0] == "*":
        return "star", s[1:]
    elif s[0] == "+":
        return "plus", s[1:]
    elif s[0] == "?":
        return "ques", s[1:]
    else:
        return "base", s

def get_next(s):
    if s[0] == "(":
        toks = ""
        i = 1
        while s[i] != ")":
            toks += s[i]
            i += 1
        assert s[i+1] == ";"
        i += 1
        
        tok_lst = toks.split(";")
    else:
        tok = ""
        i = 0 
        while s[i] != ";":
            tok += s[i]
            i += 1
        tok_lst = [tok]
    
    return tok_lst, s[i+1:]

def encode_tok(tok, dataset):
    if tok[0] == "[":
        yz = "y"
        tok = tok[1:-1]
        assert tok in dataset.word2id.keys()
        enc_tok = dataset.word2id[tok]
    elif tok == ".":
        yz = "z"
        enc_tok = -1
    else:
        yz = "z"
        try:
            enc_tok = int(tok)
        except:
            raise TypeError("Token needs to be an int or .")
    return yz, enc_tok

def init_fs():
    fs_dict = {"counter" : 0, "fss" : {}}
    init = {"id" : 0, "exit" : False, "type" : "init", "yz" : None,
            "val": None, "prev" : [], "nodes" : []} 
    fs_dict["fss"][0] = init
    return fs_dict

def make_fs(typ, yz, val, prev_lst, fs_dict):
    fs_dict["counter"] += 1
    fs_id = fs_dict["counter"]
    fs = {"id" : fs_id, "exit" : False, "type" : typ, "yz" : yz, 
          "val": val, "prev" : prev_lst, "nodes" : []}
    fs_dict["fss"][fs_id] = fs
    return fs_id

def exit_fs(end_lst, fs_dict):
    for fs_id in end_lst:
        fs_dict["fss"][fs_id]["exit"] = True

def make_base(toks, prev_lst, fs_dict, dataset):
    tok_lst = toks.split("|")
    start_lst = []
    end_lst = []
    for tok in tok_lst:
        yz, enc_tok = encode_tok(tok, dataset)
        fs_id = make_fs("base", yz, enc_tok, dc(prev_lst), fs_dict)
        start_lst.append(fs_id)
        end_lst.append(fs_id)
    return start_lst, end_lst

def make_neg(toks, prev_lst, fs_dict, dataset):
    tok_lst = toks.split("|")
    enc_tok_lst = []
    for tok in tok_lst:
        yz, enc_tok = encode_tok(tok, dataset)
        assert yz == "z"
        enc_tok_lst.append(enc_tok)
    fs_id = make_fs("neg", yz, enc_tok_lst, dc(prev_lst), fs_dict)
    return [fs_id]
    
def make_step(seq, prev_lst, fs_dict, dataset):
    typ, seq = get_type(seq)
    tok_lst, next_seq = get_next(seq)
    if typ == "base":
        assert len(tok_lst) == 1
        _, end_lst = make_base(tok_lst[0], prev_lst, fs_dict, dataset)
        return end_lst, next_seq
    
    if typ == "neg":
        assert len(tok_lst) == 1
        end_lst = make_neg(tok_lst[0], prev_lst, fs_dict, dataset)
        return end_lst, next_seq
    
    id_lsts = []
    p_lst = [prev_lst, prev_lst]
    for tok in tok_lst:
        p_lst = make_base(tok, p_lst[1], fs_dict, dataset)
        id_lsts.append(p_lst)
        
    if typ == "star":
        # self loop
        for fs_id in id_lsts[0][0]:
            fs_dict["fss"][fs_id]["prev"] += id_lsts[-1][1]
        # can skip everything
        end_lst = prev_lst + id_lsts[-1][1]
    elif typ == "plus":
        # self loop
        for fs_id in id_lsts[0][0]:
            fs_dict["fss"][fs_id]["prev"] += id_lsts[-1][1]
        end_lst = id_lsts[-1][1]
    elif typ == "ques":
        # can skip everything
        end_lst = prev_lst + id_lsts[-1][1]
        
    return end_lst, next_seq

def resolve_fs(seq, fs_dict, dataset):
    prev_lst = [0]
    
    last_seq = dc(seq)
    while seq != "":
        last_seq = dc(seq)
        prev_lst, seq = make_step(seq, prev_lst, fs_dict, dataset)
        
    assert last_seq[0] not in ["+", "?", "*"]
    
    exit_fs(prev_lst, fs_dict)