from copy import deepcopy as dc

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

def strip_paren(s):
    if s[0] == "(" and s[-1] == ")":
        return s[1:-1]
    else:
        return s
    
def get_type(s):
    if s[0] == "!":
        return "neg", strip_paren(s[1:])
    elif s[0] == "*":
        return "star", strip_paren(s[1:])
    elif s[0] == "+":
        return "plus", strip_paren(s[1:])
    elif s[0] == "?":
        return "ques", strip_paren(s[1:])
    else:
        base = True
        for delim in [";", "|"]:
            if delim in s:
                base = False
        if base:
            return "base", s
        else:
            return "nested", strip_paren(s)          

def get_next(s, delim):
    tok = ""
    nesting = 0
    i = 0
    while True:
        if s[i] == delim and nesting == 0:
            break
            
        if s[i] == "(":
            nesting += 1
        elif s[i] == ")":
            nesting -= 1
        
        tok += s[i]
        i += 1
    return tok, s[i+1:]

def get_all(s, delim):
    tok_lst = []
    while s!= "":
        tok, s = get_next(s, delim)
        tok_lst.append(tok)
    return tok_lst

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
        
def make_base(tok, prev_lst, fs_dict, dataset):
    yz, enc_tok = encode_tok(tok, dataset)
    fs_id = make_fs("base", yz, enc_tok, dc(prev_lst), fs_dict)
    return fs_id

def make_neg(tok, prev_lst, fs_dict, dataset):
    tok_lst = tok.split("|")
    enc_tok_lst = []
    for tok in tok_lst:
        yz, enc_tok = encode_tok(tok, dataset)
        assert yz == "z"
        enc_tok_lst.append(enc_tok)
    fs_id = make_fs("neg", yz, enc_tok_lst, dc(prev_lst), fs_dict)
    return fs_id

def resolve_fs(seq, prev_lst, fs_dict, dataset):
    tok_lst = get_all(seq, ";")
    
    if len(tok_lst) == 1:        
        or_lst = get_all(tok_lst[0] + "|", "|")
        if len(or_lst) > 1:
            str_lst_lst = []
            end_lst_lst = []
            for or_tok in or_lst:
                str_lst, end_lst = resolve_fs(or_tok + ";", 
                                              dc(prev_lst), fs_dict, 
                                              dataset)
                str_lst_lst += str_lst
                end_lst_lst += end_lst
            return str_lst_lst, end_lst_lst
        
        tok = or_lst[0]
        typ, tok = get_type(tok)
        
        if typ == "base":
            fs_id = make_base(tok, prev_lst, fs_dict, dataset)
            return [fs_id], [fs_id]
        elif typ == "neg":
            fs_id = make_neg(tok, prev_lst, fs_dict, dataset)
            return [fs_id], [fs_id]
        else:
            st_lst, end_lst = resolve_fs(tok + ";", prev_lst, 
                                          fs_dict, dataset)
            if typ == "star":
                # self loop
                for fs_id in st_lst:
                    fs_dict["fss"][fs_id]["prev"] += end_lst
                # can skip everything    
                end_lst += prev_lst
            elif typ == "plus":
                # self loop
                for fs_id in st_lst:
                    fs_dict["fss"][fs_id]["prev"] += end_lst
            elif typ == "ques":
                # can skip everything    
                end_lst += prev_lst
            return st_lst, end_lst
    else:
        st_lst_lst = []
        for tok in tok_lst:
            st_lst, end_lst = resolve_fs(tok + ";", prev_lst,
                                     fs_dict, dataset)
            st_lst_lst.append(st_lst)
            prev_lst = dc(end_lst)
        return dc(st_lst_lst[0]), dc(end_lst)   
    
def make_fst(seq, dataset):
    tok_lst = get_all(seq, ";")
    assert tok_lst[-1][0] not in ["+", "?", "*"] 
    
    fs_dict = init_fs()
    _, exit_lst = resolve_fs(seq, [0], fs_dict, dataset)
    exit_fs(exit_lst, fs_dict)
    
    return fs_dict