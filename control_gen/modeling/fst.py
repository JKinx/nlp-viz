from copy import deepcopy as dc

cid2alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alpha2cid = {}
count = 0
for c in cid2alpha:
    alpha2cid[c] = count
    count += 1
    
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
            enc_tok = alpha2cid[tok]
        except:
            raise TypeError("Token needs to be an alphabet or .")
    return yz, enc_tok

def strip_paren(s):
    if s[0] == "(" and s[-1] == ")":
        return s[1:-1]
    else:
        return s
    
def get_type(s):
    if s[0] == "!":
        return "neg", strip_paren(s[1:])
    elif s[-1] == "*":
        return "star", strip_paren(s[:-1])
    elif s[-1] == "+":
        return "plus", strip_paren(s[:-1])
    elif s[-1] == "?":
        return "ques", strip_paren(s[:-1])
    else:
        if len(s) == 1:
            return "base", s
        elif s[0] == "[" and s[-1] == "]":
            return "base", s
        else:
            return "nested", strip_paren(s)         

def get_next_or(s):
    tok = ""
    nesting = 0
    i = 0
    while True:
        if s[i] == "|" and nesting == 0:
            break
            
        if s[i] == "(":
            nesting += 1
        elif s[i] == ")":
            nesting -= 1
        
        tok += s[i]
        i += 1
    return tok, s[i+1:]

def get_all_or(s):
    tok_lst = []
    s += "|"
    while s!= "":
        tok, s = get_next_or(s)
        tok_lst.append(tok)
    return tok_lst

def get_next(s):
    # single token
    if s[0] in cid2alpha + ".":
        if s[1] in ["*", "+", "?"]:
            return s[0:2], s[2:]
        else:
            return s[0], s[1:]
    # negation
    elif s[0] == "!":
        tok, s2 = get_next(s[1:])
        return "!" + tok, s2
    # single word
    elif s[0] == "[":
        tok = "["
        i = 1
        while s[i] != "]":
            tok += s[i]
            i += 1
        tok += "]"
        return tok, s[i+1:]
    else:
        assert s[0] == "("
        
        tok = "("
        nesting = 0
        i = 1
        while True:
            if s[i] == ")" and nesting == 0:
                tok += ")"
                i += 1
                break
            if s[i] == "(":
                nesting += 1
            elif s[i] == ")":
                nesting -= 1
            tok += s[i]
            i += 1
        if s[i] in ["*", "+", "?"]:
            return tok + s[i], s[i+1:]
        else:
            return tok, s[i:]
        
def get_all(s):
    tok_lst = []
    s += "0"
    while s!= "0":
        tok, s = get_next(s)
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
    # break by | (OR operator)
    tok_lst = get_all_or(seq)
    
    if len(tok_lst) == 1:
        # break sequence
        and_lst = get_all(tok_lst[0])
        
        # recurse is more than one token
        if len(and_lst) > 1:
            st_lst_lst = []
            for tok in and_lst:
                st_lst, end_lst = resolve_fs(tok, prev_lst,
                                         fs_dict, dataset)
                st_lst_lst.append(st_lst)
                prev_lst = dc(end_lst)
            return dc(st_lst_lst[0]), dc(end_lst)
        
        tok = and_lst[0]
        typ, tok = get_type(tok)
        
        if typ == "base":
            fs_id = make_base(tok, prev_lst, fs_dict, dataset)
            return [fs_id], [fs_id]
        elif typ == "neg":
            fs_id = make_neg(tok, prev_lst, fs_dict, dataset)
            return [fs_id], [fs_id]
        else:
            st_lst, end_lst = resolve_fs(tok, prev_lst, 
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
        str_lst_lst = []
        end_lst_lst = []
        for tok in tok_lst:
            str_lst, end_lst = resolve_fs(tok, dc(prev_lst), 
                                          fs_dict, dataset)
            str_lst_lst += str_lst
            end_lst_lst += end_lst
        return str_lst_lst, end_lst_lst
     
def make_fst(seq, dataset):    
    assert seq[-1] not in ["+", "?", "*"] 
    
    fs_dict = init_fs()
    _, exit_lst = resolve_fs(seq, [0], fs_dict, dataset)
    exit_fs(exit_lst, fs_dict)
    
    return fs_dict