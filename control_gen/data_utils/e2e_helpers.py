import torch

def dlex(sent_lst, keys_lst, vals_lst):
    sent_dlex_lst = []
    for sent, keys, vals in zip(sent_lst, keys_lst, vals_lst):
        sent_dlex = []
        for w in sent:
            in_table = False
            for k, v in zip(keys, vals):
                if(w == v):
                    sent_dlex.append(k)
                    in_table = True
                    break
            if(in_table == False):
                sent_dlex.append(w)
        sent_dlex_lst.append(torch.tensor(sent_dlex).to(sent.device))
    return torch.stack(sent_dlex_lst)

e2e_key_names = ['area',
                 'customer_rating',
                 'eattype',
                 'familyfriendly',
                 'food',
                 'name',
                 'near',
                 'pricerange']


def e2e_dict_to_kvs(entry):
    """entry is a dict of keys and vals"""
    kvs = []
    for key in entry:
        assert key in e2e_key_names
        val = entry[key].split(" ")
        
        for i in range(len(val)):
            k = "_" + key + "_" + str(i)
            v = val[i]
            kvs.append((k, v))

    return kvs


def e2e_decode_out(dataset, ys, zs):
    batch_size = len(ys)
    sents = []
    states = []

    for idx in range(batch_size):
        sent = [dataset.id2word[el] for el in ys[idx]]
        sents.append(sent)
        states.append(zs[idx])

    return sents, states