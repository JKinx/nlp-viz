cid2alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def decode_yz(dataset, ys, zs):
    batch_size = len(ys)
    sents = []
    states = []

    for idx in range(batch_size):
        sent = [dataset.id2word[el] for el in ys[idx]]
        state = [cid2alpha[el] for el in zs[idx]]
        sents.append(sent)
        states.append(state)

    return sents, states

def decode_bt(dataset, bts):
    batch_size = len(bts)
    
    batch = []
    for idx in range(batch_size):
        bt = bts[idx]
        decoded_bt = []

        for row in bt:
            decoded_bt.append([])
            for connection in row:
                connection["y"] = dataset.id2word[connection["y_raw"]]
                connection["z"] = cid2alpha[connection["z_raw"]]
                decoded_bt[-1].append(connection)
                
        batch.append(decoded_bt)
    
    return batch