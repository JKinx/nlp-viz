""" Helper functions for parsing and preparing dataset from
the dateSet dataset
"""
from num2words import num2words
import torch

def dateSet_tuple_to_kvs(entry):
    day, month, year = entry
    assert day > 0 and day <= 31
    assert month > 0 and month <= 12
    assert year >= 2000 and year <= 2020

    if day < 10:
        day = "0 " + str(day)
    else:
        day = " ".join([el for el in str(day)])
    [day0, day1] = day.split(" ")
    month0 = str(month)
    year0 = str(year)

    return [("_day_0", day0), ("_day_1", day1), ("_month_0", month0), 
            ("_year_0", year0)]


def dateSet_decode_out(dataset, ys, zs):
    batch_size = len(ys)
    sents = []
    states = []

    for idx in range(batch_size):
        sent = [dataset.id2word[el] for el in ys[idx]]
        sents.append(sent)
        states.append(zs[idx])

    return sents, states


months = ["january", "february", "march", "april", "may", "june", "july", 
          "august", "september", "october", "november", "december"]
days_numerical = [num2words(n).replace("-", " ") for n in range(1, 32)]
days_ordinal = [num2words(n, ordinal=True).replace("-", " ") 
                    for n in range(1, 32)]
days = []
for day in days_ordinal + days_numerical:
    days += day.split()
days = list(set(days))
years = [str(year) for year in range(2000,2021)]

def dateSet_dlex(sent_lst, dataset):
    day = dataset.word2id["_day_"]
    month = dataset.word2id["_month_"]
    year = dataset.word2id["_year_"]
    sent_dlex_lst = []
    for sent in sent_lst:
        sent_dlex = []
        for w_id in sent:
            w = dataset.id2word[w_id.item()]
            if w in years:
                sent_dlex.append(year)
            elif w in months:
                sent_dlex.append(month)
            elif w in days:
                sent_dlex.append(day)
            else:
                sent_dlex.append(w_id)
        sent_dlex_lst.append(torch.tensor(sent_dlex).to(sent.device))
    return torch.stack(sent_dlex_lst)
