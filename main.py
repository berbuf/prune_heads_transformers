import modif_bert
import transformers
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import spacy
import json

def gen_sent(corpus):
    nlp = spacy.load("en_core_web_sm")
    for doc in corpus:
        for sent in nlp(doc).sents:
            yield list(zip(*[ [t.text.lower(), t.pos_, t.dep_] for t in sent ]))

def encode_ids(text, tokenizer):
    target_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    tok = [ tokenizer._convert_id_to_token(t) for t in target_ids[0].numpy() ]
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    for index_target in range(1, len(input_ids[0]) - 1):
        if target_ids[0][index_target] == 100:
            continue
        input_ids[0][index_target] = 103
        yield input_ids, target_ids, index_target, tok
        input_ids[0][index_target] = target_ids[0][index_target]

def loss_cross_entropy(loss_fn, prediction_scores, target_ids, index_target):
    input = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
    target = target_ids[0][index_target].view(-1)
    return loss_fn(input, target)

def print_tokens(prediction_scores, tokenizer, index_target):
    a = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
    d = np.argsort(-a[0].detach().cpu().numpy())
    for i in d[:10]:
        print ("{:<20}{:<20}{}".format(tokenizer._convert_id_to_token(i), i, a[0][i]))
    print ()

def norm1(model):
    norm_l1 = 0
    for param in model.parameters():
        if param.requires_grad:
            norm_l1 += param.norm(p=1)
    return norm_l1

def clip_value(model, min=0., max=10.):
    for param in model.parameters():
        if param.requires_grad:
            param.data.clamp_(min=min, max=max)

def init_gates(model, lr, index_target):
    for param in model.parameters():
        param.requires_grad = False
    model.init_head_gates(index_target)
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def gates_values(model):
    gates = list(filter(lambda p: p.requires_grad, model.parameters()))
    return np.array([ e.cpu().detach().numpy().flatten() for e in gates ])

def att_prob_values(model):
    att_probs = model.retrieve_attention_prob()
    return [ [ 0 if sum(x) == 0 else x for x in  e.cpu().detach().numpy().tolist() ] for e in att_probs]

def get_target(target_ids, model, index_target):
    with torch.no_grad():
        prediction_scores = model(target_ids)
        return prediction_scores

def prune_heads(tokenizer, model, sigma, index_target, target_ids, input_ids, epochs, loss_fn, optimizer):
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction_scores = model(input_ids)
        l = loss_cross_entropy(loss_fn, prediction_scores, target_ids, index_target)
        n = norm1(model)
        l += (sigma * n)
        l.backward()
        optimizer.step()
        clip_value(model)
    print ("gates remaining: {}".format(n.item()))
    print_tokens(prediction_scores, tokenizer, index_target)
    return gates_values(model), model.retrieve_attention_prob()

def main():
    lr = 0.1
    sigma = 0.1
    epochs = 100

    weights = 'bert-base-uncased'

    tokenizer = transformers.BertTokenizer.from_pretrained(weights)

    model = modif_bert.BertForMaskedLM.from_pretrained(weights)
    model.to('cuda')

    KD_loss = nn.KLDivLoss(reduction='batchmean')
    CE_loss = nn.CrossEntropyLoss()

    former_texts = []
    with open("./res.json", "r") as f:
        former_texts = [ " ".join(json.loads(l)["text"]) for l in f.readlines() ]

    corpus = open("../corpus.txt").readlines()
    for text, pos, dep in gen_sent(corpus):

        print (" ".join(text))
        target_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        if " ".join([ tokenizer._convert_id_to_token(t) for t in target_ids[0].numpy() ]) in former_texts:
            continue

        res = []
        for input_ids, target_ids, index_target, tokens in encode_ids(text, tokenizer):
            
    
            print ()
            print (tokens[index_target], pos[index_target-1], dep[index_target-1])

            optimizer = init_gates(model, lr, index_target)

            target_ids, input_ids = target_ids.to("cuda"), input_ids.to("cuda")
            gates, att_probs = prune_heads(tokenizer, model, sigma, index_target, target_ids, input_ids, epochs, CE_loss, optimizer)
            res += [{"gates": gates.flatten().tolist(), "attention": att_prob_values(model), "token": tokens[index_target], "pos": pos[index_target-1], "dep": dep[index_target-1] }]

        with open("./res.json", "a") as f:
            f.write(json.dumps({"text": tokens, "tokens": res }) + "\n")

main()
