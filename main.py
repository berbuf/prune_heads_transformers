import modif_bert
import transformers_pruned
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

def mask_unk(text):
    target_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    mask = [1] * len(target_ids[0])
    for i in range(1, len(target_ids[0]) - 1):
        if target_ids[i] == 100:   
            mask[i] = 0
    return mask

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

def loss_distrib(loss_fn, prediction_scores, teacher):
    input = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
    student = F.log_softmax(input, dim=-1)
    return loss_fn(student, teacher)

def loss_cross_entropy(loss_fn, prediction_scores, target_ids, index_target):
    input = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
    target = target_ids[0][index_target].view(-1)
    return loss_fn(input, target)

def print_tokens(prediction_scores, tokenizer, index_target):
    a = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
    d = np.argsort(-a[0].detach().numpy())
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

def init_gates(model, lr):
    for param in model.parameters():
        param.requires_grad = False
    model.init_head_gates()
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def gates_values(model):
    gates = list(filter(lambda p: p.requires_grad, model.parameters()))
    return np.array([ e.cpu().detach().numpy().flatten() for e in gates ])

def round_gates(gates):
    gates[gates < 0.5] = 0
    gates[gates > 0.5] = 1
    return gates

def get_target(target_ids, language_model, bert_model, index_target):
    with torch.no_grad():
        outputs = bert_model(target_ids)
        prediction_scores = language_model(outputs[0])
        #target_scores = prediction_scores[0][index_target].view(-1, prediction_scores.shape[2])
        #teacher = F.softmax(target_scores, dim=-1)
        return prediction_scores#, teacher

def prune_heads(tokenizer, bert_model, language_model, sigma, index_target, target_ids, input_ids, epochs, loss_fn, optimizer):
    #prediction_scores = language_model(bert_model(input_ids)[0])
    #print_tokens(prediction_scores, tokenizer, index_target)
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction_scores = language_model(bert_model(input_ids)[0])
        l = loss_cross_entropy(loss_fn, prediction_scores, target_ids, index_target)
        n = norm1(bert_model)
        l += (sigma * n)
        l.backward()
        optimizer.step()
        clip_value(bert_model)
    #print_tokens(prediction_scores, tokenizer, index_target)
    print ("{}".format(n.item()))
    return gates_values(bert_model)    

def main():
    lr = 0.1
    sigma = 0.1
    epochs = 50

    weights = 'bert-base-uncased'

    tokenizer = transformers_pruned.transformers.BertTokenizer.from_pretrained(weights)

    model = modif_bert.BertForMaskedLM.from_pretrained(weights)
    bert_model = model.bert
    language_model = model.cls

    KD_loss = nn.KLDivLoss(reduction='batchmean')
    CE_loss = nn.CrossEntropyLoss()

    corpus = open("../corpus.txt").readlines()
    for text, pos, dep in gen_sent(corpus):
        
        #mask = mask_unk(text)
        print (" ".join(text))

        res = []
        for input_ids, target_ids, index_target, tokens in encode_ids(text, tokenizer):
            
            print ()
            print (tokens[index_target], pos[index_target-1], dep[index_target-1])
            #print ("*#*\n")

            optimizer = init_gates(bert_model, lr)
            #target_scores = get_target(target_ids, language_model, bert_model, index_target)
            #print_tokens(target_scores, tokenizer, index_target)

            gates = prune_heads(tokenizer, bert_model, language_model, sigma, index_target, target_ids, input_ids, epochs, CE_loss, optimizer)
            #print (round_gates(gates.copy()).flatten())
            #print ()
            res += [{"gates": gates.flatten().tolist(), "token": tokens[index_target], "pos": pos[index_target-1], "dep": dep[index_target-1] }]

        with open("./res.json", "a") as f:
            f.write(json.dumps({"text": tokens, "tokens": res }) + "\n")

main()
