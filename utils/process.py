import time
from tqdm import tqdm

import pickle

import torch
from torch.optim import Adam

from sklearn.metrics import confusion_matrix

from utils.help import NormalMetric, ReferMetric
from utils.help import iterable_support, expand_list

from transformers import AdamW
from nn import vat

def _save_confusion_matrix(sent_matrix, act_matrix, cm_name):

    # For the emotions/sentiment

    with open(cm_name + 'sent_matrix.pickle', 'wb') as handle:
        pickle.dump(sent_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # For the dialog act
    with open(cm_name + 'act_matrix.pickle', 'wb') as handle:
        pickle.dump(act_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None

def training(model, data_iter, max_grad=10.0, bert_lr=1e-5, pretrained_model="none"):

    model.train()

    # using pretrain model need to change optimizer (Adam -> AdamW).
    if pretrained_model != "none":
        optimizer = AdamW(model.parameters(), lr=bert_lr, correct_bias=False)
    
    else:
        optimizer = Adam(model.parameters(), weight_decay=1e-8)
    
    time_start, total_loss = time.time(), 0.0

    for data_batch in tqdm(data_iter, ncols=50):

        batch_loss = model.measure(*data_batch)
        total_loss += batch_loss.cpu().item()

        optimizer.zero_grad()
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad
        )

        optimizer.step()

    time_con = time.time() - time_start
    return total_loss, time_con

def vat_training(model, data_iter, max_grad=10.0, bert_lr=1e-5, pretrained_model="none"):

    model.train()
    perturbation_level = 'bilstm_layer'

    # using pretrain model need to change optimizer (Adam -> AdamW).
    if pretrained_model != "none":
        optimizer = AdamW(model.parameters(), lr=bert_lr, correct_bias=False)
    
    else:
        optimizer = Adam(model.parameters(), weight_decay=1e-8)

    time_start, total_loss = time.time(), 0.0

    for utt_list, _, _, adj_list, adj_full_list, adj_id_list in tqdm(data_iter, ncols=50):

        vat_loss = vat.perform_vat(model, perturbation_level, utt_list, adj_list, adj_full_list, adj_id_list)

        total_loss += vat_loss.cpu().item()

        optimizer.zero_grad()
        vat_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad
        )

        optimizer.step()

    time_con = time.time() - time_start
    return total_loss, time_con


def evaluate(model, data_iter, mastodon_metric, cm_name):

    model.eval()

    gold_sent, pred_sent = [], []
    gold_act, pred_act = [], []
    time_start = time.time()

    for utt, sent, act, adj, adj_full, adj_I in tqdm(data_iter, ncols=50):
        gold_sent.extend(sent)
        gold_act.extend(act)

        with torch.no_grad():
            p_sent, p_act = model.predict(utt, adj, adj_full, adj_I)
        
        pred_sent.extend(p_sent)
        pred_act.extend(p_act)

    if not mastodon_metric:

        reference = ReferMetric(
            len(model.sent_vocab), len(model.act_vocab),
            model.sent_vocab.index("+"), model.sent_vocab.index("-")
        )

    else:
        reference = NormalMetric()

    pred_sent = iterable_support(model.sent_vocab.index, pred_sent)
    gold_sent = iterable_support(model.sent_vocab.index, gold_sent)

    pred_act = iterable_support(model.act_vocab.index, pred_act)
    gold_act = iterable_support(model.act_vocab.index, gold_act)

    pred_sent = expand_list(pred_sent)
    gold_sent = expand_list(gold_sent)

    pred_act = expand_list(pred_act)
    gold_act = expand_list(gold_act)

    sent_f1, sent_r, sent_p = reference.validate_emot(pred_sent, gold_sent)
    act_f1, act_r, act_p = reference.validate_act(pred_act, gold_act)

    if cm_name is not None:

        # Get the confusion matrix
        act_matrix = confusion_matrix(gold_act, pred_act)
        sent_matrix = confusion_matrix(gold_sent, pred_sent)

        # Save the confusion matrix
        _save_confusion_matrix(sent_matrix, act_matrix, cm_name)

    time_con = time.time() - time_start
    return sent_f1, sent_r, sent_p, act_f1, act_r, act_p, time_con
