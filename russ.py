
SIMPLE_WRITE_PATH = '<PATH_TO_WRITE_SIMPLIFICATIONS_IN_CHUNKS>'
UNIGRAM_PROBS_FILE = '<FILE_CONTAINING_UNIGRAM_PROBS_FOR_VOCAB>' # used for computing SLOR

predicates = []
import os
import re
import json
import torch
import string
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from nltk.tree import Tree

import math
import spacy
from spacy.tokens.span import Span
import en_core_web_md
nlp = en_core_web_md.load()

from get_token_mapping import *
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial


import re
import string
from collections import Counter
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc


USE_CUDA = torch.cuda.is_available()
device = torch.device(conf.device)

phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'SBAR']

predicate_dictionary = {'Abduct, hijack, or take hostage': ['kidnapped', 'abducting', 'abducted', 'captured'],
'Accuse': ['blame', 'blaming', 'accused', 'alleged', 'accusing'],
'Apologize': ['apologize', 'apology'],
'Assassinate': ['carried out assassination of', 'assassinate'],
'Bring lawsuit against': ['is suing someone', 'sued', 'has sued', 'filed a suit against'],
'Demonstrate or rally': ['condemn', 'protest', 'demonstrate'],
'Arrest, detain, or charge with legal action': ['arrested', 'sentenced', 'detained', 'nabbed', 'captured', 'arresting', 'capture', 'jailed', 'routinely arrested', 'prosecuted', 'convicted'],
'Use conventional military force': ['killed', 'shelled', 'combating', 'shells', 'strikes', 'strike', 'kill']
}


class config(object):

    emb_dim        = 128
    dep_dim        = 128
    pos_dim        = 128
    vocab_size     = 15000
    hidden_dim     = 512
    num_layers     = 1
    num_directions = 2

    batch_size     = 16
    lr             = 0.00025
    decay_rate     = 0.99
    device         = 0
    epochs         = 0
    patience       = 5

conf = config()

def get_vocab(DATA_DIR):

    with open(os.path.join(DATA_DIR, 'qa_dataset_combined_train_word2id.json'), 'r') as f:
        word2id = json.load(f)

    with open(os.path.join(DATA_DIR, 'qa_dataset_combined_train_dep2id.json'), 'r') as f:
        dep2id = json.load(f)

    with open(os.path.join(DATA_DIR, 'qa_dataset_combined_train_pos2id.json'), 'r') as f:
        pos2id = json.load(f)

    with open(os.path.join(DATA_DIR, 'qa_dataset_combined_train_freqvocab.json'), 'r') as f:
        freq_vocab = json.load(f)


    return word2id,     dep2id,        pos2id,     freq_vocab

def prepare_sequence(word_seq_id, dep_seq_id, pos_seq_id, target_seq_id):
    seq_lens = [len(seq_id) for seq_id in word_seq_id]
    max_l    =  max(seq_lens)
    indexs = np.argsort(seq_lens)[::-1].tolist()
    # NOTE: We assume that the candidates are already sorted by non-increasing order of length

    word_seq_id=np.array(word_seq_id)
    dep_seq_id=np.array(dep_seq_id)
    pos_seq_id=np.array(pos_seq_id)
    target_seq_id=np.array(target_seq_id)

    word_seq_id1=[]
    dep_seq_id1=[]
    pos_seq_id1=[]
    target_seq_id1=[]
    mask=[]
    for w_seq, d_seq, p_seq, t_seq in zip(word_seq_id, dep_seq_id, pos_seq_id, target_seq_id):
        if len(w_seq)!=len(d_seq) or len(d_seq)!=len(p_seq) or len(p_seq)!=len(w_seq):
            print("sth wrong with w_seq, d_seq, p_seq")
        if not isinstance(w_seq, list):
            w_seq = w_seq.tolist()
            d_seq = d_seq.tolist()
            p_seq = p_seq.tolist()
            t_seq = t_seq.tolist()
        word_seq_id1.append(w_seq + [0]*(max_l-len(w_seq)))
        dep_seq_id1.append(d_seq +[0]*(max_l-len(d_seq)))
        pos_seq_id1.append(p_seq + [0]*(max_l-len(p_seq)))
        if conf.num_directions == 2:
            target_seq_id1.append(t_seq+[0]*(max_l-len(t_seq)-2))
            mask.append([1]*(len(w_seq)-2)+[0]*(max_l-len(w_seq)))
        else:
            target_seq_id1.append(t_seq+[0]*(max_l-len(t_seq)))
            mask.append([1]*len(w_seq)+[0]*(max_l-len(w_seq)))

    word_seq_tensor = torch.LongTensor(word_seq_id1).cuda(device)
    dep_seq_tensor = torch.LongTensor(dep_seq_id1).cuda(device)
    pos_seq_tensor = torch.LongTensor(pos_seq_id1).cuda(device)
    targ_seq_tensor = torch.LongTensor(target_seq_id1).cuda(device)

    return word_seq_tensor,\
           dep_seq_tensor, \
           pos_seq_tensor, \
           targ_seq_tensor, \
           seq_lens,\
           indexs,\
           np.array(mask),\
           max_l


class vanilla_RNN(nn.Module):
    def __init__(self, freq_vocab, word2id, dep2id, pos2id):
        super(vanilla_RNN, self).__init__()
        self.vocab_size = len(freq_vocab)
        self.word_embeddings = nn.Embedding(len(word2id)+1, conf.emb_dim)
        self.dep_embeddings  = nn.Embedding(len(dep2id)+1,  conf.dep_dim)
        self.pos_embeddings  = nn.Embedding(len(pos2id)+1,   conf.pos_dim)
        self.init_all_embeddings()

        #initialize RNN
        self.rnn = nn.RNN(conf.emb_dim+conf.dep_dim+conf.pos_dim,
                          conf.hidden_dim, conf.num_layers,
                          batch_first=True, bidirectional=True if conf.num_directions==2 else False)
        self.params_init(self.rnn.named_parameters())

        #initialize linear
        self.linear = nn.Linear(conf.hidden_dim*conf.num_directions, self.vocab_size+1)
        self.params_init(self.linear.named_parameters())

        self.freq_vocab=freq_vocab
        self.word2id=word2id
        self.dep2id=dep2id
        self.pos2id=pos2id

    def init_hidden(self, batch_size):
        h0 = torch.zeros(conf.num_layers*conf.num_directions, batch_size, conf.hidden_dim)
        return h0.cuda(device) if USE_CUDA else h0

    def init_all_embeddings(self):
        self.word_embeddings.weight = nn.init.xavier_uniform(self.word_embeddings.weight)
        self.dep_embeddings.weight = nn.init.xavier_uniform(self.dep_embeddings.weight)
        self.pos_embeddings.weight = nn.init.xavier_uniform(self.pos_embeddings.weight)

    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                print(name)
                nn.init.kaiming_normal(param, a=0, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal(param)


    def forward(self, word_seq_id, dep_seq_id, pos_seq_id, target_seq_ids, h0, is_training=False):
        word_padded_ids, dep_padded_ids, pos_padded_ids, target_padded_ids, seq_lens, indexs, mask, max_l = \
        prepare_sequence(word_seq_id, dep_seq_id, pos_seq_id, target_seq_ids)

        word_vecs = self.word_embeddings(word_padded_ids)
        dep_vecs  = self.dep_embeddings(dep_padded_ids)
        pos_vecs  = self.pos_embeddings(pos_padded_ids)
        input_x = torch.cat((word_vecs, dep_vecs, pos_vecs), 2)

        '''
        input_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_x, seq_lens, batch_first=True)
        out_pack, hx = self.rnn(input_seq_packed, self.hidden)
        out, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)
        '''
        out, hx = self.rnn(input_x, h0)

        if conf.num_directions==2:
            forward_out, backward_out = out[:, :-2, :conf.hidden_dim], out[:, 2:, conf.hidden_dim:]
            out_cat = torch.cat((forward_out, backward_out), dim=-1)

        logits = self.linear(out_cat if conf.num_directions==2 else out )
        probs=0
        if is_training==False:
            probs = F.softmax(logits, dim=2)

        return logits, probs, word_padded_ids, target_padded_ids, indexs, mask


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    start = 0
    end = max_len - 1
    if start > end:
        end = 0
    seq_range = torch.range(start, end).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda(device)

    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    target_flat = target.view(-1, 1)
    losses_flat = torch.gather(log_probs_flat, dim=1, index=target_flat) # negative sign removed

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
#     loss = losses.sum() / length.float().sum() # this calculates the average loss for the batch
    loss = torch.sum(losses, dim=1) / length  # get individual loss for each instance
    return loss


def generate_candidates(predictor, sent):
    """
    From a constituency parse tree removes phrase subtrees
    Returns a list of simplification candidates
    """
    op = predictor.predict_json({"sentence": sent})
    t = Tree.fromstring(op['trees'])

    phrases = []
    pos = t.treepositions()
    for i in range(len(pos)-1,1,-1):
        if not isinstance(t[pos[i]], str):
            if t[pos[i]].label() in phrase_tags:
                phrases.append(t[pos[i]].leaves())


    cand_set = set()
    # remove
    for cand in phrases:
        new = sent.replace(' '.join(cand), '')
        l = len(new.split(' '))
        if l > 5:
            cand_set.add((new, l))

    # extract
    extractions = []
    pos = t.treepositions()
    for i in range(len(pos)-1,1,-1):
        if not isinstance(t[pos[i]], str):
            if t[pos[i]].label() in ['S', 'SBAR']:
                extractions.append(t[pos[i]].leaves())
    for cand in extractions:
        new = ' '.join(cand)
        l = len(new.split(' '))
        if l > 5:
            cand_set.add((new, l))

    sorted_cand =  sorted(list(cand_set), key=lambda x: x[1], reverse=True)  # sort according to non-increasing length

    return zip(*sorted_cand) # returns sentences and lengths


def tokenize(sentences, word2id,     dep2id,        pos2id,     freq_vocab):
    """
    Tokenize sentences for Syntax-LM
    """
    vocab_set=set(freq_vocab)
    word_seq = []
    dep_seq = []
    pos_seq = []

    for sentence in sentences:
        sentence = ' '.join(sentence.split())
        sentence = [w.strip(string.punctuation) for w in sentence.split(' ')]
        final  = []
        for w in sentence:
            if '-' in w:
                w = '@'.join(w.split('-'))
            if "'" in w:
                w = '@'.join(w.split("'"))
            if '/' in w:
                w = ' '.join(w.split('/'))
            if w != '':
                final.append(w)
        sentence = ' '.join(final)
        doc = nlp(sentence)
        word_seq.append([tok.text for tok in doc])
        pos_seq.append([tok.pos_ for tok in doc])
        dep_seq.append([tok.dep_ for tok in doc])

    word_seq_id=[]
    dep_seq_id=[]
    pos_seq_id=[]
    target_seq_id=[]
    for w_seq_i, d_seq_i, p_seq_i in zip(word_seq, dep_seq, pos_seq):
        temp_w=[]
        temp_dep=[]
        temp_pos=[]
        for w_i, d_i, p_i in zip(w_seq_i, d_seq_i, p_seq_i):
            if w_i in vocab_set:
                temp_w.append(word2id[w_i])
            else:
                temp_w.append(len(freq_vocab))

            temp_dep.append(dep2id[d_i])
            temp_pos.append(pos2id[p_i])

        word_seq_id.append(temp_w)
        dep_seq_id.append(temp_dep)
        pos_seq_id.append(temp_pos)
        target_seq_id.append(temp_w[1:-1])

    return word_seq_id, dep_seq_id, pos_seq_id, target_seq_id


def get_unigram_prob_value(unigram_prob, word):
    if unigram_prob is None:
        return 1
    elif word in unigram_prob:
        return unigram_prob[word]
    else:
        return 1

def get_unigram_prob(sent, unigram_prob):
	prob = 1.0
	for i in sent.split(' '):
		prob += math.log(get_unigram_prob_value(unigram_prob, i))
	return prob/(len(sent.split(' ')))


def lm_score(candidates, lm_model, word2id, dep2id, pos2id, freq_vocab):
    """
    Calculates the SLOR score for candidates
    """
    # tokenize candidates
    word_seq_id, dep_seq_id, pos_seq_id, target_seq_id = tokenize(candidates, word2id,     dep2id,        pos2id,     freq_vocab)
    batch_size = len(candidates)
    h0 = lm_model.init_hidden(batch_size)
    with torch.no_grad():
        logits, probs, word_padded_ids, target_padded_ids, indexs, mask = \
        lm_model(word_seq_id[:batch_size],
                 dep_seq_id[:batch_size],
                 pos_seq_id[:batch_size],
                 target_seq_id[:batch_size],
                 h0,
                 is_training=False)

    try:
        seq_lens = torch.sum(torch.LongTensor(mask).cuda(device), 1)

        test_loss = compute_loss(logits, target_padded_ids, seq_lens)

        with open(UNIGRAM_PROBS_FILE, 'r') as f:
            unigram_probs = json.load(f)

        unigram_scores = [get_unigram_prob(cand, unigram_probs) for cand in candidates]
        slor = test_loss - torch.tensor(unigram_scores).to(device)  # lm sent prob [sum(ln(w_i))] - unigram sent prob [sum]
        scores =  10e4*torch.exp(slor)
        return scores.cpu().detach().numpy()

    except:
        print('Notorious bug encountered!')
        return np.zeros(len(candidates))


def f1_score(prediction, ground_truth):
    """
    F1 at character level
    """
    prediction_tokens = [ch for ch in normalize_answer(prediction)]
    ground_truth_tokens = [ch for ch in normalize_answer(ground_truth)]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_score(predictor_qa, candidates, q_actor, q_target, ans_actor, ans_target):
    for i, cand in enumerate(candidates):
        if len(cand.split(' ')) < 3:
             candidates.pop(i)

    L_simple = len(candidates)
    batch_json_actor = []
    batch_json_target = []
    for i in range(L_simple):
        batch_json_actor.append({"context": candidates[i], "question": q_actor})
        batch_json_target.append({"context": candidates[i], "question": q_target})

    actor_pred = predictor_qa.predict_batch_json(batch_json_actor)
    targ_pred = predictor_qa.predict_batch_json(batch_json_target)

    scores_actor = [actor_pred[i]['best_span_scores'] if 'best_span_scores' in actor_pred[i] else 0.0 for i in range(L_simple)]
    scores_target = [targ_pred[i]['best_span_scores'] if 'best_span_scores' in targ_pred[i] else 0.0 for i in range(L_simple)]

    return np.array(scores_actor), np.array(scores_target)

def predicate_score(candidates, predicates):
    """
    Check if atleast one predicate is present in a candidate
    """
    scores = []
    for cand in candidates:
        preds_present = False
        for pred in predicates:
            if re.search(r'\b{}'.format(pred), cand):
                preds_present = True
                break;
        if preds_present:
            scores.append(1)
        else:
            scores.append(0)
    return np.array(scores)

def length_score(candidates):
    return np.array([1/len(cand.split(' ')) for cand in candidates])

def entity_score(candidates, actor, target):
    scores = []
    for cand in candidates:
        if re.search(r'\b{}'.format(actor), cand) and re.search(r'\b{}'.format(target), cand):
             scores.append(1)
        else:
            scores.append(0)
    return np.array(scores)

def func_2parallelize(train_sentences, lm_model=None, word2id=None, dep2id=None, pos2id=None, freq_vocab=None):
    not_simplified = 0
    predictor_qa = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-05-26.tar.gz")
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    records = []
    name_recorded = False
    batch=1
    for i, data in enumerate(train_sentences):
        idx = data['idx'] # idx maintains idx of original dataset sentences

        if not name_recorded:
            name = idx
            name_recorded = True

        if 'event' not in data:
            print(idx)
            print('Record dropped no event!')
            continue;
        sent = data['sentence']
        q_act = data['q_actor']
        q_targ = data['q_target']
        actor =  data['actor']['name']
        target =  data['target']['name']
        predicates = predicate_dictionary[data['event']]

        prev_max = 0.0
        simplification_cand = sent
        num_iter = 0

        while num_iter<=5: # set a max threshold for number of iterations
            num_iter += 1
            try:
                candidates, lengths = generate_candidates(predictor, simplification_cand)
            except:
                print('Could not generate candidates')
                break;

            if len(candidates) >= 1:
                num_cands = min(15, len(candidates))
                candidates = candidates[-num_cands:]

                f_qa_act, f_qa_targ  = qa_score(predictor_qa, candidates, q_act, q_targ, actor, target)

                f_entity = entity_score(candidates, actor, target)

                f_predicate = predicate_score(candidates, predicates)
                f_lm = lm_score(candidates, lm_model, word2id, dep2id, pos2id, freq_vocab)
                total_score = np.power(f_predicate, 1)*np.power(f_qa_act, 5)*np.power(f_qa_targ, 1)*np.power(f_entity, 1)*np.power(f_lm, 1.5)

                curr_max = np.max(total_score)

                if curr_max > prev_max:
                    train_sentences[i]['simple'] = candidates[np.argmax(total_score)]
                    simplification_cand = candidates[np.argmax(total_score)]
                    prev_max = curr_max
                else:
                    break;

            if 'simple' not in train_sentences[i] or type(train_sentences[i]['simple']) != str:
                not_simplified+=1
                if type(train_sentences[i]['simple']) != str:
                    print('Red herring at idx: {}'.format(idx))
                train_sentences[i]['simple'] = 'blank'
                print('Parsing error occurred')
                break;
        records.append(train_sentences[i])
        if i%100 == 0 and i>0:
            with open(os.path.join(SIMPLE_WRITE_PATH, 'qa_dataset_combined_train_simple_edit_iterative_multi_reward_{}_{}.json'.format(name, batch)), 'w+') as f:
                for record in records[(batch-1)*100:(batch)*100]:
                    json.dump(record, f)
                    f.write('\n')
            batch += 1

    print('Total records: {}'.format(len(train_sentences)))


if __name__ == '__main__':
    DATA_DIR = '<DIR_CONTAINING_THE_GENERATED_QA_DATASET>'

    not_simplified = 0
    word2id,     dep2id,        pos2id,     freq_vocab = get_vocab(DATA_DIR)

    print('Loading model for score computation...')

    lm_model = vanilla_RNN(freq_vocab, word2id, dep2id, pos2id)
    lm_model.load_state_dict(torch.load(os.path.join(WRITE_PATH, 'lm_qa_combined_b8.pt')))
    lm_model.eval()
    if USE_CUDA:
        lm_model = lm_model.cuda(device)

    simple_map = dict() # sentence, simplification candidates
    records = []

    with open(os.path.join(DATA_DIR, 'qa_dataset_combined_train.json'), 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            data['idx'] = i
            records.append(data)

    func_2parallelize(records, lm_model=lm_model, word2id=word2id, dep2id=dep2id, pos2id=pos2id, freq_vocab=freq_vocab)
