-"""
BERT with a token classification head on the top.
"""
import torch
import os
from os import listdir
from os.path import isfile, join
import json

from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, BertConfig
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np

from tqdm import tqdm
import time
import string
import math
import datetime
import pandas as pd

HOME_DIR = '<PATH_TO_HOME_DIR>'
DATA_DIR = '{}/qa_dataset_simple_multiproc_2proc/'.format(HOME_DIR)
WRITE_DIR = '<PATH_TO_WRITE_DIR>'
# base_device_no = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MAX_LEN = 512
DATA_PARALLEL = True

class Preprocessor:
    def __init__(self, corpus_dir, train_file='dataset.txt', test_file='test.txt'):
        records = []
        df = pd.DataFrame()
        for fi in os.listdir(data_dir):
            if fi.split('.')[-1] == 'json':
                df_ = pd.read_json(os.path.join(data_dir, fi), lines=True)
                df = pd.concat([df, df_])

        df = df[df['simple']!='blank']
        print('Total number of records read = {}'.format(len(df)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        df_test = pd.read_json(os.path.join(DATA_DIR, 'qa_dataset_16_test.json'), lines=True)
        _ctx_te, _input_type_ids_te, _attn_mask_te, _ans_st_te, _ans_end_te = self.get_data(df_test)


        self.X_te = _ctx_te
        self.te_input_type_ids = _input_type_ids_te
        self.te_attn_mask = _attn_mask_te
        self.te_st_pos = _ans_st_te
        self.te_end_pos = _ans_end_te


    def find_ans_idx(self, ctx_toks, ans_toks):
        """
        Find the start and end indices of corrected ans tokens in context tokens
        """
        i = 0
        ans_found = False
        ans_start = -1
        ans_end = -1
        ans_toks_ = []
        for tok in ans_toks:
            if tok != '-':
                ans_toks_.append(tok)
        ans_toks = ans_toks_
        while i < len(ctx_toks):
            st = i
            j = 0
            while j < len(ans_toks) and (ctx_toks[i] == ans_toks[j] or ctx_toks[i][:-1] == ans_toks[j]) :
                j+=1
                i+=1
            if j == len(ans_toks):  # ans found
                ans_start = st
                ans_end = st+j
                ans_found = True
                break;

            if not ans_found:
                i+=1
            else:
                break;


        return ans_start, ans_end

    def correct_answer(self, ctx, name):
        """
        Correct the name to match with token in ctx
        """
        corrected_name = []
        for i, tok in enumerate(name.split()):
            ctx_tokens = ctx.split()
            correction_found = False
            for j, ctx_tok in enumerate(ctx_tokens):
                if (ctx_tok=='personnel' and tok=='person') or (ctx_tok=='children' and tok=='child') or ((ctx_tok=='minister' or ctx=='ministry') and (tok=='minist' or tok=='minis')) or (ctx_tok=='governorate' and tok=='governor') or (ctx_tok=='congressman' and tok=='congress') or (ctx_tok=='businessman' and tok=='business' or tok=='man') or (ctx_tok=='gangster' and tok=='gang') or (ctx_tok=='fishermen' and tok=='fisherm'):
                    correction_found = True
                    corrected_name.append(ctx_tok)
                    break;
                elif ctx_tok[:-1] == tok and tok != 'al': # ctx_tok=='protestors', tok=='protestor': exception for the prefix al as in al shabab
                    if i>0 and corrected_name[-1]==ctx_tokens[j-1]: # if this is the second token check if the prev token  in the ctx matches the previously corrected answer tok
                        correction_found = True
                        corrected_name.append(ctx_tok)
                    elif i==0: # first token
                        correction_found = True
                        corrected_name.append(ctx_tok)
                    break;
                elif ctx_tok == tok:
                    corrected_name.append(ctx_tok)
                    break;

        return ' '.join(corrected_name)

    def exact_match(self, name, ctx):
        """
        check if str name appears as an exact match in str ctx
        """
        name_toks = name.split()
        ctx_toks  = ctx.split()
        new_name = []
        i=0
        while i < len(ctx_toks):
            j=0
            while j < len(name_toks) and i < len(ctx_toks) and name_toks[j] == ctx_toks[i]:
                i+=1
                j+=1
            if j==len(name_toks):
                return True
            else:
                i+=1

        return False

    def get_tokens_and_masks(self, ctx, question):
        """
        Return padding input tokens, token type ids and attention mask
        """
        ctx_toks = self.tokenizer.tokenize(ctx)
        ques_toks = self.tokenizer.tokenize(question)
        inp = ['[CLS]'] + ques_toks + ['[SEP]'] + ctx_toks
        inp = inp[:BERT_MAX_LEN]
        L = len(inp)
        tok_type = [0 for _ in range(len(ques_toks)+2)] + [1 for _ in range(len(ctx_toks))]
        tok_type = tok_type[:BERT_MAX_LEN]
        mask = [1 for _ in range(L)]
        if L < BERT_MAX_LEN:  # padding
            inp += ['[PAD]' for _ in range(BERT_MAX_LEN-L)]
            tok_type += [0 for _ in range(BERT_MAX_LEN-L)]
            mask += [0 for _ in range(BERT_MAX_LEN-L)]
        if len(tok_type) > 512:
            import pdb; pdb.set_trace()
        return inp, tok_type, mask

    def get_data(self, df):
        df = df.reset_index(drop=True)
        dropped = 0
        questions, contexts, ans_start, ans_end = [], [], [], []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        skipped_indices = set()
        correct_actors = []
        correct_targets = []
        for idx, row in tqdm(df.iterrows(), desc="parsing training data..."):
            print(idx)
            ctx = row['sentence']
            question_act = row['q_actor']
            question_targ = row['q_target']

            actor_name = ' '.join([tok.encode("ascii", "ignore").decode().strip(string.punctuation) for tok in row['actor']['name'].split()])
            target_name = ' '.join([tok.encode("ascii", "ignore").decode().strip(string.punctuation) for tok in row['target']['name'].split()])

            if actor_name == 'eu':
                actor_name = 'european union'
            if target_name == 'eu':
                target_name = 'european union'

            if actor_name == 'interior minister':
                actor_name = 'minister of interior'

            if target_name == 'interior minister':
                target_name = 'minister of interior'

            if idx==5568:
                ctx = ctx.replace(';', ' ')

            if not self.exact_match(actor_name, ctx):
                actor_name = self.correct_answer(ctx, row['actor']['name'].replace('-', ' '))

            if not self.exact_match(target_name, ctx):
                target_name = self.correct_answer(ctx, row['target']['name'].replace('-', ' '))


            correct_actors.append(actor_name)
            correct_targets.append(target_name)
            # tokenize things
            act_inp, act_type_ids, act_mask = self.get_tokens_and_masks(ctx, question_act)
            targ_inp, targ_type_ids, targ_mask = self.get_tokens_and_masks(ctx, question_targ)

            ans_act_toks = self.tokenizer.tokenize(actor_name)
            ans_targ_toks = self.tokenizer.tokenize(target_name)

            st, end = self.find_ans_idx(act_inp, ans_act_toks)
            if end-st > 0: # actor valid
                ans_start.append(st)
                ans_end.append(end)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(act_inp))
                token_type_ids.append(act_type_ids)
                attention_mask.append(act_mask)
            else:
                dropped += 1
                skipped_indices.add(idx)
                print(row['actor']['name'])
                print('Answer not found - edge case')
                actor_valid = False

            st, end = self.find_ans_idx(targ_inp, ans_targ_toks)
            if end-st > 0:
                ans_start.append(st)
                ans_end.append(end)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(targ_inp))
                token_type_ids.append(targ_type_ids)
                attention_mask.append(targ_mask)

            else:
                dropped += 1
                skipped_indices.add(idx)
                print(row['actor']['name'])
                print('Answer not found - edge case')
                targ_valid = False



        data = {}
        data['input_ids'] = torch.tensor(input_ids)
        data['token_type_ids'] = torch.tensor(token_type_ids)
        data['attention_mask'] = torch.tensor(attention_mask)
        data['ans_start_idx'] = torch.tensor(ans_start)
        data['ans_end_idx'] = torch.tensor(ans_end)
        data['skipped_indices'] = torch.tensor(list(skipped_indices))
        torch.save(data, os.path.join(data_dir, 'processed_data/qa_dataset_combined_processed_tensors.pt'))
        return data['input_ids'], data['token_type_ids'], data['attention_mask'], torch.tensor(ans_start), torch.tensor(ans_end)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def batch_predict(preprocessor, model):
    batch_size = 8
    dataset = TensorDataset(preprocessor.X_te, preprocessor.te_input_type_ids, preprocessor.te_attn_mask, preprocessor.te_st_pos, preprocessor.te_end_pos)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    total_loss = 0
    model.cuda()
    model.eval()
    all_predictions = []
    all_labels = []
    all_start_indices = []
    for step, (inp_ids, inp_type_ids, inp_mask, st_pos, end_pos) in enumerate(dataloader):
        model.zero_grad()

        output = model(inp_ids.to(device), token_type_ids=inp_type_ids.to(device), attention_mask=inp_mask.to(device), start_positions=st_pos.to(device), end_positions=end_pos.to(device))
        loss = output.loss

        if DATA_PARALLEL:
            loss = loss.mean()

        total_loss += loss.item()

        start_scores = output.start_logits
        end_scores = output.end_logits
        start_scores = start_scores.cpu()
        end_scores = end_scores.cpu()

        candidate_scores = torch.max(start_scores, dim=1)[0] + torch.max(end_scores,dim=1)[0]
        answers = []
        start_indices = start_scores.argmax(dim=1)
        end_indices = end_scores.argmax(dim=1)

        for b_i in range(min(batch_size, inp_ids.shape[0])):
            all_predictions.append(preprocessor.tokenizer.convert_ids_to_tokens(inp_ids[b_i][start_indices[b_i]:end_indices[b_i]]))
            all_labels.append(preprocessor.tokenizer.convert_ids_to_tokens(inp_ids[b_i][st_pos[b_i]:end_pos[b_i]]))

    import pdb; pdb.set_trace()
    output_df = pd.DataFrame(columns=['predictions', 'labels'])
    output_df['predictions'] = pd.Series(all_predictions)
    output_df['labels'] = pd.Series(all_labels)

    output_df.to_csv(os.path.join(WRITE_DIR,qa_dataset_16_test_bert_results.csv'))
    print('Test Loss: {}'.format(total_loss/len(output_df)))
    print('Done')


def train():
    preprocessor = Preprocessor(data_dir)
    batch_size = 16
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', return_dict=True)
    if DATA_PARALLEL:
        model = torch.nn.DataParallel(model,  device_ids=[0, 1, 2])
    model.to(device="cuda:0")

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.module.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.module.classifier.named_parameters()) # for data parallel add model.module
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    max_epochs = 20
    max_grad_norm = 1.0


    # Create the DataLoader for our dataset.
    dataset = TensorDataset(preprocessor.X_tr, preprocessor.tr_input_type_ids, preprocessor.tr_attn_mask, preprocessor.tr_st_pos, preprocessor.tr_end_pos)
    dataset_size = len(dataset)
    validation_split = 0.2
    shuffle = True
    indices = list(range(dataset_size))
    random_seed = 42
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = math.floor(validation_split*dataset_size)
    val_indices, train_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_loader) * max_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_val_loss = None
    patience = 0
    max_patience = 3
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    all_train_losses = []
    all_val_losses = []
    for epoch in range(max_epochs):
        print("Epoch: {}".format(epoch))
        total_loss = 0
        model.train()
        t0 = time.time()
        loss_values = []
        for step, (inp_ids, inp_type_ids, inp_mask, st_pos, end_pos) in enumerate(train_loader):
            model.zero_grad()

            output = model(inp_ids.to(device), token_type_ids=inp_type_ids.to(device), attention_mask=inp_mask.to(device), start_positions=st_pos.to(device), end_positions=end_pos.to(device))
            loss = output.loss

            if DATA_PARALLEL:
                loss = loss.mean()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.   Train Loss: {:}'.format(step, len(train_loader), elapsed, loss))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_loader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        all_train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        for step, (inp_ids, inp_type_ids, inp_mask, st_pos, end_pos) in enumerate(validation_loader):
            with torch.no_grad():
                outputs = model(inp_ids.to(device), token_type_ids=inp_type_ids.to(device), attention_mask=inp_mask.to(device), start_positions=st_pos.to(device), end_positions=end_pos.to(device))
                loss = outputs.loss
                if DATA_PARALLEL:
                    loss = loss.mean()
            total_val_loss += loss.item()
            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Val Loss: {:}'.format(step, len(validation_loader), elapsed, loss))

        average_val_loss = total_val_loss/len(validation_loader)
        all_val_losses.append(average_val_loss)
        if (not best_val_loss) or (average_val_loss < best_val_loss):
            best_val_loss = average_val_loss
            torch.save(model.module.state_dict(), os.path.join(WRITE_DIR,qa_dataset_combined_icews_bert.pt'))
        else:
            patience += 1
        if patience == max_patience:
            break;

    print('Total Epochs passed: {}'.format(epoch))
    print('All train losses: {}'.format(' '.join(list(map(str, all_train_losses)))))
    print('All Val losses :{}'.format(' '.join(list(map(str, all_val_losses)))))

if __name__=='__main__':
    train()

    # inference
    preprocessor = Preprocessor(data_dir)
    bert_config = BertConfig.from_pretrained("bert-base-uncased", output_attentions = False,
    output_hidden_states = False, return_dict=True)
    model = BertForQuestionAnswering(bert_config)
    model.load_state_dict(torch.load(os.path.join(HOME_DIR, qa_dataset_combined_icews_bert.pt')))
    batch_predict(preprocessor, model)
