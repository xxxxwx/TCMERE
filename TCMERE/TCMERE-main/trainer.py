import os
import copy
import utils
import torch
import time
import sys
import random
import math
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from constants import *
from transformers import *
from data import load_data
from scorer import evaluate
from models import JointModel
from argparse import ArgumentParser
#CUDA_VISIBLE_DEVICES='0,1'
# Main Functions
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def main():
    sys.stdout = Logger("b.log", sys.stdout)
    # sys.stderr = Logger("a.log", sys.stderr)     # redirect std err, if necessary
    # now it works
    # print('print something')
    # print("*" * 3)
    # # sys.stdout.write("???")
    import time
    time.sleep(2)
    #print("other things")


def train(configs):
    tokenizer = AutoTokenizer.from_pretrained('./TCMroberta')
    train, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
    model = JointModel(configs)
    print('Train Size = {} | Dev Size = {}'.format(len(train), len(dev)))
    print('Initialize a new model | {} parameters'.format(get_n_params(model)))
    best_dev_score, best_dev_m_score, best_dev_rel_score = 0, 0, 0
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        checkpoint = torch.load(PRETRAINED_MODEL, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded a pretrained model')
        printC('Evaluation on the dev set')
        dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
        best_dev_score = (dev_m_score + dev_rel_score) / 2.0

    # Prepare the optimizer and the scheduler
    num_train_docs = len(train)
    num_epoch_steps = math.ceil(num_train_docs / configs['batch_size'])
    num_train_steps = int(num_epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start training
    accumulated_loss = RunningAverage()
    iters, batch_loss = 0, 0
    for i in range(configs['epochs']):
        print('Starting epoch {}'.format(i+1), flush=True)
        model.in_ned_pretraining = i < configs['ned_pretrain_epochs']
        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        for train_idx in tqdm(train_indices):
            iters += 1
            tensorized_example = [b.to(model.device) for b in train[train_idx].example]
            tensorized_example.append(train[train_idx].all_relations)
            tensorized_example.append(train[train_idx])
            tensorized_example.append(True) # is_training
            iter_loss = model(*tensorized_example)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, accumulated_loss()), flush=True)
                accumulated_loss = RunningAverage()

        # Evaluation after each epochi
        
        with torch.no_grad():
            print('Evaluation on the dev set')
            print('Evaluation on the dev set', configs['gnn_mode'])
            dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
            dev_score = (dev_m_score + dev_rel_score) / 2.0

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_dev_m_score = dev_m_score
            best_dev_rel_score = dev_rel_score
            # Save the model
            save_path = join(configs['save_dir'], 'model_{}.pt'.format(configs['split_nb']))
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)
        
    return {'all': best_dev_score, 'mention': best_dev_m_score, 'relation': best_dev_rel_score}

if __name__ == "__main__":
    main()
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.txt'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    parser.add_argument('-d', '--dataset', default=BIORELEX, choices=DATASETS)
    parser.add_argument('-s', '--split_nb', default=0) # Only affect ADE dataset
    args = parser.parse_args()
    args.split_nb = int(args.split_nb)

    # Start training
    configs = prepare_configs(args.config_name, args.dataset, args.split_nb)
    train(configs)
