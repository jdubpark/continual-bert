import argparse
import os
import numpy as np
import re
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer

from covidnlp.data_loader import load_tasks, TaskLoader
from covidnlp.model import Model, ActiveColumnModel, KnowledgeBaseModel, BertModelAC, BertModelKB


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data_dir', '-D', dest='data_dir', help='Directory containing preprocessed special BERT data')
parser.add_argument('--dump_dir', '-O', dest='dump_dir', help='Root directory to dump all logs and checkpoints')
parser.add_argument('--load_ckpt_dir', '-C', dest='load_ckpt_dir', help='Directory to load checkpoints')

# train
parser.add_argument('--n_epoch', default=3, type=int)
parser.add_argument('--batch', '--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--gradient_accumulation_steps', default=50, type=int)
parser.add_argument('--task_start', default=0, type=int, help='Task start index, from 0')

# data
parser.add_argument('--max_pos', default=512, type=int)
parser.add_argument('--max_tgt_len', default=140, type=int)

# hyperparameter
# parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--wd', '--weight_decay', dest='weight_decay', default=0.01)
parser.add_argument('--max_grad_norm', default=5.0, type=float)
parser.add_argument('--warmup_prop', default=0.05, type=float, help='Linear warmup proportion')

# knowledge distillation
parser.add_argument('--temperature', default=2.0, type=float, help='Temperature for the softmax temperature')
parser.add_argument('--alpha_ce', default=0.5, type=float, help='Linear weight for the distillation loss, must be >=0')

# EWC
parser.add_argument('--ewc_lambda', default=15., type=float)
parser.add_argument('--ewc_gamma', default=.99, type=float)
parser.add_argument('--ewc_fisher_n', default=None, type=int)
parser.add_argument('--ewc_emp_fi', default=False, type=bool)

# Extract Stack
parser.add_argument('--extract_num_layers', default=2, type=int)
parser.add_argument('--extract_dropout_prob', default=.1, type=float)
parser.add_argument('--extract_layer_norm_eps', default=1e-12, type=float)


class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s))>0:
            return True
    return False


def run_pred(sent_scores, mask, src_txt):
    sent_scores_t = sent_scores + mask.float()
    sent_scores_t = sent_scores_t.cpu().data.numpy()
    selected_ids = np.argsort(-sent_scores_t, 1)

    pred = []
    for i, idx in enumerate(selected_ids):
        _pred = []
        if len(src_txt[i]) == 0:
            continue
        for j in selected_ids[i][:len(src_txt[i])]:
            if j >= len(src_txt[i]):
                continue
            candidate = src_txt[i][j].strip()
            if not _block_tri(candidate,_pred):
                _pred.append(candidate)
            if len(_pred) == 10:
                break

        _pred = re.sub('</q>', '', ' '.join(_pred))
        pred.append(_pred)
    return pred


if __name__ == '__main__':
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.dump_dir = os.path.abspath(args.dump_dir)
    pred_dump_dir = os.path.join(args.dump_dir, 'dump_pred')
    if not os.path.exists(pred_dump_dir):
        os.makedirs(pred_dump_dir)
    args.load_ckpt_dir = os.path.abspath(args.load_ckpt_dir)
    pre_path_kb = os.path.join(args.load_ckpt_dir, 'KB')
    pre_path_ac = os.path.join(args.load_ckpt_dir, 'AC')

    args_ = Map(vars(args))

    device = torch.device('cpu')
    config = BertConfig.from_pretrained('bert-base-uncased')

    BertKB = BertModelKB.from_pretrained(pre_path_kb, config=config)
    print(f'... Bert KB loaded from pretrained, path: {pre_path_kb}')
    BertAC = BertModelAC.from_pretrained(pre_path_ac, config=config, knowledge_base=BertKB, device=device)
    print(f'... Bert AC loaded from pretrained, path: {pre_path_ac}')

    KB = KnowledgeBaseModel(BertKB, device, args_) # student
    AC = ActiveColumnModel(BertAC, args_) # teacher

    # loss
    loss_progress_fn = nn.BCELoss()

    model = Model(
        AC=AC, KB=KB, optimAC=None, optimKB=None,
        loss_progress_fn=loss_progress_fn, device=device, args=args_)

    model.AC.eval()
    model.KB.eval()

    tasks = load_tasks(args.data_dir, 'train', shuffle=False)
    task_loader = TaskLoader(tasks, batch_size=args.batch_size, shuffle=True, is_test=False,
                             max_pos=args.max_pos, max_tgt_len=args.max_tgt_len)

    for i, (data_loader, data_loader_ewc) in enumerate(task_loader):
        # data_generator = data_loader()
        # dg_len = len(data_generator)

        pbar = tqdm(data_loader_ewc, total=len(data_loader_ewc), desc='==> Evaluation')
        print('... (1) Testing with {} data'.format(len(data_loader_ewc)))
        for batch in pbar:
            # src, tgt, segs, clss, mask_src, mask_tgt, mask_cls = self.read_batch(batch)
            batch_ = (batch.src, batch.segs, batch.clss, batch.mask_src, batch.mask_cls, batch.src_sent_labels, batch.src_txt, batch.tgt_txt)
            src, segs, clss, mask_src, mask_cls, src_sent_labels, src_txt, tgt_txt = batch_
            fpath = batch.fname[0]

            sent_scores, mask = model.AC(*batch_)

            pred = run_pred(sent_scores, mask, src_txt)
            pred_txt = '\n'.join(pred)
            # print(' \n'.join(pred))

            fname = fpath.split('/')[-1].split('.')[0]
            fname = f'{fname}.txt'
            fdump = os.path.join(pred_dump_dir, fname)
            with open(fdump, 'w') as f:
                f.write(pred_txt)
