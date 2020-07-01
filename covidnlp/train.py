import argparse
import logging
import os
from tqdm import tqdm

import torch
import torch.optim as O
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup, AdamW

from covidnlp.data_loader import load_tasks, TaskLoader
from covidnlp.model import Model, ActiveColumnModel, KnowledgeBaseModel, BertModelAC, BertModelKB
from covidnlp.utils import init_gpu_args, set_seed


parser = argparse.ArgumentParser(description='Train')

# dirs
parser.add_argument('--data_dir', '-D', dest='data_dir', help='Directory containing preprocessed special BERT data')
parser.add_argument('--dump_dir', '-O', dest='dump_dir', help='Root directory to dump all logs and checkpoints')
parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints')
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

parser.add_argument('--n_gpu', default=1, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--seed', default=666, type=int)

parser.add_argument('--log_interval', default=250, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)


class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    args = parser.parse_args()
    init_gpu_args(args)
    set_seed(args)

    args.data_dir = os.path.abspath(args.data_dir)
    args.dump_dir = os.path.abspath(args.dump_dir)
    if args.load_ckpt_dir is not None:
        if not os.path.exists(args.load_ckpt_dir):
            print(f'Load checkpoint directory does not exist - {args.load_ckpt_dir}')
            args.load_ckpt_dir = None
        else:
            args.load_ckpt_dir = os.path.abspath(args.load_ckpt_dir)
            pre_path_kb = os.path.join(args.load_ckpt_dir, 'KB')
            pre_path_ac = os.path.join(args.load_ckpt_dir, 'AC')


    for dname in ['logs', 'checkpoints']:
        sdir = os.path.join(args.dump_dir, dname)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
    # args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args_ = Map(vars(args))


    # device = torch.device(f'cuda:{args.local_rank}' if args.n_gpu > 0
    #                       and torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if args.n_gpu > 0 and torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.deterministric = True

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # task_loader = TaskLoader(data_dir=args.data_dir, task_size=1, batch_size=args.batch_size,
    #                          max_pos=args.max_pos, max_tgt_len=args.max_tgt_len, shuffle=True, is_test=False)

    tasks = load_tasks(args.data_dir, 'train', shuffle=False)
    task_loader = TaskLoader(tasks, batch_size=args.batch_size, shuffle=True, is_test=False,
                             max_pos=args.max_pos, max_tgt_len=args.max_tgt_len)

    config = BertConfig.from_pretrained('bert-base-uncased')
    # print(config)
    # config.max_position_embeddings = args.max_pos

    if args.load_ckpt_dir is not None:
        BertKB = BertModelKB.from_pretrained(pre_path_kb, config=config)
        print(f'... Bert KB loaded from pretrained, path: {pre_path_kb}')
        BertAC = BertModelAC.from_pretrained(pre_path_ac, config=config, knowledge_base=BertKB, device=device)
        print(f'... Bert AC loaded from pretrained, path: {pre_path_ac}')
    else:
        BertKB = BertModelKB(config=config)
        BertAC = BertModelAC(config=config, knowledge_base=BertKB, device=device)

    KB = KnowledgeBaseModel(BertKB, device, args_) # student
    AC = ActiveColumnModel(BertAC, args_) # teacher

    # optim
    no_decay = ['LayerNorm.weight']
    optim_kb_p = [{
            'params': [p for n, p in KB.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay,
        }, {
            'params': [p for n, p in KB.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        }]
    optim_ac_p = [p for p in AC.parameters() if p.requires_grad]
    optim = {
        'KB': AdamW(optim_kb_p, lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2)),
        'AC': AdamW(optim_ac_p, lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2)),
    }

    # loss
    loss_progress_fn = nn.BCELoss()

    model = Model(
        AC=AC, KB=KB, optimAC=optim['AC'], optimKB=optim['KB'],
        loss_progress_fn=loss_progress_fn, device=device, args=args_)

    if args.n_gpu > 0:
        # model.AC.to(f'cuda:{args.local_rank}')
        # model.KB.to(f'cuda:{args.local_rank}')
        model.AC.to('cuda')
        model.KB.to('cuda')

    if args.multi_gpu:
        torch.distributed.barrier()

    # print(model.AC)
    # print(model.KB)

    torch.cuda.empty_cache()

    for i, (data_loader, data_loader_ewc) in enumerate(task_loader):
        if i < args.task_start:
            print(f"... Skipping task {i}")
            continue
        print(f"... Loading new task into the model, {i}")
        model.new_task(data_loader, i)
        model.KB.new_task(data_loader_ewc)

        """ Progress step
        """
        # reset for normal training for AC
        model.AC.train()
        model.KB.eval()

        for _ in tqdm(range(args.n_epoch), desc='==> Progress Epoch', disable=args.local_rank not in [-1, 0]):
            model.progress()

        # if args.is_master:
        #     print('... Saving last checkpoint of Progress step')
        #     # self.save_checkpoint(checkpoint_name='pytorch_model.pt')
        #     print('... Progress on task {} is done'.format(i))

        """ Compress step
        """
        model.AC.eval() # teacher
        model.KB.train() # student

        ###
        # Should perform compress only once or for n epoch?
        ###
        for _ in tqdm(range(args.n_epoch), desc='==> Compress Epoch', disable=args.local_rank not in [-1, 0]):
            model.compress()

        """ Fisher estimation
        """
        # upon completing each task run, estimate fisher for KB
        # so the first task doesn't have any EWC loss (which is desired)
        model.KB.estimate_fisher()

        # if args.is_master:
        #     print('... Saving last checkpoint of Compress step')
        #     # self.save_checkpoint(checkpoint_name='pytorch_model_com.pt')
        #     print('... Compress on task {} is done'.format(i))
