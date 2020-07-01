import logging
import math
import os
import psutil
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup, AdamW

from .extract_stack import Classifier, TransformerStack

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


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


class SharedStructure(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bert = bert
        self.extract_stack = TransformerStack(self.bert.config.hidden_size, args)

        if args.max_pos > 512:
            # Bert is pretrained on 512 token (10%), so over 512 needs customization
            c_pos_embeddings = nn.Embedding(args.max_pos, self.bert.config.hidden_size)
            c_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
            c_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.embeddings.position_embeddings = c_pos_embeddings

        # initialize parameters of encoder stack
        for p in self.extract_stack.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, segs, clss, mask_src, mask_cls, src_sent_labels=None, src_txt=None, tgt_txt=None):
        # (src, segs, clss, mask_src, mask_cls, src_sent_labels)
        # (input_ids, token_type_ids, clss, attention_mask, mask_src, src_sent_labels)
        top_vec, _ = self.bert(input_ids=src, token_type_ids=segs, attention_mask=mask_src)
        # print(top_vec, clss)
        # print(top_vec.size(), clss.size())
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # print(sents_vec.size(), (mask_cls[:, :, None].float()).size())
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        # print(sents_vec.size(), mask_cls.size())
        sent_scores = self.extract_stack(sents_vec, mask_cls).squeeze(-1)
        # print(sent_scores)
        return sent_scores, mask_cls


class ActiveColumnModel(SharedStructure):
    """ Model for learning on new task T without any restrictions
    """
    def __init__(self, bert, args):
        super(ActiveColumnModel, self).__init__(bert, args)
        self.bert = bert


class KnowledgeBaseModel(SharedStructure):
    def __init__(self, bert, device, args):
        super(KnowledgeBaseModel, self).__init__(bert, args)
        self.bert = bert
        self.device = device
        self.n_gpu = args.n_gpu
        self.local_rank = args.local_rank

        # online EWC config
        self.ewc_lambda = args.ewc_lambda # how strong to weight EWC-loss - "regularization strength"
        self.ewc_gamma = args.ewc_gamma # decay-term for old tasks' contribution to qudratic term
        self.ewc_fisher_n = args.ewc_fisher_n # sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.ewc_emp_fi = args.ewc_emp_fi # if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.ewc_task_count = 0

    def new_task(self, data_loader_ewc):
        # just for loading new data loader
        self.data_loader_ewc = data_loader_ewc
        self.dl_len = len(data_loader_ewc)

    def estimate_fisher(self, label=None):
        # prepare to store estimated Fisher Information matrix
        self.est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.est_fisher_info[n] = p.detach().clone().zero_()

        self.eval()

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        pbar = tqdm(self.data_loader_ewc, total=self.dl_len, desc='==> EWC Estimate', disable=self.local_rank not in [-1, 0])
        idx = 0
        start = time.time()
        print('... (3.1) EWC estimate fisher with {} data'.format(self.dl_len))
        for batch in pbar:
            # break from for-loop if max number of samples has been reached
            if self.ewc_fisher_n is not None:
                if idx >= self.ewc_fisher_n:
                    break

            batch = (batch.src, batch.segs, batch.clss, batch.mask_src, batch.mask_cls, batch.src_sent_labels)
            if self.n_gpu > 0:
                # batch = tuple(t.to(f'cuda:{self.local_rank}') for t in batch)
                batch = tuple(t.to('cuda') for t in batch)

            logit, _ = self(*batch)

            if self.ewc_emp_fi and label is not None:
                # use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([label], device=self.device) if type(label) == int else label
            else:
                # use predicted label to calculate loglikelihood:
                # label = logit.max(1)[1]
                label = logit.squeeze(0).type(torch.LongTensor).to(self.device) # just [C] for nll_loss, where C is logits

            # log softmax then negative log-likelihood
            try:
                negloglikelihood = F.nll_loss(F.log_softmax(logit, dim=1).view(-1, 1), label)
            except IndexError:
                negloglikelihood = F.nll_loss(F.log_softmax(logit, dim=0).view(-1, 1), label)
            # print(negloglikelihood)

            # calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        self.est_fisher_info[n] += p.grad.detach() ** 2

            idx += 1
            pbar.update()
        pbar.close()

        end = time.time()
        print('... Ending (3.1), took {} seconds'.format(end-start))

        start = time.time()
        print('... (3.2) EWC estimate fisher, after steps')
        # normalize by sample size (final idx) used for estimation
        # allows computing the updates F^* based on the relative importance
        #   of weights in a network (treating each task equally)
        self.est_fisher_info = {n: p / idx for n, p in self.est_fisher_info.items()}

        # consolidate new Fisher Info with the old one
        # (as buffer to disable optimiziation)
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # mode (MAP parameter estimate)
                self.register_buffer('{}_mean'.format(n),
                                     p.detach().clone())

                # precision (approximated by diagonal Fisher Information matrix)
                if self.ewc_task_count > 0:
                    # if task is not first, consolidate with old params by
                    # self.gamma * existing_values, where gamma is importance
                    existing_values = getattr(self, '{}_fisher'.format(n))
                    # F_{i}^{*} = \gamma F_{i-1}^{*} + F_{i}
                    self.est_fisher_info[n] += self.ewc_gamma * existing_values

                self.register_buffer('{}_fisher'.format(n),
                                     self.est_fisher_info[n])

        self.ewc_task_count = 1 # set EWC-loss as now calculatable

        end = time.time()
        print('... Ending (3.2), took {} seconds'.format(end-start))

        # back to training
        self.train()

    def ewc_loss(self):
        """
        Calculate online EWC loss (P&C) for task k
        All \theta are of KB, omitted for brevity.
        :math:
        E[KL(\pi_{k}(\dot|x)||\pi^{KB}(\dot|x))]
            + online EWC

        where:
        E is expectation over dataset
        \pi_{k}(\dot|x) is prediction of AC (after learning on task K)
        \pi^{KB}(\dot|x)) is prediction of KB

        :math: `online EWC`
        ## -\log p(T_{i}|\theta) +
        \frac{1}{2} \lVert \theta -
            \theta_{k-1}^{*} \rVert_{\gamma F_{k-1}^{*}}^{2}

        where:
        \gamma < 1 is a hyperparameter that decays the Fisher Information matrices
            associated with the previous presentation of task i

        If :math:`\theta_{i}^{*}` is the optimized MAP parameter and :math:`\F_{i}` is
        the optimized Fisher for task i, the overall Fisher is updated as:
        :math:
        F_{i}^{*} = \gamma F_{i-1}^* + F_{i}

        This method treats all tasks equivalently and thus avoid needing to identify the
        task labels
        """
        if self.ewc_task_count > 0:
            losses = []
            # online, only 1 iteration
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # retrieve the consolidated mean and fisher information
                    # mode - mean: (latest) MAP estimate
                    # precision - F: running sum of the Fisher Information matrices
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher = getattr(self, '{}_fisher'.format(n))

                    # wrap in variables
                    # mean = Variable(mean)
                    # fisher = Variable(fisher)

                    # apply decay-term to the running sum of the Fisher Information matrices
                    fisher = self.ewc_gamma * fisher

                    # calculate EWC loss
                    loss = (fisher * (p - mean) ** 2).sum()
                    losses.append(loss)

            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet (first task)
            return torch.tensor(0., device=self.device)


class Model(nn.Module):
    def __init__(self, AC, KB, optimAC, optimKB, loss_progress_fn, device, args):
        super().__init__()
        self.device = device
        self.is_master = args.is_master
        self.max_grad_norm = args.max_grad_norm
        self.grad_accm_steps = args.gradient_accumulation_steps
        self.n_epoch = args.n_epoch # total epoch
        self.warmup_prop = args.warmup_prop
        self.n_gpu = args.n_gpu
        self.multi_gpu = args.multi_gpu
        self.local_rank = args.local_rank
        self.log_interval = args.log_interval
        self.checkpoint_interval = args.checkpoint_interval

        self.dump_dir = args.dump_dir
        self.log_dir = os.path.join(self.dump_dir, 'logs')
        self.ckpt_dir = os.path.join(self.dump_dir, 'checkpoints')

        self.alpha_ce = args.alpha_ce
        self.temperature = args.temperature

        self.AC = AC # teacher
        self.KB = KB # student

        self.task_count = -1 # initializing with new task will automatically add 1
        self.loss_pg_fn = loss_progress_fn

        self.optimAC = optimAC
        self.optimKB = optimKB

        if self.is_master:
            print('... Initializing Tensorboard')
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
            # self.tensorboard.add_text(tag="config/training", text_string=str(self.args), global_step=0)
            # self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)


    def new_task(self, task_data_loader, idx):
        self.epoch = {'pg': 0, 'cp': 0}
        self.n_iter = {'pg': 0, 'cp': 0}
        self.n_total_iter = {'pg': 0, 'cp': 0}
        self.n_sequences_epoch = {'pg': 0, 'cp': 0}
        self.n_optim = {'pg': 0, 'cp': 0}
        self.last_log = 0
        self.total_loss_epoch = {'pg': 0, 'cp': 0}
        self.last_loss = {'pg': 0, 'cp': 0}
        # only used for compress
        self.last_loss_kd = 0
        self.last_loss_ewc = 0

        self.data_loader = task_data_loader
        self.data_generator = task_data_loader()
        # def len must come before iter(obj) - also, it's fixed for data loader so no need to update
        self.dg_len = len(self.data_generator)
        self.data_generator = iter(self.data_generator)

        self.num_steps_epoch = self.dg_len
        num_train_optimization_steps = (
            int(self.num_steps_epoch / self.grad_accm_steps * self.n_epoch) + 1
        )
        # self.warmup_steps = math.ceil(num_train_optimization_steps * self.warmup_prop)
        self.warmup_steps = math.ceil(self.num_steps_epoch * self.warmup_prop)
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
        self.schedulerAC = get_linear_schedule_with_warmup(
            self.optimAC, num_warmup_steps=self.warmup_steps, num_training_steps=num_train_optimization_steps
        )
        self.schedulerKB = get_linear_schedule_with_warmup(
            self.optimKB, num_warmup_steps=self.warmup_steps, num_training_steps=num_train_optimization_steps
        )

        self.task_count += 1
        if isinstance(idx, int):
            self.task_count += idx

        print('... New task ({}) loaded'.format(self.task_count))
        print('... -- Total epoch steps: {}'.format(self.num_steps_epoch))
        print('... -- Total optimization iterations: {}'.format(num_train_optimization_steps))
        print('... -- Warmup epoch steps: {}'.format(self.warmup_steps))
        print('... -- Please note that optimization is performed every {} steps'.format(self.grad_accm_steps))

    def read_batch(self, batch):
        batch_ = (batch.src, batch.segs, batch.clss, batch.mask_src, batch.mask_cls, batch.src_sent_labels)
        # batch_ = (batch.src, batch.tgt, batch.segs, batch.clss, batch.mask_src, batch.mask_tgt, batch.mask_cls)
        return tuple(t.to(self.device) for t in batch_) + (batch.src_txt,) + (batch.tgt_txt,)

    def loss_kd(self, s_outputs, s_labels, t_outputs):
        """ Knowledge Distillation Loss
            KL Divergence for PyTorch comparing the softmaxs of teacher
            and student expects the input tensor to be log probabilities
        """
        T = self.temperature
        try:
            loss_kl = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(s_outputs / T, dim=1),
                F.softmax(t_outputs / T, dim=1)) * T ** 2
        except IndexError:
            loss_kl = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(s_outputs / T, dim=0),
                F.softmax(t_outputs / T, dim=0)) * T ** 2
        # binary cross entropy
        loss_ce = nn.BCELoss()(s_outputs, s_labels.float())

        loss = self.alpha_ce * loss_kl + (1. - self.alpha_ce) * loss_ce
        return loss

    def progress(self):
        """
        PROGRESS:
        Training of Active Column, with layerwise connectionbs from Knowledge Base
        to reuse previously learned features (positive forward transfer)
        """
        self.last_log = time.time()

        # reuse data loader after exhausting it
        if self.epoch['pg'] > 0:
            try:
                next(self.data_generator)
            except StopIteration:
                self.data_generator = iter(self.data_loader())

        pbar = tqdm(self.data_generator, total=self.dg_len, desc='==> Progress steps',
                    disable=self.local_rank not in [-1, 0])
        print('... (1) Progress training with {} data'.format(self.dg_len))
        for batch in pbar:
            # src, tgt, segs, clss, mask_src, mask_tgt, mask_cls = self.read_batch(batch)
            batch = self.read_batch(batch)
            src, segs, clss, mask_src, mask_cls, src_sent_labels, src_txt, tgt_txt = batch

            sent_scores, mask = self.AC(*batch)

            loss = self.loss_pg_fn(sent_scores, src_sent_labels.float())
            loss = (loss * mask.float()).sum()
            # norm = float(loss.numel())
            # loss.div(norm).backward()

            self.total_loss_epoch['pg'] += loss.item()
            self.last_loss['pg'] = loss.item()

            # Check for NaN
            if (loss != loss).data.any():
                print('NaN detected')
                exit()

            """
            Normalization on the loss (gradient accumulation or distributed training), followed by
            backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
            Also update the metrics for tensorboard.
            """
            if self.multi_gpu:
                loss = loss.mean()
            if self.grad_accm_steps > 1:
                loss = loss / self.grad_accm_steps

            loss.backward()

            self.iter('pg', (sent_scores, mask, src_txt))
            if self.n_iter['pg'] % self.grad_accm_steps == 0:
                clip_grad_norm_(self.AC.parameters(), self.max_grad_norm)
                self.optimAC.step()
                self.optimAC.zero_grad()
                self.schedulerAC.step()
                self.n_optim['pg'] += 1

            self.n_sequences_epoch['pg'] += src.size(1)

            pbar.update()
            pbar.set_postfix({
                # Last loss
                'Loss': f"{self.last_loss['pg']:.2f}",
                # Avgerage accumulated loss
                'Avg_loss': f"{self.total_loss_epoch['pg']/self.n_iter['pg']:.2f}",
                'Optim': self.n_optim['pg'],
                'Seq': self.n_sequences_epoch['pg'],
                })
        pbar.close()
        self.end_epoch('pg')

    def compress(self):
        """
        COMPRESS:
        This is when parameters learned from the active column get distilled (knowledge distillation)
        into the knowledge base (KB), with online EWC to minimize catastrophic forgetting.
        Parameters of the active column is frozen and distilled into the knowledge base with the equation:

        :math:
        E[KL(\pi_{k}(\dot|x)) || \pi^{KB}(\dot|x)] +
            \frac{1}{2} \lVert \theta^{KB} - \theta_{k-1}^{KB} \rVert_{\gamma F_{k-1}^{*}}^{2}

        where:
        ^{KB} denotes Knoledge Base
        x is the input
        \pi_{k}(\dot|x) is prediction of the active column after learning task k
        \pi^{KB}(\dot|x) is the prediction of the knowledge base
        E denotes expectation over the dataset under the active column
        \theta^{KB}
        \theta_{k-1}^{KB} is the latest MAP parameter on previous tasks (online EWC)
        \gamma < 1 is a hyperparameter that decays the Fisher Information matrices
            associated with the previous presentation of task i
        F_{k-1}^{*} is the running sum of the diagonal Fisher matrix on previous tasks (online EWC)

        Note:
        \pi_{k} is fized throughout the process to that learnt on task k
        """
        self.last_log = time.time()

        # since compress comes after progress, data generator is already used (exhausted).
        # so create a new generator
        self.data_generator = iter(self.data_loader())

        # reuse data loader after exhausting it
        if self.epoch['cp'] > 0:
            try:
                next(self.data_generator)
            except StopIteration:
                self.data_generator = iter(self.data_loader())

        pbar = tqdm(self.data_generator, total=self.dg_len, desc='==> Compress steps',
                    disable=self.local_rank not in [-1, 0])
        print('... (2) Compress training with {} data'.format(self.dg_len))
        for batch in pbar:
            batch = self.read_batch(batch)
            src, segs, clss, mask_src, mask_cls, src_sent_labels, src_txt, tgt_txt = batch

            """
            One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
            and possibly a parameter update (depending on the gradient accumulation).
            """
            # student
            s_sent_scores, s_mask = self.KB(*batch)
            # teacher
            with torch.no_grad():
                t_sent_scores, t_mask = (x.detach() for x in self.AC(*batch))

            assert s_sent_scores.size() == t_sent_scores.size()

            # print(s_sent_scores.size(), batch[5].size(), t_sent_scores.size())
            # KD Loss (batch[5] = src_sent_labels)
            loss_kd = self.loss_kd(s_sent_scores, batch[5], t_sent_scores)
            # EWC loss
            loss_ewc = self.KB.ewc_loss()
            loss = loss_kd + loss_ewc

            self.total_loss_epoch['cp'] += loss.item()
            self.last_loss['cp'] = loss.item()
            self.last_loss_kd = loss_kd.item()
            self.last_loss_ewc = loss_ewc.item()

            """
            Normalization on the loss (gradient accumulation or distributed training), followed by
            backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
            Also update the metrics for tensorboard.
            """
            # Check for NaN
            if (loss != loss).data.any():
                print("NaN detected")
                exit()

            if self.multi_gpu:
                loss = loss.mean()
            if self.grad_accm_steps > 1:
                loss = loss / self.grad_accm_steps

            loss.backward()

            self.iter('cp', (s_sent_scores, s_mask, src_txt))
            if self.n_iter['cp'] % self.grad_accm_steps == 0:
                clip_grad_norm_(self.KB.parameters(), self.max_grad_norm)
                self.optimKB.step()
                self.optimKB.zero_grad()
                self.schedulerKB.step()
                self.n_optim['cp'] += 1

            self.n_sequences_epoch['cp'] += src.size(1)

            pbar.update()
            pbar.set_postfix({
                # Last loss
                'Loss': f"{self.last_loss['cp']:.4f}",
                # Avgerage accumulated loss
                'Avg_loss': f"{self.total_loss_epoch['cp']/self.n_iter['cp']:.4f}",
                'Optim': self.n_optim['cp'],
                'Seq': self.n_sequences_epoch['cp'],
                })

        pbar.close()
        self.end_epoch('cp')

    def iter(self, type, pred_args=None):
        """ Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter[type] += 1
        self.n_total_iter[type] += 1

        if self.n_total_iter[type] % self.log_interval == 0:
            # print(self.n_total_iter[type], self.n_iter[type])
            self.log_tensorboard(type, pred_args)
            self.last_log = time.time()
        # if self.n_total_iter[type] % self.checkpoint_interval == 0:
        #     self.save_checkpoint(type)

    def run_pred(self, sent_scores, mask, src_txt):
        # print(src.size(), sent_scores.size())
        # print(src_sent_labels.size(), mask.size())
        sent_scores_t = sent_scores + mask.float()
        # print('sent scores', sent_scores)
        # print('mask', mask.float())
        # print('sent * mask', sent_scores_t)
        sent_scores_t = sent_scores_t.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores_t, 1) # sort sent scores from highest to lowest
        # print('sel ids', sent_scores_t, selected_ids)

        pred = []
        # print('src txt: {} ---\n'.format(len(src_txt[0])))
        # print(len(src_txt[0]))
        for i, idx in enumerate(selected_ids):
            _pred = []
            # print(i, idx)
            # print(src_txt[i])
            if len(src_txt[i]) == 0:
                continue
            # print(selected_ids[i][:len(src_txt[i])])
            for j in selected_ids[i][:len(src_txt[i])]:
                # print(i, j)
                if j >= len(src_txt[i]):
                    continue
                candidate = src_txt[i][j].strip()
                # if(self.args.block_trigram):
                if not _block_tri(candidate,_pred):
                    _pred.append(candidate)
                # else:
                #     _pred.append(candidate)

                # if not cal_oracle and not self.args.recall_eval and len(_pred) == 3:
                if len(_pred) == 10:
                    break

            _pred = '<q>'.join(_pred)
            # if self.args.recall_eval:
            #     _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])
            pred.append(_pred)

        return pred

    def log_tensorboard(self, type, pred_args=None):
        """ Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        global_steps = self.n_total_iter[type]

        if type == 'pg': # progress
            nps = self.AC.named_parameters()
            scheduler = self.schedulerAC
        else: # compress
            nps = self.KB.named_parameters()
            scheduler = self.schedulerKB

        for n, p in nps:
            self.tensorboard.add_scalar(tag=type+'_param_mean/'+n, scalar_value=p.data.mean(), global_step=global_steps)
            self.tensorboard.add_scalar(tag=type+'_param_std/'+n, scalar_value=p.data.std(), global_step=global_steps)
            if p.grad is None:
                continue
            self.tensorboard.add_scalar(tag=type+'_grad_mean/' + n, scalar_value=p.grad.data.mean(), global_step=global_steps)
            self.tensorboard.add_scalar(tag=type+'_grad_std/' + n, scalar_value=p.grad.data.std(), global_step=global_steps)

        if pred_args is not None:
            sent_scores, mask, src_txt = pred_args
            pred = ' \n'.join(self.run_pred(sent_scores, mask, src_txt))
            self.tensorboard.add_text(tag=type+'_pred/s{}'.format(global_steps), text_string=pred)

        self.tensorboard.add_scalar(
            tag=type+'/losses/cum_avg_loss_epoch',
            scalar_value=self.total_loss_epoch[type] / max(self.n_iter[type], 1),
            global_step=self.n_total_iter[type],
        )
        self.tensorboard.add_scalar(tag=type+'/losses/loss', scalar_value=self.last_loss[type], global_step=global_steps)
        if type == 'cp': # compress-only loss (CE and EWC)
            self.tensorboard.add_scalar(tag=type+'/losses/loss_kd', scalar_value=self.last_loss_kd, global_step=global_steps)
            self.tensorboard.add_scalar(tag=type+'/losses/loss_ewc', scalar_value=self.last_loss_ewc, global_step=global_steps)
        self.tensorboard.add_scalar(tag=type+'/learning_rate', scalar_value=scheduler.get_last_lr()[-1], global_step=global_steps)
        self.tensorboard.add_scalar(
            tag=type+'/global/memory_usage',
            scalar_value=psutil.virtual_memory()._asdict()['used'] / 1_000_000,
            global_step=global_steps,
        )
        self.tensorboard.add_scalar(tag=type+'/global/speed', scalar_value=time.time() - self.last_log, global_step=global_steps)

    def end_epoch(self, type):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logger and checkpoint saving.
        """
        print(f'{self.n_sequences_epoch[type]} sequences have been trained during epoch {self.epoch[type]}.')

        # print(type, self.total_loss_epoch, self.n_iter)
        n_it = self.n_iter[type]
        if self.is_master:
            ckpt_folder = 'model_task_{}_epoch_{}'.format(self.task_count, self.epoch[type])
            self.save_checkpoint(ckpt_folder=ckpt_folder)
            self.tensorboard.add_scalar(
                tag='epoch/{}/loss'.format(type),
                scalar_value=(self.total_loss_epoch[type] / n_it if n_it != 0 else 0),
                global_step=self.epoch[type]
            )

        self.epoch[type] += 1
        self.n_sequences_epoch[type] = 0
        self.n_iter[type] = 0
        self.total_loss_epoch[type] = 0

    def save_checkpoint(self, ckpt_folder):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return

        ckpt_dir = os.path.join(self.ckpt_dir, ckpt_folder)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        start = time.time()
        print('... (SC) Saving model checkpoint to {}'.format(ckpt_dir))
        for mname in ['KB', 'AC']:
            # mdl_to_save = self[mname].module if hasattr(self[mname], 'module') else self[mname]
            mdl_to_save = getattr(self, mname, False)
            mdl_ckpt_dir = os.path.join(ckpt_dir, mname)
            if not os.path.exists(mdl_ckpt_dir):
                os.makedirs(mdl_ckpt_dir)
            mdl_to_save.bert.save_pretrained(mdl_ckpt_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(mdl_ckpt_dir, 'model.pt'))

        end = time.time()
        print('... Ending (SC), time elapsed {}'.format(end-start))
