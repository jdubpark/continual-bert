import bisect
import gc
import glob
import os
import logging
import random

import torch


logger = logging.getLogger(__name__)


def load_tasks(data_dir, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        # logger.info('Loading %s dataset from %s, number of examples: %d' %
        #             (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    # pts_dir = os.path.join(data_dir, '{}.[0-9]*.pt'.format(corpus_type))
    pts_dir = os.path.join(data_dir, '[0-9]*.pt')
    pts = sorted(glob.glob(pts_dir))
    if pts:
        if shuffle:
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = os.path.join(data_dir, '0.pt')
        yield _lazy_dataset_loader(pt, corpus_type)


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]
            fname = [x[7] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = ~(src == 0) # inverse src
            mask_tgt = ~(tgt == 0)

            clss = torch.tensor(self._pad(pre_clss, -1))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'src', src)
            setattr(self, 'tgt', tgt)
            setattr(self, 'segs', segs)
            setattr(self, 'clss', clss)
            setattr(self, 'mask_src', mask_src)
            setattr(self, 'mask_tgt', mask_tgt)
            setattr(self, 'mask_cls', mask_cls)
            setattr(self, 'src_sent_labels', src_sent_labels)
            setattr(self, 'fname', fname)

            # if is_test:
            src_txt = [x[5] for x in data]
            setattr(self, 'src_txt', src_txt)
            tgt_txt = [x[6] for x in data]
            setattr(self, 'tgt_txt', tgt_txt)

    def __len__(self):
        return self.batch_size


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class TaskLoader(object):
    def __init__(self, tasks, batch_size, max_pos, max_tgt_len, shuffle=True, is_test=False):
        self.tasks = tasks
        self.batch_size = batch_size
        self.max_pos = max_pos
        self.max_tgt_len = max_tgt_len
        self.shuffle = shuffle
        self.is_test = is_test

        self.task_iter = (d for d in self.tasks)
        self.cur_task_iter = self._next_dataset_iterator()

        assert self.cur_task_iter is not None

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        while self.cur_task_iter is not None:
            for batch in self.cur_task_iter:
                yield batch
            self.cur_task_iter = self._next_dataset_iterator()

    def _next_dataset_iterator(self):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, 'cur_task'):
                self.cur_task = None
                gc.collect()
                del self.cur_task
                gc.collect()

            self.cur_task = next(self.task_iter)
        except StopIteration:
            return None

        return (
            # main, return as function to enable creating new iterator
            lambda: DataIterator(
                dataset=self.cur_task, batch_size=self.batch_size, max_pos=self.max_pos,
                max_tgt_len=self.max_tgt_len, shuffle=self.shuffle, is_test=self.is_test),
            # ewc
            DataIterator(
                dataset=self.cur_task, batch_size=1, max_pos=self.max_pos,
                max_tgt_len=self.max_tgt_len, shuffle=self.shuffle, is_test=self.is_test)
        ),


class DataIterator(object):
    def __init__(self, dataset, batch_size, max_pos, max_tgt_len, shuffle=True, is_test=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_pos = max_pos
        self.max_tgt_len = max_tgt_len
        self.shuffle = shuffle
        self.is_test = is_test

        self.sort_key = lambda x: len(x[1])
        self.batch_size_fn = simple_batch_size_fn

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self.batches = self.create_batches()
        for idx, minibatch in enumerate(self.batches):
            yield Batch(minibatch, self.is_test) # batch

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        return self.dataset

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['labels'] if 'labels' in ex else ex['src_sent_labels']
        segs = ex['segs']
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        fname = ex['fname']

        end_id = [src[-1]]
        src = src[:-1][:self.max_pos - 1] + end_id
        segs = segs[:self.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt, fname

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex['src']) == 0:
                continue
            ex = self.preprocess(ex, self.is_test)
            if ex is None:
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = list(self.batch(p_batch, self.batch_size))
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b
